import glob
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

from .batch import batch_compensate
from .calibration import (
    CalibrationModel,
    auto_build_calibration,
    build_calibration,
    quality_report,
    scan_calibration_candidates,
)
from .compensation import apply_compensation, write_comp_csv
from .io import read_pre_csv, select_segments
from .plot import local_enu, plot_2d_comparison, plot_calibration_quality, plot_compensation_timeseries


class MagHeadingApp(tk.Tk):
    """Desktop GUI wrapping the scan/calibrate/check/compensate pipeline.

    The main plot's SpanSelector mirrors the vendor LabVIEW tool's cursor-drag
    segment selection, but backed by the quantitative scan/quality_report
    machinery instead of eyeballing a curve.
    """

    def __init__(self, pre_path=None):
        super().__init__()
        self.title("MagArrow Heading Calibration")
        self.geometry("1200x800")

        self.df = None
        self.model = None
        self.start_idx = None
        self.end_idx = None
        self.segments = []

        self._build_layout()
        if pre_path:
            self._load_pre(pre_path)

    # ------------------------------------------------------------------ layout

    def _build_layout(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(top, text="Load -pre.csv...", command=self._on_load_clicked).pack(side=tk.LEFT)
        self.path_label = ttk.Label(top, text="(no file loaded)")
        self.path_label.pack(side=tk.LEFT, padx=8)

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_select = ttk.Frame(self.notebook)
        self.tab_quality = ttk.Frame(self.notebook)
        self.tab_2d = ttk.Frame(self.notebook)
        self.tab_batch = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_select, text="Segment Selection")
        self.notebook.add(self.tab_quality, text="Calibration Quality")
        self.notebook.add(self.tab_2d, text="2D Comparison")
        self.notebook.add(self.tab_batch, text="Batch")
        self._build_batch_tab(self.tab_batch)

        self.select_fig, (self.select_ax, self.path_ax) = plt.subplots(1, 2, figsize=(11, 4.2))
        self.select_canvas = FigureCanvasTkAgg(self.select_fig, master=self.tab_select)
        NavigationToolbar2Tk(self.select_canvas, self.tab_select).pack(side=tk.TOP, fill=tk.X)
        self.select_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.span_selector = None
        self._has_path = False
        self._path_x = self._path_y = None
        self._path_highlight = None
        self._path_segments_highlight = None
        self._segment_patches = []

        self.quality_canvas_widget = None
        self.plot2d_canvas_widget = None
        self.timeseries_canvas_widget = None

        # The control panel keeps growing new sections (Auto-build, multi-segment
        # accumulation, ...) and a fixed-height Frame silently clips whatever
        # doesn't fit -- with no scrollbar, buttons/entries below the clip line
        # are simply unreachable. A scrollable Canvas fixes this permanently
        # instead of needing another window-height bump every time a section
        # is added.
        right_container = ttk.Frame(body, width=340)
        right_container.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))
        right_container.pack_propagate(False)

        right_canvas = tk.Canvas(right_container, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_container, orient=tk.VERTICAL, command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")

        def _on_right_configure(_event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))

        right.bind("<Configure>", _on_right_configure)

        def _on_canvas_configure(event):
            right_canvas.itemconfig(right_window, width=event.width)

        right_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        self._build_controls(right)

        # Bind the wheel on the panel and each of its descendants rather than via
        # bind_all/<Enter>/<Leave>: Tk delivers <Leave> to a container as soon as
        # the pointer moves onto a child, so an unbind-on-Leave scheme switches
        # the wheel off over the Entries and Buttons that cover most of the panel
        # -- exactly where scrolling is needed. Widget-level binds also avoid
        # touching the matplotlib canvas's own wheel handling. Listboxes and Text
        # widgets are skipped so they keep scrolling their own contents.
        def _bind_wheel(widget):
            if not isinstance(widget, (tk.Listbox, tk.Text)):
                widget.bind("<MouseWheel>", _on_mousewheel)
            for child in widget.winfo_children():
                _bind_wheel(child)

        _bind_wheel(right_canvas)
        _bind_wheel(right)

        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH)
        ttk.Label(bottom, text="Log:").pack(anchor=tk.W, padx=6)
        self.log_text = tk.Text(bottom, height=6, state=tk.DISABLED)
        self.log_text.pack(fill=tk.X, padx=6, pady=(0, 6))

    def _build_controls(self, parent):
        sel = ttk.LabelFrame(parent, text="Selection")
        sel.pack(fill=tk.X, pady=4)
        self.selection_label = ttk.Label(sel, text="start=- end=- (drag on plot to select)")
        self.selection_label.pack(anchor=tk.W, padx=4, pady=4)

        ttk.Button(sel, text="Add segment", command=self._on_add_segment_clicked).pack(
            fill=tk.X, padx=4, pady=1
        )
        ttk.Button(sel, text="Remove selected segment", command=self._on_remove_segment_clicked).pack(
            fill=tk.X, padx=4, pady=1
        )
        ttk.Button(sel, text="Clear all segments", command=self._on_clear_segments_clicked).pack(
            fill=tk.X, padx=4, pady=1
        )

        self.segments_list = tk.Listbox(sel, height=4)
        self.segments_list.pack(fill=tk.X, padx=4, pady=(2, 2))

        self.center_segments_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            sel,
            text="center segments (recommended)",
            variable=self.center_segments_var,
        ).pack(anchor=tk.W, padx=4, pady=(0, 4))

        params = ttk.LabelFrame(parent, text="Model parameters")
        params.pack(fill=tk.X, pady=4)
        self.method_var = tk.StringVar(value="nearest")
        ttk.Label(params, text="method").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Combobox(params, textvariable=self.method_var, values=["nearest", "spline"],
                     state="readonly", width=10).grid(row=0, column=1, padx=4, pady=2)

        self.grid_res_var = tk.DoubleVar(value=5.0)
        ttk.Label(params, text="grid res (deg)").grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(params, textvariable=self.grid_res_var, width=10).grid(row=1, column=1, padx=4, pady=2)

        self.coverage_var = tk.DoubleVar(value=10.0)
        ttk.Label(params, text="coverage thresh (deg)").grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(params, textvariable=self.coverage_var, width=10).grid(row=2, column=1, padx=4, pady=2)

        self.max_var_var = tk.DoubleVar(value=2.0)
        ttk.Label(params, text="max variation (nT)").grid(row=3, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(params, textvariable=self.max_var_var, width=10).grid(row=3, column=1, padx=4, pady=2)

        self.trim_outliers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params, text="trim outliers", variable=self.trim_outliers_var).grid(
            row=4, column=0, sticky=tk.W, padx=4, pady=2
        )
        self.outlier_mad_var = tk.DoubleVar(value=3.0)
        ttk.Entry(params, textvariable=self.outlier_mad_var, width=10).grid(
            row=4, column=1, padx=4, pady=2
        )

        scan = ttk.LabelFrame(parent, text="Scan for candidates")
        scan.pack(fill=tk.X, pady=4)
        self.window_size_var = tk.IntVar(value=3000)
        ttk.Label(scan, text="window size").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(scan, textvariable=self.window_size_var, width=10).grid(row=0, column=1, padx=4, pady=2)
        ttk.Button(scan, text="Scan", command=self._on_scan_clicked).grid(row=1, column=0, columnspan=2, pady=4)
        self.candidates_list = tk.Listbox(scan, height=6)
        self.candidates_list.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        self.candidates_list.bind("<<ListboxSelect>>", self._on_candidate_selected)
        self._candidates = []

        auto = ttk.LabelFrame(parent, text="Auto-build")
        auto.pack(fill=tk.X, pady=4)
        self.auto_window_sizes_var = tk.StringVar(value="3000,1500,750,300")
        ttk.Label(auto, text="window sizes").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(auto, textvariable=self.auto_window_sizes_var, width=10).grid(row=0, column=1, padx=4, pady=2)
        self.auto_coverage_target_var = tk.DoubleVar(value=0.1)
        ttk.Label(auto, text="coverage target").grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(auto, textvariable=self.auto_coverage_target_var, width=10).grid(row=1, column=1, padx=4, pady=2)
        self.auto_max_segments_var = tk.IntVar(value=15)
        ttk.Label(auto, text="max segments").grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(auto, textvariable=self.auto_max_segments_var, width=10).grid(row=2, column=1, padx=4, pady=2)
        ttk.Button(auto, text="Auto-build calibration", command=self._on_auto_build_clicked).grid(
            row=3, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=(4, 2)
        )

        cal = ttk.LabelFrame(parent, text="Calibration")
        cal.pack(fill=tk.X, pady=4)
        ttk.Button(cal, text="Build calibration", command=self._on_build_clicked).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(cal, text="Save calibration...", command=self._on_save_cal_clicked).pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(cal, text="Load calibration...", command=self._on_load_cal_clicked).pack(fill=tk.X, padx=4, pady=2)
        self.report_text = tk.Text(cal, height=10, width=36, state=tk.DISABLED)
        self.report_text.pack(fill=tk.X, padx=4, pady=4)

        comp = ttk.LabelFrame(parent, text="Compensation")
        comp.pack(fill=tk.X, pady=4)
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(comp, text="overwrite Mag (else add MagComp)",
                         variable=self.overwrite_var).pack(anchor=tk.W, padx=4)
        ttk.Button(comp, text="Compensate & save...", command=self._on_compensate_clicked).pack(
            fill=tk.X, padx=4, pady=4
        )

    def _build_batch_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(top, text="Add files...", command=self._on_batch_add_files).pack(side=tk.LEFT)
        ttk.Button(top, text="Add folder...", command=self._on_batch_add_folder).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(top, text="Remove selected", command=self._on_batch_remove_selected).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(top, text="Clear", command=self._on_batch_clear).pack(side=tk.LEFT, padx=(6, 0))

        mid = ttk.Frame(parent)
        mid.pack(side=tk.TOP, fill=tk.X, padx=6)
        self.batch_files_list = tk.Listbox(mid, height=5, selectmode=tk.EXTENDED)
        self.batch_files_list.pack(fill=tk.X)

        run_row = ttk.Frame(parent)
        run_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Label(run_row, text="output dir (optional):").pack(side=tk.LEFT)
        self.batch_out_dir_var = tk.StringVar(value="")
        ttk.Entry(run_row, textvariable=self.batch_out_dir_var, width=40).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(run_row, text="Browse...", command=self._on_batch_browse_out_dir).pack(
            side=tk.LEFT
        )
        ttk.Button(run_row, text="Compensate All", command=self._on_batch_run_clicked).pack(
            side=tk.LEFT, padx=(12, 0)
        )

        columns = ("file", "n", "raw_noise", "comp_noise", "improve")
        self.batch_tree = ttk.Treeview(parent, columns=columns, show="headings", height=14)
        headings = {
            "file": "file",
            "n": "n samples",
            "raw_noise": "raw motion noise (nT)",
            "comp_noise": "compensated motion noise (nT)",
            "improve": "improvement",
        }
        for col in columns:
            self.batch_tree.heading(col, text=headings[col])
            self.batch_tree.column(col, width=160 if col == "file" else 130, anchor=tk.W)
        self.batch_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self.batch_summary_label = ttk.Label(parent, text="")
        self.batch_summary_label.pack(anchor=tk.W, padx=6, pady=(0, 6))

    # ------------------------------------------------------------------ logging

    def _log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _error(self, message):
        self._log(f"ERROR: {message}")
        messagebox.showerror("Error", message)

    # ------------------------------------------------------------------ data loading

    def _on_load_clicked(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self._load_pre(path)

    def _load_pre(self, path):
        try:
            self.df = read_pre_csv(path)
        except Exception as exc:
            self._error(f"failed to load {path}: {exc}")
            return
        self.path_label.config(text=path)
        self._log(f"loaded {path} ({len(self.df)} samples)")
        self._reset_for_new_file()
        self._draw_selection_plot()

    def _reset_for_new_file(self):
        """Scan candidates, selection, and any built model all refer to whatever
        file was loaded when they were computed. Without this, picking a stale
        candidate after loading a different file silently builds a "calibration"
        whose reported start/end indices belong to the new file but whose p2p in
        the list is still the old file's number -- a real bug that was making
        results look inconsistent (list said good, Build said bad, or vice versa).
        """
        self.start_idx = None
        self.end_idx = None
        self.model = None
        self._candidates = []
        self.candidates_list.delete(0, tk.END)
        self.segments = []
        self.segments_list.delete(0, tk.END)
        self._segment_patches = []
        self.selection_label.config(text="start=- end=- (drag on plot to select)")

        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.config(state=tk.DISABLED)

        if self.quality_canvas_widget is not None:
            self.quality_canvas_widget.destroy()
            self.quality_canvas_widget = None
        if self.plot2d_canvas_widget is not None:
            self.plot2d_canvas_widget.destroy()
            self.plot2d_canvas_widget = None
        if self.timeseries_canvas_widget is not None:
            self.timeseries_canvas_widget.destroy()
            self.timeseries_canvas_widget = None

    def _draw_selection_plot(self):
        self.select_ax.clear()
        idx = np.arange(len(self.df))
        self.select_ax.plot(idx, self.df["Mag"].to_numpy(), linewidth=0.8, color="tab:blue")
        self.select_ax.set_xlabel("sample index")
        self.select_ax.set_ylabel("Mag (nT)")
        self.select_ax.set_title("drag to select a calibration segment")

        self.span_selector = SpanSelector(
            self.select_ax,
            self._on_span_selected,
            "horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.2, facecolor="tab:orange"),
        )

        self._segment_patches = []
        self._add_segment_overlay_patches()

        self.path_ax.clear()
        self._path_highlight = None
        self._path_segments_highlight = None
        self._has_path = "Latitude" in self.df.columns and "Longitude" in self.df.columns
        if self._has_path:
            self._path_x, self._path_y = local_enu(self.df["Latitude"], self.df["Longitude"])
            self.path_ax.plot(self._path_x, self._path_y, "-", color="lightgray",
                               linewidth=0.8, zorder=1)
            (self._path_segments_highlight,) = self.path_ax.plot([], [], ".", color="tab:green",
                                                                   markersize=3, zorder=2)
            (self._path_highlight,) = self.path_ax.plot([], [], ".", color="tab:red",
                                                          markersize=3, zorder=3)
            self.path_ax.set_xlabel("East (m)")
            self.path_ax.set_ylabel("North (m)")
            self.path_ax.set_title("flight path (pending=red, added segments=green)")
            self.path_ax.set_aspect("equal")
        else:
            self._path_x = self._path_y = None
            self.path_ax.text(0.5, 0.5, "no Latitude/Longitude columns",
                               ha="center", va="center", transform=self.path_ax.transAxes)
            self.path_ax.set_xticks([])
            self.path_ax.set_yticks([])

        self._update_path_highlight()
        self.select_canvas.draw_idle()

    def _update_path_highlight(self):
        if not self._has_path:
            return
        if self._path_highlight is not None:
            if self.start_idx is None or self.end_idx is None:
                self._path_highlight.set_data([], [])
            else:
                self._path_highlight.set_data(
                    self._path_x[self.start_idx : self.end_idx + 1],
                    self._path_y[self.start_idx : self.end_idx + 1],
                )
        if self._path_segments_highlight is not None:
            if self.segments:
                idx = np.concatenate([np.arange(s, e + 1) for s, e in self.segments])
                self._path_segments_highlight.set_data(self._path_x[idx], self._path_y[idx])
            else:
                self._path_segments_highlight.set_data([], [])

    def _add_segment_overlay_patches(self):
        for start, end in self.segments:
            patch = self.select_ax.axvspan(start, end, color="tab:green", alpha=0.25, zorder=0)
            self._segment_patches.append(patch)

    def _refresh_segment_overlays(self):
        for patch in self._segment_patches:
            patch.remove()
        self._segment_patches = []
        self._add_segment_overlay_patches()
        self._update_path_highlight()
        self.select_canvas.draw_idle()

    # ------------------------------------------------------------------ multi-segment accumulation

    def _on_add_segment_clicked(self):
        if self.df is None:
            self._error("load a -pre.csv file first")
            return
        if self.start_idx is None or self.end_idx is None:
            self._error("drag on the plot (or pick a scan candidate) to select a segment "
                         "before adding it")
            return
        self.segments.append((self.start_idx, self.end_idx))
        self.segments_list.insert(
            tk.END, f"{self.start_idx}-{self.end_idx} ({self.end_idx - self.start_idx + 1} samples)"
        )
        self._log(f"added segment {self.start_idx}-{self.end_idx} "
                  f"({len(self.segments)} segment(s) total)")
        self.start_idx = None
        self.end_idx = None
        self.selection_label.config(text="start=- end=- (drag on plot to select)")
        try:
            self.span_selector.extents = (0, 0)
        except Exception:
            pass
        self._refresh_segment_overlays()

    def _on_remove_segment_clicked(self):
        for i in reversed(self.segments_list.curselection()):
            self.segments_list.delete(i)
            del self.segments[i]
        self._refresh_segment_overlays()

    def _on_clear_segments_clicked(self):
        self.segments_list.delete(0, tk.END)
        self.segments = []
        self._refresh_segment_overlays()

    def _on_span_selected(self, xmin, xmax):
        n = len(self.df)
        self.start_idx = max(0, int(round(xmin)))
        self.end_idx = min(n - 1, int(round(xmax)))
        if self.end_idx <= self.start_idx:
            return
        self.selection_label.config(text=f"start={self.start_idx} end={self.end_idx} "
                                          f"({self.end_idx - self.start_idx + 1} samples)")
        self._update_path_highlight()
        self.select_canvas.draw_idle()

    # ------------------------------------------------------------------ scan

    def _on_scan_clicked(self):
        if self.df is None:
            self._error("load a -pre.csv file first")
            return
        # Reading a Tk numeric var raises TclError when its entry holds text, and
        # the GUI normally runs under pythonw (no console) via the .bat launcher,
        # so an escaping exception would make the button look simply dead.
        try:
            window_size = self.window_size_var.get()
        except tk.TclError:
            self._error("window size must be a whole number of samples")
            return
        if window_size > len(self.df):
            self._error(f"window size {window_size} exceeds this file's {len(self.df)} "
                         f"samples; try {max(2, len(self.df) // 4)} or smaller")
            return
        self._log(f"scanning window_size={window_size} ...")
        self.update_idletasks()
        try:
            self._candidates = scan_calibration_candidates(
                self.df,
                window_size=window_size,
                method=self.method_var.get(),
                coverage_threshold_deg=self.coverage_var.get(),
                grid_resolution_deg=self.grid_res_var.get(),
                trim_outliers=self.trim_outliers_var.get(),
                outlier_mad_threshold=self.outlier_mad_var.get(),
                max_variation=self.max_var_var.get(),
                top_n=15,
            )
        except Exception as exc:
            self._error(f"scan failed: {exc}")
            return

        self.candidates_list.delete(0, tk.END)
        for c in self._candidates:
            dom = "dom-ok" if c["dominant_heading_covered"] else "dom-MISS"
            label = (f"{c['start']:>6}-{c['end']:<6} p2p={c['self_compensation_p2p_nT']:.3g}nT "
                     f"uncov={c['survey_uncovered_fraction']:.1%} {dom}")
            self.candidates_list.insert(tk.END, label)
        self._log(f"found {len(self._candidates)} candidates")

    def _on_candidate_selected(self, _event):
        selection = self.candidates_list.curselection()
        if not selection:
            return
        c = self._candidates[selection[0]]
        self.start_idx, self.end_idx = c["start"], c["end"]
        self.selection_label.config(
            text=f"start={self.start_idx} end={self.end_idx} ({self.end_idx - self.start_idx + 1} samples)"
        )
        self.span_selector.extents = (self.start_idx, self.end_idx)
        self._update_path_highlight()
        self.select_canvas.draw_idle()

    # ------------------------------------------------------------------ auto-build

    def _on_auto_build_clicked(self):
        if self.df is None:
            self._error("load a -pre.csv file first")
            return
        try:
            window_sizes = [int(x.strip()) for x in self.auto_window_sizes_var.get().split(",")
                             if x.strip()]
            if not window_sizes:
                raise ValueError("window sizes list is empty")
        except ValueError as exc:
            self._error(f"invalid window sizes ('{self.auto_window_sizes_var.get()}'): {exc}")
            return
        except tk.TclError:
            self._error("window sizes must be whole numbers separated by commas")
            return

        self._log(f"auto-build: trying window sizes {window_sizes} (widest first) ...")
        self.update_idletasks()
        try:
            result = auto_build_calibration(
                self.df,
                window_size=window_sizes,
                method=self.method_var.get(),
                coverage_threshold_deg=self.coverage_var.get(),
                grid_resolution_deg=self.grid_res_var.get(),
                max_variation=self.max_var_var.get(),
                outlier_mad_threshold=self.outlier_mad_var.get(),
                max_candidates=self.auto_max_segments_var.get(),
                coverage_target=self.auto_coverage_target_var.get(),
            )
        except tk.TclError:
            self._error("coverage target / max segments / model parameters must be numbers")
            return
        except Exception as exc:
            self._error(f"auto-build failed: {exc}")
            return

        for attempt in result["attempts"]:
            status = "PASSED" if attempt["passed"] else "did not pass"
            n_accepted = sum(1 for e in attempt["log"] if e["accepted"])
            n_rejected = len(attempt["log"]) - n_accepted
            self._log(f"  window_size={attempt['window_size']}: {status} "
                      f"({n_accepted} accepted, {n_rejected} rejected)")

        if not result["segments"] or result["model"] is None:
            self._error("auto-build could not assemble any passing calibration at the given "
                         "window sizes / max variation. Try picking segments manually instead.")
            return

        # populate the accumulated-segments UI exactly as if these had been picked
        # and added by hand, so the result is visible and still editable
        self.segments = list(result["segments"])
        self.segments_list.delete(0, tk.END)
        for s, e in self.segments:
            self.segments_list.insert(tk.END, f"{s}-{e} ({e - s + 1} samples)")
        self.start_idx = None
        self.end_idx = None
        self.selection_label.config(text="start=- end=- (drag on plot to select)")
        self._refresh_segment_overlays()

        self.model = result["model"]
        self._show_report(result["report"])
        self._update_quality_tab()

        if result["passed"]:
            self._log(f"auto-build PASSED with window_size={result['window_size']}, "
                      f"{len(self.segments)} segment(s) -- opening save dialog")
            self._on_save_cal_clicked()
        else:
            self._log(f"auto-build did NOT pass (best-effort shown: window_size="
                      f"{result['window_size']}, {len(self.segments)} segment(s))")
            messagebox.showwarning(
                "Auto-build did not pass",
                "The best-effort auto-build result did not meet the pass criteria "
                "(self-compensation p2p / dominant heading coverage / coverage target).\n\n"
                "Review the quality report on the right and the Calibration Quality tab "
                "before deciding whether to save it manually, or pick segments yourself.",
            )

    # ------------------------------------------------------------------ calibration

    def _on_build_clicked(self):
        if self.df is None:
            self._error("load a -pre.csv file first")
            return
        if self.segments:
            ranges = list(self.segments)
        elif self.start_idx is not None and self.end_idx is not None:
            ranges = [(self.start_idx, self.end_idx)]
        else:
            self._error("select a segment first (drag on the plot, or pick a scan candidate), "
                         "or add one or more segments with 'Add segment'")
            return

        try:
            segment = select_segments(
                self.df, ranges, center_segments=self.center_segments_var.get()
            )
            self.model = build_calibration(
                segment,
                method=self.method_var.get(),
                coverage_threshold_deg=self.coverage_var.get(),
                grid_resolution_deg=self.grid_res_var.get(),
                trim_outliers=self.trim_outliers_var.get(),
                outlier_mad_threshold=self.outlier_mad_var.get(),
                metadata={"segments": [f"{s}-{e}" for s, e in ranges]},
            )
            report = quality_report(self.model, max_variation=self.max_var_var.get(), survey_df=self.df)
        except Exception as exc:
            self._error(f"calibration failed: {exc}")
            return

        self._show_report(report)
        seg_desc = ", ".join(f"{s}-{e}" for s, e in ranges)
        self._log(f"built calibration model from segment(s) {seg_desc}")
        offsets = segment.attrs.get("segment_baseline_offsets_nT")
        if offsets:
            offsets_str = ", ".join(f"{o:+.2f}" for o in offsets)
            self._log(f"  segment baseline offsets applied (nT): {offsets_str}")
        self._update_quality_tab()

    def _show_report(self, report):
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        for key, value in report.items():
            if isinstance(value, float):
                self.report_text.insert(tk.END, f"{key}: {value:.4g}\n")
            else:
                self.report_text.insert(tk.END, f"{key}: {value}\n")
        self.report_text.config(state=tk.DISABLED)

    def _update_quality_tab(self):
        if self.quality_canvas_widget is not None:
            self.quality_canvas_widget.destroy()
        fig = plot_calibration_quality(self.model, title="self-compensation")
        canvas = FigureCanvasTkAgg(fig, master=self.tab_quality)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.quality_canvas_widget = canvas.get_tk_widget()

    def _on_save_cal_clicked(self):
        if self.model is None:
            self._error("build or load a calibration model first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("npz", "*.npz")])
        if not path:
            return
        self.model.save(path)
        self._log(f"calibration model saved to {path}")

    def _on_load_cal_clicked(self):
        path = filedialog.askopenfilename(filetypes=[("npz", "*.npz"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.model = CalibrationModel.load(path)
        except Exception as exc:
            self._error(f"failed to load {path}: {exc}")
            return
        self._log(f"calibration model loaded from {path}")
        report = quality_report(
            self.model, max_variation=self.max_var_var.get(),
            survey_df=self.df if self.df is not None else None,
        )
        self._show_report(report)
        self._update_quality_tab()

    # ------------------------------------------------------------------ compensation

    def _on_compensate_clicked(self):
        if self.df is None:
            self._error("load a -pre.csv file first")
            return
        if self.model is None:
            self._error("build or load a calibration model first")
            return

        out_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out_path:
            return
        try:
            compensated = apply_compensation(self.df, self.model, keep_raw=not self.overwrite_var.get())
            write_comp_csv(compensated, self.path_label.cget("text"), out_path=out_path)
        except Exception as exc:
            self._error(f"compensation failed: {exc}")
            return
        self._log(f"compensated survey written to {out_path} ({len(compensated)} samples)")
        self._update_2d_tab(compensated)

    def _update_2d_tab(self, compensated):
        if self.plot2d_canvas_widget is not None:
            self.plot2d_canvas_widget.destroy()
            self.plot2d_canvas_widget = None
        if self.timeseries_canvas_widget is not None:
            self.timeseries_canvas_widget.destroy()
            self.timeseries_canvas_widget = None

        # Mag was overwritten in place when --overwrite is on; there's no separate
        # raw column left to compare against, so only plot the one series.
        comp_col = None if self.overwrite_var.get() else "MagComp"

        try:
            ts_fig = plot_compensation_timeseries(compensated, raw_col="Mag", comp_col=comp_col)
            ts_canvas = FigureCanvasTkAgg(ts_fig, master=self.tab_2d)
            ts_canvas.draw()
            ts_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.timeseries_canvas_widget = ts_canvas.get_tk_widget()
        except Exception as exc:
            self._log(f"time series plot skipped: {exc}")

        if "Latitude" not in compensated.columns or "Longitude" not in compensated.columns:
            self._log("no Latitude/Longitude columns; skipping 2D map")
        else:
            try:
                fig = plot_2d_comparison(compensated, raw_col="Mag", comp_col=comp_col)
                canvas = FigureCanvasTkAgg(fig, master=self.tab_2d)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                self.plot2d_canvas_widget = canvas.get_tk_widget()
            except Exception as exc:
                self._log(f"2D plot skipped: {exc}")

        self.notebook.select(self.tab_2d)

    # ------------------------------------------------------------------ batch

    def _on_batch_add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        for p in paths:
            self.batch_files_list.insert(tk.END, p)

    def _on_batch_add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        found = sorted(glob.glob(os.path.join(folder, "*-pre.csv")))
        if not found:
            self._error(f"no *-pre.csv files found in {folder}")
            return
        for p in found:
            self.batch_files_list.insert(tk.END, p)
        self._log(f"added {len(found)} file(s) from {folder}")

    def _on_batch_remove_selected(self):
        for i in reversed(self.batch_files_list.curselection()):
            self.batch_files_list.delete(i)

    def _on_batch_clear(self):
        self.batch_files_list.delete(0, tk.END)
        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)
        self.batch_summary_label.config(text="")

    def _on_batch_browse_out_dir(self):
        folder = filedialog.askdirectory()
        if folder:
            self.batch_out_dir_var.set(folder)

    def _on_batch_run_clicked(self):
        if self.model is None:
            self._error("build or load a calibration model first")
            return
        paths = list(self.batch_files_list.get(0, tk.END))
        if not paths:
            self._error("add at least one -pre.csv file first")
            return

        out_dir = self.batch_out_dir_var.get().strip() or None
        self._log(f"batch-compensating {len(paths)} file(s) ...")
        self.update_idletasks()
        try:
            results = batch_compensate(
                paths, self.model, keep_raw=not self.overwrite_var.get(), out_dir=out_dir
            )
        except Exception as exc:
            self._error(f"batch compensate failed: {exc}")
            return

        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)
        for r in results:
            self.batch_tree.insert(
                "",
                tk.END,
                values=(
                    os.path.basename(r["path"]),
                    r["n_samples"],
                    f"{r['raw_motion_noise_std_nT']:.4g}",
                    f"{r['comp_motion_noise_std_nT']:.4g}",
                    f"{r['motion_noise_improvement_pct']:.1f}%",
                ),
            )

        improvements = [r["motion_noise_improvement_pct"] for r in results]
        mean_imp = sum(improvements) / len(improvements)
        poor = [os.path.basename(r["path"]) for r in results if r["motion_noise_improvement_pct"] < 5.0]
        summary = f"mean improvement: {mean_imp:.1f}% (min {min(improvements):.1f}%, max {max(improvements):.1f}%)"
        if poor:
            summary += f"  |  low improvement, consider rebuilding calibration: {', '.join(poor)}"
        self.batch_summary_label.config(text=summary)
        self._log(f"batch compensate done: {summary}")


def main(pre_path=None):
    app = MagHeadingApp(pre_path=pre_path)
    app.mainloop()


if __name__ == "__main__":
    main()
