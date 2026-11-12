import glob
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

from .batch import batch_compensate, noise_breakdown, turn_line_masks
from .calibration import (
    CalibrationModel,
    auto_build_calibration,
    build_calibration,
    quality_report,
    scan_calibration_candidates,
)
from .compensation import apply_compensation, infer_sample_rate_hz, write_comp_csv
from .io import comp_output_path, read_pre_csv, select_segments
from .plot import (
    local_enu,
    plot_2d_comparison,
    plot_calibration_quality,
    plot_compensation_timeseries,
    plot_heading_coverage,
)
from .survey_leveling import level_survey_files

DEFAULT_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".mag_heading_gui.json")

# Friendlier labels for the quality report; keys not listed are shown as-is.
REPORT_LABELS = {
    "method": "method",
    "n_calibration_samples": "calibration samples",
    "n_grid_bins": "grid bins",
    "grid_resolution_deg": "grid resolution (deg)",
    "median_samples_per_bin": "median samples / bin",
    "bins_with_single_sample_fraction": "single-sample bins",
    "self_compensation_p2p_nT": "self-comp p2p (nT)",
    "self_compensation_std_nT": "self-comp std (nT)",
    "max_variation_nT": "max variation (nT)",
    "self_compensation_pass": "self-comp pass",
    "survey_heading_coverage_threshold_deg": "coverage threshold (deg)",
    "survey_uncovered_fraction": "survey uncovered",
    "survey_chord_distance_median": "chord dist median",
    "survey_chord_distance_p95": "chord dist p95",
    "survey_chord_distance_max": "chord dist max",
    "survey_dominant_heading_deg": "dominant heading (polar, az)",
    "survey_dominant_heading_fraction": "dominant heading share",
    "survey_dominant_heading_covered": "dominant heading covered",
}
FRACTION_KEYS = {"bins_with_single_sample_fraction", "survey_uncovered_fraction",
                 "survey_dominant_heading_fraction"}


def find_pre_files(folder):
    """Survey files in `folder`: CSVs with '-pre' in the name (vendor Module 1
    output, e.g. '...-pre.csv' or '...-pre-LPF.csv'), excluding this tool's own
    '-comp' / '-leveled' outputs so a re-run never compensates twice."""
    found = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        name = os.path.basename(path).lower()
        if "-pre" in name and "-comp" not in name and "-leveled" not in name:
            found.append(path)
    return found


def _fmt_pct(value):
    return "-" if value is None else f"{value:+.1f}%"


def evaluate_report(report, max_variation, coverage_target):
    """PASS/FAIL checks shown at the top of the quality report: the same three
    criteria auto-build stops on (self-comp p2p, dominant heading covered,
    survey uncovered fraction). Returns a list of (label, passed, detail);
    checks that need a loaded survey are skipped when the report has none.
    """
    checks = []
    p2p = report.get("self_compensation_p2p_nT")
    if p2p is not None and max_variation is not None:
        checks.append(("self-comp p2p", bool(p2p <= max_variation),
                       f"{p2p:.3g} nT (limit {max_variation:g})"))
    if "survey_dominant_heading_covered" in report:
        checks.append(("dominant heading", bool(report["survey_dominant_heading_covered"]),
                       f"{report['survey_dominant_heading_deg']} "
                       f"({report['survey_dominant_heading_fraction']:.0%} of survey)"))
    if "survey_uncovered_fraction" in report and coverage_target is not None:
        unc = report["survey_uncovered_fraction"]
        checks.append(("survey coverage", bool(unc <= coverage_target),
                       f"{unc:.1%} uncovered (target {coverage_target:.0%})"))
    return checks


class MagHeadingApp(tk.Tk):
    """Desktop GUI wrapping the scan/calibrate/check/compensate/level pipeline.

    The main plot's SpanSelector mirrors the vendor LabVIEW tool's cursor-drag
    segment selection, but backed by the quantitative scan/quality_report
    machinery instead of eyeballing a curve.

    Slow steps (scan, auto-build, compensation preview, batch, leveling) run on a
    worker thread so the window stays responsive; `sync_tasks=True` runs them
    inline instead (used by the tests).
    """

    def __init__(self, pre_path=None, settings_path=DEFAULT_SETTINGS_PATH, sync_tasks=False):
        super().__init__()
        self.title("MagArrow Heading Calibration")
        self.geometry("1400x900")
        self.minsize(1000, 650)

        self.df = None
        self.pre_path = None
        self.model = None
        self.model_source = None
        self.start_idx = None
        self.end_idx = None
        self.segments = []
        self.compensated = None

        self.settings_path = settings_path
        self.sync_tasks = sync_tasks
        self._busy = None
        self._task_queue = queue.Queue()
        self._last_dir = None

        self._build_layout()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if pre_path:
            self._load_pre(pre_path)

    # ------------------------------------------------------------------ layout

    def _build_layout(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(top, text="Load -pre.csv...", command=self._on_load_clicked).pack(side=tk.LEFT)
        self.path_label = ttk.Label(top, text="(no file loaded)")
        self.path_label.pack(side=tk.LEFT, padx=8)

        self.progress = ttk.Progressbar(top, mode="indeterminate", length=160)
        self.progress.pack(side=tk.RIGHT)
        self.status_label = ttk.Label(top, text="ready", foreground="gray30")
        self.status_label.pack(side=tk.RIGHT, padx=8)
        self.model_label = ttk.Label(top, text="calibration: none", foreground="gray30")
        self.model_label.pack(side=tk.RIGHT, padx=12)

        # Pack order matters: Tk's packer hands out space in packing order, so the
        # log (bottom) and control panel (right) are packed BEFORE the expanding
        # notebook. Packed after it, a large figure in any tab claimed all the room
        # and pushed the panel off the right edge and the log off the bottom.
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bottom, text="Log:").pack(anchor=tk.W, padx=6)
        log_frame = ttk.Frame(bottom)
        log_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.log_text = tk.Text(log_frame, height=7, state=tk.DISABLED)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.log_text.tag_configure("error", foreground="firebrick")
        self.log_text.tag_configure("warn", foreground="darkorange3")

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        # The control panel keeps growing new sections and a fixed-height Frame
        # silently clips whatever doesn't fit, so it lives in a scrollable Canvas.
        right_container = ttk.Frame(body, width=370)
        right_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        right_container.pack_propagate(False)

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
        self.notebook.add(self.tab_2d, text="Compensation Result")
        self.notebook.add(self.tab_batch, text="Batch / Leveling")
        self._build_quality_tab(self.tab_quality)
        self._build_2d_tab(self.tab_2d)
        self._build_batch_tab(self.tab_batch)

        self.select_fig, (self.select_ax, self.path_ax) = plt.subplots(
            1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]}
        )
        plt.close(self.select_fig)  # embedded below; keep it out of pyplot's registry
        self.select_canvas = FigureCanvasTkAgg(self.select_fig, master=self.tab_select)
        NavigationToolbar2Tk(self.select_canvas, self.tab_select).pack(side=tk.TOP, fill=tk.X)
        self.select_canvas.get_tk_widget().configure(width=300, height=200)
        self.select_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.span_selector = None
        self._has_path = False
        self._path_x = self._path_y = None
        self._path_highlight = None
        self._path_segments_highlight = None
        self._segment_patches = []

        right_canvas = tk.Canvas(right_container, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_container, orient=tk.VERTICAL, command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(right_canvas)
        right_window = right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda _e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>", lambda e: right_canvas.itemconfig(right_window, width=e.width))

        def _on_mousewheel(event):
            right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        self._build_controls(right)

        # Bind the wheel on the panel and each of its descendants rather than via
        # bind_all/<Enter>/<Leave>: Tk delivers <Leave> to a container as soon as
        # the pointer moves onto a child, so an unbind-on-Leave scheme switches
        # the wheel off over the Entries and Buttons that cover most of the panel.
        # Listboxes and Text widgets keep scrolling their own contents.
        def _bind_wheel(widget):
            if not isinstance(widget, (tk.Listbox, tk.Text)):
                widget.bind("<MouseWheel>", _on_mousewheel)
            for child in widget.winfo_children():
                _bind_wheel(child)

        _bind_wheel(right_canvas)
        _bind_wheel(right)

    def _labeled_entry(self, parent, row, label, var, width=12):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky=tk.EW, padx=4, pady=2)
        return entry

    def _build_controls(self, parent):
        sel = ttk.LabelFrame(parent, text="1. Segment selection")
        sel.pack(fill=tk.X, pady=4)
        self.selection_label = ttk.Label(sel, text="start=- end=- (drag on plot to select)")
        self.selection_label.pack(anchor=tk.W, padx=4, pady=4)
        row = ttk.Frame(sel)
        row.pack(fill=tk.X, padx=4, pady=1)
        ttk.Button(row, text="Add segment", command=self._on_add_segment_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Remove", command=self._on_remove_segment_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row, text="Clear", command=self._on_clear_segments_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        self.segments_list = tk.Listbox(sel, height=4)
        self.segments_list.pack(fill=tk.X, padx=4, pady=(2, 2))
        self.center_segments_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sel, text="center segments (recommended)",
                        variable=self.center_segments_var).pack(anchor=tk.W, padx=4, pady=(0, 4))

        params = ttk.LabelFrame(parent, text="Model parameters")
        params.pack(fill=tk.X, pady=4)
        params.columnconfigure(1, weight=1)
        self.method_var = tk.StringVar(value="nearest")
        ttk.Label(params, text="method").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        ttk.Combobox(params, textvariable=self.method_var, values=["nearest", "spline"],
                     state="readonly", width=10).grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        self.grid_res_var = tk.DoubleVar(value=5.0)
        self._labeled_entry(params, 1, "grid res (deg)", self.grid_res_var)
        self.coverage_var = tk.DoubleVar(value=10.0)
        self._labeled_entry(params, 2, "coverage thresh (deg)", self.coverage_var)
        self.max_var_var = tk.DoubleVar(value=2.0)
        self._labeled_entry(params, 3, "max variation (nT)", self.max_var_var)
        self.trim_outliers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params, text="trim outliers (MAD)", variable=self.trim_outliers_var).grid(
            row=4, column=0, sticky=tk.W, padx=4, pady=2)
        self.outlier_mad_var = tk.DoubleVar(value=3.0)
        ttk.Entry(params, textvariable=self.outlier_mad_var, width=12).grid(
            row=4, column=1, sticky=tk.EW, padx=4, pady=2)

        scan = ttk.LabelFrame(parent, text="Scan for candidates")
        scan.pack(fill=tk.X, pady=4)
        scan.columnconfigure(1, weight=1)
        self.window_size_var = tk.IntVar(value=3000)
        self._labeled_entry(scan, 0, "window size (samples)", self.window_size_var)
        ttk.Button(scan, text="Scan", command=self._on_scan_clicked).grid(
            row=1, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=4)
        self.candidates_list = tk.Listbox(scan, height=6, font=("Consolas", 9))
        self.candidates_list.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=2)
        self.candidates_list.bind("<<ListboxSelect>>", self._on_candidate_selected)
        self._candidates = []

        auto = ttk.LabelFrame(parent, text="Auto-build")
        auto.pack(fill=tk.X, pady=4)
        auto.columnconfigure(1, weight=1)
        self.auto_window_sizes_var = tk.StringVar(value="3000,1500,750,300")
        self._labeled_entry(auto, 0, "window sizes", self.auto_window_sizes_var, width=18)
        self.auto_coverage_target_var = tk.DoubleVar(value=0.1)
        self._labeled_entry(auto, 1, "coverage target (0-1)", self.auto_coverage_target_var)
        self.auto_max_segments_var = tk.IntVar(value=15)
        self._labeled_entry(auto, 2, "max segments", self.auto_max_segments_var)
        ttk.Button(auto, text="Auto-build calibration", command=self._on_auto_build_clicked).grid(
            row=3, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=(4, 2))

        cal = ttk.LabelFrame(parent, text="2. Calibration")
        cal.pack(fill=tk.X, pady=4)
        ttk.Button(cal, text="Build calibration", command=self._on_build_clicked).pack(
            fill=tk.X, padx=4, pady=2)
        row = ttk.Frame(cal)
        row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Button(row, text="Save...", command=self._on_save_cal_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Load...", command=self._on_load_cal_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(row, text="Export vendor -cal.csv...", command=self._on_export_vendor_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        self.report_text = tk.Text(cal, height=16, width=40, state=tk.DISABLED, font=("Consolas", 9))
        self.report_text.pack(fill=tk.X, padx=4, pady=4)
        self.report_text.tag_configure("pass", foreground="forest green", font=("Consolas", 9, "bold"))
        self.report_text.tag_configure("fail", foreground="firebrick", font=("Consolas", 9, "bold"))
        self.report_text.tag_configure("dim", foreground="gray40")

        comp = ttk.LabelFrame(parent, text="3. Compensation")
        comp.pack(fill=tk.X, pady=4)
        comp.columnconfigure(1, weight=1)
        self.smooth_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(comp, text="smooth correction, Hz", variable=self.smooth_enabled_var).grid(
            row=0, column=0, sticky=tk.W, padx=4, pady=2)
        self.smooth_hz_var = tk.DoubleVar(value=1.0)
        ttk.Entry(comp, textvariable=self.smooth_hz_var, width=12).grid(
            row=0, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Label(comp, text="low-passes the correction so it doesn't add\n"
                             "compass noise on straight lines (1 Hz: M350)",
                  foreground="gray40").grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=4)
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(comp, text="overwrite Mag (else add MagComp)",
                        variable=self.overwrite_var).grid(row=2, column=0, columnspan=2,
                                                          sticky=tk.W, padx=4, pady=2)
        row = ttk.Frame(comp)
        row.grid(row=3, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=4)
        ttk.Button(row, text="Preview", command=self._on_preview_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Compensate && save...", command=self._on_compensate_clicked).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

    def _build_quality_tab(self, parent):
        self.quality_summary = tk.Text(parent, height=2, state=tk.DISABLED, relief=tk.FLAT,
                                       background=self.cget("background"), font=("Segoe UI", 10))
        self.quality_summary.tag_configure("pass", foreground="forest green", font=("Segoe UI", 10, "bold"))
        self.quality_summary.tag_configure("fail", foreground="firebrick", font=("Segoe UI", 10, "bold"))
        self.quality_summary.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        self.quality_panes = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        self.quality_panes.pack(fill=tk.BOTH, expand=True)
        self._quality_frames = []

    def _build_2d_tab(self, parent):
        self.result_summary = ttk.Label(parent, text="Preview or compensate to see the result here.",
                                        font=("Segoe UI", 10), justify=tk.LEFT)
        self.result_summary.pack(side=tk.TOP, anchor=tk.W, padx=6, pady=4)
        self.result_panes = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        self.result_panes.pack(fill=tk.BOTH, expand=True)
        self._result_frames = []

    def _build_batch_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(top, text="Add files...", command=self._on_batch_add_files).pack(side=tk.LEFT)
        ttk.Button(top, text="Add folder...", command=self._on_batch_add_folder).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Remove selected", command=self._on_batch_remove_selected).pack(
            side=tk.LEFT, padx=(6, 0))
        ttk.Button(top, text="Clear", command=self._on_batch_clear).pack(side=tk.LEFT, padx=(6, 0))
        self.batch_count_label = ttk.Label(top, text="0 file(s)")
        self.batch_count_label.pack(side=tk.LEFT, padx=12)

        mid = ttk.Frame(parent)
        mid.pack(side=tk.TOP, fill=tk.X, padx=6)
        self.batch_files_list = tk.Listbox(mid, height=5, selectmode=tk.EXTENDED)
        files_scroll = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.batch_files_list.yview)
        self.batch_files_list.configure(yscrollcommand=files_scroll.set)
        files_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.batch_files_list.pack(fill=tk.X)

        run_row = ttk.Frame(parent)
        run_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Label(run_row, text="output dir (optional):").pack(side=tk.LEFT)
        self.batch_out_dir_var = tk.StringVar(value="")
        ttk.Entry(run_row, textvariable=self.batch_out_dir_var, width=50).pack(side=tk.LEFT, padx=4)
        ttk.Button(run_row, text="Browse...", command=self._on_batch_browse_out_dir).pack(side=tk.LEFT)
        ttk.Button(run_row, text="Compensate All", command=self._on_batch_run_clicked).pack(
            side=tk.LEFT, padx=(12, 0))
        ttk.Label(run_row, text="(uses the calibration and smoothing set on the right)",
                  foreground="gray40").pack(side=tk.LEFT, padx=6)

        columns = ("file", "n", "all", "turns", "lines", "turn_frac", "status")
        self.batch_tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        headings = {
            "file": "file",
            "n": "samples",
            "all": "noise change, all",
            "turns": "turns only",
            "lines": "lines only",
            "turn_frac": "turn share",
            "status": "status / output",
        }
        widths = {"file": 260, "n": 70, "all": 110, "turns": 90, "lines": 90, "turn_frac": 80, "status": 300}
        for col in columns:
            self.batch_tree.heading(col, text=headings[col])
            self.batch_tree.column(col, width=widths[col], anchor=tk.W, stretch=col in ("file", "status"))
        self.batch_tree.tag_configure("error", foreground="firebrick")
        self.batch_tree.tag_configure("warn", foreground="darkorange3")
        self.batch_tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 2))
        ttk.Label(parent, text="noise = sample-to-sample (motion) noise; positive % = reduced. "
                               "Turns are where heading error is largest; lines must not get worse.",
                  foreground="gray40").pack(anchor=tk.W, padx=6)
        self.batch_summary_label = ttk.Label(parent, text="", wraplength=900, justify=tk.LEFT)
        self.batch_summary_label.pack(anchor=tk.W, padx=6, pady=(2, 6))

        lev = ttk.LabelFrame(parent, text="Forward/reverse line leveling (whole survey, files above)")
        lev.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.level_compensate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lev, text="compensate with the current calibration first "
                                  "(leave off for files that are already -comp.csv)",
                        variable=self.level_compensate_var).grid(row=0, column=0, columnspan=3,
                                                                 sticky=tk.W, padx=4, pady=2)
        self.level_force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(lev, text="force (add MagLeveled even when the estimate is flagged unreliable)",
                        variable=self.level_force_var).grid(row=1, column=0, columnspan=3,
                                                            sticky=tk.W, padx=4, pady=2)
        ttk.Button(lev, text="Level survey", command=self._on_level_survey_clicked).grid(
            row=2, column=0, sticky=tk.W, padx=4, pady=4)
        self.level_result_label = ttk.Label(lev, text="", justify=tk.LEFT)
        self.level_result_label.grid(row=2, column=1, sticky=tk.W, padx=8)

    # ------------------------------------------------------------------ logging / tasks

    def _log(self, message, tag=None):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n", (tag,) if tag else ())
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _error(self, message):
        self._log(f"ERROR: {message}", "error")
        messagebox.showerror("Error", message)

    def _run_task(self, label, work, on_done):
        """Run `work(report)` off the UI thread, then `on_done(result)` on it.

        `report(message)` may be called from the worker to append to the log and
        status line. Only one task runs at a time; clicks while busy are ignored
        with a log note rather than queued, so results can't arrive out of order.
        """
        if self._busy:
            self._log(f"busy with '{self._busy}' -- wait for it to finish", "warn")
            return False
        self._busy = label
        self.status_label.config(text=f"{label} ...")
        self.progress.start(12)
        self.config(cursor="watch")

        if self.sync_tasks:
            try:
                result = work(self._log)
            except Exception as exc:
                self._finish_task()
                self._error(f"{label} failed: {exc}")
                return True
            self._finish_task()
            on_done(result)
            return True

        def runner():
            try:
                result = work(lambda msg: self._task_queue.put(("log", msg)))
                self._task_queue.put(("done", result))
            except Exception as exc:  # reported on the UI thread
                self._task_queue.put(("error", exc))

        threading.Thread(target=runner, daemon=True).start()
        self.after(100, lambda: self._poll_task(label, on_done))
        return True

    def _poll_task(self, label, on_done):
        try:
            while True:
                kind, payload = self._task_queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                    self.status_label.config(text=f"{label}: {payload[:60]}")
                elif kind == "done":
                    self._finish_task()
                    on_done(payload)
                    return
                elif kind == "error":
                    self._finish_task()
                    self._error(f"{label} failed: {payload}")
                    return
        except queue.Empty:
            pass
        self.after(100, lambda: self._poll_task(label, on_done))

    def _finish_task(self):
        self._busy = None
        self.progress.stop()
        self.status_label.config(text="ready")
        self.config(cursor="")

    def _embed_figure(self, fig, panes, frames, weight=1, toolbar=True):
        """Put a Figure (from plot.py) into a PanedWindow with its own toolbar.

        The canvas widget's requested size is set small on purpose: a
        FigureCanvasTkAgg otherwise asks Tk for the figure's full inch size, and
        that request is what used to push the control panel and log off-screen.
        It still fills whatever space the pane gives it.
        """
        plt.close(fig)  # drop pyplot's reference; the Tk canvas owns it now
        # plot.py lays figures out once at their nominal size; re-run the layout
        # on every draw so axis labels aren't clipped when a pane is resized.
        fig.set_layout_engine("constrained")
        frame = ttk.Frame(panes)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        if toolbar:
            NavigationToolbar2Tk(canvas, frame).pack(side=tk.TOP, fill=tk.X)
        widget = canvas.get_tk_widget()
        widget.configure(width=300, height=150)
        widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        panes.add(frame, weight=weight)
        frames.append(frame)
        return frame

    @staticmethod
    def _clear_frames(panes, frames):
        for frame in frames:
            panes.forget(frame)
            frame.destroy()
        frames.clear()

    # ------------------------------------------------------------------ settings

    SETTINGS_VARS = (
        "method_var", "grid_res_var", "coverage_var", "max_var_var", "trim_outliers_var",
        "outlier_mad_var", "window_size_var", "auto_window_sizes_var", "auto_coverage_target_var",
        "auto_max_segments_var", "center_segments_var", "smooth_enabled_var", "smooth_hz_var",
        "overwrite_var", "batch_out_dir_var",
    )

    def _load_settings(self):
        if not self.settings_path:
            return
        try:
            with open(self.settings_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        for name in self.SETTINGS_VARS:
            if name in data:
                try:
                    getattr(self, name).set(data[name])
                except (tk.TclError, TypeError, ValueError):
                    pass
        self._last_dir = data.get("last_dir")

    def _save_settings(self):
        if not self.settings_path:
            return
        data = {"last_dir": self._last_dir}
        for name in self.SETTINGS_VARS:
            try:
                data[name] = getattr(self, name).get()
            except tk.TclError:
                pass  # half-typed entry: keep the previous saved value out of it
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1)
        except OSError:
            pass

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _remember_dir(self, path):
        if path:
            path = os.path.abspath(path)
            self._last_dir = path if os.path.isdir(path) else os.path.dirname(path)

    def _initial_dir(self):
        if self.pre_path:
            return os.path.dirname(self.pre_path)
        return self._last_dir if self._last_dir and os.path.isdir(self._last_dir) else None

    # ------------------------------------------------------------------ data loading

    def _on_load_clicked(self):
        path = filedialog.askopenfilename(initialdir=self._initial_dir(),
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self._load_pre(path)

    def _load_pre(self, path):
        try:
            self.df = read_pre_csv(path)
        except Exception as exc:
            self._error(f"failed to load {path}: {exc}")
            return
        self.pre_path = path
        self._remember_dir(path)
        self.path_label.config(text=path)
        rate = infer_sample_rate_hz(self.df)
        rate_str = f", {rate:.3g} Hz" if rate else ", sample rate unknown (no Time column)"
        self._log(f"loaded {path} ({len(self.df)} samples{rate_str})")
        mag = self.df["Mag"].to_numpy(dtype=float)
        med = float(np.nanmedian(mag)) if len(mag) else float("nan")
        if not 15000 <= med <= 80000:
            # io.py's index_col=False guard against column shifts: an implausible
            # Mag is the first visible symptom of a mis-parsed file.
            self._log(f"WARNING: median Mag is {med:.6g} nT -- not a plausible total field; "
                      "check the file's columns", "warn")
        self._reset_for_new_file()
        self._draw_selection_plot()
        if self.model is not None:
            self._log("calibration kept from before -- report below re-checked against this file")
            self._refresh_report()

    def _reset_for_new_file(self):
        """Scan candidates, segment selection and previews refer to whichever file
        was loaded when they were computed, so they are cleared on load. Without
        this, picking a stale candidate after loading a different file silently
        builds a "calibration" from the new file's samples at the old file's
        indices.

        The calibration model itself is kept: it is independent of the survey
        file (one calibration per project is applied to every file), and
        clearing it made "load calibration, then load the file to compensate"
        silently lose the calibration.
        """
        self.start_idx = None
        self.end_idx = None
        self.compensated = None
        self._candidates = []
        self.candidates_list.delete(0, tk.END)
        self.segments = []
        self.segments_list.delete(0, tk.END)
        self._segment_patches = []
        self.selection_label.config(text="start=- end=- (drag on plot to select)")
        self._clear_frames(self.result_panes, self._result_frames)
        self.result_summary.config(text="Preview or compensate to see the result here.")

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
            self.path_ax.plot(self._path_x, self._path_y, "-", color="lightgray", linewidth=0.8, zorder=1)
            (self._path_segments_highlight,) = self.path_ax.plot([], [], ".", color="tab:green",
                                                                   markersize=3, zorder=2)
            (self._path_highlight,) = self.path_ax.plot([], [], ".", color="tab:red", markersize=3, zorder=3)
            self.path_ax.set_xlabel("East (m)")
            self.path_ax.set_ylabel("North (m)")
            self.path_ax.set_title("flight path (pending=red, added=green)")
            self.path_ax.set_aspect("equal", adjustable="datalim")
        else:
            self._path_x = self._path_y = None
            self.path_ax.text(0.5, 0.5, "no Latitude/Longitude columns",
                              ha="center", va="center", transform=self.path_ax.transAxes)
            self.path_ax.set_xticks([])
            self.path_ax.set_yticks([])

        self._update_path_highlight()
        self.select_fig.tight_layout()
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
            self._error("drag on the plot (or pick a scan candidate) to select a segment before adding it")
            return
        self.segments.append((self.start_idx, self.end_idx))
        self.segments_list.insert(
            tk.END, f"{self.start_idx}-{self.end_idx} ({self.end_idx - self.start_idx + 1} samples)"
        )
        self._log(f"added segment {self.start_idx}-{self.end_idx} ({len(self.segments)} segment(s) total)")
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
        # the GUI normally runs under pythonw (no console), so an escaping
        # exception would make the button look simply dead.
        try:
            window_size = self.window_size_var.get()
            kwargs = dict(
                window_size=window_size,
                method=self.method_var.get(),
                coverage_threshold_deg=self.coverage_var.get(),
                grid_resolution_deg=self.grid_res_var.get(),
                trim_outliers=self.trim_outliers_var.get(),
                outlier_mad_threshold=self.outlier_mad_var.get(),
                max_variation=self.max_var_var.get(),
                top_n=15,
            )
        except tk.TclError:
            self._error("window size and model parameters must be numbers")
            return
        if window_size > len(self.df):
            self._error(f"window size {window_size} exceeds this file's {len(self.df)} "
                        f"samples; try {max(2, len(self.df) // 4)} or smaller")
            return
        df = self.df

        def work(report):
            report(f"scanning window_size={window_size} ...")
            return scan_calibration_candidates(df, **kwargs)

        def done(candidates):
            if df is not self.df:
                self._log("scan finished for a file that is no longer loaded -- discarded", "warn")
                return
            self._candidates = candidates
            self.candidates_list.delete(0, tk.END)
            for c in candidates:
                dom = "dom-ok" if c["dominant_heading_covered"] else "dom-MISS"
                self.candidates_list.insert(
                    tk.END,
                    f"{c['start']:>6}-{c['end']:<6} p2p={c['self_compensation_p2p_nT']:<5.3g} "
                    f"unc={c['survey_uncovered_fraction']:>4.0%} {dom}",
                )
            self._log(f"found {len(candidates)} candidates")

        self._run_task("scan", work, done)

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
            window_sizes = [int(x.strip()) for x in self.auto_window_sizes_var.get().split(",") if x.strip()]
            if not window_sizes:
                raise ValueError("window sizes list is empty")
        except ValueError as exc:
            self._error(f"invalid window sizes ('{self.auto_window_sizes_var.get()}'): {exc}")
            return
        try:
            kwargs = dict(
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
        df = self.df

        def work(report):
            report(f"auto-build: trying window sizes {window_sizes} (widest first) ...")
            return auto_build_calibration(df, **kwargs)

        def done(result):
            if df is not self.df:
                self._log("auto-build finished for a file that is no longer loaded -- discarded", "warn")
                return
            for attempt in result["attempts"]:
                status = "PASSED" if attempt["passed"] else "did not pass"
                n_accepted = sum(1 for e in attempt["log"] if e["accepted"])
                self._log(f"  window_size={attempt['window_size']}: {status} "
                          f"({n_accepted} accepted, {len(attempt['log']) - n_accepted} rejected)")

            if not result["segments"] or result["model"] is None:
                self._error("auto-build could not assemble any passing calibration at the given "
                            "window sizes / max variation. Try picking segments manually instead.")
                return

            # populate the accumulated-segments UI exactly as if these had been
            # picked and added by hand, so the result is visible and still editable
            self.segments = list(result["segments"])
            self.segments_list.delete(0, tk.END)
            for s, e in self.segments:
                self.segments_list.insert(tk.END, f"{s}-{e} ({e - s + 1} samples)")
            self.start_idx = None
            self.end_idx = None
            self.selection_label.config(text="start=- end=- (drag on plot to select)")
            self._refresh_segment_overlays()

            self._set_model(result["model"], f"auto-build of {os.path.basename(self.pre_path or '')}")
            self._show_report(result["report"])
            self._update_quality_tab()

            if result["passed"]:
                self._log(f"auto-build PASSED with window_size={result['window_size']}, "
                          f"{len(self.segments)} segment(s) -- opening save dialog")
                self._on_save_cal_clicked()
            else:
                self._log(f"auto-build did NOT pass (best-effort shown: window_size="
                          f"{result['window_size']}, {len(self.segments)} segment(s))", "warn")
                messagebox.showwarning(
                    "Auto-build did not pass",
                    "The best-effort auto-build result did not meet the pass criteria "
                    "(self-compensation p2p / dominant heading coverage / coverage target).\n\n"
                    "Review the quality report on the right and the Calibration Quality tab "
                    "before deciding whether to save it manually, or pick segments yourself.",
                )

        self._run_task("auto-build", work, done)

    # ------------------------------------------------------------------ calibration

    def _set_model(self, model, source):
        self.model = model
        self.model_source = source
        self.model_label.config(text=f"calibration: {source} ({model.method}, "
                                     f"{len(model.mag_cal)} samples)")

    def _on_build_clicked(self):
        if self.df is None:
            self._error("load a -pre.csv file first")
            return
        if self.segments:
            ranges = list(self.segments)
            pending = (self.start_idx, self.end_idx)
            if self.start_idx is not None and pending not in self.segments:
                self._log(f"note: the pending selection {pending[0]}-{pending[1]} is NOT included -- "
                          "click 'Add segment' first to use it", "warn")
        elif self.start_idx is not None and self.end_idx is not None:
            ranges = [(self.start_idx, self.end_idx)]
        else:
            self._error("select a segment first (drag on the plot, or pick a scan candidate), "
                        "or add one or more segments with 'Add segment'")
            return

        try:
            segment = select_segments(self.df, ranges, center_segments=self.center_segments_var.get())
            model = build_calibration(
                segment,
                method=self.method_var.get(),
                coverage_threshold_deg=self.coverage_var.get(),
                grid_resolution_deg=self.grid_res_var.get(),
                trim_outliers=self.trim_outliers_var.get(),
                outlier_mad_threshold=self.outlier_mad_var.get(),
                metadata={"segments": [f"{s}-{e}" for s, e in ranges],
                          "source_file": os.path.basename(self.pre_path or "")},
            )
            report = quality_report(model, max_variation=self.max_var_var.get(), survey_df=self.df)
        except Exception as exc:
            self._error(f"calibration failed: {exc}")
            return

        self._set_model(model, f"built from {os.path.basename(self.pre_path or '')}")
        self._show_report(report)
        seg_desc = ", ".join(f"{s}-{e}" for s, e in ranges)
        self._log(f"built calibration model from segment(s) {seg_desc}")
        offsets = segment.attrs.get("segment_baseline_offsets_nT")
        if offsets:
            self._log(f"  segment baseline offsets applied (nT): {', '.join(f'{o:+.2f}' for o in offsets)}")
        self._update_quality_tab()

    def _report_thresholds(self):
        try:
            max_var = self.max_var_var.get()
        except tk.TclError:
            max_var = None
        try:
            target = self.auto_coverage_target_var.get()
        except tk.TclError:
            target = None
        return max_var, target

    def _refresh_report(self):
        max_var, _ = self._report_thresholds()
        report = quality_report(self.model, max_variation=max_var, survey_df=self.df)
        self._show_report(report)
        self._update_quality_tab()

    def _show_report(self, report):
        max_var, target = self._report_thresholds()
        checks = evaluate_report(report, max_var, target)
        overall = all(ok for _, ok, _ in checks) if checks else None

        t = self.report_text
        t.config(state=tk.NORMAL)
        t.delete("1.0", tk.END)
        if overall is not None:
            t.insert(tk.END, "PASS\n" if overall else "FAIL\n", "pass" if overall else "fail")
            for label, ok, detail in checks:
                t.insert(tk.END, f"{'OK ' if ok else 'NO '} {label}: ", "pass" if ok else "fail")
                t.insert(tk.END, f"{detail}\n")
            if "survey_uncovered_fraction" not in report:
                t.insert(tk.END, "(load a survey file to check coverage)\n", "dim")
            t.insert(tk.END, "\n")
        for key, value in report.items():
            label = REPORT_LABELS.get(key, key)
            if key in FRACTION_KEYS and isinstance(value, float):
                text = f"{value:.1%}"
            elif isinstance(value, float):
                text = f"{value:.4g}"
            else:
                text = str(value)
            t.insert(tk.END, f"{label}: ", "dim")
            t.insert(tk.END, f"{text}\n")
        t.config(state=tk.DISABLED)

        q = self.quality_summary
        q.config(state=tk.NORMAL)
        q.delete("1.0", tk.END)
        if overall is not None:
            q.insert(tk.END, "PASS  " if overall else "FAIL  ", "pass" if overall else "fail")
            q.insert(tk.END, "   ".join(f"{label}: {detail}" for label, _, detail in checks) + "\n")
        q.insert(tk.END, "A pass here is necessary, not sufficient: confirm the turn-only noise "
                         "improvement on the Compensation Result / Batch tabs before trusting it.")
        q.config(state=tk.DISABLED)

    def _update_quality_tab(self):
        self._clear_frames(self.quality_panes, self._quality_frames)
        self._embed_figure(plot_calibration_quality(self.model, title="self-compensation"),
                           self.quality_panes, self._quality_frames, weight=1)
        try:
            cov_fig = plot_heading_coverage(self.model, survey_df=self.df)
            self._embed_figure(cov_fig, self.quality_panes, self._quality_frames, weight=1)
        except Exception as exc:
            self._log(f"coverage plot skipped: {exc}", "warn")

    def _on_save_cal_clicked(self):
        if self.model is None:
            self._error("build or load a calibration model first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("npz", "*.npz")],
                                            initialdir=self._initial_dir())
        if not path:
            return
        self.model.save(path)
        self._remember_dir(path)
        self._log(f"calibration model saved to {path}")

    def _on_load_cal_clicked(self):
        path = filedialog.askopenfilename(filetypes=[("npz", "*.npz"), ("All files", "*.*")],
                                          initialdir=self._initial_dir())
        if not path:
            return
        try:
            model = CalibrationModel.load(path)
        except Exception as exc:
            self._error(f"failed to load {path}: {exc}")
            return
        self._remember_dir(path)
        self._set_model(model, os.path.basename(path))
        self._log(f"calibration model loaded from {path}")
        if model.metadata:
            self._log(f"  metadata: {model.metadata}")
        self._refresh_report()

    def _on_export_vendor_clicked(self):
        if self.model is None:
            self._error("build or load a calibration model first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialdir=self._initial_dir(), initialfile="vendor-cal.csv")
        if not path:
            return
        try:
            self.model.export_vendor_cal_csv(path)
        except Exception as exc:
            self._error(f"vendor export failed: {exc}")
            return
        self._log(f"vendor-format -cal.csv exported to {path}")

    # ------------------------------------------------------------------ compensation

    def _smooth_hz(self):
        """The smoothing cutoff to use, or None. Raises ValueError on a bad entry."""
        if not self.smooth_enabled_var.get():
            return None
        try:
            hz = float(self.smooth_hz_var.get())
        except tk.TclError:
            raise ValueError("smoothing cutoff must be a number (Hz)")
        if hz <= 0:
            raise ValueError("smoothing cutoff must be > 0 Hz")
        return hz

    @staticmethod
    def _compensate_df(df, model, smooth_hz, keep_raw):
        """apply_compensation with a clear fallback when smoothing can't run.

        Returns (compensated_df, smoothing_used_hz_or_None, note)."""
        if smooth_hz is not None and infer_sample_rate_hz(df) is None:
            comp = apply_compensation(df, model, keep_raw=keep_raw)
            return comp, None, "smoothing OFF: no usable Time column to get the sample rate"
        comp = apply_compensation(df, model, keep_raw=keep_raw, smooth_hz=smooth_hz)
        return comp, smooth_hz, None

    def _check_ready_to_compensate(self):
        if self.df is None:
            self._error("load a -pre.csv file first")
            return None
        if self.model is None:
            self._error("build or load a calibration model first")
            return None
        try:
            return {"smooth_hz": self._smooth_hz()}
        except ValueError as exc:
            self._error(str(exc))
            return None

    def _start_compensation(self, out_path=None):
        opts = self._check_ready_to_compensate()
        if opts is None:
            return
        df, model, keep_raw = self.df, self.model, not self.overwrite_var.get()
        pre_path = self.pre_path

        def work(report):
            report("compensating ...")
            comp, used_hz, note = self._compensate_df(df, model, opts["smooth_hz"], keep_raw)
            comp_col = "MagComp" if keep_raw else "Mag"
            breakdown = noise_breakdown(df["Mag"].to_numpy(), comp[comp_col].to_numpy(),
                                        turn_line_masks(df))
            if out_path:
                write_comp_csv(comp, pre_path, out_path=out_path)
            return comp, used_hz, note, breakdown

        def done(result):
            comp, used_hz, note, breakdown = result
            if df is not self.df:
                self._log("compensation finished for a file that is no longer loaded -- discarded", "warn")
                return
            self.compensated = comp
            if note:
                self._log(note, "warn")
            smooth = f"smoothing {used_hz:g} Hz" if used_hz else "no smoothing"
            self._log(f"noise change ({smooth}): all {_fmt_pct(breakdown['motion_noise_improvement_pct'])}, "
                      f"turns {_fmt_pct(breakdown['turn_improvement_pct'])}, "
                      f"lines {_fmt_pct(breakdown['line_improvement_pct'])}")
            if out_path:
                self._log(f"compensated survey written to {out_path} ({len(comp)} samples)")
            self._update_2d_tab(comp, breakdown, smooth, saved_to=out_path)

        self._run_task("compensation", work, done)

    def _on_preview_clicked(self):
        self._start_compensation(out_path=None)

    def _on_compensate_clicked(self):
        if self._check_ready_to_compensate() is None:
            return
        default = comp_output_path(self.pre_path) if self.pre_path else "survey-comp.csv"
        out_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialdir=os.path.dirname(default) or None, initialfile=os.path.basename(default),
        )
        if not out_path:
            return
        self._remember_dir(out_path)
        self._start_compensation(out_path=out_path)

    def _update_2d_tab(self, compensated, breakdown=None, smooth_desc="", saved_to=None):
        self._clear_frames(self.result_panes, self._result_frames)

        # Mag was overwritten in place when overwrite is on; there's no separate
        # raw column left to compare against, so only plot the one series.
        comp_col = None if self.overwrite_var.get() else "MagComp"

        if breakdown is not None:
            lines = [
                f"motion-noise change ({smooth_desc}):  all {_fmt_pct(breakdown['motion_noise_improvement_pct'])}"
                f"   turns {_fmt_pct(breakdown['turn_improvement_pct'])}"
                f"   lines {_fmt_pct(breakdown['line_improvement_pct'])}"
                + (f"   (turns are {breakdown['turn_fraction']:.0%} of samples)"
                   if breakdown["turn_fraction"] is not None else "   (no GPS: turn/line split unavailable)"),
                f"saved to {saved_to}" if saved_to else "preview only -- not saved",
            ]
            self.result_summary.config(text="\n".join(lines))

        try:
            ts_fig = plot_compensation_timeseries(compensated, raw_col="Mag", comp_col=comp_col)
            self._embed_figure(ts_fig, self.result_panes, self._result_frames, weight=1)
        except Exception as exc:
            self._log(f"time series plot skipped: {exc}", "warn")

        if "Latitude" not in compensated.columns or "Longitude" not in compensated.columns:
            self._log("no Latitude/Longitude columns; skipping 2D map")
        else:
            try:
                fig = plot_2d_comparison(compensated, raw_col="Mag", comp_col=comp_col)
                self._embed_figure(fig, self.result_panes, self._result_frames, weight=2)
            except Exception as exc:
                self._log(f"2D plot skipped: {exc}", "warn")

        self.notebook.select(self.tab_2d)

    # ------------------------------------------------------------------ batch

    def _batch_paths(self):
        return list(self.batch_files_list.get(0, tk.END))

    def _update_batch_count(self):
        self.batch_count_label.config(text=f"{self.batch_files_list.size()} file(s)")

    def _on_batch_add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                                            initialdir=self._initial_dir())
        existing = set(self._batch_paths())
        for p in paths:
            if p not in existing:
                self.batch_files_list.insert(tk.END, p)
        if paths:
            self._remember_dir(paths[0])
        self._update_batch_count()

    def _on_batch_add_folder(self):
        folder = filedialog.askdirectory(initialdir=self._initial_dir())
        if not folder:
            return
        found = find_pre_files(folder)
        if not found:
            self._error(f"no -pre CSV files found in {folder} (looked for names containing '-pre' "
                        "that are not -comp/-leveled outputs)")
            return
        existing = set(self._batch_paths())
        added = [p for p in found if p not in existing]
        for p in added:
            self.batch_files_list.insert(tk.END, p)
        self._last_dir = folder
        self._log(f"added {len(added)} file(s) from {folder}")
        self._update_batch_count()

    def _on_batch_remove_selected(self):
        for i in reversed(self.batch_files_list.curselection()):
            self.batch_files_list.delete(i)
        self._update_batch_count()

    def _on_batch_clear(self):
        self.batch_files_list.delete(0, tk.END)
        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)
        self.batch_summary_label.config(text="")
        self._update_batch_count()

    def _on_batch_browse_out_dir(self):
        folder = filedialog.askdirectory(initialdir=self._initial_dir())
        if folder:
            self.batch_out_dir_var.set(folder)

    def _on_batch_run_clicked(self):
        if self.model is None:
            self._error("build or load a calibration model first")
            return
        paths = self._batch_paths()
        if not paths:
            self._error("add at least one -pre.csv file first")
            return
        try:
            smooth_hz = self._smooth_hz()
        except ValueError as exc:
            self._error(str(exc))
            return
        out_dir = self.batch_out_dir_var.get().strip() or None
        model, keep_raw = self.model, not self.overwrite_var.get()
        smooth_desc = f"smoothing {smooth_hz:g} Hz" if smooth_hz else "no smoothing"

        def work(report):
            report(f"batch-compensating {len(paths)} file(s), {smooth_desc} ...")
            return batch_compensate(
                paths, model, keep_raw=keep_raw, out_dir=out_dir, smooth_hz=smooth_hz,
                progress=lambda i, n, p: report(f"{i + 1}/{n} {os.path.basename(p)}"),
            )

        self._run_task("batch", work, lambda results: self._show_batch_results(results, smooth_desc))

    def _show_batch_results(self, results, smooth_desc=""):
        for row in self.batch_tree.get_children():
            self.batch_tree.delete(row)
        ok = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        for r in results:
            name = os.path.basename(r["path"])
            if "error" in r:
                self.batch_tree.insert("", tk.END, values=(name, "", "", "", "", "", f"FAILED: {r['error']}"),
                                       tags=("error",))
                continue
            worse = [v for v in (r["motion_noise_improvement_pct"], r["turn_improvement_pct"],
                                 r["line_improvement_pct"]) if v is not None and v < 0]
            self.batch_tree.insert(
                "", tk.END,
                values=(
                    name, r["n_samples"],
                    _fmt_pct(r["motion_noise_improvement_pct"]),
                    _fmt_pct(r["turn_improvement_pct"]),
                    _fmt_pct(r["line_improvement_pct"]),
                    "-" if r["turn_fraction"] is None else f"{r['turn_fraction']:.0%}",
                    os.path.basename(r["out_path"]),
                ),
                tags=("warn",) if worse else (),
            )

        if not ok:
            self.batch_summary_label.config(text=f"all {len(failed)} file(s) failed -- see the table")
            self._log(f"batch compensate: all {len(failed)} file(s) failed", "error")
            return

        def _mean(key):
            vals = [r[key] for r in ok if r[key] is not None]
            return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

        mean_all, _ = _mean("motion_noise_improvement_pct")
        mean_turn, n_turn = _mean("turn_improvement_pct")
        mean_line, n_line = _mean("line_improvement_pct")
        summary = (f"{smooth_desc}: mean noise change all {_fmt_pct(mean_all)}, "
                   f"turns {_fmt_pct(mean_turn)} ({n_turn} files), lines {_fmt_pct(mean_line)} ({n_line} files)")
        turn_neg = [os.path.basename(r["path"]) for r in ok
                    if r["turn_improvement_pct"] is not None and r["turn_improvement_pct"] < 0]
        line_neg = [os.path.basename(r["path"]) for r in ok
                    if r["line_improvement_pct"] is not None and r["line_improvement_pct"] < 0]
        if turn_neg:
            summary += f"\nturns got WORSE in {len(turn_neg)} file(s) -- calibration may not fit this survey"
        if line_neg:
            hint = "" if "smoothing" in smooth_desc and "no smoothing" not in smooth_desc \
                else " -- try turning smoothing (1 Hz) on"
            summary += f"\nlines got noisier in {len(line_neg)} file(s){hint}"
        if failed:
            summary += f"\n{len(failed)} file(s) FAILED (red rows); the others were written"
        self.batch_summary_label.config(text=summary)
        self._log(f"batch compensate done: {summary}", "warn" if (turn_neg or line_neg or failed) else None)

    # ------------------------------------------------------------------ leveling

    def _on_level_survey_clicked(self):
        paths = self._batch_paths()
        if not paths:
            self._error("add the survey's files to the list above first")
            return
        model = None
        smooth_hz = None
        if self.level_compensate_var.get():
            if self.model is None:
                self._error("'compensate first' is on but no calibration is built or loaded")
                return
            model = self.model
            try:
                smooth_hz = self._smooth_hz()
            except ValueError as exc:
                self._error(str(exc))
                return
        out_dir = self.batch_out_dir_var.get().strip() or None
        force = self.level_force_var.get()

        def work(report):
            return level_survey_files(paths, model=model, smooth_hz=smooth_hz,
                                      out_dir=out_dir, force=force, log=report)

        def done(summary):
            result = summary["result"]
            if summary["status"] == "missing_column":
                text = f"column '{summary['value_col']}' missing in some files -- see log"
            elif result is None:
                text = "nothing computed -- see log"
            elif not result.applied:
                text = f"no correction: {result.reason}"
            else:
                text = (f"offset {result.offset_nt:+.2f} nT (spread {result.offset_spread_nt:.2f} nT), "
                        f"{summary['n_lines']} lines on '{summary['value_col']}' -- "
                        + ("MagLeveled ADDED" if summary["add_leveled"] else "MagLeveled not added (unreliable)"))
            if summary["written"]:
                text += f"\n{len(summary['written'])} file(s) written"
            self.level_result_label.config(text=text)

        self._run_task("level survey", work, done)


def main(pre_path=None):
    app = MagHeadingApp(pre_path=pre_path)
    app.mainloop()


if __name__ == "__main__":
    main()
