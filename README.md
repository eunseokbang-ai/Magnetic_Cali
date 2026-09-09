# MagArrow Heading Error Calibration / Compensation

Geometrics MagArrow 세슘 자력계 시스템의 heading error(방위 오차)를 보정하기 위한
캘리브레이션 파일 생성 및 compensation 프로그램. Geometrics가 제공하는 LabVIEW
"Survey_Data_Processing" 프로그램의 Module 3(Heading Error Calibration File
Generation)과 Module 4(Heading Error Compensation)에 해당하는 부분을, 정량적인
품질 지표와 함께 재구현한다. Module 1(베이스국 보정), 2(노이즈 저감), 5(파일 결합)는
벤더 프로그램을 그대로 사용한다.

## 배경

MagArrow는 두 개의 직교 배치된 전광학 세슘 자력계를 결합해 dead zone을 없앤 대신,
heading effect(자력계 방향에 따른 총자기장 판독값의 미세한 변화)가 커진다. 이 오차는
헤딩 θ(polar), φ(azimuth)의 함수 ΔB(θ, φ)로 근사할 수 있으며, 낮은 자기장 구배
지역에서 다양한 헤딩을 포함하도록 비행한 캘리브레이션 구간으로부터 이 함수를 구해
survey 데이터에서 빼주면 된다 (Zhang et al., 2022, *The Leading Edge*,
"Motion noise reduction in drone magnetometry using a dead-zone-free
total-field magnetometer system").

벤더 프로그램은 GUI 커서로 캘리브레이션 구간을 수작업으로 선택해야 하고, 품질 판단은
자기평탄도(정량) + 헤딩 커버리지(정성/시각 비교)에 의존한다. 이 프로그램은 같은 원리를
쓰되 헤딩 커버리지까지 정량 지표로 만들고, 논문에서 제안한 spline 보간(벤더의
nearest-neighbor 대비 개선안)을 선택적으로 제공한다.

## 데이터 흐름

```
MagArrow 원본 자료
    │  (Survey Manager로 .csv 변환)
    ▼
벤더 프로그램 Module 1: Survey Data Preparation
    │  (구간 자르기 + 베이스국 보정)
    ▼
  <survey>-pre.csv   (62열: Mag, CompassX/Y/Z, GPS/IMU/NMEA ...)
    │
    ├──▶ mag_heading scan        ──▶ 후보 캘리브레이션 구간 랭킹 (start/end 추천)
    │
    ├──▶ mag_heading calibrate   ──▶ <project>.cal.npz  (캘리브레이션 모델, 프로젝트당 1개)
    │                                        │
    └──▶ mag_heading compensate ◀────────────┘
              │
              ▼
        <survey>-pre-comp.csv  (MagComp 컬럼 추가)
              │
              ▼
   벤더 프로그램 Module 2 (노이즈 저감/Line Leveling), Module 5 (파일 결합)
```

## 설치

```bash
cd C:\Mag_Heading
pip install -r requirements.txt
pip install -e .
```

## 사용법

```bash
# 0. (선택) 좋은 캘리브레이션 구간을 수작업으로 찾기 어려울 때: 슬라이딩 윈도우로
#    후보 구간을 스캔해 자기평탄도(p2p) + survey의 주 헤딩(드론의 주 비행 방향) 커버 여부로
#    랭킹을 매겨 추천받는다. dom_cov 컬럼이 그 주 헤딩 커버 여부다. --window-size 기본값은
#    3000(10Hz 기준 ~5분) — 턴 구간의 헤딩 스윙 전체를 담을 만큼 넓게 잡은 것이다.
python -m mag_heading scan --pre <survey>-pre.csv --max-variation 2.0

# 1. -pre.csv에서 캘리브레이션 구간(인덱스 범위, scan 추천값 또는 직접 선택)으로
#    캘리브레이션 모델 생성. --trim-outliers를 주면 각 각도 그리드 칸(bin) 안에서
#    heading과 무관한 튀는 값(구배/드론 간섭)만 골라 제거한다 — 칸을 통째로 비우지는
#    않으므로 헤딩 커버리지는 그대로 유지된다 (자세한 근거는 매뉴얼 8.5절 참고).
python -m mag_heading calibrate --pre <survey>-pre.csv --start <start_idx> --end <end_idx> \
    --method nearest --trim-outliers --out project.cal.npz

# 1b. 한 구간만으로는 "자기평탄도"와 "헤딩 다양성"을 동시에 만족하기 어려울 때:
#     --start/--end를 여러 번 짝지어 주면(짧고 개별적으로 깨끗한 여러 구간을 조합)
#     헤딩이 합산된다 (실제 홍천 프로젝트의 제조사 캘리브레이션 파일이 이런 식으로
#     여러 비연속 구간을 이어붙인 것으로 확인됨). 서로 다른 구간은 실제 지자기장
#     레벨도 다른 경우가 많으므로 (같은 파일 안에서도 수백~수천 샘플 떨어진 구간은
#     ~수십 nT 차이가 흔함) 기본적으로 구간별 median을 빼서 공통 기준선으로 맞춘 뒤
#     합친다 (--no-center-segments로 끌 수 있음, 자세한 근거는 매뉴얼 8.6절 참고).
python -m mag_heading calibrate --pre <survey>-pre.csv \
    --start <start1> <start2> <start3> --end <end1> <end2> <end3> \
    --method spline --trim-outliers --out project.cal.npz

# 1c. 위 0~1b 과정(구간 찾기 → 품질 확인 → 안 되면 구간 바꿔서 재시도) 자체를 자동화한 것.
#     벤더 GUI의 "Calibration Data Quality Check" 화면에서 사용자가 직접 반복하는 pick →
#     Pass/Fail 확인 → 다른 구간 시도 루프를, 이 프로그램의 정량 지표로 자동 반복하고
#     실제로 통과(Pass)했을 때만 파일로 저장한다(--force-save 없이는 미달 상태로 저장 안 함).
#     --window-size는 넓은 것부터 순서대로 시도하다 실패하면 더 좁은 값으로 자동 대체된다
#     (기본 3000,1500,750,300) — 얼마나 넓은 구간까지 "깨끗함"을 유지할 수 있는지는
#     현장의 실제 자기장 구배에 달린 문제라 한 값으로 못 박을 수 없기 때문이다. 실측 결과:
#     홍천 자료에서 3000/1500샘플 구간은 전부 실패(p2p 최대 68nT, 실제 공간적 구배 때문)했지만
#     750샘플로 자동 대체되어 서로 다른 두 지점의 구간 2개를 조합해 p2p 1.4nT, 헤딩 미달률
#     5.1%로 자동 통과했다 (turn-only 검증으로 재확인한 실개선율 +7.9%, 자세한 근거는
#     매뉴얼 8.7절 참고).
#     헤딩 커버리지는 탐색 중에만 survey 샘플 일부(기본 20000개)로 추정해 속도를 올리고,
#     최종 보고 수치와 통과 판정은 항상 전체 survey로 정확히 다시 계산한다
#     (--coverage-sample-limit 0을 주면 탐색도 전수 계산; 실측 28.4초→7.4초, 결과는 동일).
python -m mag_heading auto-calibrate --pre <survey>-pre.csv \
    --method spline --max-variation 2.0 --out project.cal.npz

# 2. 캘리브레이션 품질 확인 (자기평탄도 pass/fail + survey 대비 헤딩 커버리지 +
#    survey 주 헤딩 커버 여부)
python -m mag_heading check --cal project.cal.npz --pre <survey>-pre.csv --max-variation 2.0

# 3. 같은 프로젝트의 모든 -pre 파일에 compensation 적용 (기본값이 원본 Mag을 보존하고
#    MagComp 컬럼을 추가함; 원본을 덮어쓰려면 --overwrite)
python -m mag_heading compensate --pre <survey>-pre.csv --cal project.cal.npz
```

`--method spline`으로 두 방법(nearest-neighbor / thin-plate spline)의 `check` 결과를
비교해 어느 쪽이 해당 프로젝트에 더 적합한지 선택할 수 있다. `HaeNam_Mag` 샘플 데이터로
검증했을 때는 nearest 2.38nT → spline 1.77nT (self-compensation p2p)로 spline이 더
우수했다.

**"캘리브레이션 자료는 많을수록 좋은가?"** — 아니다. 벤더 매뉴얼과 Zhang et al. 논문 모두
자기평탄도와 헤딩 커버리지가 서로 상충한다고 명시한다. 중요한 건 자료 개수가 아니라
① heading 다양성(특히 survey의 주 비행 방향이 반드시 담길 것)과 ② 그 변화가 순전히
heading 때문이어야 한다는 것이다. `--trim-outliers`는 이미 충분히 대표되는(반복 관측이
있는) 헤딩에서 튀는 값만 골라 제거하고, 특정 헤딩의 유일한 관측치는 튀어 보여도 절대
지우지 않는다 — 지워버리면 바로 그 헤딩에서 compensation이 안 되는 구멍이 생기기 때문.

## 플롯

```bash
# 캘리브레이션 구간을 고르기 전에 전체 heading 분포를 훑어볼 때
python -m mag_heading plot-heading --pre <survey>-pre.csv --out heading.png

# 캘리브레이션 자기평탄도 확인 (raw vs self-compensated, Zhang et al. Fig. 5a 스타일)
python -m mag_heading plot-calibration --cal project.cal.npz --out cal_quality.png

# compensation 전후 2D 지도 비교 (GPS 좌표가 있는 -pre.csv 또는 -comp.csv 사용,
# --comp-col이 없는 파일이면 raw만 그림)
python -m mag_heading plot2d --pre <survey>-pre-comp.csv --out map.png

# compensation이 전체 survey에서 실제로 효과가 있었는지 시간영역에서 확인
# (벤더 프로그램 Module 4 "MagArrow Data Plot"과 동일한 스타일 — raw/compensated 시계열 중첩)
python -m mag_heading plot-timeseries --pre <survey>-pre-comp.csv --out timeseries.png
```

`plot-timeseries`는 self-compensation 플롯(캘리브레이션 구간 자체)과 달리 compensation을
**적용한 survey 파일 전체**에서 raw와 compensated를 겹쳐 그린다 — 캘리브레이션이 실제
survey에 얼마나 효과가 있었는지 눈으로 바로 확인하는 용도다. 개선 폭이 실제 지자기 신호
크기(수십~수백 nT)에 비해 작은 heading effect(보통 수 nT 이내)라서, 큰 스케일로 보면
raw/compensated가 거의 겹쳐 보일 수 있다 — 확대(줌인)해서 봐야 차이가 드러난다.

## 여러 파일 일괄 처리 (batch-compensate)

캘리브레이션 파일을 만든 뒤, 현장에서 취득한 모든 파일에 한꺼번에 적용해서 파일별로
실제로 얼마나 개선됐는지 확인한다. 헤딩 오차 자체보다 **스윙(swing)에 의한 값의 기복을
줄이는 것**이 이 도구의 핵심 목적이므로, 원시(raw) 자기장 표준편차가 아니라
sample-to-sample 1차 차분의 표준편차(`motion_noise_std_nT`)로 개선율을 측정한다 —
실제 이상체 신호(느리게 변하는 성분)에 오염되지 않는 지표다.

```bash
python -m mag_heading batch-compensate --pre-glob "D:/HaeNam_Mag/20260715/*-pre.csv" \
    --cal project.cal.npz --out-dir D:/HaeNam_Mag/20260715/compensated
```

파일별 개선율(%)이 표로 출력되고, 5% 미만인 파일은 "캘리브레이션 재검토 필요" 목록에
따로 표시된다. 여러 파일에 걸쳐 개선율이 고르게 좋으면 그 캘리브레이션 파일을 채택하고,
특정 파일에서만 개선이 약하면 그 구간의 헤딩이 캘리브레이션에 충분히 담기지 않았을
가능성이 높다 — `scan`으로 다시 구간을 찾아본다.

## 측선간 헤딩/방향 오차 레벨링 (level-survey)

`compensate`/`batch-compensate`의 ΔB(θ,φ) 보정은 **한 파일 안에서** 그 순간의 자세로부터
헤딩 효과를 제거한다. 하지만 이 보정이 캘리브레이션 비행에서 충분히 다루지 못한 특정 자세
(예: 실제 survey의 정방향/역방향 측선이 캘리브레이션 비행보다 훨씬 큰 pitch 차이로 날았을 때)
근처에서는, survey 전체에 걸쳐 **정방향과 역방향 측선 사이에 남는 일정한 오프셋**이 생길 수
있다 — 그리드에서 줄무늬(striping)로 보이는 원인 중 하나다. 이 오프셋은 한 파일의 자세
이력만으로는 알 수 없고, **survey 전체의 반대 방향 측선들을 직접 비교**해야 추정할 수 있다.

`level-survey`는 여러 파일(하나의 survey 전체)을 GPS 좌표로 합쳐 측선을 자동 검출하고,
공간적으로 겹치는 여러 국소 구역에서 "실제 지형 경사"와 "방향에 따른 계통 오차"를 분리해
낸 뒤, 그 오차가 정말로 방향에 의한 것인지(구역별 오프셋의 산포가 오프셋 자체보다 작은지)
자체 판정한다. 신뢰할 수 없다고 판단되면(산포가 오프셋보다 크면) **기본적으로 아무 것도
쓰지 않고 경고만 출력**한다 — `--force`로 강제할 수 있지만, 그 전에 출력된 경고를 확인해야
한다.

```bash
# --cal을 주면 compensate까지 한 번에 한다 (캘리브레이션 → 레벨링)
python -m mag_heading level-survey --pre-glob "D:/HaeNam_Mag/20260715/*-pre.csv" \
    --cal project.cal.npz --smooth-hz 1.0 --out-dir D:/HaeNam_Mag/20260715/leveled

# 이미 compensate까지 끝낸 -comp.csv들에 레벨링만 추가로 적용
python -m mag_heading level-survey --pre-glob "D:/HaeNam_Mag/20260715/compensated/*-comp.csv" \
    --out-dir D:/HaeNam_Mag/20260715/leveled
```

레벨링할 값은 `--value-col`로 지정하며, 기본값은 모든 파일에 `MagComp`가 있으면 `MagComp`,
아니면 `Mag`이다. 적용되면 원본 값에 방향별 상수 오프셋을 더한 `MagLeveled` 컬럼이 각
파일에 새로 추가되어 저장된다(기존 `Mag`/`MagComp`는 그대로 남는다 — vendor 프로그램의
"Keep Raw Data" 관례와 동일). 이 방법은 `processing/leveling.py`의 "local_plane" 방식(별도
자력탐사 자료처리 앱에서 이미 실측 검증된 방법)을 그대로 이식한 것이다.

## GUI

**`Run MagHeading GUI.bat`를 더블클릭하면 된다** — 터미널/명령어 입력 없이 바로 실행된다.
(원한다면 바탕화면에 바로가기를 만들어 둬도 된다: 파일을 마우스 오른쪽 클릭 →
"바로 가기 만들기" 후 바탕화면으로 옮기면 된다.)

명령줄에서 직접 실행하려면:

```bash
python -m mag_heading gui --pre <survey>-pre.csv
```

Tkinter 기반 경량 데스크톱 GUI(별도 설치 불필요). Mag 시계열 플롯 위에서 드래그해
캘리브레이션 구간을 고르거나(벤더 프로그램의 커서 방식과 동일), `scan` 결과 리스트에서
후보를 골라 바로 적용할 수 있다. 시계열 옆에는 GPS 좌표 기반 **비행 경로 플롯이 나란히
표시되고, 선택한 구간이 경로 위에 빨간 점으로 하이라이트**되어 그 구간이 측선상 어디에
해당하는지 바로 보인다(GPS 좌표가 없는 파일은 이 패널을 생략한다). 드래그로 구간을 고른 뒤
**"Add segment"를 누르면 그 구간이 목록에 쌓이고**(시계열 위에는 초록색 띠로, 경로 위에는
초록 점으로 표시됨 — 아직 추가 안 한 현재 드래그 선택은 빨간 점), 이렇게 여러 구간을
반복해서 추가한 뒤 "Build calibration"을 누르면 CLI의 다구간(`--start`/`--end` 여러 개) 기능과
동일하게 그 구간들을 모두 이어붙여 하나의 캘리브레이션으로 만든다("Remove selected segment"/
"Clear all segments"로 수정 가능). 이때 **"center segments" 체크박스**(기본 켜짐)가 각
구간의 실제 지자기장 레벨 차이를 보정해준다 — 서로 떨어진 구간을 보정 없이 그냥 합치면
그 공간적 차이가 heading 효과로 오인되어 모델이 심하게 망가질 수 있음이 실측으로 확인됐다
(자세한 근거는 매뉴얼 8.6절 참고). 목록에 추가한 구간이 없으면 기존처럼 현재 드래그
선택 하나만으로 모델을 만든다(단일 구간 워크플로는 그대로 유지됨). 모델을 만들면 품질
리포트와 self-compensation 플롯이 바로 갱신되고(우측 패널의 "trim outliers" 체크박스로
이상치 제거도 켤 수 있다), 캘리브레이션 모델 저장/불러오기,
compensation 적용 후에는 "2D Comparison" 탭에 **raw vs compensated 시간영역 그래프(벤더
프로그램 Module 4 스타일)가 위에, 2D 지도 비교가 아래에** 함께 뜬다 — compensation이 실제로
효과가 있었는지 시계열로 바로 확인할 수 있다. **"Batch" 탭**에서는 현장에서
취득한 여러 파일을 한꺼번에 불러와 현재 모델로 compensation을 적용하고, 파일별 motion-noise
개선율을 표로 비교해 캘리브레이션을 채택할지 다시 만들지 판단할 수 있다.

(참고: 원래 계획은 Streamlit이었으나, 이 환경에서 Windows 긴 경로(long path) 제한으로
streamlit 설치가 실패해 Tkinter로 전환함 — 표준 라이브러리만 사용하므로 이 문제를
피해간다.)

## 사용자 매뉴얼

`docs/MagHeading_User_Guide.pdf` — 설치, 명령어 레퍼런스, 실전 워크플로 예제, 품질 지표
해석 가이드, GUI 사용법, heading error compensation 이론 부록, 문제 해결을 담은 전체 매뉴얼.
CLI/동작이 바뀌면 재생성한다:

```bash
pip install reportlab   # 또는 requirements-dev.txt
python docs/build_manual.py
```

## 폴더 구조

```
src/mag_heading/   패키지 소스
tests/             단위 테스트
docs/              사용자 매뉴얼 PDF + 생성 스크립트 + 참조 그림
data/              샘플/작업 데이터 (git 미추적)
```
