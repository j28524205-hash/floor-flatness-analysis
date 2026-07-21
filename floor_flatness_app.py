import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata
import io
import os
from fpdf import FPDF

st.set_page_config(page_title="바닥면 평탄성 분석", layout="wide", page_icon="📐")

if "xyz" not in st.session_state:
    st.session_state.xyz = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ── 블루프린트 테마: Plotly 차트 기본 스타일 ──────────────────
_blueprint = go.layout.Template()
_blueprint.layout.paper_bgcolor = "#0b1e30"
_blueprint.layout.plot_bgcolor = "#0b1e30"
_blueprint.layout.font = dict(family="IBM Plex Mono, monospace", color="#c9dced", size=12)
_axis = dict(gridcolor="rgba(110,190,255,.15)", zerolinecolor="rgba(110,190,255,.3)",
             linecolor="rgba(143,168,196,.4)", color="#8fa9c7")
_blueprint.layout.xaxis = _axis
_blueprint.layout.yaxis = _axis
_blueprint.layout.scene = dict(
    xaxis=dict(backgroundcolor="#0b1e30", gridcolor="rgba(110,190,255,.15)", color="#8fa9c7"),
    yaxis=dict(backgroundcolor="#0b1e30", gridcolor="rgba(110,190,255,.15)", color="#8fa9c7"),
    zaxis=dict(backgroundcolor="#0b1e30", gridcolor="rgba(110,190,255,.15)", color="#8fa9c7"),
)
_blueprint.layout.legend = dict(font=dict(color="#c9dced"))
_blueprint.layout.colorway = ["#48d8f0", "#ffa53d", "#4fa3ff", "#ff5577", "#3ee08a"]
pio.templates["blueprint"] = _blueprint
pio.templates.default = "blueprint"

# ── 블루프린트 테마: 전역 CSS ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Big+Shoulders:wght@600;700;900&family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');

:root{
  --bp-bg:#081420; --bp-bg2:#0b1e30; --bp-ink:#e7f1fb; --bp-dim:#8fa9c7;
  --bp-border:rgba(143,168,196,.25); --bp-cyan:#48d8f0; --bp-amber:#ffb648;
}

html, body, [class*="css"]{ font-family:'IBM Plex Sans KR','IBM Plex Sans',sans-serif; }

[data-testid="stAppViewContainer"]{
  background:
    linear-gradient(rgba(110,190,255,.06) 1px, transparent 1px) 0 0/40px 40px,
    linear-gradient(90deg, rgba(110,190,255,.06) 1px, transparent 1px) 0 0/40px 40px,
    radial-gradient(1000px 600px at 10% -10%, #123456 0%, transparent 60%),
    var(--bp-bg);
  color: var(--bp-ink);
}
[data-testid="stHeader"]{ background:rgba(8,20,32,.7); }
.block-container{ padding-top:2rem; max-width:1180px; }

h1, h2, h3{
  font-family:'Big Shoulders','IBM Plex Sans KR',sans-serif !important;
  font-weight:800 !important; letter-spacing:.02em; text-transform:uppercase;
  color: var(--bp-ink) !important;
}
h1{ border-bottom:1px solid var(--bp-border); padding-bottom:.5rem; }

p, label, span, div{ color: var(--bp-ink); }

[data-testid="stMetric"]{
  background:rgba(255,255,255,.02); border:1px solid var(--bp-border); border-radius:6px;
  padding:12px 16px;
}
[data-testid="stMetricValue"]{ font-family:'IBM Plex Mono',monospace !important; color:var(--bp-cyan); }
[data-testid="stMetricLabel"]{ color:var(--bp-dim) !important; text-transform:uppercase; font-size:11px !important; letter-spacing:.05em; }

.stButton > button, .stDownloadButton > button{
  font-family:'IBM Plex Mono',monospace; letter-spacing:.03em;
  background:rgba(255,255,255,.03); color:var(--bp-ink);
  border:1px solid var(--bp-border); border-radius:5px;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  border-color:var(--bp-cyan); color:var(--bp-cyan); background:rgba(72,216,240,.08);
}
.stButton > button[kind="primary"]{
  background:linear-gradient(180deg,#57e3f7,#2fb8d4); color:#052430; font-weight:700; border:none;
}
.stButton > button[kind="primary"]:hover{ color:#052430; filter:brightness(1.08); }

[data-testid="stExpander"]{ border:1px solid var(--bp-border) !important; border-radius:6px !important; background:rgba(255,255,255,.015); }

input, textarea, [data-baseweb="input"] > div{
  font-family:'IBM Plex Mono',monospace !important;
  background:rgba(0,0,0,.25) !important; color:var(--bp-ink) !important;
  border-color: var(--bp-border) !important;
}

[data-testid="stDataFrame"]{ border:1px solid var(--bp-border); border-radius:6px; }

[data-testid="stTabs"] button{ font-family:'IBM Plex Mono',monospace; color:var(--bp-dim); }
[data-testid="stTabs"] button[aria-selected="true"]{ color:var(--bp-cyan) !important; border-bottom-color:var(--bp-cyan) !important; }

[data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.02) !important; border:1.5px dashed var(--bp-border) !important;
}

[data-testid="stAlert"]{ border-radius:6px; font-size:13.5px; }

[data-testid="stCaptionContainer"], .stCaption{ color:var(--bp-dim) !important; }

code, pre, [data-testid="stCodeBlock"]{
  font-family:'IBM Plex Mono',monospace !important;
  background:#050d16 !important; color:#9fd4de !important; border:1px solid var(--bp-border) !important;
}

hr{ border-color: var(--bp-border); }
</style>
""", unsafe_allow_html=True)


# ── 분석 함수 ──────────────────────────────────────────────

def remove_outliers(points, k=20, std_ratio=2.0, max_removal_ratio=0.3):
    """Returns (cleaned_points, removed_ratio, status).

    status: "ok" (정상 제거) | "no_outliers" (제거할 이상치 없음)
            | "skipped_too_many" (제거 예상 비율이 과도해 안전장치 발동, 원본 유지)
    """
    n0 = len(points)
    k = min(k, n0 - 1)
    if k < 1:
        return points, 0.0, "no_outliers"
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_d = dists[:, 1:].mean(axis=1)
    std = mean_d.std()
    if std < 1e-12:
        # 모든 포인트의 이웃 거리가 동일(완전 격자 등) → 이상치 없음, 그대로 반환
        return points, 0.0, "no_outliers"
    thresh = mean_d.mean() + std_ratio * std
    mask = mean_d < thresh
    removed_ratio = 1 - mask.mean()
    # SOR이 데이터 특성과 맞지 않아 너무 많이 제거하는 경우, 이상치 제거보다는
    # 파라미터/데이터 불일치일 가능성이 높으므로 원본을 그대로 반환한다.
    if removed_ratio > max_removal_ratio:
        return points, removed_ratio, "skipped_too_many"
    return points[mask], removed_ratio, "ok"


def separate_floor(points, z_margin, normal_angle_thresh, mode="OR"):
    """
    한계: 기본값(OR)은 z_mask와 angle_mask를 OR로 결합하므로, 바닥이 아닌 다른 수평면
    (테이블 상판, 계단 디딤판 등)이 스캔 범위에 포함되면 바닥으로 오분류될 수 있다.
    논문(2.2절 2단계)에 기술된 방법론이며, 단일 평면 바닥 스캔을 전제로 한 단순화이다.
    다른 수평 구조물이 섞인 스캔이면 mode="AND"(더 엄격한 분리)를 사용할 수 있다.
    """
    z_min = points[:, 2].min()
    z_mask = points[:, 2] <= z_min + z_margin

    # 법선 추정 (k=10 이웃, 전체 포인트에 대해 벡터화 — 대규모 점군에서 Python
    # 반복문 방식은 수십만~수백만 점에서 매우 느려 배포 서버가 응답 없음으로
    # 재시작되는 원인이 되었음. NearestNeighbors 조회 자체는 그대로 두고,
    # 이후 이웃별 공분산행렬 계산과 고유값분해만 전부 배치로 처리한다.
    n = len(points)
    k = min(10, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, idx = nbrs.kneighbors(points)
    neighbor_idx = idx[:, 1:]                                  # (n, k), 자기 자신 제외
    nb = points[neighbor_idx]                                  # (n, k, 3)
    centered = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k    # (n, 3, 3)
    _, eigvecs = np.linalg.eigh(cov)                           # 고유값 오름차순
    normals = eigvecs[:, :, 0]                                  # 최소고유값 방향 = 법선
    angle = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0, 1)))
    angle_mask = angle < normal_angle_thresh

    floor_mask = (z_mask & angle_mask) if mode == "AND" else (z_mask | angle_mask)
    return floor_mask


def fit_plane_ransac(points, threshold_m, n_iter=1000, seed=42, min_inlier_ratio=0.3):
    rng = np.random.default_rng(seed)
    n = len(points)
    best_plane, best_count = None, 0
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = -np.dot(normal, p1)
        dists = np.abs(points @ normal + d)
        count = (dists < threshold_m).sum()
        if count > best_count:
            best_count = count
            best_plane = np.append(normal, d)
    if best_plane is None:
        raise ValueError(
            "RANSAC이 유효한 평면을 찾지 못했습니다. 포인트가 거의 일직선상에 있거나 "
            "중복점이 너무 많을 수 있습니다."
        )

    min_inliers = max(30, int(n * min_inlier_ratio))
    if best_count < min_inliers:
        raise ValueError(
            f"기준 평면을 안정적으로 추정할 만큼 인라이어가 부족합니다 "
            f"({best_count}개 / 최소 {min_inliers}개 필요). 바닥면이 실제로 평평한지, "
            "또는 바닥 분리 파라미터가 적절한지 확인해보세요."
        )

    # 최적 3점 평면은 랜덤 샘플일 뿐이므로, 해당 평면의 인라이어 전체를 모아
    # 최소제곱(SVD) 평면으로 재적합해 정확도를 높인다.
    a, b, c, d = best_plane
    dists = np.abs(points @ np.array([a, b, c]) + d)
    inliers = points[dists < threshold_m]
    if len(inliers) >= 3:
        centroid = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        d = -np.dot(normal, centroid)
        best_plane = np.append(normal, d)

    # 법선이 위를 향하도록 (양수 = 돌출, 음수 = 함몰)
    if best_plane[2] < 0:
        best_plane = -best_plane
    return best_plane


def detect_cracks(points, k=15):
    """
    법선 벡터 기반 크랙 탐지 (곡률 + 법선 불연속성 + 선형성 결합)

    한계: norm01()이 해당 데이터셋 내부의 최소/최대값으로 0~1 정규화를 하므로,
    같은 curv_thresh(크랙 민감도) 값이라도 데이터셋마다 실제로 걸러내는 기준이
    달라질 수 있다. 서로 다른 스캔 파일 간 결과를 절대적으로 비교하려면
    이 상대적 정규화 방식을 절대/분위수 기준으로 바꿔야 한다.
    """
    # 이웃 조회를 제외한 전 과정을 벡터화한다(사유는 separate_floor 주석 참고).
    n = len(points)
    k = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, idx = nbrs.kneighbors(points)
    neighbor_idx = idx[:, 1:]                                   # (n, k)

    nb = points[neighbor_idx]                                   # (n, k, 3)
    centered = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k     # (n, 3, 3)
    eigvals, eigvecs = np.linalg.eigh(cov)                       # 오름차순
    eigvals = np.abs(eigvals)

    normals = eigvecs[:, :, 0].copy()                            # 최소고유값 방향
    flip = normals[:, 2] < 0
    normals[flip] = -normals[flip]

    curvatures = eigvals[:, 0] / (eigvals.sum(axis=1) + 1e-10)
    # 선형성(linearity): 크랙은 한 방향으로만 곡률이 높음. 변수명은 anisotropy이나
    # 지역 기하특징 표기 관례상 이 식은 비등방성((λ2-λ0)/λ2)이 아니라 선형성에 해당함
    anisotropy = (eigvals[:, 2] - eigvals[:, 1]) / (eigvals[:, 2] + 1e-10)

    # 법선 불연속성: 이웃 법선과의 평균 각도 차이
    nb_normals = normals[neighbor_idx]                           # (n, k, 3)
    cos_a = np.clip(np.abs(np.einsum('nkj,nj->nk', nb_normals, normals)), 0, 1)
    discontinuity = np.degrees(np.arccos(cos_a)).mean(axis=1)

    def norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-10)

    # 결합 크랙 점수 (0~1): 불연속성 40% + 곡률 40% + 선형성 20%
    score = (norm01(discontinuity) * 0.4 +
             norm01(curvatures) * 0.4 +
             norm01(anisotropy) * 0.2)

    return score


def find_korean_font():
    """OS별 한글 지원 폰트를 탐색해 (정규체, 굵은체) 경로를 반환한다."""
    candidates = [
        # Windows
        ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"),
        # Linux (Noto Sans CJK / Nanum Gothic 등 일반적인 설치 경로)
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
         "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        # macOS
        ("/Library/Fonts/AppleGothic.ttf", "/Library/Fonts/AppleGothic.ttf"),
    ]
    for regular, bold in candidates:
        if os.path.exists(regular):
            return regular, (bold if os.path.exists(bold) else regular)
    return None, None


def generate_pdf(report_text, summary_rows, worst_rows_data, grade_icon, grade_str, bad_pct,
                 fig_heatmap=None, fig_hist=None, fig_surf=None):
    font_regular, font_bold = find_korean_font()
    if font_regular is None:
        raise RuntimeError(
            "한글을 지원하는 폰트를 찾을 수 없습니다. Windows는 맑은 고딕, Linux는 "
            "Noto Sans CJK 또는 나눔고딕 설치가 필요합니다."
        )
    pdf = FPDF()
    pdf.add_font("K", fname=font_regular)
    pdf.add_font("KB", fname=font_bold)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    # 제목
    pdf.set_font("KB", size=16)
    pdf.cell(0, 12, "바닥면 평탄성 분석 결과 보고서", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("K", size=10)
    pdf.cell(0, 6, "Floor Flatness Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(4)

    # 보고서 본문 (섹션별로)
    for line in report_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
            continue
        if stripped.startswith("==="):
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            continue
        if stripped.startswith("---"):
            pdf.set_draw_color(200, 200, 200)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.set_draw_color(0, 0, 0)
            continue
        if stripped.startswith("[ ") and stripped.endswith(" ]"):
            pdf.ln(2)
            pdf.set_font("KB", size=11)
            pdf.cell(0, 7, stripped, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("K", size=10)
        else:
            pdf.set_font("K", size=10)
            pdf.cell(0, 6, line.rstrip(), new_x="LMARGIN", new_y="NEXT")

    # 요약 테이블
    pdf.ln(4)
    pdf.set_font("KB", size=11)
    pdf.cell(0, 8, "[ 분류별 상세 요약 ]", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("K", size=9)

    headers = ["분류", "포인트 수", "비율(%)", "평균편차(mm)", "최대편차(mm)", "표준편차(mm)"]
    col_w   = [40, 25, 20, 32, 32, 32]
    pdf.set_fill_color(230, 230, 230)
    for h, w in zip(headers, col_w):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_fill_color(255, 255, 255)
    for row in summary_rows:
        vals = [row["분류"], str(row["포인트 수"]), row["비율 (%)"],
                row["평균 편차 (mm)"], row["최대 편차 (mm)"], row["표준편차 (mm)"]]
        for v, w in zip(vals, col_w):
            pdf.cell(w, 6, v, border=1, align="C")
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("KB", size=12)
    pdf.cell(0, 10, f"종합 등급 : {grade_str}  (불량 비율 {bad_pct:.1f}%)",
             new_x="LMARGIN", new_y="NEXT", align="C")

    # ── 시각화 이미지 삽입 ──
    def insert_fig(fig, title, width=170):
        if fig is None:
            return
        try:
            img_bytes = fig.to_image(format="png", width=900, height=500, scale=1.5)
            tmp = io.BytesIO(img_bytes)
            pdf.add_page()
            pdf.set_font("KB", size=12)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
            pdf.image(tmp, x=20, w=width)
        except Exception as e:
            # 조용히 무시하면 사용자가 그림 누락을 알 수 없으므로 명확히 실패시킨다
            # (호출부에서 "PDF 생성 실패" 경고로 사용자에게 표시됨).
            raise RuntimeError(f"'{title}' 이미지를 PDF에 삽입하지 못했습니다: {e}") from e

    insert_fig(fig_heatmap, "[ 2D 평탄성 히트맵 ]")
    insert_fig(fig_hist,    "[ 편차 분포 그래프 ]")
    insert_fig(fig_surf,    "[ 3D 표면 시각화 ]")

    out = io.BytesIO()
    pdf.output(out)
    return out.getvalue()


def point_to_plane_dist(points, plane):
    a, b, c, d = plane
    return (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)


@st.cache_data(show_spinner=False)
def safe_griddata(xy, values, mesh):
    """선형 보간이 실패(일직선/삼각분할 불가 등)하면 nearest로 재시도한다.

    Returns:
        (보간 결과, 실제 사용된 method 문자열 "linear" | "nearest")

    `st.cache_data`로 캐싱한다: 이 함수는 대규모 점군(수만~수십만 점)에서
    Qhull 삼각분할 비용이 커서 느린데, session_state로 분석 결과를 유지하는
    방식(Phase 10.9)으로 바꾼 뒤에는 Z 과장 배율 슬라이더처럼 이 결과와
    무관한 위젯을 조작해도 화면 전체가 매번 재실행되면서 이 함수도 매번
    다시 호출됐다. 캐싱 없이는 실제 대규모 데이터에서 위젯을 조작할 때마다
    무거운 보간을 반복 실행해 배포 서버가 응답 없음으로 죽는 원인이 됐다.
    입력(xy, values, mesh)이 그대로면 캐시를 재사용해 즉시 반환된다.
    """
    x, y = xy
    try:
        return griddata((x, y), values, mesh, method="linear"), "linear"
    except Exception:
        return griddata((x, y), values, mesh, method="nearest"), "nearest"


# RANSAC 인라이어 임계값은 판정 기준(threshold_mm)과 분리된 고정값을 사용한다.
# 판정 기준을 바꿔도 기준 평면 자체는 흔들리지 않도록 하기 위함.
RANSAC_INLIER_MM = 5.0
MIN_POINTS = 30
# Streamlit Community Cloud 무료 티어 메모리 한도(약 1GB) 안에서 안전하게 돌도록
# 하는 분석용 포인트 수 상한. 실제 스캐너로 취득한 수십만~수백만 점 데이터가
# 이 한도를 넘기면 메모리 부족으로 앱이 강제 재시작되는 문제가 있었음.
MAX_ANALYSIS_POINTS = 300_000


def run_analysis(xyz, z_margin, normal_angle, threshold_mm, curv_thresh,
                  ransac_inlier_mm=RANSAC_INLIER_MM, floor_mode="OR",
                  min_inlier_ratio=0.3, sor_std_ratio=2.0, ransac_n_iter=1000,
                  max_analysis_points=MAX_ANALYSIS_POINTS):
    """이상치 제거 → 바닥 분리 → 평면 추정 → 균열 탐지 → 포인트 분류까지 전체 파이프라인 실행."""
    downsampled_from = None
    if len(xyz) > max_analysis_points:
        downsampled_from = len(xyz)
        rng = np.random.default_rng(42)
        sel = rng.choice(len(xyz), max_analysis_points, replace=False)
        xyz = xyz[sel]

    clean, outlier_removed_ratio, outlier_status = remove_outliers(xyz, std_ratio=sor_std_ratio)
    if len(clean) < MIN_POINTS:
        raise ValueError(
            f"이상치 제거 후 남은 포인트({len(clean):,}개)가 너무 적습니다. "
            "원본 데이터에 노이즈가 많거나 포인트 수가 부족할 수 있습니다."
        )

    floor_mask = separate_floor(clean, z_margin, normal_angle, mode=floor_mode)
    floor_pts = clean[floor_mask]
    wall_pts = clean[~floor_mask]

    if len(floor_pts) < 4:
        raise ValueError(
            "바닥면으로 분리된 포인트가 너무 적습니다 "
            "(Z 여유값 또는 법선 각도 임계값을 조정해보세요)."
        )

    plane = fit_plane_ransac(floor_pts, ransac_inlier_mm / 1000, n_iter=ransac_n_iter,
                              min_inlier_ratio=min_inlier_ratio)
    signed_mm = point_to_plane_dist(floor_pts, plane) * 1000
    abs_mm = np.abs(signed_mm)
    curvs = detect_cracks(floor_pts)

    crack_mask = curvs > curv_thresh
    protrude_mask = (signed_mm > threshold_mm) & ~crack_mask
    depress_mask = (signed_mm < -threshold_mm) & ~crack_mask
    good_mask = ~crack_mask & ~protrude_mask & ~depress_mask

    labels = np.full(len(floor_pts), "A_정상", dtype=object)
    labels[protrude_mask] = "B1_돌출"
    labels[depress_mask] = "B2_함몰"
    labels[crack_mask] = "C_크랙"

    return dict(
        floor_pts=floor_pts, wall_pts=wall_pts,
        signed_mm=signed_mm, abs_mm=abs_mm, curvs=curvs,
        crack_mask=crack_mask, protrude_mask=protrude_mask,
        depress_mask=depress_mask, good_mask=good_mask, labels=labels,
        outlier_removed_ratio=outlier_removed_ratio, outlier_status=outlier_status,
        downsampled_from=downsampled_from,
    )


# ── UI ────────────────────────────────────────────────────

st.title("🏗️ 바닥면 평탄성 분석")
st.warning(
    "⚠️ 연구용 프로토타입입니다. 공식 KCS 적합성 판정이나 "
    "현장 최종 판정 근거로 사용할 수 없습니다. "
    "2026-07-21 합성 데이터 ground truth 대조 검증(코드 커밋 기준, `validation/` 폴더) 결과, "
    "이 버전의 기본 파라미터로는 균열(C) 탐지 recall이 0.4%(사실상 미탐지)였습니다 — 코드나 파라미터가 "
    "바뀌면 이 수치도 달라질 수 있으니 최신 검증 결과는 PRD.md 7절을 확인하세요. "
    "업로드하는 점군 데이터에 민감한 위치·구조 정보가 포함되지 않도록 주의하세요."
)

# ── 데이터 입력 ──
st.subheader("📁 데이터 입력")
uploaded = st.file_uploader("CSV 파일 불러오기", type=["csv", "txt", "xyz"],
                             help="컬럼 형식: X, Y, Z, 헤더 포함")
unit_choice = st.radio("좌표 단위", ["m (미터)", "mm (밀리미터)"], horizontal=True,
                        help="아래 모든 기준값(Z 여유값, 허용 편차 등)은 m 단위입니다. "
                             "mm 단위 파일이면 이걸 선택해야 자동으로 m로 환산합니다.")
use_sample = st.button("샘플 데이터 사용 (바닥+벽체 포함)")

if uploaded:
    # st.file_uploader()는 st.button()과 달리 한 번 업로드되면 파일을 계속 들고
    # 있어서, 이 블록이 슬라이더 조작 등 다른 위젯으로 인한 재실행 때마다 매번
    # 다시 실행된다. 그때마다 "새 파일이 들어왔다"고 착각해 아래에서
    # analysis_result를 초기화해버리면, 분석 후 슬라이더만 움직여도 결과가
    # 사라지는 문제가 생긴다. 파일 서명이 실제로 바뀐 경우에만 재처리한다.
    file_sig = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    is_new_upload = st.session_state.get("last_uploaded_sig") != file_sig
else:
    is_new_upload = False

if uploaded and is_new_upload:
    st.session_state.last_uploaded_sig = file_sig
    try:
        raw = uploaded.read()
        # 1차: numpy loadtxt (공백/탭 구분, 헤더 없음)
        try:
            data = np.loadtxt(io.BytesIO(raw))
        except Exception:
            # 2차: pandas 자동 구분자 감지
            try:
                data = pd.read_csv(io.BytesIO(raw), sep=None, engine="python",
                                   header=None, comment="#").values.astype(float)
            except Exception:
                # 3차: 헤더 있는 CSV
                data = pd.read_csv(io.BytesIO(raw)).iloc[:, :3].values.astype(float)

        if data.ndim != 2 or data.shape[1] < 3:
            st.error(f"로드 실패: 최소 3개 컬럼(X, Y, Z)이 필요합니다. "
                     f"감지된 형태: {data.shape}")
            st.stop()

        xyz_candidate = data[:, :3].astype(float)

        n_bad = ~np.isfinite(xyz_candidate).all(axis=1)
        if n_bad.any():
            xyz_candidate = xyz_candidate[~n_bad]
            st.warning(f"⚠️ NaN/Inf가 포함된 {n_bad.sum():,}개 포인트를 제외했습니다.")

        if len(xyz_candidate) == 0:
            st.error("로드 실패: 유효한 포인트가 없습니다.")
            st.stop()

        if unit_choice.startswith("mm"):
            xyz_candidate = xyz_candidate / 1000.0
            st.info("ℹ️ mm → m 단위로 환산했습니다.")
        else:
            # 명시적으로 m를 선택했더라도, 좌표 범위가 비정상적으로 크면
            # mm 파일을 잘못 선택했을 가능성을 마지막으로 한 번 더 경고한다.
            span = xyz_candidate.max(axis=0) - xyz_candidate.min(axis=0)
            x_span, y_span, z_span = span
            if z_span > 50 or max(x_span, y_span) > 500:
                st.warning(f"⚠️ 좌표 범위가 비정상적으로 큽니다 (X:{x_span:.1f}, Y:{y_span:.1f}, "
                           f"Z:{z_span:.1f}). mm 단위 파일인데 'm'를 선택하지 않았는지 확인해보세요.")

        st.session_state.xyz = xyz_candidate
        st.session_state.xyz_unit_note = unit_choice
        st.session_state.analysis_result = None  # 새 데이터 로드 시 이전 분석 결과는 무효화
        st.success(f"✅ 로드됨: {len(st.session_state.xyz):,}개 포인트 ({data.shape[1]}개 컬럼, "
                   f"단위: {unit_choice} → m로 처리됨)")
    except Exception as e:
        st.error(f"로드 실패: {e}\n\n지원 형식: X Y Z (공백/탭/쉼표 구분, 헤더 있/없 모두 가능)")

if use_sample:
    rng = np.random.default_rng(42)
    n_f, n_w = 500, 250
    fx = rng.uniform(0, 5, n_f)
    fy = rng.uniform(0, 5, n_f)
    fz = rng.normal(0, 0.002, n_f)
    fz += np.exp(-((fx - 2.5)**2 + (fy - 2.5)**2) / 0.5) * 0.015
    floor = np.column_stack([fx, fy, fz])
    wx = rng.uniform(0, 5, n_w)
    wz = rng.uniform(0.05, 1.0, n_w)
    wy = np.where(rng.random(n_w) > 0.5, 0.0, 5.0)
    wall = np.column_stack([wx, wy, wz])
    st.session_state.xyz = np.vstack([floor, wall])
    st.session_state.xyz_unit_note = "m (샘플 데이터)"
    st.session_state.analysis_result = None  # 새 데이터 로드 시 이전 분석 결과는 무효화

xyz = st.session_state.xyz
if xyz is not None:
    unit_note = st.session_state.get("xyz_unit_note", "m")
    st.info(f"✅ 현재 로드된 데이터: {len(xyz):,}개 포인트 (단위: {unit_note})")

if xyz is not None:

    # ── 판정 기준 ──
    st.subheader("⚙️ 판정 기준 설정")
    c1, c2 = st.columns(2)
    threshold_mm = c1.number_input("평탄도 허용 편차 (mm)", value=3.00, min_value=0.1, step=0.1, format="%.2f")
    curv_thresh = c2.number_input("크랙 민감도 (0~1, 낮을수록 민감)", value=0.65, min_value=0.1, max_value=0.99, step=0.01, format="%.2f")
    st.caption("※ KCS 41 46 01(바닥 평탄도 허용오차 ±3mm)에서 채택한 수치 기준이며, "
               "이 값을 점-기준평면 편차에 적용한 자체 작업 기준입니다. "
               "KCS 원 규정의 3m 직선자 측정 절차를 그대로 재현한 것이 아니며, "
               "공식 적합·부적합 판정 근거로 사용할 수 없습니다.")
    st.caption("※ 크랙 민감도는 이번 스캔 데이터 내 상대적 기준이라, 다른 스캔 파일과 "
               "같은 값이어도 실제로 걸러지는 정도가 달라질 수 있습니다.")

    # ── 벽체 분리 ──
    st.subheader("🧱 벽·벽체 자동 분리")
    c1, c2 = st.columns(2)
    z_margin = c1.number_input("Z 여유값 (바닥 두께, m)", value=0.05, min_value=0.01, step=0.01, format="%.2f")
    normal_angle = c2.number_input("법선 각도 임계값 (°)", value=35.0, min_value=5.0, max_value=85.0, step=1.0)
    st.caption("※ 바닥 외 다른 수평면(테이블 상판, 계단 디딤판 등)이 스캔 범위에 있으면 "
               "바닥으로 오인식될 수 있습니다. 단일 바닥면 스캔을 전제로 합니다.")
    floor_mode_label = st.radio(
        "바닥 분리 방식", ["OR - 기본 방식 (논문 기준)", "AND - 엄격 방식 (비바닥 수평면 제외)"],
        help="OR: Z 여유값 이내이거나 수평면이면 바닥으로 인정(기본, 논문에 서술된 방식). "
             "AND: 두 조건을 모두 만족해야 바닥으로 인정 — 테이블 상판/계단 디딤판 등 "
             "다른 수평 구조물이 섞인 스캔이면 이 옵션을 권장합니다.")
    floor_mode = "AND" if floor_mode_label.startswith("AND") else "OR"

    # ── 고급 설정 ──
    with st.expander("🔧 고급 설정"):
        ransac_inlier_mm = st.number_input(
            "기준 평면 추정 허용오차 (RANSAC 인라이어, mm)",
            value=RANSAC_INLIER_MM, min_value=0.5, step=0.5, format="%.1f",
            help="판정 기준(허용 편차)과 별개로, 기준 평면을 추정할 때 '바닥'으로 볼 점의 "
                 "오차 범위입니다. 장비 정밀도나 스캔 밀도에 따라 조정하세요.")
        min_inlier_pct = st.number_input(
            "기준 평면 최소 인라이어 비율 (%)",
            value=30.0, min_value=5.0, max_value=90.0, step=5.0, format="%.0f",
            help="RANSAC이 찾은 최적 평면이라도, 이 비율 미만의 점만 지지한다면 "
                 "신뢰도가 낮다고 보고 분석을 중단합니다. 복잡한 형상의 바닥이면 낮추세요.")
        min_inlier_ratio = min_inlier_pct / 100.0
        sor_std_ratio = st.number_input(
            "이상치 제거 민감도 (표준편차 배수)",
            value=2.0, min_value=0.5, max_value=5.0, step=0.1, format="%.1f",
            help="작을수록 더 엄격하게(많이) 이상치를 제거합니다. 스캐너 노이즈가 크면 값을 키우세요.")
        ransac_n_iter = st.number_input(
            "RANSAC 반복 횟수",
            value=1000, min_value=100, max_value=10000, step=100,
            help="많을수록 더 안정적인 평면을 찾지만 느려집니다. 포인트 수가 매우 많거나 "
                 "이상치 비율이 높으면 늘려보세요.")
        max_analysis_points = st.number_input(
            "분석 최대 포인트 수 (초과 시 무작위 다운샘플링)",
            value=MAX_ANALYSIS_POINTS, min_value=10_000, max_value=2_000_000, step=50_000,
            help="실제 스캐너 데이터가 수십만~수백만 점이면 서버 메모리 한도를 넘어 "
                 "앱이 강제 재시작될 수 있습니다. 이 값을 넘으면 무작위로 샘플링해 분석합니다.")

    # ── 분석 실행 ──
    # 분석 결과를 session_state에 저장해서, 이후 슬라이더(Z 과장 배율 등) 조작으로
    # 스크립트가 재실행돼도 결과가 사라지지 않도록 한다. st.button()은 클릭된 그 순간의
    # 재실행에서만 True이므로, 결과 표시를 버튼 블록 안에 두면 다른 위젯 조작 시 결과가
    # 사라지는 문제가 있었다.
    if len(xyz) < MIN_POINTS:
        st.warning(f"⚠️ 분석에는 최소 {MIN_POINTS}개 이상의 포인트가 필요합니다. "
                   f"현재 입력된 포인트: {len(xyz):,}개")
    else:
        if st.button("▶ 분석 실행", type="primary", use_container_width=True):
            try:
                with st.spinner("분석 중..."):
                    st.session_state.analysis_result = run_analysis(
                        xyz, z_margin, normal_angle, threshold_mm, curv_thresh,
                        ransac_inlier_mm=ransac_inlier_mm, floor_mode=floor_mode,
                        min_inlier_ratio=min_inlier_ratio,
                        sor_std_ratio=sor_std_ratio, ransac_n_iter=ransac_n_iter,
                        max_analysis_points=max_analysis_points)
            except Exception as e:
                st.session_state.analysis_result = None
                st.error(f"⚠️ 분석 중 오류가 발생했습니다: {e}\n\n"
                         "포인트 배치가 지나치게 단순(일직선 등)하거나 파라미터가 데이터와 맞지 않을 수 있습니다.")

    if st.session_state.get("analysis_result") is not None:
        result = st.session_state.analysis_result
        floor_pts = result["floor_pts"]
        wall_pts = result["wall_pts"]
        signed_mm = result["signed_mm"]
        abs_mm = result["abs_mm"]
        curvs = result["curvs"]
        crack_mask = result["crack_mask"]
        protrude_mask = result["protrude_mask"]
        depress_mask = result["depress_mask"]
        good_mask = result["good_mask"]
        labels = result["labels"]
        outlier_removed_ratio = result["outlier_removed_ratio"]
        outlier_status = result["outlier_status"]
        downsampled_from = result["downsampled_from"]
        if downsampled_from is not None:
            st.info(f"ℹ️ 원본 {downsampled_from:,}개 포인트가 분석 상한을 초과해, "
                     f"무작위로 {max_analysis_points:,}개로 다운샘플링한 뒤 분석했습니다 "
                     "(고급 설정에서 상한 조정 가능).")
        if outlier_status == "ok" and outlier_removed_ratio > 0:
            st.caption(f"🧹 이상치 제거: 전체의 {outlier_removed_ratio*100:.1f}%를 노이즈로 제거했습니다.")
        elif outlier_status == "skipped_too_many":
            st.warning(f"⚠️ 이상치 제거 생략: 예상 제거율({outlier_removed_ratio*100:.1f}%)이 너무 높아 "
                       "안전장치가 발동해 원본 데이터를 그대로 사용했습니다. 데이터나 파라미터를 확인해보세요.")

        CLASS_META = {
            "A_정상":  ("🟢 정상",   "#2ecc71", good_mask),
            "B1_돌출": ("🟠 돌출",   "#e67e22", protrude_mask),
            "B2_함몰": ("🔵 함몰",   "#3498db", depress_mask),
            "C_크랙":  ("🔴 크랙",   "#e74c3c", crack_mask),
        }

        # ── 결과 요약 ──
        st.subheader("📊 분석 결과 요약")
        c1, c2, c3 = st.columns(3)
        c1.metric("총 입력점",  f"{len(xyz):,}개")
        c2.metric("벽체 제거됨", f"{len(wall_pts):,}개", f"{len(wall_pts)/len(xyz)*100:.1f}%")
        c3.metric("바닥 분석점", f"{len(floor_pts):,}개")

        c1, c2, c3, c4 = st.columns(4)
        for col, (key, (label, _, mask)) in zip([c1, c2, c3, c4], CLASS_META.items()):
            col.metric(label, f"{mask.sum():,}개", f"{mask.mean()*100:.1f}%")

        st.caption(
            f"평균 편차: {abs_mm.mean():.2f}mm | "
            f"최대 돌출: +{signed_mm.max():.2f}mm | "
            f"최대 함몰: {signed_mm.min():.2f}mm | "
            f"표준편차: {abs_mm.std():.2f}mm"
        )

        bad_pct = (~good_mask).mean() * 100
        if bad_pct <= 5:
            st.success("📋 종합 등급: ⭐ 우수 (Excellent)")
        elif bad_pct <= 15:
            st.success("📋 종합 등급: ✅ 양호 (Good)")
        elif bad_pct <= 30:
            st.warning("📋 종합 등급: ⚠️ 보통 (Fair)")
        else:
            st.error("📋 종합 등급: 🔴 불량 (Poor)")

        # ── 히트맵 ──
        st.subheader("🗺️ 2D 평탄성 히트맵")
        x, y = floor_pts[:, 0], floor_pts[:, 1]
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        fig = None
        try:
            zi, method_used = safe_griddata((x, y), abs_mm, np.meshgrid(xi, yi))
            fig = go.Figure(go.Heatmap(
                x=xi, y=yi, z=zi,
                colorscale="RdYlGn_r",
                colorbar=dict(title="편차 (mm)")
            ))
            fig.update_layout(xaxis_title="X (m)", yaxis_title="Y (m)",
                              height=400, margin=dict(l=40, r=40, t=10, b=40))
            if method_used == "nearest":
                st.caption("※ 포인트 배치상 선형 보간이 불가능해 최근접(nearest) 보간을 대신 사용했습니다 "
                           "(경계가 계단형으로 보일 수 있음).")
        except Exception:
            st.warning("⚠️ 2D 히트맵을 생성할 수 없습니다 (포인트 배치가 격자화하기 어려운 형태일 수 있습니다).")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        # ── 편차 분포 그래프 ──
        st.subheader("📈 편차 분포 (기준 대비)")
        hist_fig = go.Figure()
        for key, (label, color, mask) in CLASS_META.items():
            if mask.sum() == 0:
                continue
            hist_fig.add_trace(go.Histogram(
                x=signed_mm[mask],
                name=label,
                marker_color=color,
                opacity=0.75,
                nbinsx=40,
                hovertemplate=f"{label}<br>편차: %{{x:.2f}}mm<br>포인트: %{{y}}개<extra></extra>"
            ))
        hist_fig.add_vline(x= threshold_mm, line_dash="dash", line_color="gray",
                           annotation_text=f"+{threshold_mm}mm 기준", annotation_position="top right")
        hist_fig.add_vline(x=-threshold_mm, line_dash="dash", line_color="gray",
                           annotation_text=f"-{threshold_mm}mm 기준", annotation_position="top left")
        hist_fig.add_vline(x=0, line_color="black", line_width=1)
        hist_fig.update_layout(
            barmode="overlay",
            xaxis_title="편차 (mm)  [+ : 돌출 / - : 함몰]",
            yaxis_title="포인트 수",
            legend=dict(orientation="h", y=1.1),
            height=350, margin=dict(l=40, r=20, t=30, b=40)
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        # ── 3D 시각화 ──
        st.subheader("🔭 3D 평탄성 시각화")

        VIS_MAX = 50_000
        if len(floor_pts) > VIS_MAX:
            # 시드를 고정해 같은 분석 결과에 대해서는 위젯을 조작해도(예: Z 과장
            # 배율 슬라이더) 매번 재실행 때마다 다른 점이 뽑히지 않도록 한다.
            vis_idx = np.random.default_rng(42).choice(len(floor_pts), VIS_MAX, replace=False)
            st.caption(f"⚡ 시각화: {len(floor_pts):,}개 → {VIS_MAX:,}개로 자동 샘플링 (분석은 전체 데이터 사용)")
        else:
            vis_idx = np.arange(len(floor_pts))

        EXAG_MAX = 500
        exag = st.slider("Z 과장 배율 (높낮이 강조)", min_value=1, max_value=EXAG_MAX,
                          value=100, step=10,
                          help="실제 편차는 mm 단위라 육안으로 안 보임. 배율을 높이면 높낮이가 강조됨")

        tab1, tab2, tab3 = st.tabs(["🏔️ 3D 표면 (높낮이)", "분류별 3D 시각화", "전체 포인트 클라우드"])

        with tab1:
            # Surface 플롯: 편차값을 Z로 사용 (과장 적용)
            x, y = floor_pts[:, 0], floor_pts[:, 1]
            res = min(150, len(floor_pts) // 2)
            xi_s = np.linspace(x.min(), x.max(), res)
            yi_s = np.linspace(y.min(), y.max(), res)
            fig_surf = None
            try:
                zi_signed, method_signed = safe_griddata((x, y), signed_mm, np.meshgrid(xi_s, yi_s))
                zi_abs_s, method_abs = safe_griddata((x, y), abs_mm, np.meshgrid(xi_s, yi_s))

                # z축 표시 범위를 "슬라이더 최댓값 기준"으로 고정한다. 그렇지 않으면
                # Plotly가 aspectmode="manual" 박스 안에 항상 꽉 차도록 자동으로
                # 재조정해버려서, exag를 바꿔도 모양이 똑같아 보이고 축 눈금 숫자만
                # 바뀌는 문제가 있었다(과장 배율 슬라이더가 시각적으로 무의미해짐).
                raw_max_mm = np.nanmax(np.abs(zi_signed))
                if not np.isfinite(raw_max_mm) or raw_max_mm < 1e-9:
                    raw_max_mm = 1.0
                fixed_range_m = raw_max_mm * EXAG_MAX / 1000

                fig_surf = go.Figure(go.Surface(
                    x=xi_s, y=yi_s,
                    z=zi_signed * exag / 1000,   # mm → m 변환 후 과장
                    surfacecolor=zi_abs_s,
                    colorscale="RdYlGn_r",
                    colorbar=dict(title="편차 (mm)"),
                    hovertemplate="X: %{x:.3f}m<br>Y: %{y:.3f}m<br>편차: %{surfacecolor:.2f}mm<extra></extra>"
                ))
                fig_surf.update_layout(
                    scene=dict(
                        xaxis_title="X (m)", yaxis_title="Y (m)",
                        zaxis=dict(title=f"편차 (×{exag} 과장)", range=[-fixed_range_m, fixed_range_m]),
                        aspectmode="manual",
                        aspectratio=dict(x=1, y=1, z=0.4)
                    ),
                    height=550, margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_surf, use_container_width=True)
                st.caption(f"※ Z축은 실제 편차의 {exag}배 과장 표시 (색상은 실제 편차값 기준)")
                if "nearest" in (method_signed, method_abs):
                    st.caption("※ 포인트 배치상 선형 보간이 불가능해 최근접(nearest) 보간을 대신 사용했습니다.")
            except Exception:
                st.warning("⚠️ 3D 표면을 생성할 수 없습니다 (포인트 배치가 격자화하기 어려운 형태일 수 있습니다).")

        with tab2:
            fig3d = go.Figure()
            for key, (label, color, mask) in CLASS_META.items():
                sub = vis_idx[mask[vis_idx]]
                if len(sub) == 0:
                    continue
                fig3d.add_trace(go.Scatter3d(
                    x=floor_pts[sub, 0], y=floor_pts[sub, 1], z=floor_pts[sub, 2],
                    mode="markers", name=label,
                    marker=dict(size=3, color=color, opacity=0.85),
                    hovertemplate=(
                        f"{label}<br>"
                        "X: %{x:.3f}m<br>Y: %{y:.3f}m<br>Z: %{z:.3f}m<extra></extra>"
                    )
                ))
            fig3d.update_layout(
                scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                           aspectmode="data"),
                legend=dict(orientation="h", y=-0.05),
                height=550, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with tab3:
            fig_all = go.Figure()
            for key, (label, color, mask) in CLASS_META.items():
                sub = vis_idx[mask[vis_idx]]
                if len(sub) == 0:
                    continue
                fig_all.add_trace(go.Scatter3d(
                    x=floor_pts[sub, 0], y=floor_pts[sub, 1], z=floor_pts[sub, 2],
                    mode="markers", name=label,
                    marker=dict(size=3, color=color, opacity=0.85)
                ))
            if len(wall_pts) > 0:
                wv = wall_pts[:min(VIS_MAX, len(wall_pts))]
                fig_all.add_trace(go.Scatter3d(
                    x=wv[:, 0], y=wv[:, 1], z=wv[:, 2],
                    mode="markers", name="벽체",
                    marker=dict(size=2, color="lightgray", opacity=0.3)
                ))
            fig_all.update_layout(
                scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                           aspectmode="data"),
                legend=dict(orientation="h", y=-0.05),
                height=550, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_all, use_container_width=True)

        # ── 데이터 테이블 ──
        st.subheader("📋 데이터 테이블")
        summary_rows = []
        for key, (label, color, mask) in CLASS_META.items():
            if mask.sum() > 0:
                summary_rows.append({
                    "분류": label,
                    "포인트 수": int(mask.sum()),
                    "비율 (%)": f"{mask.mean()*100:.1f}",
                    "평균 편차 (mm)": f"{abs_mm[mask].mean():.2f}",
                    "최대 편차 (mm)": f"{abs_mm[mask].max():.2f}",
                    "표준편차 (mm)": f"{abs_mm[mask].std():.2f}",
                })

        tab_s, tab_d = st.tabs(["분류별 요약", "전체 데이터"])

        with tab_s:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        with tab_d:
            result_df = pd.DataFrame({
                "X (m)": np.round(floor_pts[:, 0], 4),
                "Y (m)": np.round(floor_pts[:, 1], 4),
                "Z (m)": np.round(floor_pts[:, 2], 4),
                "편차_절대 (mm)": np.round(abs_mm, 3),
                "편차_부호 (mm)": np.round(signed_mm, 3),
                "크랙점수_상대(파일내0-1)": np.round(curvs, 4),
                "분류": labels,
            })
            st.dataframe(result_df, use_container_width=True, height=300)

        # ── CSV 저장 ──
        buf = io.StringIO()
        result_df.to_csv(buf, index=False)
        st.download_button(
            "📥 결과 CSV 저장", buf.getvalue(),
            file_name="floor_analysis_result.csv", mime="text/csv",
            use_container_width=True
        )

        # ── 결과 보고서 ──
        st.divider()
        st.subheader("📄 종합 결과 보고서")

        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        x_range = floor_pts[:, 0].max() - floor_pts[:, 0].min()
        y_range = floor_pts[:, 1].max() - floor_pts[:, 1].min()
        area = x_range * y_range

        if bad_pct <= 5:    grade_str, grade_icon = "우수 (Excellent)", "⭐"
        elif bad_pct <= 15: grade_str, grade_icon = "양호 (Good)",      "✅"
        elif bad_pct <= 30: grade_str, grade_icon = "보통 (Fair)",      "⚠️"
        else:               grade_str, grade_icon = "불량 (Poor)",      "🔴"

        # 최대 편차 상위 5개 위치
        worst_idx = np.argsort(abs_mm)[-5:][::-1]
        worst_rows = "\n".join(
            f"  {i+1}. ({floor_pts[j,0]:.3f}m, {floor_pts[j,1]:.3f}m) "
            f"— {signed_mm[j]:+.2f}mm [{labels[j]}]"
            for i, j in enumerate(worst_idx)
        )

        report_text = f"""================================================================
            바닥면 평탄성 분석 결과 보고서
            (Floor Flatness Analysis Report)
================================================================
분석 일시  : {now}
입력 좌표 단위 : {st.session_state.get("xyz_unit_note", "m")}
처리 좌표 단위 : m (모든 기준값은 m 기준)
적용 기준  : 점-기준평면 편차 ±{threshold_mm:.1f}mm (KCS 41 46 01에서 채택한 수치 기준을 적용한 자체 작업 기준이며, KCS 원 규정의 3m 직선자 측정 절차를 재현한 것이 아니고 공식 적합·부적합 판정 근거가 아닙니다)
기준 평면 추정 허용오차 (RANSAC 인라이어) : ±{ransac_inlier_mm:.1f}mm (최소 인라이어 비율 {min_inlier_ratio*100:.0f}%, 반복 {ransac_n_iter}회)
이상치 제거 민감도 (표준편차 배수) : {sor_std_ratio:.1f}
바닥 분리 방식 : {floor_mode} ({'테이블 상판 등 비바닥 수평면 제외에 엄격' if floor_mode == 'AND' else '기본(논문 기준)'})
이상치 제거 : {outlier_removed_ratio*100:.1f}% 제거{'(안전장치로 생략됨)' if outlier_status == 'skipped_too_many' else ''}
다운샘플링 : {(f'원본 {downsampled_from:,}개 → {max_analysis_points:,}개로 무작위 다운샘플링') if downsampled_from is not None else '적용 안 함'}
※ 크랙점수는 해당 스캔 내부의 상대 지표이며, 서로 다른 파일 간 절대 비교에는 주의가 필요합니다.

────────────────────────────────────────────────────────────────
[ 1. 데이터 개요 ]
────────────────────────────────────────────────────────────────
  총 입력 포인트    : {len(xyz):,} 개
  벽체 제거 포인트  : {len(wall_pts):,} 개  ({len(wall_pts)/len(xyz)*100:.1f}%)
  바닥 분석 포인트  : {len(floor_pts):,} 개  ({len(floor_pts)/len(xyz)*100:.1f}%)
  분석 영역         : {x_range:.2f}m × {y_range:.2f}m  (≈ {area:.1f} m²)

────────────────────────────────────────────────────────────────
[ 2. 편차 통계 ]
────────────────────────────────────────────────────────────────
  평균 편차         : {abs_mm.mean():.3f} mm
  중앙값 편차       : {np.median(abs_mm):.3f} mm
  표준편차          : {abs_mm.std():.3f} mm
  최대 돌출         : +{signed_mm.max():.3f} mm
  최대 함몰         : {signed_mm.min():.3f} mm

────────────────────────────────────────────────────────────────
[ 3. 분류별 결과 ]
────────────────────────────────────────────────────────────────
  🟢 정상  (A)  : {good_mask.sum():,} 개  ({good_mask.mean()*100:.1f}%)
  🟠 돌출  (B1) : {protrude_mask.sum():,} 개  ({protrude_mask.mean()*100:.1f}%)
  🔵 함몰  (B2) : {depress_mask.sum():,} 개  ({depress_mask.mean()*100:.1f}%)
  🔴 크랙  (C)  : {crack_mask.sum():,} 개  ({crack_mask.mean()*100:.1f}%)

────────────────────────────────────────────────────────────────
[ 4. 주요 불량 위치 상위 5개 ]
────────────────────────────────────────────────────────────────
{worst_rows}

────────────────────────────────────────────────────────────────
[ 5. 종합 판정 ]
────────────────────────────────────────────────────────────────
  불량 비율         : {bad_pct:.1f}%
  종합 등급         : {grade_icon} {grade_str}

================================================================
"""

        st.code(report_text, language=None)

        c1, c2 = st.columns(2)
        c1.download_button(
            "📄 TXT 저장", report_text,
            file_name="floor_flatness_report.txt", mime="text/plain",
            use_container_width=True
        )
        try:
            pdf_bytes = generate_pdf(report_text, summary_rows, worst_idx,
                                     grade_icon, grade_str, bad_pct,
                                     fig_heatmap=fig, fig_hist=hist_fig, fig_surf=fig_surf)
            c2.download_button(
                "📑 PDF 저장", pdf_bytes,
                file_name="floor_flatness_report.pdf", mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            c2.warning(f"PDF 생성 실패: {e}")
