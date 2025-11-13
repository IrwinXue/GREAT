# -*- coding: utf-8 -*-
"""
Shielding_marked.py — 以 Clip 幾何中心位置排布（中文介面；A=下、C=上；B=右、D=左）

新增功能：
- 在**最下方**加入 UI：Mark_1 / Mark_2，可輸入兩個頻率點。
- 圖上以**虛線**顯示 Mark_1 / Mark_2 所在頻率，並在該頻率處標示
  Original（Before）與 New（After）兩條曲線的 dBm 數值（含交點標記與文字標註）。

既有設定維持：
- L 定義不變（不含 Clip 本體的淨距，外緣-外緣、端點-外緣）。
- 模型公式係數 3.9e-16（相對舊版 3.9e-15 下修 20 dB），確保 2.4–2.5 GHz 約落在 -93 ~ -91 dBm。
- 標示固定：上下錯排位移 1.60 mm、字級 9 pt。
- 頻率 UI 在圖下方；預設 Start=0.500、End=5.000、Step=0.050。
"""
import io
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import streamlit as st
from PIL import Image, ImageOps

# ===================== 基本設定 =====================
st.set_page_config(page_title="SO-DIMM（以中心排布；中文；A=下、C=上、B=右、D=左）", layout="wide")
SO_DIMM_TYPES = {
    "Second floor": {"L_mm": 77.0, "W_mm": 49.0},
    "Dual Single": {"L_mm": 154.0, "W_mm": 39.0},
    "Butterfly": {"L_mm": 86.0, "W_mm": 74.5},
}
CLIP_TYPES = {
    "6053B1063701_SO DIMM": {"L_mm": 6.0, "W_mm": 1.5},
    "6053B2437701_SO DIMM(New)": {"L_mm": 6.0, "W_mm": 1.5},
    "6053B1648901_LPDDR": {"L_mm": 6.5, "W_mm": 1.0},
}

# 模型係數（Original/New 共用）
alpha_model = 1.0
R_m_model   = 3.0
R_load      = 50.0  # ohm

# 影像偵測相關參數
PX_PER_MM       = 10
EDGE_BAND_RATIO = 0.12
MIN_LEN_RATIO   = 0.5
MERGE_GAP_RATIO = 0.15

# 標示固定參數
SDY_MM_FIXED     = 1.60  # 標示上下錯排位移（mm）
FONT_SIZE_FIXED  = 9     # 標示字級（pt）

# ===================== UI：板件與 Clip =====================
c1, c2 = st.columns(2)
with c1:
    so_keys = list(SO_DIMM_TYPES.keys())
    so_default_idx = so_keys.index("Dual Single") if "Dual Single" in so_keys else 0
    so_choice = st.selectbox("SO-DIMM 種類", so_keys, index=so_default_idx)
    so_size = SO_DIMM_TYPES[so_choice]
with c2:
    clip_keys = list(CLIP_TYPES.keys())
    clip_default_idx = clip_keys.index("6053B2437701_SO DIMM(New)") if "6053B2437701_SO DIMM(New)" in clip_keys else 0
    clip_choice = st.selectbox("Clip 種類", clip_keys, index=clip_default_idx)
    clip_size = CLIP_TYPES[clip_choice]

# ===================== 工具函式 =====================
def to_float_list(text: str):
    """逗號/空白分隔 → float list（支援中文逗號）"""
    if not text:
        return []
    parts = [p.strip() for p in text.replace("，", ",").split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            for s in [s for s in p.split() if s]:
                try:
                    out.append(float(s))
                except ValueError:
                    pass
    return out

def fmt_val(x):
    try:
        xv = float(x)
        return f"{int(round(xv))}" if abs(xv - round(xv)) < 1e-6 else f"{xv:.2f}"
    except:
        return str(x)

# 中心(mm) → 有效縫隙序列（水平用板長 L；垂直用板寬 W）
# ✅ L 的定義：回傳 gaps 為「不含 Clip 本體」的淨距（相鄰 Clips 外緣間距、端點到最近 Clip 外緣距離）
def centers_to_gaps(centers_mm, board_len_mm, clip_len_mm):
    if not centers_mm:
        return []
    half = clip_len_mm / 2.0  # 由中心推得外緣
    C = sorted([max(half, min(float(c), max(board_len_mm - half, half))) for c in centers_mm])
    gaps = []
    # 端點 → 第一個 Clip 外緣
    first_left = C[0] - half
    gaps.append(max(0.0, first_left - 0.0))
    # 相鄰 Clips 外緣 → 外緣
    for i in range(1, len(C)):
        prev_right = C[i-1] + half
        this_left  = C[i]   - half
        gaps.append(max(0.0, this_left - prev_right))
    # 最後一個 Clip 外緣 → 端點
    last_right = C[-1] + half
    gaps.append(max(0.0, board_len_mm - last_right))
    return gaps

# 模型：由 gaps 算 Vrms；其中 L = gaps 中每段淨距（不含 Clip 本體），模型採用 sum(L^3)
# 係數採 3.9e-16（相對舊版 3.9e-15 下修 20 dB）
def model_sum_gaps(freq_ghz, gap_list_mm, alpha, R_m):
    f_hz = np.asarray(freq_ghz) * 1e9
    R    = max(R_m, 1e-12)
    arg  = 1 + 0.66 * alpha
    if arg <= 0:
        raise ValueError("ln(1 + 0.66*α) 需 > 0。")
    f_pow   = f_hz ** 1.5
    sum_L3  = np.sum((np.asarray(gap_list_mm, dtype=float) * 1e-3) ** 3) if gap_list_mm else 0.0
    coeff   = 3.9e-16 / (np.log(arg) * R)
    Vrms    = coeff * sum_L3 * f_pow
    return Vrms

def to_dBm_from_Vrms(Vrms, R_load_ohm=50.0):
    Vrms_safe = np.maximum(Vrms, 1e-30)  # 極小值保護
    P_mW      = (Vrms_safe ** 2) / R_load_ohm * 1e3
    return 10 * np.log10(P_mW)

# 影像：上傳 → 等比裁切 → 偵測窄帶
def prep_bg_for_board(img: Image.Image, L_mm: float, W_mm: float, pxmm: int = PX_PER_MM):
    target_px = (max(int(L_mm * pxmm), 32), max(int(W_mm * pxmm), 32))
    resample = getattr(Image, "Resampling", None)
    lanczos  = Image.LANCZOS if resample is None else Image.Resampling.LANCZOS
    img_fit  = ImageOps.fit(img.convert("RGB"), target_px, method=lanczos, centering=(0.5, 0.5))
    return np.asarray(img_fit)

def make_color_mask(arr):
    R = arr[..., 0].astype(np.int16)
    G = arr[..., 1].astype(np.int16)
    B = arr[..., 2].astype(np.int16)
    red_mask   = (R >= 150) & (G <= 90) & (B <= 90)
    white_mask = (R >= 220) & (G >= 220) & (B >= 220)
    return red_mask | white_mask

def find_runs_1d(bool_vec, min_len=3, merge_gap=2):
    runs, in_run, start = [], False, None
    for i, v in enumerate(bool_vec):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            runs.append((start, i-1))
    if in_run:
        runs.append((start, len(bool_vec)-1))
    merged = []
    for s, e in runs:
        if not merged:
            merged.append([s, e]); continue
        ps, pe = merged[-1]
        if s - pe <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged if (e - s + 1) >= min_len]
    return merged

def detect_edge_runs(mask, edge_band_ratio=EDGE_BAND_RATIO):
    H, W    = mask.shape
    band_h  = max(int(H * edge_band_ratio), 4)
    band_w  = max(int(W * edge_band_ratio), 4)
    top_x   = mask[:band_h, :].any(axis=0)
    bottom_x= mask[-band_h:, :].any(axis=0)
    left_y  = mask[:, :band_w ].any(axis=1)
    right_y = mask[:, -band_w:].any(axis=1)
    return top_x, bottom_x, left_y, right_y

# run 中心(px) → 中心(mm)（限制到 [half, board_len-half]）
def runs_to_centers_mm(runs, pxmm, clip_L_mm, board_len_mm):
    centers, half = [], clip_L_mm / 2.0
    for s, e in runs:
        center_px = (s + e) / 2.0
        center_mm = center_px / pxmm
        center_mm = max(half, min(center_mm, max(board_len_mm - half, half)))
        centers.append(center_mm)
    centers.sort()
    return centers

# ===================== 畫板（以中心為主） =====================
def draw_board_with_centers(
    ax, title, board_L_mm, board_W_mm,
    centers_dict,
    clip_L_mm, clip_W_mm,
    face_color="#3f6db3", clip_color="#f5a623",
    bg_arr=None, bg_alpha=1.0,
    board_alpha=0.92, show_board_face=True,
    # 覆寫中心（Original 用）
    A_centers_override=None,  # 下邊：X 中心（原點左下）
    C_centers_override=None,  # 上邊：X 中心（原點左上）
    B_centers_override=None,  # 右邊：Y 中心（原點右上）
    D_centers_override=None   # 左邊：Y 中心（原點左上）
):
    ax.clear()
    ax.set_aspect('equal', adjustable='box')
    margin = max(clip_W_mm * 0.6, 3.0)
    ax.set_xlim(-clip_W_mm/2 - margin, board_L_mm + clip_W_mm/2 + margin)
    ax.set_ylim(-clip_W_mm/2 - margin, board_W_mm + clip_W_mm/2 + margin)
    ax.axis('off')

    if bg_arr is not None:
        # 重要：使用 origin='upper'，避免上/下顛倒
        ax.imshow(bg_arr, extent=(0, board_L_mm, 0, board_W_mm), origin='upper', alpha=bg_alpha)

    if show_board_face:
        ax.add_patch(plt.Rectangle((0, 0), board_L_mm, board_W_mm,
                                   fc=face_color, ec=None, lw=0, alpha=board_alpha))
    ax.text(board_L_mm/2, board_W_mm/2, title, color="white", fontsize=13,
            ha="center", va="center", weight="bold")

    # 文字與外框參數（固定值）
    rad      = max(0.4, 0.15 * min(clip_L_mm, clip_W_mm))
    lab_off  = max(1.0, 0.03 * board_W_mm)      # 文字與板外緣距離
    sdy_mm   = SDY_MM_FIXED                      # 上下錯排位移（固定）
    font_sz  = FONT_SIZE_FIXED
    half     = clip_L_mm / 2.0

    def label_text(val):
        return f"{fmt_val(val)}"

    # ===== A（下邊；X 中心：左下原點） =====
    A_centers = A_centers_override if A_centers_override is not None else centers_dict.get("A", [])
    for x_c in sorted(A_centers):
        x_left = max(0.0, min(x_c - half, board_L_mm - clip_L_mm))
        ax.add_patch(FancyBboxPatch((x_left, -clip_W_mm/2), clip_L_mm, clip_W_mm,
                                    boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
    for i, x_c in enumerate(sorted(A_centers), start=1):
        y_text = -lab_off + (sdy_mm if (i % 2 == 1) else -sdy_mm)  # 奇數向上、偶數向下
        ax.text(x_c, y_text, label_text(x_c), ha="center", va="top", fontsize=font_sz, color="black")

    # ===== C（上邊；X 中心：左上原點） =====
    C_centers = C_centers_override if C_centers_override is not None else centers_dict.get("C", [])
    for x_c in sorted(C_centers):
        x_left = max(0.0, min(x_c - half, board_L_mm - clip_L_mm))
        ax.add_patch(FancyBboxPatch((x_left, board_W_mm - clip_W_mm/2), clip_L_mm, clip_W_mm,
                                    boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
    for i, x_c in enumerate(sorted(C_centers), start=1):
        y_text = board_W_mm + lab_off + (sdy_mm if (i % 2 == 0) else -sdy_mm)  # 偶數向上、奇數向下
        ax.text(x_c, y_text, label_text(x_c), ha="center", va="bottom", fontsize=font_sz, color="black")

    # ===== B（右邊；Y 中心：右上原點） =====
    B_centers = B_centers_override if B_centers_override is not None else centers_dict.get("B", [])
    for y_c in sorted(B_centers):
        y_top    = max(0.0, min(y_c - half, board_W_mm - clip_L_mm))
        y_bottom = board_W_mm - (y_top + clip_L_mm)
        y_bottom = max(0.0, min(y_bottom, board_W_mm - clip_L_mm))
        ax.add_patch(FancyBboxPatch((board_L_mm - clip_W_mm/2, y_bottom), clip_W_mm, clip_L_mm,
                                    boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
    for i, y_c in enumerate(sorted(B_centers), start=1):
        y_text = board_W_mm - y_c + (sdy_mm if (i % 2 == 1) else -sdy_mm)  # 奇數向上、偶數向下
        ax.text(board_L_mm + lab_off, y_text, label_text(y_c), ha="left", va="center", fontsize=font_sz, color="black")

    # ===== D（左邊；Y 中心：左上原點） =====
    D_centers = D_centers_override if D_centers_override is not None else centers_dict.get("D", [])
    for y_c in sorted(D_centers):
        y_top    = max(0.0, min(y_c - half, board_W_mm - clip_L_mm))
        y_bottom = board_W_mm - (y_top + clip_L_mm)
        y_bottom = max(0.0, min(y_bottom, board_W_mm - clip_L_mm))
        ax.add_patch(FancyBboxPatch((-clip_W_mm/2, y_bottom), clip_W_mm, clip_L_mm,
                                    boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
    for i, y_c in enumerate(sorted(D_centers), start=1):
        y_text = board_W_mm - y_c + (sdy_mm if (i % 2 == 0) else -sdy_mm)  # 偶數向上、奇數向下
        ax.text(-lab_off, y_text, label_text(y_c), ha="right", va="center", fontsize=font_sz, color="black")

# ===================== Original Design =====================
st.markdown("### Original Design ")
uploaded = st.file_uploader("上傳 JPG/PNG", type=["jpg", "jpeg", "png"], key="bg_uploader")

# 初始化中心列表（保留預設；a/b/c/d 可顯示與輸入）
for k, v in [("A_list", "10 25"), ("B_list", "10 25"), ("C_list", "10 25"), ("D_list", "10 25"),
             ("a_list", "10 25"), ("b_list", "10 25"), ("c_list", "10 25"), ("d_list", "10 25")]:
    if k not in st.session_state:
        st.session_state[k] = v

bg_before = None
A_centers_override = C_centers_override = None
B_centers_override = D_centers_override = None

if uploaded is not None:
    try:
        base      = Image.open(uploaded).convert("RGB")
        bg_before = prep_bg_for_board(base, so_size["L_mm"], so_size["W_mm"], PX_PER_MM)
        mask      = make_color_mask(bg_before)
        top_x, bottom_x, left_y, right_y = detect_edge_runs(mask, edge_band_ratio=EDGE_BAND_RATIO)
        min_len_px   = max(int(clip_size["L_mm"] * PX_PER_MM * MIN_LEN_RATIO), 3)
        merge_gap_px = max(int(clip_size["L_mm"] * PX_PER_MM * MERGE_GAP_RATIO), 2)
        top_runs    = find_runs_1d(top_x,    min_len=min_len_px, merge_gap=merge_gap_px)
        bottom_runs = find_runs_1d(bottom_x, min_len=min_len_px, merge_gap=merge_gap_px)
        left_runs   = find_runs_1d(left_y,   min_len=min_len_px, merge_gap=merge_gap_px)
        right_runs  = find_runs_1d(right_y,  min_len=min_len_px, merge_gap=merge_gap_px)
        # 映射：A ← Bottom、C ← Top、B ← Right、D ← Left
        A_centers_override = runs_to_centers_mm(bottom_runs, PX_PER_MM, clip_size["L_mm"], so_size["L_mm"])  # X
        C_centers_override = runs_to_centers_mm(top_runs,    PX_PER_MM, clip_size["L_mm"], so_size["L_mm"])  # X
        B_centers_override = runs_to_centers_mm(right_runs,  PX_PER_MM, clip_size["L_mm"], so_size["W_mm"])  # Y
        D_centers_override = runs_to_centers_mm(left_runs,   PX_PER_MM, clip_size["L_mm"], so_size["W_mm"])  # Y
        # 更新 session_state：Original（A/B/C/D）
        st.session_state["A_list"] = " ".join(fmt_val(x) for x in A_centers_override) if A_centers_override else st.session_state["A_list"]
        st.session_state["B_list"] = " ".join(fmt_val(x) for x in B_centers_override) if B_centers_override else st.session_state["B_list"]
        st.session_state["C_list"] = " ".join(fmt_val(x) for x in C_centers_override) if C_centers_override else st.session_state["C_list"]
        st.session_state["D_list"] = " ".join(fmt_val(x) for x in D_centers_override) if D_centers_override else st.session_state["D_list"]
        # 同步 New 預設：a/b/c/d ← 偵測到的 A/B/C/D
        st.session_state["a_list"] = st.session_state["A_list"]
        st.session_state["b_list"] = st.session_state["B_list"]
        st.session_state["c_list"] = st.session_state["C_list"]
        st.session_state["d_list"] = st.session_state["D_list"]
    except Exception as e:
        st.error(f"圖片處理/偵測失敗：{e}")

# ===================== Clip 中心位置輸入（保留 a/b/c/d） =====================
st.markdown("### New Design — 請手動修改")
with st.container():
    colA, colB = st.columns(2)
    with colA:
        d_list = st.text_input("左邊", value=st.session_state["d_list"])
        b_list = st.text_input("右邊", value=st.session_state["b_list"])
    with colB:
        c_list = st.text_input("上邊", value=st.session_state["c_list"])
        a_list = st.text_input("下邊", value=st.session_state["a_list"])

# 解析 New 的文字輸入
a_vals = to_float_list(a_list)
b_vals = to_float_list(b_list)
c_vals = to_float_list(c_list)
d_vals = to_float_list(d_list)

# Original 值（A/B/C/D）：優先使用偵測 override；若無則用 session_state
def session_or_default_list(key: str):
    return to_float_list(st.session_state.get(key, "10 25"))

A_vals = A_centers_override if A_centers_override is not None else session_or_default_list("A_list")
B_vals = B_centers_override if B_centers_override is not None else session_or_default_list("B_list")
C_vals = C_centers_override if C_centers_override is not None else session_or_default_list("C_list")
D_vals = D_centers_override if D_centers_override is not None else session_or_default_list("D_list")

# gaps（供模型） —— L 為「不含 Clip 本體」的淨距
# Original
A_gaps = centers_to_gaps(A_vals, so_size["L_mm"], clip_size["L_mm"])  # 下邊 → 水平
C_gaps = centers_to_gaps(C_vals, so_size["L_mm"], clip_size["L_mm"])  # 上邊 → 水平
B_gaps = centers_to_gaps(B_vals, so_size["W_mm"], clip_size["L_mm"])  # 右邊 → 垂直
D_gaps = centers_to_gaps(D_vals, so_size["W_mm"], clip_size["L_mm"])  # 左邊 → 垂直
gaps_original_mm = A_gaps + B_gaps + C_gaps + D_gaps

# New
a_gaps = centers_to_gaps(a_vals, so_size["L_mm"], clip_size["L_mm"])  # 下邊（水平）
c_gaps = centers_to_gaps(c_vals, so_size["L_mm"], clip_size["L_mm"])  # 上邊（水平）
b_gaps = centers_to_gaps(b_vals, so_size["W_mm"], clip_size["L_mm"])  # 右邊（垂直）
d_gaps = centers_to_gaps(d_vals, so_size["W_mm"], clip_size["L_mm"])  # 左邊（垂直）
gaps_new_mm = a_gaps + b_gaps + c_gaps + d_gaps

# ===================== 畫面：板視覺 =====================
dc1, dc2 = st.columns(2)
with dc1:
    st.subheader("Original")
    fig_b, ax_b = plt.subplots(figsize=(6.2, 6.2 * so_size["W_mm"] / max(so_size["L_mm"], 1.0)))
    draw_board_with_centers(
        ax_b, "Original",
        board_L_mm=so_size["L_mm"], board_W_mm=so_size["W_mm"],
        centers_dict={"A": A_vals, "B": B_vals, "C": C_vals, "D": D_vals},
        clip_L_mm=clip_size["L_mm"], clip_W_mm=clip_size["W_mm"],
        face_color="#3f6db3", clip_color="#f5a623",
        bg_arr=bg_before, bg_alpha=1.0,
        board_alpha=0.0 if bg_before is not None else 0.92,
        show_board_face=(bg_before is None),
        A_centers_override=A_centers_override,
        C_centers_override=C_centers_override,
        B_centers_override=B_centers_override,
        D_centers_override=D_centers_override
    )
    st.pyplot(fig_b, use_container_width=True)

with dc2:
    st.subheader("New")
    fig_a, ax_a = plt.subplots(figsize=(6.2, 6.2 * so_size["W_mm"] / max(so_size["L_mm"], 1.0)))
    draw_board_with_centers(
        ax_a, "New",
        board_L_mm=so_size["L_mm"], board_W_mm=so_size["W_mm"],
        centers_dict={"A": a_vals, "B": b_vals, "C": c_vals, "D": d_vals},
        clip_L_mm=clip_size["L_mm"], clip_W_mm=clip_size["W_mm"],
        face_color="#3f6db3", clip_color="#f5a623",
        bg_arr=None, bg_alpha=1.0,
        board_alpha=0.92, show_board_face=True
    )
    st.pyplot(fig_a, use_container_width=True)

st.markdown("---")

# ===================== 模型曲線（先畫圖，UI 放在圖的下方） =====================
# 預設值如圖：Start=0.500, End=5.000, Step=0.050
f_start_default = 0.500
f_end_default   = 5.000
f_step_default  = 0.050

# 讀取或設定預設頻率（從 session_state 讀取）
f_start = st.session_state.get("f_start", f_start_default)
f_end   = st.session_state.get("f_end",   f_end_default)
f_step  = st.session_state.get("f_step",  f_step_default)

# 頻率區間驗證
if f_end <= f_start:
    f_end = f_start + max(f_step, 1e-3)
if f_step <= 0:
    f_step = 0.01

n_steps = max(int(math.floor((f_end - f_start) / f_step)) + 1, 2)
freq    = np.linspace(f_start, f_end, n_steps)

# 計算曲線（依公式如實呈現）
V_original = model_sum_gaps(freq, gaps_original_mm, alpha_model, R_m_model)
V_new      = model_sum_gaps(freq, gaps_new_mm,      alpha_model, R_m_model)
dB_original = to_dBm_from_Vrms(V_original, R_load)
dB_new      = to_dBm_from_Vrms(V_new,      R_load)

# ===== 讀取 Mark_1 / Mark_2 目前值（預設在 2.400 / 2.500 GHz）=====
mark1_default = 2.400
mark2_default = 2.500
mark1 = st.session_state.get("mark1", mark1_default)
mark2 = st.session_state.get("mark2", mark2_default)

# 工具：在給定頻率上取曲線值（線性插值）
def eval_at_mark(mark_ghz, freq, curve_dB):
    if len(freq) < 2:
        return float(mark_ghz), float(curve_dB[0]) if len(curve_dB) else float("nan")
    mark = float(np.clip(mark_ghz, freq[0], freq[-1]))
    val  = float(np.interp(mark, freq, curve_dB))
    return mark, val

m1, orig_m1 = eval_at_mark(mark1, freq, dB_original)
_,  new_m1  = eval_at_mark(mark1, freq, dB_new)
m2, orig_m2 = eval_at_mark(mark2, freq, dB_original)
_,  new_m2  = eval_at_mark(mark2, freq, dB_new)

# 畫圖 + 虛線 + 交點 + 數值標註
fig, ax = plt.subplots(figsize=(9, 3.8))
color_original = "#1f77b4"
color_new      = "#ff7f0e"
color_m1_line  = "#2ca02c"  # 綠
color_m2_line  = "#d62728"  # 紅

ax.plot(freq, dB_original, label="Original", color=color_original, lw=2)
ax.plot(freq, dB_new,      label="New",      color=color_new,      lw=2)

# Mark_1 虛線與標註
ax.axvline(m1, color=color_m1_line, ls="--", lw=1.5, label=f"Mark_1: {m1:.3f} GHz")
ax.scatter([m1], [orig_m1], color=color_original, s=30, zorder=5)
ax.scatter([m1], [new_m1],  color=color_new,      s=30, zorder=5)
ax.annotate(f"Orig {orig_m1:.2f} dBm", xy=(m1, orig_m1), xytext=(5, 10),
            textcoords="offset points", color=color_original, fontsize=9, bbox=dict(fc="white", alpha=0.7, ec=color_original))
ax.annotate(f"New  {new_m1:.2f} dBm", xy=(m1, new_m1), xytext=(5, -15),
            textcoords="offset points", color=color_new, fontsize=9, bbox=dict(fc="white", alpha=0.7, ec=color_new))

# Mark_2 虛線與標註
ax.axvline(m2, color=color_m2_line, ls="--", lw=1.5, label=f"Mark_2: {m2:.3f} GHz")
ax.scatter([m2], [orig_m2], color=color_original, s=30, zorder=5)
ax.scatter([m2], [new_m2],  color=color_new,      s=30, zorder=5)
ax.annotate(f"Orig {orig_m2:.2f} dBm", xy=(m2, orig_m2), xytext=(5, 10),
            textcoords="offset points", color=color_original, fontsize=9, bbox=dict(fc="white", alpha=0.7, ec=color_original))
ax.annotate(f"New  {new_m2:.2f} dBm", xy=(m2, new_m2), xytext=(5, -15),
            textcoords="offset points", color=color_new, fontsize=9, bbox=dict(fc="white", alpha=0.7, ec=color_new))

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Level (dBm)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
st.pyplot(fig, use_container_width=True)

# ======= 頻率設定 UI（在圖的下方，帶預設值）=======
f1, f2, f3 = st.columns([1, 1, 1])
with f1:
    st.session_state["f_start"] = st.number_input("Start (GHz)", value=float(f_start), step=0.1, format="%.3f", key="f_start_input")
with f2:
    st.session_state["f_end"]   = st.number_input("End (GHz)",   value=float(f_end),   step=0.1, format="%.3f", key="f_end_input")
with f3:
    st.session_state["f_step"]  = st.number_input("Step (GHz)",  value=float(f_step),  step=0.01, format="%.3f", key="f_step_input")

# 同步回主要鍵值（供下次重繪使用）
st.session_state["f_start"] = st.session_state["f_start_input"]
st.session_state["f_end"]   = st.session_state["f_end_input"]
st.session_state["f_step"]  = st.session_state["f_step_input"]

# ======= 最下方：Mark_1 / Mark_2 輸入（頻率），即時重繪會用到 =======
st.markdown("---")
mcol1, mcol2 = st.columns([1, 1])
with mcol1:
    st.session_state["mark1"] = st.number_input("Mark_1 (GHz)", value=float(mark1), step=0.01, format="%.3f", key="mark1_input")
with mcol2:
    st.session_state["mark2"] = st.number_input("Mark_2 (GHz)", value=float(mark2), step=0.01, format="%.3f", key="mark2_input")

# 同步回主要鍵值
st.session_state["mark1"] = st.session_state["mark1_input"]
st.session_state["mark2"] = st.session_state["mark2_input"]