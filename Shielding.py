
# -*- coding: utf-8 -*-
"""
Shielding.py — Clip位置優化

- 四周邊帶模式：Top/Bottom/Left/Right 各自做自適應門檻
- 二值化採「顏色亮度 (HSV) 或 邊緣能量」任一成立
- 連通元件 + 狹長矩形形狀過濾 + 近距合併 + 不相互碰觸
- 可選自動調參（在目前值附近微掃描），不硬編目標數
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import streamlit as st
from PIL import Image
from collections import deque

# ===================== 基本設定 =====================
st.set_page_config(page_title="SO-DIMM（HSV+Edge；中文；A=下、C=上、B=右、D=左）", layout="wide")

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

alpha_model = 1.0
R_m_model = 3.0
R_load = 50.0  # ohm

# ===================== 參數（側欄；依你截圖設為預設） =====================
PX_PER_MM = 10
EDGE_BAND_RATIO = st.sidebar.slider("邊帶厚度比例", 0.05, 0.30, 0.16, 0.01)

st.sidebar.markdown("### 顏色/亮度門檻（HSV百分位）")
PERC_V = st.sidebar.slider("亮度V百分位", 60, 99, 90, 1)
PERC_S = st.sidebar.slider("飽和S百分位", 40, 95, 70, 1)

st.sidebar.markdown("### 邊緣能量門檻")
RECT_STD_K = st.sidebar.slider("STD_K（mean + k*std）", 0.5, 3.0, 1.10, 0.05)

st.sidebar.markdown("### 形狀與面積過濾")
ASPECT_MIN = st.sidebar.slider("最小長寬比（狹長）", 2.0, 10.0, 4.50, 0.1)
MIN_AREA_PX = st.sidebar.number_input("最小面積（px）", value=80, min_value=10, max_value=5000, step=10)
MAX_THICK_FACTOR = st.sidebar.slider("最大厚度係數（相對Clip寬 px 的倍數）", 1.5, 5.0, 2.50, 0.1)
FILL_MIN = st.sidebar.slider("最小填充率（area/(w*h)）", 0.10, 1.00, 0.45, 0.05)  # 容忍只有邊線

st.sidebar.markdown("### 合併與安全距")
MERGE_GAP_RATIO = st.sidebar.slider("中心合併容差（相對Clip長度）", 0.05, 0.40, 0.12, 0.01)
NON_TOUCH_MARGIN = st.sidebar.slider("不相互碰觸（加碼安全距，mm）", 0.0, 1.0, 0.0, 0.05)

AUTO_TUNE = st.sidebar.checkbox("自動調參（小範圍微掃描）", value=True)
SHOW_DEBUG_BOXES = st.sidebar.checkbox("Original 疊圖顯示偵測框", value=False)

# 標示固定參數
SDY_MM_FIXED = 1.60
FONT_SIZE_FIXED = 9

# ===================== 工具函式 =====================
def prep_bg_for_board(img: Image.Image, L_mm: float, W_mm: float, pxmm: int = PX_PER_MM):
    target_px = (max(int(L_mm * pxmm), 32), max(int(W_mm * pxmm), 32))
    resample = getattr(Image, "Resampling", None)
    lanczos = Image.LANCZOS if resample is None else Image.Resampling.LANCZOS
    img_stretched = img.convert("RGB").resize(target_px, lanczos)
    return np.asarray(img_stretched)

def rgb_to_hsv_np(arr_rgb):
    # arr: HxWx3 in [0..255]
    arr = arr_rgb.astype(np.float32) / 255.0
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    diff = mx - mn + 1e-6
    s = diff / (mx + 1e-6)
    v = mx
    return s, v  # 只取 S, V

def edge_energy(Y):
    dx = np.abs(np.diff(Y, axis=1, prepend=Y[:, :1]))
    dy = np.abs(np.diff(Y, axis=0, prepend=Y[:1, :]))
    return dx + dy

def label_components(bin_img):
    H, W = bin_img.shape
    labels = -np.ones((H, W), dtype=np.int32)
    comps = []
    lab_id = 0
    q = deque()
    for y in range(H):
        for x in range(W):
            if not bin_img[y, x] or labels[y, x] >= 0:
                continue
            labels[y, x] = lab_id
            q.append((y, x))
            minx = maxx = x
            miny = maxy = y
            area = 0
            while q:
                cy, cx = q.pop()
                area += 1
                if cx < minx: minx = cx
                if cx > maxx: maxx = cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                for ny in (cy-1, cy, cy+1):
                    for nx in (cx-1, cx, cx+1):
                        if ny < 0 or ny >= H or nx < 0 or nx >= W: continue
                        if labels[ny, nx] >= 0: continue
                        if bin_img[ny, nx]:
                            labels[ny, nx] = lab_id
                            q.append((ny, nx))
            comps.append((minx, maxx, miny, maxy, area))
            lab_id += 1
    return comps

def merge_close_centers_mm(centers, clip_L_mm, merge_gap_ratio):
    if not centers: return []
    thr_mm = clip_L_mm * merge_gap_ratio
    out = []
    for c in sorted(centers):
        if not out or abs(c - out[-1]) > thr_mm:
            out.append(c)
        else:
            out[-1] = 0.5 * (out[-1] + c)
    return out

def enforce_non_touching(centers, clip_L_mm, board_len_mm, margin_mm=0.0):
    if not centers: return []
    half = clip_L_mm / 2.0
    c_sorted = sorted(max(half, min(float(c), board_len_mm - half)) for c in centers)
    out = []
    last = None
    for c in c_sorted:
        if last is None or (c - last) >= (clip_L_mm + margin_mm):
            out.append(c); last = c
        else:
            # 忽略後到者
            continue
    return out

def to_float_list(text: str):
    if not text: return []
    parts = [p.strip() for p in text.replace("，", ",").split(",")]
    out = []
    for p in parts:
        if not p: continue
        try: out.append(float(p))
        except ValueError:
            for s in [s for s in p.split() if s]:
                try: out.append(float(s))
                except ValueError: pass
    return out

def fmt_val(x):
    try:
        xv = float(x)
        return f"{int(round(xv))}" if abs(xv - round(xv)) < 1e-6 else f"{xv:.2f}"
    except: return str(x)

def centers_to_gaps(centers_mm, board_len_mm, clip_len_mm):
    if not centers_mm: return []
    half = clip_len_mm / 2.0
    C = sorted([max(half, min(float(c), board_len_mm - half)) for c in centers_mm])
    gaps = []
    first_left = C[0] - half
    gaps.append(max(0.0, first_left))
    for i in range(1, len(C)):
        prev_right = C[i-1] + half
        this_left = C[i] - half
        gaps.append(max(0.0, this_left - prev_right))
    last_right = C[-1] + half
    gaps.append(max(0.0, board_len_mm - last_right))
    return gaps

def model_sum_gaps(freq_ghz, gap_list_mm, alpha, R_m):
    f_hz = np.asarray(freq_ghz) * 1e9
    R = max(R_m, 1e-12)
    arg = 1 + 0.66 * alpha
    if arg <= 0: raise ValueError("ln(1 + 0.66*α) 需 > 0。")
    f_pow = f_hz ** 1.5
    sum_L3 = np.sum((np.asarray(gap_list_mm, dtype=float) * 1e-3) ** 3) if gap_list_mm else 0.0
    coeff = 3.9e-16 / (np.log(arg) * R)
    Vrms = coeff * sum_L3 * f_pow
    return Vrms

def to_dBm_from_Vrms(Vrms, R_load_ohm=50.0):
    Vrms_safe = np.maximum(Vrms, 1e-30)
    P_mW = (Vrms_safe ** 2) / R_load_ohm * 1e3
    return 10 * np.log10(P_mW)

# ===================== 偵測：HSV + Edge（四邊） =====================
def detect_rect_centers_mm(arr_rgb, edge_band_ratio, pxmm, clip_L_mm, clip_W_mm, board_L_mm, board_W_mm,
                           perc_V=90, perc_S=70, std_k=1.10, aspect_min=4.50, min_area_px=80,
                           max_thick_factor=2.50, fill_min=0.45, merge_gap_ratio=0.12,
                           non_touch_margin_mm=0.0, debug_collect=False):
    """
    回傳 A/B/C/D 四邊中心（mm）與（可選）偵測框（全圖座標）。
    """
    H, W, _ = arr_rgb.shape
    band_h = max(int(H * edge_band_ratio), 4)
    band_w = max(int(W * edge_band_ratio), 4)
    clip_L_px = max(int(clip_L_mm * pxmm), 1)
    clip_W_px = max(int(clip_W_mm * pxmm), 1)
    max_thick_px = int(max_thick_factor * clip_W_px)

    # HSV
    arr = arr_rgb.astype(np.float32)
    R = arr[...,0]; G = arr[...,1]; B = arr[...,2]
    mx = np.maximum(np.maximum(R, G), B); mn = np.minimum(np.minimum(R, G), B)
    diff = mx - mn + 1e-6
    S = (diff / (mx + 1e-6)) / 255.0
    V = (mx / 255.0)

    # Gray + Edge
    Y = 0.299*R + 0.587*G + 0.114*B
    dx = np.abs(np.diff(Y, axis=1, prepend=Y[:, :1]))
    dy = np.abs(np.diff(Y, axis=0, prepend=Y[:1, :]))
    E = dx + dy

    def band_thr_color(v_band, s_band):
        v_thr = float(np.percentile(v_band.reshape(-1), perc_V))
        s_thr = float(np.percentile(s_band.reshape(-1), perc_S))
        return v_thr, s_thr

    def band_thr_edge(e_band):
        m, s = float(np.mean(e_band)), float(np.std(e_band))
        return m + std_k * s

    def filter_and_collect(bin_band, offset_xy, horizontal=True):
        comps = label_components(bin_band)
        centers_mm = []
        boxes = []
        for minx, maxx, miny, maxy, area in comps:
            w = maxx - minx + 1
            h = maxy - miny + 1
            if area < min_area_px: 
                continue
            if (horizontal and h > max_thick_px) or ((not horizontal) and w > max_thick_px):
                continue
            aspect = (w / max(h,1)) if horizontal else (h / max(w,1))
            if aspect < aspect_min:
                continue
            fill_ratio = area / float(w*h)
            if fill_ratio < fill_min:
                continue
            # 中心（px→mm）
            if horizontal:
                center_px = (minx + maxx) / 2.0
                center_mm = center_px / pxmm
            else:
                center_px = (miny + maxy) / 2.0
                center_mm = center_px / pxmm
            centers_mm.append(center_mm)
            if debug_collect:
                ox, oy = offset_xy
                boxes.append((minx + ox, miny + oy, w, h))
        centers_mm = merge_close_centers_mm(centers_mm, clip_L_mm, merge_gap_ratio)
        if horizontal:
            centers_mm = enforce_non_touching(centers_mm, clip_L_mm, board_L_mm, margin_mm=non_touch_margin_mm)
        else:
            centers_mm = enforce_non_touching(centers_mm, clip_L_mm, board_W_mm, margin_mm=non_touch_margin_mm)
        return centers_mm, boxes

    # ===== TOP =====
    top_S, top_V, top_E = S[:band_h,:], V[:band_h,:], E[:band_h,:]
    v_thr_top, s_thr_top = band_thr_color(top_V, top_S)
    thr_edge_top = band_thr_edge(top_E)
    top_bin = ((top_V >= v_thr_top) & (top_S >= s_thr_top)) | (top_E >= thr_edge_top)
    C_list, top_boxes = filter_and_collect(top_bin, (0, 0), horizontal=True)

    # ===== BOTTOM =====
    bottom_S, bottom_V, bottom_E = S[-band_h:,:], V[-band_h:,:], E[-band_h:,:]
    v_thr_bottom, s_thr_bottom = band_thr_color(bottom_V, bottom_S)
    thr_edge_bottom = band_thr_edge(bottom_E)
    bottom_bin = ((bottom_V >= v_thr_bottom) & (bottom_S >= s_thr_bottom)) | (bottom_E >= thr_edge_bottom)
    A_list, bottom_boxes = filter_and_collect(bottom_bin, (0, H - band_h), horizontal=True)

    # ===== LEFT =====
    left_S, left_V, left_E = S[:, :band_w], V[:, :band_w], E[:, :band_w]
    v_thr_left, s_thr_left = band_thr_color(left_V, left_S)
    thr_edge_left = band_thr_edge(left_E)
    left_bin = ((left_V >= v_thr_left) & (left_S >= s_thr_left)) | (left_E >= thr_edge_left)
    D_list, left_boxes = filter_and_collect(left_bin, (0, 0), horizontal=False)

    # ===== RIGHT =====
    right_S, right_V, right_E = S[:, -band_w:], V[:, -band_w:], E[:, -band_w:]
    v_thr_right, s_thr_right = band_thr_color(right_V, right_S)
    thr_edge_right = band_thr_edge(right_E)
    right_bin = ((right_V >= v_thr_right) & (right_S >= s_thr_right)) | (right_E >= thr_edge_right)
    B_list, right_boxes = filter_and_collect(right_bin, (W - band_w, 0), horizontal=False)

    debug_boxes = {
        "top": top_boxes,
        "bottom": bottom_boxes,
        "left": left_boxes,
        "right": right_boxes,
        "band_h": band_h, "band_w": band_w
    } if debug_collect else None

    return A_list, B_list, C_list, D_list, debug_boxes

# ===================== UI：板件與 Clip =====================
c1, c2 = st.columns(2)
with c1:
    so_keys = list(SO_DIMM_TYPES.keys())
    so_choice = st.selectbox("SO-DIMM 種類", so_keys, index=so_keys.index("Dual Single") if "Dual Single" in so_keys else 0)
    so_size = SO_DIMM_TYPES[so_choice]
with c2:
    clip_keys = list(CLIP_TYPES.keys())
    clip_choice = st.selectbox("Clip 種類", clip_keys, index=clip_keys.index("6053B2437701_SO DIMM(New)") if "6053B2437701_SO DIMM(New)" in clip_keys else 0)
    clip_size = CLIP_TYPES[clip_choice]

# 初始化中心列表
for k, v in [("A_list", "10 25"), ("B_list", "10 25"), ("C_list", "10 25"), ("D_list", "10 25"),
             ("a_list", "10 25"), ("b_list", "10 25"), ("c_list", "10 25"), ("d_list", "10 25")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ===================== Original Design（偵測） =====================
st.markdown("### Original Design")
uploaded = st.file_uploader("上傳 JPG/PNG", type=["jpg", "jpeg", "png"], key="bg_uploader")

bg_before = None
A_centers_override = B_centers_override = C_centers_override = D_centers_override = None
debug_boxes = None

if uploaded is not None:
    try:
        base = Image.open(uploaded).convert("RGB")
        bg_before = prep_bg_for_board(base, so_size["L_mm"], so_size["W_mm"], PX_PER_MM)

        # —— 主偵測（HSV + Edge；無特例）
        A_rect, B_rect, C_rect, D_rect, debug_boxes = detect_rect_centers_mm(
            bg_before, EDGE_BAND_RATIO, PX_PER_MM,
            clip_size["L_mm"], clip_size["W_mm"],
            so_size["L_mm"], so_size["W_mm"],
            perc_V=PERC_V, perc_S=PERC_S, std_k=RECT_STD_K,
            aspect_min=ASPECT_MIN, min_area_px=MIN_AREA_PX,
            max_thick_factor=MAX_THICK_FACTOR, fill_min=FILL_MIN,
            merge_gap_ratio=MERGE_GAP_RATIO, non_touch_margin_mm=NON_TOUCH_MARGIN,
            debug_collect=SHOW_DEBUG_BOXES
        )

        A_centers_override, B_centers_override, C_centers_override, D_centers_override = A_rect, B_rect, C_rect, D_rect

        # —— 自動調參（可選；預設啟用）
        if AUTO_TUNE:
            best_score = -1
            best_sets = (A_rect, B_rect, C_rect, D_rect)
            rect_k_list = [RECT_STD_K * s for s in [0.9, 1.0, 1.1]]
            aspect_list = [max(2.0, ASPECT_MIN - 0.5), ASPECT_MIN, ASPECT_MIN + 0.5]
            area_list = [max(20, MIN_AREA_PX - 40), MIN_AREA_PX, MIN_AREA_PX + 40]
            percV_list = [max(60, PERC_V - 5), PERC_V, min(99, PERC_V + 5)]
            percS_list = [max(40, PERC_S - 5), PERC_S, min(95, PERC_S + 5)]
            for rk in rect_k_list:
                for asp in aspect_list:
                    for ar in area_list:
                        for pv in percV_list:
                            for ps in percS_list:
                                Ar, Br, Cr, Dr, _ = detect_rect_centers_mm(
                                    bg_before, EDGE_BAND_RATIO, PX_PER_MM,
                                    clip_size["L_mm"], clip_size["W_mm"],
                                    so_size["L_mm"], so_size["W_mm"],
                                    perc_V=pv, perc_S=ps, std_k=rk,
                                    aspect_min=asp, min_area_px=ar,
                                    max_thick_factor=MAX_THICK_FACTOR, fill_min=FILL_MIN,
                                    merge_gap_ratio=MERGE_GAP_RATIO, non_touch_margin_mm=NON_TOUCH_MARGIN,
                                    debug_collect=False
                                )
                                # 評分
                                def score_one(lst, clip_L, board_len):
                                    if not lst: return 0.0
                                    lst = sorted(lst)
                                    s = len(lst)
                                    s -= 0.2 * sum(1 for c in lst if (c < clip_L/2 + 0.5) or (c > board_len - clip_L/2 - 0.5))
                                    for i in range(1, len(lst)):
                                        if (lst[i] - lst[i-1]) < clip_L:
                                            s -= 0.5
                                    return s
                                score = (score_one(Ar, clip_size["L_mm"], so_size["L_mm"]) +
                                         score_one(Cr, clip_size["L_mm"], so_size["L_mm"]) +
                                         score_one(Br, clip_size["L_mm"], so_size["W_mm"]) +
                                         score_one(Dr, clip_size["L_mm"], so_size["W_mm"]))
                                if score > best_score:
                                    best_score = score
                                    best_sets = (Ar, Br, Cr, Dr)
            A_centers_override, B_centers_override, C_centers_override, D_centers_override = best_sets

        # 更新 session_state
        if A_centers_override: st.session_state["A_list"] = " ".join(fmt_val(x) for x in A_centers_override)
        if B_centers_override: st.session_state["B_list"] = " ".join(fmt_val(x) for x in B_centers_override)
        if C_centers_override: st.session_state["C_list"] = " ".join(fmt_val(x) for x in C_centers_override)
        if D_centers_override: st.session_state["D_list"] = " ".join(fmt_val(x) for x in D_centers_override)

        st.session_state["a_list"] = st.session_state["A_list"]
        st.session_state["b_list"] = st.session_state["B_list"]
        st.session_state["c_list"] = st.session_state["C_list"]
        st.session_state["d_list"] = st.session_state["D_list"]

    except Exception as e:
        st.error(f"圖片處理/偵測失敗：{e}")

# ===================== New Design（手動可修） =====================
st.markdown("### New Design — 請手動修改")
with st.container():
    colA, colB = st.columns(2)
    with colA:
        d_list = st.text_input("左邊", value=st.session_state["d_list"])
        b_list = st.text_input("右邊", value=st.session_state["b_list"])
    with colB:
        c_list = st.text_input("上邊", value=st.session_state["c_list"])
        a_list = st.text_input("下邊", value=st.session_state["a_list"])

a_vals = to_float_list(a_list)
b_vals = to_float_list(b_list)
c_vals = to_float_list(c_list)
d_vals = to_float_list(d_list)

def session_or_default_list(key: str):
    return to_float_list(st.session_state.get(key, "10 25"))

A_vals = A_centers_override if A_centers_override else session_or_default_list("A_list")
B_vals = B_centers_override if B_centers_override else session_or_default_list("B_list")
C_vals = C_centers_override if C_centers_override else session_or_default_list("C_list")
D_vals = D_centers_override if D_centers_override else session_or_default_list("D_list")

# gaps（供模型）
A_gaps = centers_to_gaps(A_vals, so_size["L_mm"], clip_size["L_mm"])
C_gaps = centers_to_gaps(C_vals, so_size["L_mm"], clip_size["L_mm"])
B_gaps = centers_to_gaps(B_vals, so_size["W_mm"], clip_size["L_mm"])
D_gaps = centers_to_gaps(D_vals, so_size["W_mm"], clip_size["L_mm"])
gaps_original_mm = A_gaps + B_gaps + C_gaps + D_gaps

a_gaps = centers_to_gaps(a_vals, so_size["L_mm"], clip_size["L_mm"])
c_gaps = centers_to_gaps(c_vals, so_size["L_mm"], clip_size["L_mm"])
b_gaps = centers_to_gaps(b_vals, so_size["W_mm"], clip_size["L_mm"])
d_gaps = centers_to_gaps(d_vals, so_size["W_mm"], clip_size["L_mm"])
gaps_new_mm = a_gaps + b_gaps + c_gaps + d_gaps

# ===================== 畫面：板視覺 =====================
dc1, dc2 = st.columns(2)
with dc1:
    st.subheader("Original")
    fig_b, ax_b = plt.subplots(figsize=(6.2, 6.2 * so_size["W_mm"] / max(so_size["L_mm"], 1.0)))

    def draw_board_with_centers(
        ax, title, board_L_mm, board_W_mm, centers_dict,
        clip_L_mm, clip_W_mm, face_color="#3f6db3", clip_color="#f5a623",
        bg_arr=None, bg_alpha=1.0, board_alpha=0.92, show_board_face=True,
        rect_boxes=None, pxmm: int = PX_PER_MM  # 修補：加入 pxmm
    ):
        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        margin = max(clip_W_mm * 0.6, 3.0)
        ax.set_xlim(-clip_W_mm/2 - margin, board_L_mm + clip_W_mm/2 + margin)
        ax.set_ylim(-clip_W_mm/2 - margin, board_W_mm + clip_W_mm/2 + margin)
        ax.axis('off')

        if bg_arr is not None:
            ax.imshow(bg_arr, extent=(0, board_L_mm, 0, board_W_mm), origin='upper', alpha=bg_alpha)
        if show_board_face:
            ax.add_patch(plt.Rectangle((0, 0), board_L_mm, board_W_mm, fc=face_color, ec=None, lw=0, alpha=board_alpha))
        ax.text(board_L_mm/2, board_W_mm/2, title, color="white", fontsize=13, ha="center", va="center", weight="bold")

        rad = max(0.4, 0.15 * min(clip_L_mm, clip_W_mm))
        lab_off = max(1.0, 0.03 * board_W_mm)
        sdy_mm = SDY_MM_FIXED
        font_sz = FONT_SIZE_FIXED
        half = clip_L_mm / 2.0
        def label_text(val): return f"{fmt_val(val)}"

        # A 下邊
        for i, x_c in enumerate(sorted(centers_dict.get("A", []))):
            x_left = max(0.0, min(x_c - half, board_L_mm - clip_L_mm))
            ax.add_patch(FancyBboxPatch((x_left, -clip_W_mm/2), clip_L_mm, clip_W_mm,
                                        boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
            y_text = -lab_off + (sdy_mm if (i % 2 == 0) else -sdy_mm)
            ax.text(x_c, y_text, label_text(x_c), ha="center", va="top", fontsize=font_sz, color="black")
        # C 上邊
        for i, x_c in enumerate(sorted(centers_dict.get("C", []))):
            x_left = max(0.0, min(x_c - half, board_L_mm - clip_L_mm))
            ax.add_patch(FancyBboxPatch((x_left, board_W_mm - clip_W_mm/2), clip_L_mm, clip_W_mm,
                                        boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
            y_text = board_W_mm + lab_off + (sdy_mm if (i % 2 == 1) else -sdy_mm)
            ax.text(x_c, y_text, label_text(x_c), ha="center", va="bottom", fontsize=font_sz, color="black")
        # B 右邊
        for i, y_c in enumerate(sorted(centers_dict.get("B", []))):
            y_top = max(0.0, min(y_c - half, board_W_mm - clip_L_mm))
            y_bottom = board_W_mm - (y_top + clip_L_mm)
            y_bottom = max(0.0, min(y_bottom, board_W_mm - clip_L_mm))
            ax.add_patch(FancyBboxPatch((board_L_mm - clip_W_mm/2, y_bottom), clip_W_mm, clip_L_mm,
                                        boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
            y_text = board_W_mm - y_c + (sdy_mm if (i % 2 == 0) else -sdy_mm)
            ax.text(board_L_mm + lab_off, y_text, label_text(y_c), ha="left", va="center", fontsize=font_sz, color="black")
        # D 左邊
        for i, y_c in enumerate(sorted(centers_dict.get("D", []))):
            y_top = max(0.0, min(y_c - half, board_W_mm - clip_L_mm))
            y_bottom = board_W_mm - (y_top + clip_L_mm)
            y_bottom = max(0.0, min(y_bottom, board_W_mm - clip_L_mm))
            ax.add_patch(FancyBboxPatch((-clip_W_mm/2, y_bottom), clip_W_mm, clip_L_mm,
                                        boxstyle=f"round,pad=0,rounding_size={rad}", linewidth=0, facecolor=clip_color))
            y_text = board_W_mm - y_c + (sdy_mm if (i % 2 == 1) else -sdy_mm)
            ax.text(-lab_off, y_text, label_text(y_c), ha="right", va="center", fontsize=font_sz, color="black")

        # 疊圖：偵測框（全圖座標，px → mm）
        if rect_boxes:
            box_ec = "#00ff88"
            for (x, y, w, h) in rect_boxes.get("top", []):
                ax.add_patch(plt.Rectangle((x/pxmm, y/pxmm), w/pxmm, h/pxmm, fill=False, ec=box_ec, lw=1.2))
            for (x, y, w, h) in rect_boxes.get("bottom", []):
                ax.add_patch(plt.Rectangle((x/pxmm, y/pxmm), w/pxmm, h/pxmm, fill=False, ec=box_ec, lw=1.2))
            for (x, y, w, h) in rect_boxes.get("left", []):
                ax.add_patch(plt.Rectangle((x/pxmm, y/pxmm), w/pxmm, h/pxmm, fill=False, ec=box_ec, lw=1.2))
            for (x, y, w, h) in rect_boxes.get("right", []):
                ax.add_patch(plt.Rectangle((x/pxmm, y/pxmm), w/pxmm, h/pxmm, fill=False, ec=box_ec, lw=1.2))

    rect_boxes_to_draw = debug_boxes if SHOW_DEBUG_BOXES else None
    draw_board_with_centers(
        ax_b, "Original",
        board_L_mm=so_size["L_mm"], board_W_mm=so_size["W_mm"],
        centers_dict={"A": A_vals, "B": B_vals, "C": C_vals, "D": D_vals},
        clip_L_mm=clip_size["L_mm"], clip_W_mm=clip_size["W_mm"],
        face_color="#3f6db3", clip_color="#f5a623",
        bg_arr=bg_before, bg_alpha=1.0,
        board_alpha=0.0 if bg_before is not None else 0.92,
        show_board_face=(bg_before is None),
        rect_boxes=rect_boxes_to_draw,
        pxmm=PX_PER_MM
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
        board_alpha=0.92, show_board_face=True,
        rect_boxes=None,
        pxmm=PX_PER_MM
    )
    st.pyplot(fig_a, use_container_width=True)

# 摘要
st.markdown("---")
st.info(
    f"Top(C)={len(C_vals)} / Bottom(A)={len(A_vals)} / Left(D)={len(D_vals)} / Right(B)={len(B_vals)}；"
    f"Total={len(A_vals)+len(B_vals)+len(C_vals)+len(D_vals)}"
)

# ===================== 模型曲線 =====================
f_start_default = 0.500
f_end_default = 5.000
f_step_default = 0.050
f_start = st.session_state.get("f_start", f_start_default)
f_end = st.session_state.get("f_end", f_end_default)
f_step = st.session_state.get("f_step", f_step_default)
if f_end <= f_start: f_end = f_start + max(f_step, 1e-3)
if f_step <= 0: f_step = 0.01
n_steps = max(int(math.floor((f_end - f_start) / f_step)) + 1, 2)
freq = np.linspace(f_start, f_end, n_steps)

V_original = model_sum_gaps(freq, gaps_original_mm, alpha_model, R_m_model)
V_new = model_sum_gaps(freq, gaps_new_mm, alpha_model, R_m_model)
dB_original = to_dBm_from_Vrms(V_original, R_load)
dB_new = to_dBm_from_Vrms(V_new, R_load)

mark1_default = 2.400
mark2_default = 2.500
mark1 = st.session_state.get("mark1", mark1_default)
mark2 = st.session_state.get("mark2", mark2_default)

def eval_at_mark(mark_ghz, freq, curve_dB):
    if len(freq) < 2:
        return float(mark_ghz), float(curve_dB[0]) if len(curve_dB) else float("nan")
    mark = float(np.clip(mark_ghz, freq[0], freq[-1]))
    val = float(np.interp(mark, freq, curve_dB))
    return mark, val

m1, orig_m1 = eval_at_mark(mark1, freq, dB_original)
_, new_m1 = eval_at_mark(mark1, freq, dB_new)
m2, orig_m2 = eval_at_mark(mark2, freq, dB_original)
_, new_m2 = eval_at_mark(mark2, freq, dB_new)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(freq, dB_original, label="Original", color="#1f77b4", lw=2)
ax.plot(freq, dB_new, label="New", color="#ff7f0e", lw=2)
ax.axvline(m1, color="#2ca02c", ls="--", lw=1.5, label=f"Mark_1: {m1:.3f} GHz")
ax.scatter([m1], [orig_m1], color="#1f77b4", s=30, zorder=5)
ax.scatter([m1], [new_m1], color="#ff7f0e", s=30, zorder=5)
ax.annotate(f"Orig {orig_m1:.2f} dBm", xy=(m1, orig_m1), xytext=(5, 10),
            textcoords="offset points", color="#1f77b4", fontsize=9,
            bbox=dict(fc="white", alpha=0.7, ec="#1f77b4"))
ax.annotate(f"New {new_m1:.2f} dBm", xy=(m1, new_m1), xytext=(5, -15),
            textcoords="offset points", color="#ff7f0e", fontsize=9,
            bbox=dict(fc="white", alpha=0.7, ec="#ff7f0e"))
ax.axvline(m2, color="#d62728", ls="--", lw=1.5, label=f"Mark_2: {m2:.3f} GHz")
ax.scatter([m2], [orig_m2], color="#1f77b4", s=30, zorder=5)
ax.scatter([m2], [new_m2], color="#ff7f0e", s=30, zorder=5)
ax.annotate(f"Orig {orig_m2:.2f} dBm", xy=(m2, orig_m2), xytext=(5, 10),
            textcoords="offset points", color="#1f77b4", fontsize=9,
            bbox=dict(fc="white", alpha=0.7, ec="#1f77b4"))
ax.annotate(f"New {new_m2:.2f} dBm", xy=(m2, new_m2), xytext=(5, -15),
            textcoords="offset points", color="#ff7f0e", fontsize=9,
            bbox=dict(fc="white", alpha=0.7, ec="#ff7f0e"))
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Level (dBm)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
st.pyplot(fig, use_container_width=True)
