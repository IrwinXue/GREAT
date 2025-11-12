# -*- coding: utf-8 -*-
# SL_gen.py
import io
import os
import re
from typing import Tuple, List, Set, Dict, Union
import difflib
import pandas as pd
import streamlit as st

# ========= Streamlit 頁面設定（需最先呼叫） =========
st.set_page_config(page_title="cpn_rep：一鍵查詢", layout="wide")

# ====== 可調參數 ======
W_TOKEN_MAIN = 0.7
W_TEXT_MAIN = 0.3
W_OVERLAP = 0.7
W_JACCARD = 0.3
BONUS_CORE_MATCH = 0.15
PENALTY_FUNC_CONFLICT = 0.15

# ========= 讀取 cpn_rep.txt =========
def load_cpn_rep(src_bytes_or_path: Union[str, bytes]) -> pd.DataFrame:
    """
    從報表（含表頭 'REFDES,PIN_NUMBER,...'）載入為 DataFrame。
    支援 bytes（上傳）與路徑（同目錄）。
    """
    if isinstance(src_bytes_or_path, bytes):
        raw_text = src_bytes_or_path.decode("utf-8", errors="ignore")
    else:
        with open(src_bytes_or_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

    lines = raw_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("REFDES,PIN_NUMBER"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("找不到表頭 'REFDES,PIN_NUMBER,...'")

    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text), engine="python")

    # 清理欄位空白
    for c in ["REFDES", "PIN_NUMBER", "NET_NAME", "PIN_NAME", "PIN_TYPE", "COMP_DEVICE_TYPE"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 關鍵欄位不得為空
    df = df.dropna(subset=["REFDES", "PIN_NUMBER", "NET_NAME"])
    return df

# ========= REFDES[pin] 解析 =========
PIN_PATTERNS = [
    r"^\s*([A-Za-z]+\d+)\s*\[\s*([A-Za-z0-9\-\_/#]+)\s*\]\s*$",  # CN1401[30]
    r"^\s*([A-Za-z]+\d+)\s*[:\- ]\s*([A-Za-z0-9\-\_/#]+)\s*$",   # CN1401:30 / CN1401-30
    r"^\s*([A-Za-z]+\d+)\s*$",                                   # CN1401
]
def parse_ref_pin(text: str) -> Tuple[Union[str, None], Union[str, None]]:
    for pat in PIN_PATTERNS:
        m = re.match(pat, text)
        if m:
            if len(m.groups()) == 2:
                return m.group(1).upper(), m.group(2)
            else:
                return m.group(1).upper(), None
    return None, None

# ========= Token / 同義字 / 停用 =========
TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")
STOPWORDS: Set[str] = {"WWAN","WLAN","CN","LS","HS","HPD","DBG","RT","SX","FB","A","B","C","L","R","P","N"}
SYN_MAP: Dict[str, str] = {
    "RESET":"RST","RSTB":"RST","RST#":"RST","RSTN":"RST",
    "UIM":"UIM","USIM":"UIM","SIM":"UIM","UICC":"UIM",
    "CLK":"CLK","CLOCK":"CLK",
    "PWR":"PWR","POWER":"PWR","VDD":"PWR","VCC":"PWR",
}
FUNCTION_CONFLICT_PAIRS = [("RST","CLK")]

def to_tokens(s: str) -> List[str]:
    if not s:
        return []
    raw = TOKEN_SPLIT_RE.split(s.upper())
    tokens = []
    for t in raw:
        if not t or t in STOPWORDS:
            continue
        t = SYN_MAP.get(t, t)
        tokens.append(t)
    return tokens

# ========= 相似度計算 =========
def token_overlap_min(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    denom = min(len(A), len(B))
    return inter / denom if denom else 0.0

def token_jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def core_match_bonus(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if A and B and A == B and len(A) >= 2:
        return BONUS_CORE_MATCH
    return 0.0

def function_conflict_penalty(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    for x, y in FUNCTION_CONFLICT_PAIRS:
        if (x in A and y in B) or (y in A and x in B):
            return PENALTY_FUNC_CONFLICT
    return 0.0

def score_net_against_query(net: str, query: str) -> float:
    tq = to_tokens(query)
    tn = to_tokens(net)
    s_overlap = token_overlap_min(tq, tn)
    s_jac = token_jaccard(tq, tn)
    token_score = W_OVERLAP * s_overlap + W_JACCARD * s_jac
    try:
        s_text = difflib.SequenceMatcher(None, net or "", query).ratio()
    except Exception:
        s_text = 0.0
    bonus = core_match_bonus(tq, tn)
    penalty = function_conflict_penalty(tq, tn)
    s = W_TOKEN_MAIN * token_score + W_TEXT_MAIN * s_text + bonus - penalty
    return max(0.0, min(1.0, s))

@st.cache_data(show_spinner=False)
def compute_similarity_series(df: pd.DataFrame, query: str) -> pd.Series:
    nets = df["NET_NAME"].astype(str)
    return nets.apply(lambda net: score_net_against_query(net, query))

# ========= 主介面 =========
st.title("cpn_rep.txt｜整合查詢：輸入 起點REFDES[pin] 與 終點REFDES，直接列出候選（前10筆）")

uploaded = st.file_uploader("請上傳 cpn_rep.txt（或使用同目錄檔案）", type=["txt"])
use_local = st.toggle("使用同目錄 cpn_rep.txt", value=True)
src: Union[str, bytes, None] = None
if uploaded is not None:
    src = uploaded.getvalue()
elif use_local and os.path.exists("cpn_rep.txt"):
    src = "cpn_rep.txt"
if src is None:
    st.info("請上傳檔案或將 cpn_rep.txt 放在與本 .py 同一資料夾。")
    st.stop()

# 讀檔
try:
    df = load_cpn_rep(src)
except Exception as e:
    st.error(f"讀檔失敗：{e}")
    st.stop()

st.success(f"已載入 {len(df):,} 筆資料，REFDES 種類 {df['REFDES'].nunique():,}。")

# 查詢輸入
with st.container():
    st.subheader("查詢條件（單一步驟）")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        user_str = st.text_input("起點 REFDES[pin]（例如：CN1401[30]）", value="CN1401[30]").strip()
    with col2:
        end_refdes = st.text_input("終點 REFDES（例如：U500）", value="U500").strip().upper()

refdes_src, pin_no = parse_ref_pin(user_str)
if refdes_src is None:
    st.warning("起點格式不符，請輸入像 CN1401[30] 或 CN1401:30。")
    st.stop()

pins_df = df.loc[df["REFDES"] == refdes_src].copy()
if pins_df.empty:
    st.warning(f"找不到起點 REFDES：{refdes_src}")
    st.stop()

if not end_refdes:
    st.warning("請輸入終點 REFDES。")
    st.stop()

if pin_no is None:
    all_pins = sorted(pins_df["PIN_NUMBER"].unique(), key=lambda x: (len(str(x)), str(x)))
    pin_no = st.selectbox("你只輸入了 REFDES，請選擇 pin number", options=all_pins)

target = pins_df.loc[pins_df["PIN_NUMBER"] == str(pin_no)].copy()
if target.empty:
    st.warning(f"{refdes_src} 找不到 pin {pin_no}，請確認。")
    st.stop()

nets = sorted(target["NET_NAME"].unique(), key=lambda x: (len(str(x)), str(x)))
net_for_similarity = nets[0] if len(nets) == 1 else st.selectbox("選擇作為查詢的 NET_NAME", options=nets)
if not net_for_similarity:
    st.warning("未取得可用的 NET_NAME。")
    st.stop()

# 計算相似度並準備候選（起點/終點/所有 RLC）
sim_series = compute_similarity_series(df, net_for_similarity)
work_df = df.copy()
work_df["similarity"] = sim_series

ref_upper = work_df["REFDES"].str.upper().str.strip()
mask_start = (ref_upper == refdes_src)
mask_end = (ref_upper == end_refdes)
mask_rlc = ref_upper.str.match(r'^[RLC]\d+', na=False)

candidates = work_df.loc[mask_start | mask_end | mask_rlc].copy()
candidates_sorted = candidates.sort_values(
    ["similarity", "NET_NAME", "REFDES", "PIN_NUMBER"],
    ascending=[False, True, True, True]
).head(10)

# 互動勾選欄位
candidates_sorted.insert(0, "串接", False)
candidates_sorted.insert(1, "非串接", False)

SHOW_COLS = ["串接", "非串接", "similarity", "REFDES", "PIN_NUMBER", "COMP_DEVICE_TYPE", "NET_NAME"]

st.caption("候選集 = 起點本身 + 終點本身 + 所有 R/L/C（^ [RLC]\\d+）")
st.markdown(
    f"**查詢字串（NET_NAME）**：`{net_for_similarity}` ｜ **起點**：`{refdes_src}[{pin_no}]` ｜ **終點**：`{end_refdes}`"
)

edited = st.data_editor(
    candidates_sorted[SHOW_COLS],
    key="edit_candidates",
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "串接": st.column_config.CheckboxColumn("串接", help="勾選此列為『串接』"),
        "非串接": st.column_config.CheckboxColumn("非串接", help="勾選此列為『非串接』"),
        "similarity": st.column_config.NumberColumn("similarity", format="%.3f", help="字串相似度分數"),
        "REFDES": st.column_config.TextColumn("REFDES"),
        "PIN_NUMBER": st.column_config.TextColumn("PIN_NUMBER"),
        "COMP_DEVICE_TYPE": st.column_config.TextColumn("COMP_DEVICE_TYPE"),
        "NET_NAME": st.column_config.TextColumn("NET_NAME"),
    }
)

# 互斥保護：同列同時勾選時，優先保留『串接』
both_true = edited["串接"] & edited["非串接"]
if both_true.any():
    st.warning("偵測到有列同時勾選『串接』與『非串接』；已自動取消『非串接』以維持互斥。")
    edited.loc[both_true, "非串接"] = False

# -------- 非串接補齊「另一腳」的工具函式 --------
def build_nonseries_with_pair(edited_df: pd.DataFrame, base_df_with_sim: pd.DataFrame) -> pd.DataFrame:
    """
    針對「非串接」：
    - 取出勾選的列；
    - 若該 REFDES 在原始資料中恰為 2 腳（二腳件），則自動補上「另一個 PIN」；
    - 以 REFDES + PIN_NUMBER 去重。
    """
    non_sel = edited_df.loc[edited_df["非串接"]].copy()
    if non_sel.empty:
        return non_sel

    mate_rows: List[pd.DataFrame] = []

    base_df = base_df_with_sim.copy()
    base_df["REFDES_UP"] = base_df["REFDES"].str.upper().str.strip()
    base_df["PIN_NUMBER_STR"] = base_df["PIN_NUMBER"].astype(str)

    for _, row in non_sel.iterrows():
        ref = str(row["REFDES"]).upper().strip()
        pin = str(row["PIN_NUMBER"]).strip()

        ref_all = base_df.loc[base_df["REFDES_UP"] == ref].copy()
        # 僅在恰好兩個 pin 時補齊
        if ref_all["PIN_NUMBER_STR"].nunique() == 2:
            mate = ref_all.loc[ref_all["PIN_NUMBER_STR"] != pin].copy()
            if not mate.empty:
                mate = mate.assign(串接=False, 非串接=True)
                mate = mate[["串接", "非串接", "similarity", "REFDES", "PIN_NUMBER", "COMP_DEVICE_TYPE", "NET_NAME"]]
                mate_rows.append(mate)

    if mate_rows:
        mates_cat = pd.concat(mate_rows, ignore_index=True)
        non_sel_show = non_sel[["串接", "非串接", "similarity", "REFDES", "PIN_NUMBER", "COMP_DEVICE_TYPE", "NET_NAME"]]
        out = pd.concat([non_sel_show, mates_cat], ignore_index=True)
        out = out.drop_duplicates(subset=["REFDES", "PIN_NUMBER"], keep="first")
        return out
    else:
        return non_sel[["串接", "非串接", "similarity", "REFDES", "PIN_NUMBER", "COMP_DEVICE_TYPE", "NET_NAME"]]

# 生成兩個輸出表格
series_df = edited.loc[edited["串接"], SHOW_COLS].copy()
nonseries_df = build_nonseries_with_pair(edited, work_df)

# 顯示：串接表格
st.subheader("串接：已勾選的項目")
if series_df.empty:
    st.info("目前尚未勾選任何『串接』項目。")
else:
    st.dataframe(series_df, use_container_width=True)

series_csv = series_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "下載 串接 清單（CSV）",
    data=series_csv,
    file_name="mapped_refdes_pins_series.csv",
    mime="text/csv",
    disabled=series_df.empty
)

# 顯示：非串接表格（含自動補齊另一腳）
st.subheader("非串接：已勾選的項目")
if nonseries_df.empty:
    st.info("目前尚未勾選任何『非串接』項目。")
else:
    st.dataframe(nonseries_df, use_container_width=True)

nonseries_csv = nonseries_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "下載 非串接 清單（CSV）",
    data=nonseries_csv,
    file_name="mapped_refdes_pins_nonseries.csv",
    mime="text/csv",
    disabled=nonseries_df.empty
)

# -------------------------------
# Script_Language：生成邏輯（起點/終點保留 REFDES，其餘泛化）
# -------------------------------
VALUE_PAT = re.compile(r'(\d+(?:\.\d+)?)\s*(PF|NF|UF|pF|nF|uF|OHM|Ω|mH|uH|nH|H)', re.IGNORECASE)

def extract_value(comp_type: str) -> Union[str, None]:
    """
    從 COMP_DEVICE_TYPE 抽取數值+單位（例如：18PF、4.7UF、10Ω、1uH）
    會將 Ω 正規化為 OHM，大小寫正規化為大寫。
    """
    if not isinstance(comp_type, str):
        return None
    m = VALUE_PAT.search(comp_type)
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    unit_norm = unit.upper().replace("Ω", "OHM")
    unit_norm = {
        "PF":"PF", "NF":"NF", "UF":"UF",
        "NH":"NH", "UH":"UH", "MH":"MH", "H":"H",
        "OHM":"OHM"
    }.get(unit_norm, unit_norm)
    return f"{num}{unit_norm}"

def prefer_net_order(nets: List[str]) -> List[str]:
    """ 讓 DGND 若存在，優先排第一，其餘用字母序。 """
    nets = list(set(nets))
    nets.sort()
    if "DGND" in nets:
        nets.remove("DGND")
        nets = ["DGND"] + nets
    return nets

def build_script_language(series_df: pd.DataFrame,
                          nonseries_df: pd.DataFrame,
                          start_refdes: str,
                          end_refdes: str) -> str:
    """
    生成 DSL 腳本：
    - 對『起點 REFDES』與『終點 REFDES』：**保留 REFDES 條件**（不泛化）。
    - 其他 REFDES：**泛化**（不含 REFDES 條件）。

    非串接（二腳件）示例：
      - 起/終點：EXISTS C (VALUE == "18PF" AND CONNECTS("DGND") AND CONNECTS("UIM_RST_CN") AND REFDES == "C1451")
      - 其他：   EXISTS C (VALUE == "18PF" AND CONNECTS("DGND") AND CONNECTS("UIM_RST_CN"))

    串接（泛化模板）示例：
      - 起/終點：EXISTS R (VALUE == "10OHM" AND ON_NET IN {"NET1","NET2"} AND REFDES == "R123")
      - 其他：   EXISTS R (VALUE == "10OHM" AND ON_NET IN {"NET1","NET2"})
    """
    start_up = (start_refdes or "").upper().strip()
    end_up   = (end_refdes or "").upper().strip()

    lines: List[str] = []
    lines.append("# Script_Language generated from selections")
    lines.append(f'# 起點保留 REFDES：{start_up} ｜ 終點保留 REFDES：{end_up}')

    # ---- 非串接（二腳件）----
    lines.append("# 類別：非串接（二腳件，依起/終點決定是否保留 REFDES）")
    if not nonseries_df.empty:
        for refdes, group in nonseries_df.groupby("REFDES"):
            refdes_up = str(refdes).upper().strip()
            comp = group["COMP_DEVICE_TYPE"].iloc[0]
            val = extract_value(comp)
            nets = prefer_net_order(group["NET_NAME"].tolist())
            kind = "C" if refdes_up.startswith("C") else ("R" if refdes_up.startswith("R") else ("L" if refdes_up.startswith("L") else "X"))

            if len(nets) >= 2:
                include_refdes = (refdes_up == start_up) or (refdes_up == end_up)
                conds = []
                if val:
                    conds.append(f'VALUE == "{val}"')
                for net in nets[:2]:  # 二腳件取前兩個網路
                    conds.append(f'CONNECTS("{net}")')
                if include_refdes:
                    conds.append(f'REFDES == "{refdes_up}"')
                lines.append(f'EXISTS {kind} ({ " AND ".join(conds) })')

    # ---- 串接 ----
    lines.append("")
    lines.append("# 類別：串接（依起/終點決定是否保留 REFDES）")
    if not series_df.empty:
        for refdes, group in series_df.groupby("REFDES"):
            refdes_up = str(refdes).upper().strip()
            comp = group["COMP_DEVICE_TYPE"].iloc[0]
            val = extract_value(comp)
            nets = sorted(set(group["NET_NAME"].tolist()))
            kind = "C" if refdes_up.startswith("C") else ("R" if refdes_up.startswith("R") else ("L" if refdes_up.startswith("L") else "X"))
            include_refdes = (refdes_up == start_up) or (refdes_up == end_up)

            conds = []
            if val:
                conds.append(f'VALUE == "{val}"')
            if nets:
                conds.append("ON_NET IN {" + ", ".join([f'"{n}"' for n in nets]) + "}")
            if include_refdes:
                conds.append(f'REFDES == "{refdes_up}"')

            if conds:
                lines.append(f'EXISTS {kind} ({ " AND ".join(conds) })')
            else:
                # 理論上不會進來，保留完整性
                lines.append(f'EXISTS {kind}')

    # 人類可讀摘要
    lines.append("")
    lines.append("# 人類可讀摘要：")
    if not nonseries_df.empty:
        for refdes, group in nonseries_df.groupby("REFDES"):
            refdes_up = str(refdes).upper().strip()
            comp = group["COMP_DEVICE_TYPE"].iloc[0]
            val = extract_value(comp)
            nets = prefer_net_order(group["NET_NAME"].tolist())
            if len(nets) >= 2 and refdes_up.startswith("C"):
                if (refdes_up == start_up) or (refdes_up == end_up):
                    lines.append(f'- 在起/終點：找到 C（{val or "未知容值"}）同時連接 {nets[0]} 與 {nets[1]}，REFDES={refdes_up}')
                else:
                    lines.append(f'- 泛化：找到任一 C（{val or "未知容值"}）同時連接 {nets[0]} 與 {nets[1]}')

    return "\n".join(lines)

# 生成 Script_Language 與下載
st.subheader("邏輯腳本｜Script_Language（起/終點保留 REFDES，其餘泛化）")
if st.button("Script_Language：生成邏輯判斷式"):
    script_txt = build_script_language(series_df, nonseries_df, refdes_src, end_refdes)
    st.code(script_txt, language="text")
    st.download_button(
        "下載 Script_Language.txt",
        data=script_txt.encode("utf-8"),
        file_name="Script_Language.txt",
        mime="text/plain"
    )
