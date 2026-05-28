from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils.preprocessing import PROJECT_ROOT, load_csv, load_processed_dataset


st.set_page_config(page_title="데이터 확인", layout="wide")
st.title("데이터 확인")

raw_dir = PROJECT_ROOT / "data" / "raw"
processed = load_processed_dataset()

st.subheader("가공 데이터")
if processed.empty:
    st.info("`data/processed/final_dataset.csv` 파일이 아직 없습니다.")
else:
    st.dataframe(processed.tail(100), use_container_width=True)
    st.line_chart(processed.set_index("date").select_dtypes("number"))

st.subheader("원천 데이터")
csv_files = sorted(raw_dir.glob("*.csv"))
if not csv_files:
    st.info("`data/raw/` 폴더에 CSV 파일을 추가하면 여기에서 확인할 수 있습니다.")
else:
    selected = st.selectbox("파일 선택", csv_files, format_func=lambda path: Path(path).name)
    raw = load_csv(selected)
    st.dataframe(raw.head(100), use_container_width=True)

uploaded = st.file_uploader("임시 CSV 미리보기", type="csv")
if uploaded:
    preview = pd.read_csv(uploaded)
    st.dataframe(preview.head(100), use_container_width=True)
