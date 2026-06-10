"""
5_similar_periods.py — 과거 유사 시점 탐색기 (독립 페이지)

로직은 utils/similar_periods.py 에 있으며, 이 파일은
st.set_page_config 설정 후 render_similar_periods() 를 호출하기만 한다.
pages/3_forecast.py 의 "4) 유사 시점 탐색기" 메뉴와 동일한 내용을 표시한다.
"""

from __future__ import annotations

import streamlit as st
from utils.similar_periods import render_similar_periods

st.set_page_config(page_title="유사 시점 탐색기", layout="wide")

render_similar_periods()
