# =============================================================================
# 외생변수 다중공선성 확인 스크립트
# — 상관행렬 히트맵 + VIF 막대그래프
#
# 필요한 라이브러리 설치 (처음 한 번만):
#   pip install pandas numpy seaborn matplotlib statsmodels
# =============================================================================

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# ── 0. 설정 — 여기만 수정하면 됩니다 ─────────────────────────────────────────

# CSV 파일 경로 (본인 경로로 바꾸세요)
CSV_PATH = "data/processed/final_dataset.csv"
TARGET_COL = "sp500_log_return"

CANDIDATE_COLUMNS = [
    "term_spread_10y_2y",   # 원하는 변수만 남기거나 추가
    "vix_log_return",
    ...
]


# 분석할 후보 변수 목록 (자유롭게 추가·삭제하세요)
CANDIDATE_COLUMNS = [
    "term_spread_10y_2y",
    "financial_conditions",
    "vix_log_return",
    "dxy_log_return",
    "oil_log_return",
    "cpi_change",
]

# 결과 PNG 저장 폴더 (스크립트와 같은 폴더에 저장)
OUTPUT_DIR = Path(__file__).parent

# ── 1. 데이터 불러오기 ────────────────────────────────────────────────────────

print("=" * 60)
print("■ 1단계: 데이터 불러오기")
print("=" * 60)

# 파일이 없을 때 친절하게 안내하고 종료
if CSV_PATH == r"<여기에_파일경로.csv>":
    print("[오류] CSV_PATH를 실제 파일 경로로 바꿔주세요.")
    print("  예시: CSV_PATH = r'C:/Users/사용자명/Documents/data/final_dataset.csv'")
    sys.exit(1)

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] 파일을 찾을 수 없습니다: {csv_path}")
    print("  경로와 파일명을 다시 확인해 주세요.")
    sys.exit(1)

# CSV 읽기 (date 컬럼이 있으면 자동 파싱)
try:
    df_raw = pd.read_csv(csv_path, parse_dates=["date"])
    print(f"  파일 로드 완료: {len(df_raw):,}행 × {len(df_raw.columns)}열")
except Exception as e:
    print(f"[오류] CSV를 읽는 도중 문제가 발생했습니다: {e}")
    sys.exit(1)

# ── 2. 컬럼 존재 확인 + 선택 ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 2단계: 후보 변수 확인")
print("=" * 60)

available_cols = []    # 실제로 존재하는 변수
missing_cols = []      # 없는 변수

for col in CANDIDATE_COLUMNS:
    if col in df_raw.columns:
        available_cols.append(col)
    else:
        missing_cols.append(col)

# 없는 변수는 경고만 하고 계속 진행
if missing_cols:
    print(f"  [경고] 아래 변수는 CSV에 없어서 분석에서 제외합니다:")
    for c in missing_cols:
        print(f"    - {c}")

if not available_cols:
    print("[오류] 분석 가능한 변수가 하나도 없습니다. CANDIDATE_COLUMNS를 확인해 주세요.")
    sys.exit(1)

print(f"\n  분석에 사용할 변수 ({len(available_cols)}개):")
for c in available_cols:
    print(f"    ✓ {c}")

# ── 3. 전처리 ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 3단계: 전처리")
print("=" * 60)

# 선택한 컬럼만 추리고 숫자형으로 강제 변환
df = df_raw[available_cols].copy()
df = df.apply(pd.to_numeric, errors="coerce")   # 문자가 섞여 있어도 숫자로 변환

# lag 생성 과정에서 생긴 결측치(NaN) 제거
before_rows = len(df)
df = df.dropna()
after_rows = len(df)

print(f"  결측치 제거 전: {before_rows:,}행")
print(f"  결측치 제거 후: {after_rows:,}행  (제거된 행: {before_rows - after_rows:,}개)")

# 표본이 너무 작으면 경고
if after_rows < 30:
    print(f"\n  [경고] 남은 관측치({after_rows}행)가 너무 적습니다.")
    print("         VIF·상관분석 결과를 신중하게 해석하세요.")
elif after_rows < 100:
    print(f"\n  [주의] 관측치가 {after_rows}행입니다. 결과 해석 시 참고하세요.")
else:
    print(f"\n  최종 분석 표본 수: {after_rows:,}행 — 분석에 충분합니다.")

# ── 4. 피어슨 상관행렬 계산 ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 4단계: 피어슨 상관행렬 계산")
print("=" * 60)

corr_matrix = df.corr(method="pearson")
print("  상관행렬 계산 완료.")

# ── 5. VIF 계산 ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 5단계: VIF(분산팽창계수) 계산")
print("=" * 60)

# 상수항(절편)을 추가해야 VIF가 정확합니다 — 빠뜨리면 값이 틀림
df_with_const = add_constant(df, has_constant="add")

vif_list = []
for i, col in enumerate(df.columns):   # 상수항(const)은 건너뜁니다
    vif_val = variance_inflation_factor(df_with_const.values, i + 1)  # i+1: const가 0번째
    vif_list.append({"변수": col, "VIF": round(vif_val, 2)})

vif_df = pd.DataFrame(vif_list).sort_values("VIF", ascending=False).reset_index(drop=True)

print("  VIF 계산 결과 (내림차순):")
print(vif_df.to_string(index=False))

# ── 6. 시각화 — 상관행렬 히트맵 ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 6단계: 시각화")
print("=" * 60)

# 글꼴 설정 (한글이 깨지는 경우 대비)
plt.rcParams["axes.unicode_minus"] = False
try:
    plt.rcParams["font.family"] = "Malgun Gothic"   # 윈도우 한글 글꼴
except Exception:
    pass

# ─ 6-1. 상관행렬 히트맵 ─
n_vars = len(available_cols)
fig_size = max(8, n_vars * 0.9)   # 변수 수에 따라 크기 자동 조정

fig1, ax1 = plt.subplots(figsize=(fig_size, fig_size * 0.85))

sns.heatmap(
    corr_matrix,
    ax=ax1,
    annot=True,             # 각 셀에 상관계수 숫자 표시
    fmt=".2f",              # 소수점 두 자리
    vmin=-1, vmax=1,        # 색상 범위 고정
    center=0,               # 0을 중심으로 색상 발산
    cmap="RdBu_r",          # 빨강(양의 상관)↔파랑(음의 상관) 발산형 컬러맵
    linewidths=0.5,         # 셀 구분선
    square=True,            # 셀을 정사각형으로
    cbar_kws={"shrink": 0.8, "label": "피어슨 상관계수"},
)

ax1.set_title("외생변수 후보 — 피어슨 상관행렬", fontsize=14, fontweight="bold", pad=15)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.tick_params(axis="x", rotation=40, labelsize=10)
ax1.tick_params(axis="y", rotation=0,  labelsize=10)

plt.tight_layout()
heatmap_path = OUTPUT_DIR / "corr_heatmap.png"
fig1.savefig(heatmap_path, dpi=150, bbox_inches="tight")
print(f"  히트맵 저장 완료: {heatmap_path}")
plt.show()

# ─ 6-2. VIF 막대그래프 ─
fig2, ax2 = plt.subplots(figsize=(max(8, n_vars * 0.9), 5))

# VIF 내림차순으로 정렬된 색상: 10 초과=빨강, 5~10=주황, 5 미만=초록
colors = []
for v in vif_df["VIF"]:
    if v > 10:
        colors.append("#e74c3c")   # 빨강 — 심각
    elif v > 5:
        colors.append("#f39c12")   # 주황 — 주의
    else:
        colors.append("#2ecc71")   # 초록 — 양호

bars = ax2.bar(vif_df["변수"], vif_df["VIF"], color=colors, edgecolor="white", linewidth=0.8)

# 막대 위에 VIF 값 표시
for bar, val in zip(bars, vif_df["VIF"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{val:.1f}",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

# 기준선: VIF=5(주의), VIF=10(심각)
ax2.axhline(y=5,  color="#f39c12", linestyle="--", linewidth=1.5, label="VIF = 5  (주의 기준)")
ax2.axhline(y=10, color="#e74c3c", linestyle="--", linewidth=1.5, label="VIF = 10 (심각 기준)")

ax2.set_title("외생변수 후보 — VIF(분산팽창계수)", fontsize=14, fontweight="bold", pad=12)
ax2.set_xlabel("변수", fontsize=11)
ax2.set_ylabel("VIF", fontsize=11)
ax2.tick_params(axis="x", rotation=35, labelsize=9)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.legend(fontsize=10)
ax2.set_ylim(bottom=0)

plt.tight_layout()
vif_path = OUTPUT_DIR / "vif_barplot.png"
fig2.savefig(vif_path, dpi=150, bbox_inches="tight")
print(f"  VIF 막대그래프 저장 완료: {vif_path}")
plt.show()

# ── 7. 텍스트 해석 출력 ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 7단계: 해석")
print("=" * 60)

# ─ 7-1. 절대 상관계수 0.8 이상인 변수쌍 ─
print("\n◆ 절대 상관계수 0.8 이상인 변수쌍 (다중공선성 의심)")
high_corr_pairs = []
cols = corr_matrix.columns.tolist()

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):          # 중복 제거를 위해 상삼각만 봄
        r = corr_matrix.iloc[i, j]
        if abs(r) >= 0.8:
            high_corr_pairs.append((cols[i], cols[j], r))

if high_corr_pairs:
    for a, b, r in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        direction = "양(+)의 상관" if r > 0 else "음(-)의 상관"
        print(f"  • {a}  ↔  {b}   r = {r:.3f}  ({direction})")
else:
    print("  해당 없음 — 0.8 이상 강한 상관관계 쌍이 없습니다.")

# ─ 7-2. VIF 10 초과 변수 ─
print("\n◆ VIF 10 초과 변수 (다중공선성 심각)")
high_vif = vif_df[vif_df["VIF"] > 10]

if not high_vif.empty:
    for _, row in high_vif.iterrows():
        print(f"  • {row['변수']}   VIF = {row['VIF']:.2f}")
else:
    print("  해당 없음 — VIF 10 초과 변수가 없습니다.")

# ─ 7-3. 제거 검토 제안 ─
print("\n◆ 변수 제거 검토 제안")

suggestions = []

# 높은 상관쌍 중 VIF가 더 높은 것을 제거 후보로
for a, b, r in high_corr_pairs:
    vif_a = vif_df.loc[vif_df["변수"] == a, "VIF"].values
    vif_b = vif_df.loc[vif_df["변수"] == b, "VIF"].values
    vif_a = vif_a[0] if len(vif_a) > 0 else 0
    vif_b = vif_b[0] if len(vif_b) > 0 else 0
    remove_candidate = a if vif_a >= vif_b else b
    keep_candidate   = b if remove_candidate == a else a
    suggestions.append(
        f"  • '{a}'와 '{b}'의 상관계수가 {r:.2f}로 높습니다. "
        f"둘 중 VIF가 더 높은 '{remove_candidate}' 제거를 검토해 보세요 "
        f"('{keep_candidate}' 유지)."
    )

# VIF 10 초과 중 상관쌍에 없는 것 추가
for _, row in high_vif.iterrows():
    already_suggested = any(row["변수"] in s for s in suggestions)
    if not already_suggested:
        suggestions.append(
            f"  • '{row['변수']}'의 VIF({row['VIF']:.1f})가 10을 넘습니다. "
            f"다른 변수와 강하게 선형 결합된 것으로 보이므로 제거를 검토해 보세요."
        )

if suggestions:
    for s in suggestions[:4]:   # 너무 많으면 상위 4개만
        print(s)
else:
    print("  현재 후보 변수들은 심각한 다중공선성 없이 비교적 독립적입니다.")
    print("  VIF 5~10 구간 변수가 있다면 모델 성능을 보면서 선택적으로 제거하세요.")

print("\n" + "=" * 60)
print("분석 완료.")
print("=" * 60)
