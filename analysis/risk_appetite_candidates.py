# =============================================================================
# risk_appetite_candidates.py  —  NFCILEVERAGE 기반 RAI 워크포워드 검증
#
# 목적:
#   NFCILEVERAGE(레버리지 서브인덱스)로 구성한 Risk Appetite Index(RAI)가
#   기존 best_spec 대비 표본 밖에서 실제로 가치를 더하는지 검증한다.
#
#   테스트 사양:
#     1) base
#     2) base + neg_nfci_leverage(lag)       ← 레버리지 프록시 단독
#     3) (base에서 vix 제거) + RAI            ← 표준 RAI 사양
#     4) (base에서 vix+financial_cond 제거) + RAI  ← 중복 의심 시 비교용
#
# 외부 데이터(FRED):
#   - NFCILEVERAGE : Chicago Fed NFCI 레버리지 서브인덱스 (주간)
#   - BAA10Y       : Moody's Baa - 10Y Treasury 신용스프레드 (일별)
#                    ※ BAMLH0A0HYM2 (ICE BofA HY OAS)는 FRED에서 2023~ 만
#                      제공되므로 투자적격 신용스프레드(BAA10Y)로 대체
#   - VIX          : 기존 CSV에 이미 있는 컬럼 사용
#
# 필요 라이브러리 설치:
#   pip install pandas numpy statsmodels scikit-learn fredapi python-dotenv
#
# 느리면 REFIT_EVERY를 8~12로 높이세요.
# =============================================================================

import itertools
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# ── 0. 설정 — 여기만 수정하면 됩니다 ─────────────────────────────────────────

CSV_PATH   = "data/processed/final_dataset.csv"
TARGET_COL = "sp500_log_return"

# ★ 직전 워크포워드에서 채택된 최고 사양으로 교체하세요.
BEST_SPEC: list[str] = [
    "financial_conditions_lag1",   # 앵커 변수 (walk_forward_validation.py 에서 생성한 명칭)
    # "vix_log_return_lag1",       # vix가 채택됐으면 주석 해제
    # "cpi_change_lag4",           # cpi가 채택됐으면 주석 해제
]

# FRED 시리즈 ID (교체 불필요, 검증 완료)
FRED_NFCI_LEV = "NFCILEVERAGE"   # Chicago Fed NFCI 레버리지 서브인덱스 (주간)
FRED_CREDIT   = "BAA10Y"         # Moody's Baa 신용스프레드 (일별, BAMLH0A0HYM2 대체)
# ※ BAMLH0A0HYM2 교체 이유: FRED API가 2023~ 데이터만 반환 (접근 제한)
#    BAA10Y는 투자적격 등급 신용스프레드로 HY와 방향성이 동일하며 1986~로 이력이 충분

VIX_COL   = "vix"                 # 기존 CSV의 VIX 컬럼명 (재조회 불필요)
FC_COL    = "financial_conditions" # 기존 CSV의 금융여건 컬럼명

HOLDOUT_WEEKS   = 78
MIN_TRAIN_WEEKS = 260
REFIT_EVERY     = 4
WINDOW          = "expanding"

MAX_P = 2
MAX_Q = 2

OUTPUT_DIR = Path(__file__).parent

# ── 1. FRED 데이터 조회 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ NFCILEVERAGE 기반 RAI 워크포워드 검증")
print("=" * 65)

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("[오류] .env 파일에 FRED_API_KEY 가 없습니다.")
    print("  .env 파일에 아래 형식으로 추가하세요:")
    print("  FRED_API_KEY=your_key_here")
    sys.exit(1)

try:
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
except ImportError:
    print("[오류] fredapi 가 없습니다. pip install fredapi 를 실행하세요.")
    sys.exit(1)

print("\n■ 1단계: FRED 데이터 조회")
print("=" * 65)

# ---- 1-1. NFCILEVERAGE (주간, 이미 표준화됨) ----
print(f"  [{FRED_NFCI_LEV}] 조회 중...")
try:
    nfci_raw = fred.get_series(FRED_NFCI_LEV, observation_start="1999-01-01")
    nfci_raw.index = pd.to_datetime(nfci_raw.index)
    print(f"  ✓ {FRED_NFCI_LEV}: {len(nfci_raw)}행, "
          f"{nfci_raw.index[0].date()} ~ {nfci_raw.index[-1].date()}")
    print(f"    평균={nfci_raw.mean():.3f}, 표준편차={nfci_raw.std():.3f}  "
          f"(이미 표준화된 지수 — 양수=레버리지 축소/위험회피)")
except Exception as e:
    print(f"  ✗ {FRED_NFCI_LEV} 조회 실패: {e}")
    sys.exit(1)

# ---- 1-2. 신용스프레드 BAA10Y (일별) ----
print(f"\n  [{FRED_CREDIT}] 조회 중...")
print(f"    ※ BAMLH0A0HYM2(ICE BofA HY OAS)는 FRED API 접근 제한으로 2023~만 반환됨")
print(f"       → Moody's Baa 신용스프레드(BAA10Y)로 대체 (HY와 방향성 동일, 1986~ 이력)")
try:
    time.sleep(0.5)
    credit_raw = fred.get_series(FRED_CREDIT, observation_start="1999-01-01")
    credit_raw.index = pd.to_datetime(credit_raw.index)
    print(f"  ✓ {FRED_CREDIT}: {len(credit_raw)}행, "
          f"{credit_raw.index[0].date()} ~ {credit_raw.index[-1].date()}")
    print(f"    결측={credit_raw.isna().sum()}개 (주말·공휴일)")
except Exception as e:
    print(f"  ✗ {FRED_CREDIT} 조회 실패: {e}")
    sys.exit(1)

# ── 2. 기존 CSV 불러오기 ────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 2단계: 기존 CSV 불러오기")
print("=" * 65)

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] CSV를 찾을 수 없습니다: {csv_path.resolve()}")
    sys.exit(1)

df_csv = pd.read_csv(csv_path, parse_dates=["date"])
df_csv = df_csv.sort_values("date").reset_index(drop=True)
print(f"  로드: {len(df_csv):,}행, {df_csv['date'].min().date()} ~ {df_csv['date'].max().date()}")

# 필요 컬럼 존재 확인
for col in [TARGET_COL, VIX_COL, FC_COL]:
    if col not in df_csv.columns:
        print(f"  [오류] 컬럼 없음: '{col}'")
        sys.exit(1)
print(f"  ✓ 필수 컬럼 확인: {TARGET_COL}, {VIX_COL}, {FC_COL}")

# ── 3. 주간 금요일 기준으로 모든 시계열 정렬 ──────────────────────────────────────

print("\n" + "=" * 65)
print("■ 3단계: 주간 정렬 (모두 금요일 종료 기준)")
print("=" * 65)

# ---- 기준 인덱스: CSV 의 날짜 (이미 주간 금요일) ----
csv_idx = pd.to_datetime(df_csv["date"])
csv_idx_set = set(csv_idx.dt.normalize())

# ---- NFCILEVERAGE: 이미 주간 금요일 → 날짜만 정규화 ----
# NFCI는 금요일 기준이나 간혹 목·수 발표도 있어 W-FRI로 리샘플
nfci_weekly = (
    nfci_raw
    .resample("W-FRI").last()    # 금요일 값 사용
    .ffill()                      # 누락된 주는 직전 값으로 채움
)
nfci_weekly.name = "nfcileverage"
print(f"  NFCILEVERAGE → 주간: {len(nfci_weekly)}행")

# ---- BAA10Y: 일별 → 주간 금요일 (마지막 영업일 값, 이후 ffill) ----
credit_weekly = (
    credit_raw
    .ffill()                      # 일별 결측(공휴일) 먼저 채움
    .resample("W-FRI").last()     # 금요일 값
    .ffill()
)
credit_weekly.name = "credit_spread"
print(f"  BAA10Y → 주간: {len(credit_weekly)}행")

# ---- VIX: CSV에서 직접 사용 (이미 주간) ----
vix_series = df_csv.set_index("date")[VIX_COL]
vix_series.index = pd.to_datetime(vix_series.index).normalize()
vix_series.name = "vix_raw"

# ---- CSV 기준으로 외부 시리즈 join ----
df_base = df_csv[["date", TARGET_COL, VIX_COL, FC_COL]].copy()
df_base["date"] = pd.to_datetime(df_base["date"])
df_base = df_base.set_index("date")

nfci_al  = nfci_weekly.reindex(df_base.index, method="ffill")
credit_al = credit_weekly.reindex(df_base.index, method="ffill")

df_aligned = df_base.copy()
df_aligned["nfcileverage"]  = nfci_al
df_aligned["credit_spread"] = credit_al
df_aligned = df_aligned.reset_index()

n_valid = df_aligned[["nfcileverage","credit_spread"]].notna().all(axis=1).sum()
print(f"  JOIN 후 유효 행: {n_valid:,} / 전체 {len(df_aligned):,}행")

# ── 4. 새 변수 생성 (누수 없음) ──────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 4단계: 새 변수 생성 (시점 t는 t-1까지의 정보만 사용)")
print("=" * 65)

# ---- 4-1. neg_nfci_leverage ----
# NFCILEVERAGE 양수 = 레버리지 긴축/위험회피이므로 부호 반전
# → 값↑ = 레버리지 확대 = 위험선호  (마진뎁트 YoY 자리 대체)
df_aligned["neg_nfci_leverage"] = -1 * df_aligned["nfcileverage"]

# 1주 lag → 시점 t에서 t-1 값 = 이미 알려진 정보 (누수 없음)
df_aligned["neg_nfci_leverage_lag1"] = df_aligned["neg_nfci_leverage"].shift(1)

print("  ✓ neg_nfci_leverage_lag1 생성")
print("    공식: -1 × NFCILEVERAGE → shift(1)")
print("    해석: 값↑ = 레버리지 확대 = 위험선호 의미")

# ---- 4-2. RAI: 성분별 팽창 z-score → 합성 ----
# 핵심: 각 성분을 '먼저' 표준화한 뒤 합성 (스케일 큰 성분 지배 방지)
# 팽창 z-score: 시점 t에서 0..t-1 구간의 평균·표준편차 사용 (shift(1) → 누수 없음)
# ※ NFCILEVERAGE가 이미 표준화돼 있어도 반드시 다시 표준화 (in-fold 일관성)
# ※ 대안: 고정 ±1 가중 대신 PCA 가중을 쓰려면 아래 주석 코드를 참고하세요.

def expanding_zscore(s: pd.Series) -> pd.Series:
    """
    시점 t에서 0..t-1 구간의 통계로 z-score를 계산한다 (lookahead 없음).
    shift(1)로 현재 시점을 통계 계산에서 제외하므로 데이터 누수가 없다.
    """
    # 0..t-1 구간의 누적 평균·표준편차 (현재 시점 제외)
    exp_mean = s.expanding().mean().shift(1)
    exp_std  = s.expanding().std(ddof=1).shift(1)
    # 표준편차가 0에 가까우면 NaN 처리
    safe_std = exp_std.where(exp_std > 1e-8, np.nan)
    return (s - exp_mean) / safe_std


z_neg_lev = expanding_zscore(df_aligned["neg_nfci_leverage"])
z_credit  = expanding_zscore(df_aligned["credit_spread"])
z_vix     = expanding_zscore(df_aligned[VIX_COL])

# RAI = z(레버리지 프록시) - z(신용스프레드) - z(VIX)
# 신용스프레드↑/VIX↑는 위험회피이므로 빼서 부호 통일
df_aligned["rai_raw"] = z_neg_lev - z_credit - z_vix

# lag-1: 모델에 투입할 때 t 시점에서 t-1 값 사용
df_aligned["rai_lag1"] = df_aligned["rai_raw"].shift(1)

print("\n  ✓ RAI 생성 완료")
print("    공식: z(neg_nfci_leverage) - z(BAA10Y_credit) - z(VIX)")
print("    표준화: 팽창 z-score (각 t에서 0..t-1 구간 통계, shift(1) 적용)")

# ── PCA 대안 ── (기본은 고정 ±1 가중)
# from sklearn.decomposition import PCA
# 아래 주석을 해제하면 PCA 가중을 사용합니다.
# comp_mat = pd.concat([z_neg_lev, -z_credit, -z_vix], axis=1).dropna()
# pca = PCA(n_components=1)
# pca_scores = pca.fit_transform(comp_mat)
# df_aligned.loc[comp_mat.index, "rai_raw"] = pca_scores[:, 0]
# print(f"    PCA 설명분산비: {pca.explained_variance_ratio_[0]:.3f}")

# ── 5. BEST_SPEC 컬럼 재생성 (없는 경우) ─────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 5단계: BEST_SPEC 컬럼 확인 및 생성")
print("=" * 65)

import re as _re
for feat in BEST_SPEC:
    if feat in df_aligned.columns:
        print(f"  ✓ '{feat}' 이미 존재")
        continue
    m = _re.match(r"^(.+)_lag(\d+)$", feat)
    if m:
        base_col, lag_n = m.group(1), int(m.group(2))
        if base_col in df_aligned.columns:
            df_aligned[feat] = df_aligned[base_col].shift(lag_n)
            print(f"  ✓ '{feat}' 생성 (shift({lag_n}) of '{base_col}')")
        elif base_col in df_csv.columns:
            df_aligned[feat] = df_csv[base_col].shift(lag_n).values
            print(f"  ✓ '{feat}' 생성 (CSV shift({lag_n}) of '{base_col}')")
        else:
            print(f"  ✗ '{feat}' 생성 실패 — '{base_col}' 없음")
    else:
        print(f"  ✗ '{feat}' — 규칙 파악 불가. BEST_SPEC을 확인하세요.")

valid_best = [c for c in BEST_SPEC if c in df_aligned.columns]
missing_best = [c for c in BEST_SPEC if c not in df_aligned.columns]
if missing_best:
    print(f"  ⚠ 제외된 BEST_SPEC: {missing_best}")
print(f"  → 실제 사용 BEST_SPEC: {valid_best}")

# ── 6. 단위근 검정 (ADF) ──────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 6단계: 새 변수 단위근 검정 (ADF)")
print("  귀무가설: 단위근 있음(비정상). p > 0.05 이면 차분 고려.")
print("=" * 65)

NEW_FEAT_MAP: dict[str, str] = {}   # {최종 컬럼명: 설명}


def run_adf(series: pd.Series, name: str) -> tuple[bool, float]:
    """ADF 검정 실행. (비정상 여부, p값) 반환."""
    clean = series.dropna()
    if len(clean) < 30:
        print(f"  {name}: 데이터 부족 ({len(clean)}행) — 건너뜀")
        return True, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = adfuller(clean, maxlag=12, autolag="AIC")
        pval = res[1]
        stat = res[0]
        status = "비정상 ⚠" if pval > 0.05 else "정상  ✓"
        print(f"  {name:<30}: ADF={stat:+.3f}, p={pval:.4f} → [{status}]")
        return pval > 0.05, pval
    except Exception as ex:
        print(f"  {name}: ADF 실패 ({ex})")
        return True, 1.0


for raw_col, model_col in [
    ("neg_nfci_leverage", "neg_nfci_leverage_lag1"),
    ("rai_raw",           "rai_lag1"),
]:
    nonstat, _ = run_adf(df_aligned[raw_col], raw_col)

    if nonstat:
        diff_col = f"d_{raw_col}"
        df_aligned[diff_col] = df_aligned[raw_col].diff()
        print(f"    → 1차 차분 적용 후 재검정:")
        nonstat2, _ = run_adf(df_aligned[diff_col], diff_col)

        # lag-1 재생성
        lag_col = f"{diff_col}_lag1"
        df_aligned[lag_col] = df_aligned[diff_col].shift(1)
        NEW_FEAT_MAP[lag_col] = f"1차 차분 후 lag-1  (원래: {raw_col})"
        print(f"    → 처리: 1차 차분 후 lag-1  →  '{lag_col}' 사용")
    else:
        NEW_FEAT_MAP[model_col] = f"수준값 lag-1  (정상 확인)"
        print(f"    → 처리: 수준값 그대로 lag-1  →  '{model_col}' 사용")

# 최종적으로 쓸 컬럼 이름 확정
nfci_feat = next(
    (k for k in NEW_FEAT_MAP if "nfci" in k or "neg_nfci" in k), None
)
rai_feat = next(
    (k for k in NEW_FEAT_MAP if "rai" in k), None
)
print(f"\n  최종 레버리지 컬럼: {nfci_feat}")
print(f"  최종 RAI 컬럼    : {rai_feat}")

# ── 7. 중복·다공선성 점검 (상관 + VIF) ────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 7단계: 중복 점검 — 상관계수 + VIF")
print("  (RAI ≈ financial_conditions 면 '같은 정보의 재포장' 가능성)")
print("=" * 65)

fc_lag_col = "financial_conditions_lag1"
if fc_lag_col not in df_aligned.columns:
    # CSV에서 lag 생성
    df_aligned[fc_lag_col] = df_aligned[FC_COL].shift(1)

check_cols = [c for c in [nfci_feat, rai_feat, fc_lag_col] if c and c in df_aligned.columns]
df_check = df_aligned[check_cols].dropna()

if len(df_check) > 0:
    # 상관계수
    corr_mat = df_check.corr()
    print("\n  [상관계수 행렬]")
    print(corr_mat.round(3).to_string())

    # VIF (다공선성 지수: 10 이상 = 강한 다공선성, 5~10 = 주의)
    print("\n  [VIF — 분산팽창지수]")
    print("  VIF > 10 : 강한 다공선성 → 함께 쓰지 말 것")
    print("  VIF 5~10 : 주의")
    print("  VIF < 5  : 양호")
    import statsmodels.api as sm
    X_vif = sm.add_constant(df_check.astype(float))
    vif_rows = []
    for i, col in enumerate(df_check.columns):
        try:
            vif_val = variance_inflation_factor(X_vif.values, i + 1)
            vif_rows.append({"변수": col, "VIF": round(vif_val, 2)})
        except Exception:
            vif_rows.append({"변수": col, "VIF": "계산 불가"})
    vif_df = pd.DataFrame(vif_rows)
    print(vif_df.to_string(index=False))

    # 판정
    if rai_feat and fc_lag_col in check_cols:
        rai_fc_corr = abs(corr_mat.loc[rai_feat, fc_lag_col]) if rai_feat in corr_mat.index else 0
        if rai_fc_corr > 0.7:
            print(f"\n  ⚠ [경고] RAI와 financial_conditions의 상관={rai_fc_corr:.3f} (높음)")
            print(f"    → RAI가 financial_conditions의 재포장일 가능성 있음")
            print(f"    → 사양 4 (financial_conditions 제거 + RAI) 결과를 주목하세요")
        elif rai_fc_corr > 0.4:
            print(f"\n  ⚠ [주의] RAI와 financial_conditions의 상관={rai_fc_corr:.3f} (중간)")
        else:
            print(f"\n  ✓ RAI와 financial_conditions의 상관={rai_fc_corr:.3f} (낮음 — 독립 정보)")

# ── 8. 전처리 + 홀드아웃 분리 ─────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 8단계: 전처리 + 홀드아웃 분리")
print("=" * 65)

all_feat_cols = list({
    c for c in valid_best + [nfci_feat, rai_feat, fc_lag_col]
    if c is not None and c in df_aligned.columns
})
keep_cols = ["date", TARGET_COL] + all_feat_cols
df_use = df_aligned[[c for c in keep_cols if c in df_aligned.columns]].copy()
df_use = df_use.apply(
    lambda col: pd.to_numeric(col, errors="coerce") if col.name != "date" else col
)

before = len(df_use)
df_clean = df_use.dropna().reset_index(drop=True)
after = len(df_clean)
print(f"  결측 제거: {before:,}행 → {after:,}행")
print(f"  기간: {df_clean['date'].min().date()} ~ {df_clean['date'].max().date()}")

if after < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 20:
    print(f"[오류] 데이터 부족. 최소 {MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 20}주 필요.")
    sys.exit(1)

df_wf   = df_clean.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_full = df_clean.reset_index(drop=True)

print(f"  워크포워드: {len(df_wf):,}주  ({df_wf['date'].iloc[0].date()} ~ {df_wf['date'].iloc[-1].date()})")
print(f"  홀드아웃  : {HOLDOUT_WEEKS}주  ({df_clean['date'].iloc[-HOLDOUT_WEEKS].date()} ~ {df_clean['date'].iloc[-1].date()})")

# ── 9. 사양 정의 ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 9단계: 검증 사양 정의")
print("=" * 65)


def remove_cols_containing(cols: list[str], keywords: list[str]) -> list[str]:
    """keywords 중 하나라도 포함된 컬럼을 제거한다."""
    return [c for c in cols if not any(kw.lower() in c.lower() for kw in keywords)]


# base에서 vix 제거 버전 (RAI에 VIX가 포함되어 이중계상 방지)
base_no_vix = remove_cols_containing(valid_best, ["vix"])
# base에서 vix + financial_conditions 제거 버전 (VIF 높을 때 비교용)
base_no_vix_fc = remove_cols_containing(valid_best, ["vix", "financial_cond"])

SPECS: dict[str, list[str]] = {}

# 사양 1: base (기준선)
SPECS["1_base"] = valid_best

# 사양 2: base + neg_nfci_leverage 단독
if nfci_feat and nfci_feat in df_clean.columns:
    SPECS["2_base+nfci"] = valid_best + [nfci_feat]

# 사양 3: (base-vix) + RAI
if rai_feat and rai_feat in df_clean.columns:
    SPECS["3_rai_novix"] = base_no_vix + [rai_feat]

# 사양 4: (base-vix-financial_cond) + RAI  (중복 의심 시 비교)
if rai_feat and rai_feat in df_clean.columns:
    SPECS["4_rai_nofc"] = base_no_vix_fc + [rai_feat]
    if not base_no_vix_fc:
        SPECS["4_rai_nofc"] = [rai_feat]  # 기저변수 없이 RAI만

for name, cols in SPECS.items():
    print(f"  {name:<16}: {cols}")

# ── 10. 헬퍼 함수 ─────────────────────────────────────────────────────────────────


def select_arima_order(
    y: np.ndarray,
    X: np.ndarray | None = None,
    max_p: int = MAX_P,
    max_q: int = MAX_Q,
) -> tuple[int, int, int]:
    """
    AIC 기준으로 ARIMA(p, 0, q) 차수를 선택한다.
    d=0 고정: 타깃(로그수익률)이 이미 정상성이기 때문이다.
    """
    best_aic   = np.inf
    best_order = (1, 0, 1)

    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            X_df = pd.DataFrame(X) if X is not None else None
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(
                    y, exog=X_df,
                    order=(p, 0, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            if m.aic < best_aic:
                best_aic   = m.aic
                best_order = (p, 0, q)
        except Exception:
            pass

    return best_order


def compute_metrics(
    actuals:    list[float],
    preds:      list[float],
    benchmarks: list[float],
) -> dict:
    """
    표본외 R²(R2_oos)와 방향 적중률을 계산한다.
      R2_oos = 1 - SSE_모델 / SSE_벤치마크
        양수 → 모델이 역사적 평균보다 나음
        음수 → 역사적 평균보다 못함 (흔한 결과 — 솔직히 받아들일 것)
    """
    a = np.array(actuals,    dtype=float)
    p = np.array(preds,      dtype=float)
    b = np.array(benchmarks, dtype=float)

    sse_model = float(np.sum((a - p) ** 2))
    sse_bench = float(np.sum((a - b) ** 2))
    r2_oos    = (1 - sse_model / sse_bench) if sse_bench > 1e-12 else np.nan

    dir_acc   = float(np.mean(np.sign(a) == np.sign(p)))
    bench_dir = float(np.mean(np.sign(a) == np.sign(b)))

    return {
        "r2_oos": r2_oos, "dir_acc": dir_acc, "bench_dir": bench_dir,
        "n_steps": len(a),
    }


def run_walk_forward(
    df_source:   pd.DataFrame,
    target_col:  str,
    specs:       dict[str, list[str]],
    orders:      dict[str, tuple],
    min_train:   int,
    refit_every: int,
    window:      str,
    start_t:     int | None = None,
    verbose:     bool = True,
) -> dict[str, dict]:
    """
    워크포워드 1주 앞 예측 엔진.

    [데이터 누수 방지 원칙]
    ① StandardScaler 는 항상 훈련 구간에만 fit 한다.
    ② ARIMA 차수(p, q)는 고정, refit_every 마다 파라미터만 재추정한다.
    ③ 벤치마크 = 훈련 구간의 타깃 누적 평균 (역사적 평균).

    예측 구조:
      - t 행의 exog(x_t) 로 t 행의 y_t 를 예측한다.
      - lag-1 변수이므로 x_t[lag1] = 원시값_{t-1} : 시점 t-1 에 알려진 값.
      - 팽창 z-score RAI 역시 t-1 통계 → t-1 값 → lag1 = t-2 정보.
    """
    if start_t is None:
        start_t = min_train

    n       = len(df_source)
    records = {name: {"actual": [], "pred": [], "bench": []} for name in specs}
    fitted_: dict[str, object]                  = {}
    scaler_: dict[str, StandardScaler | None]   = {}

    total_steps = n - start_t
    next_print  = max(50, total_steps // 10)

    for step_idx, t in enumerate(range(start_t, n)):
        if window == "expanding":
            s = slice(0, t)
        else:
            s = slice(max(0, t - min_train), t)

        y_train  = df_source[target_col].iloc[s].values.astype(float)
        y_actual = float(df_source[target_col].iloc[t])
        y_bench  = float(np.mean(y_train))

        need_refit = (step_idx % refit_every == 0)

        for spec_name, feat_cols in specs.items():
            try:
                if need_refit or spec_name not in fitted_:
                    if feat_cols:
                        scaler = StandardScaler()
                        X_tr   = scaler.fit_transform(
                            df_source[feat_cols].iloc[s].values.astype(float)
                        )
                        scaler_[spec_name] = scaler
                    else:
                        X_tr = None
                        scaler_[spec_name] = None

                    X_tr_df = pd.DataFrame(X_tr, columns=feat_cols) if feat_cols else None
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted_[spec_name] = SARIMAX(
                            y_train,
                            exog=X_tr_df,
                            order=orders[spec_name],
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)

                if feat_cols and scaler_[spec_name] is not None:
                    x_test = scaler_[spec_name].transform(
                        df_source[feat_cols].iloc[[t]].values.astype(float)
                    )
                    x_test_df = pd.DataFrame(x_test, columns=feat_cols)
                else:
                    x_test_df = None

                y_pred = float(
                    fitted_[spec_name].forecast(steps=1, exog=x_test_df).iloc[0]
                )
                if not np.isfinite(y_pred):
                    y_pred = y_bench

            except Exception:
                y_pred = y_bench

            records[spec_name]["actual"].append(y_actual)
            records[spec_name]["pred"].append(y_pred)
            records[spec_name]["bench"].append(y_bench)

        if verbose and (step_idx + 1) >= next_print:
            pct     = (step_idx + 1) / total_steps * 100
            elapsed = time.time() - _wf_start
            print(f"    {step_idx+1:>4}/{total_steps} 스텝 ({pct:.0f}%) — {elapsed:.0f}초 경과")
            next_print += max(50, total_steps // 10)

    return records


# ── 11. ARIMA 차수 선택 ────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 10단계: ARIMA(p,0,q) 차수 선택 — 첫 훈련 윈도우에서 한 번")
print("  (이후 refit_every 스텝마다 파라미터만 재추정, 차수는 고정)")
print("=" * 65)

y_init = df_wf[TARGET_COL].values[:MIN_TRAIN_WEEKS].astype(float)
orders_: dict[str, tuple] = {}
t0 = time.time()

for spec_name, feat_cols in SPECS.items():
    if feat_cols:
        sc = StandardScaler()
        X_init = sc.fit_transform(df_wf[feat_cols].values[:MIN_TRAIN_WEEKS].astype(float))
    else:
        X_init = None
    order = select_arima_order(y_init, X_init)
    orders_[spec_name] = order
    print(f"  {spec_name:<18}: ARIMA{order}")

print(f"  차수 선택 소요: {time.time()-t0:.1f}초")

# ── 12. 워크포워드 실행 ────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 11단계: 워크포워드 실행 (홀드아웃 제외 구간)")
print(f"  윈도우: {WINDOW} | 재적합: {REFIT_EVERY}스텝 | 최소학습: {MIN_TRAIN_WEEKS}주")
total_wf = len(df_wf) - MIN_TRAIN_WEEKS
print(f"  예측 스텝: {total_wf}주 × {len(SPECS)}사양")
if total_wf > 300:
    print(f"  [안내] 느리면 REFIT_EVERY를 8~12로 높이세요.")
print("=" * 65)

_wf_start = time.time()
records_wf = run_walk_forward(
    df_source   = df_wf,
    target_col  = TARGET_COL,
    specs       = SPECS,
    orders      = orders_,
    min_train   = MIN_TRAIN_WEEKS,
    refit_every = REFIT_EVERY,
    window      = WINDOW,
)
print(f"\n  워크포워드 완료: {time.time()-_wf_start:.1f}초")

# ── 13. 결과 집계 + 사양 선택 ─────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 12단계: 결과 비교 + 사양 선택")
print("=" * 65)

rows = []

first_rec = list(records_wf.values())[0]
bm_m = compute_metrics(first_rec["actual"], first_rec["bench"], first_rec["bench"])
rows.append({
    "사양":       "0_benchmark",
    "변수 구성":  "역사적 누적 평균",
    "R2_oos":    0.0,
    "방향적중률": round(bm_m["dir_acc"], 3),
    "n_steps":   bm_m["n_steps"],
})

for spec_name, rec in records_wf.items():
    m = compute_metrics(rec["actual"], rec["pred"], rec["bench"])
    rows.append({
        "사양":       spec_name,
        "변수 구성":  " | ".join(SPECS[spec_name]) if SPECS[spec_name] else "(없음)",
        "R2_oos":    round(m["r2_oos"], 4),
        "방향적중률": round(m["dir_acc"], 3),
        "n_steps":   m["n_steps"],
    })

results_df = (
    pd.DataFrame(rows)
    .sort_values("R2_oos", ascending=False)
    .reset_index(drop=True)
)

print(f"\n  [참고] R2_oos > 0 = 벤치마크 이김,  R2_oos < 0 = 짐\n")
print(results_df[["사양", "R2_oos", "방향적중률", "n_steps"]].to_string(index=False))

# 새 변수 기여도
print("\n  [새 변수 기여도]")
base_r2 = next((r["R2_oos"] for r in rows if r["사양"] == "1_base"), np.nan)
for spec_name in ["2_base+nfci", "3_rai_novix", "4_rai_nofc"]:
    r2 = next((r["R2_oos"] for r in rows if r["사양"] == spec_name), None)
    if r2 is None or np.isnan(base_r2):
        continue
    delta = r2 - base_r2
    label = "개선" if delta > 0.005 else ("미미" if delta > 0 else ("거의 동일" if delta > -0.005 else "악화"))
    print(f"  {spec_name:<18}: base 대비 ΔR2={delta:+.4f}  ({label})")

# 최고 사양 선택
model_rows = [r for r in rows if r["사양"] != "0_benchmark"]
best_row   = max(model_rows, key=lambda x: x["R2_oos"])
best_name  = best_row["사양"]
best_cols  = SPECS[best_name]
print(f"\n  → 선택 사양: '{best_name}'  (R2_oos={best_row['R2_oos']:.4f})")

# ── 14. 최종 홀드아웃 검증 ────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 13단계: 최종 홀드아웃 검증 — 선택 사양 1회 적용")
print(f"  사양: '{best_name}'  |  홀드아웃: {HOLDOUT_WEEKS}주")
print("  ※ 이 숫자가 정직한 표본 외 성적입니다.")
print("=" * 65)

_wf_start = time.time()
records_hold = run_walk_forward(
    df_source   = df_full,
    target_col  = TARGET_COL,
    specs       = {best_name: best_cols},
    orders      = {best_name: orders_[best_name]},
    min_train   = MIN_TRAIN_WEEKS,
    refit_every = REFIT_EVERY,
    window      = WINDOW,
    start_t     = len(df_wf),
)
hold_m = compute_metrics(
    records_hold[best_name]["actual"],
    records_hold[best_name]["pred"],
    records_hold[best_name]["bench"],
)

print(f"\n  홀드아웃 R2_oos   : {hold_m['r2_oos']:+.4f}  "
      f"({'벤치마크보다 나음 ✓' if hold_m['r2_oos'] > 0 else '벤치마크보다 못함 ✗'})")
print(f"  방향 적중률       : {hold_m['dir_acc']:.3f}  "
      f"(벤치마크: {hold_m['bench_dir']:.3f})")
print(f"  스텝 수           : {hold_m['n_steps']}주  (소요: {time.time()-_wf_start:.1f}초)")

# 결과 저장
hold_df = pd.DataFrame({
    "date":      df_full["date"].iloc[len(df_wf):len(df_wf)+hold_m["n_steps"]].values,
    "actual":    records_hold[best_name]["actual"],
    "predicted": records_hold[best_name]["pred"],
    "benchmark": records_hold[best_name]["bench"],
})
out_csv = OUTPUT_DIR / "rai_holdout_predictions.csv"
hold_df.to_csv(out_csv, index=False)
print(f"\n  저장: {out_csv}")

# ── 15. 종합 요약 ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

print(f"\n  [워크포워드 결과표 (상위 → 하위)]")
print(results_df[["사양", "변수 구성", "R2_oos", "방향적중률"]].to_string(index=False))

print(f"\n  [선택 사양: '{best_name}']")
print(f"  · 워크포워드 R2_oos = {best_row['R2_oos']:.4f}")
print(f"  · 홀드아웃  R2_oos = {hold_m['r2_oos']:+.4f}")
print(f"  · 홀드아웃 방향적중률 = {hold_m['dir_acc']:.3f}")

print(f"""
  [중요 주의 사항]
  ① VIX·HY 신용스프레드는 주가와 동시(contemporaneous) 관계가 강합니다.
     lag-1 적용 시 선행 성분만 남아 예측력이 제한될 수 있습니다.
  ② RAI가 financial_conditions와 강하게 겹치면 '새 정보'가 아니라
     같은 정보의 재포장일 수 있습니다. 상관·VIF 결과를 먼저 확인하세요.
  ③ 벤치마크(역사적 평균)를 이기지 못해도 유효한 발견입니다.
     → 이 경우 "이 변수들은 주간 수익률 예측에 새 정보를 추가하지 않는다"
       는 결론도 충분히 의미 있습니다.
  ④ 신용스프레드로 사용한 BAA10Y는 투자적격(BBB) 등급 기준이며,
     고수익(HY) 스프레드(BAMLH0A0HYM2)보다 변동성이 작습니다.
     FRED API 접근 제한으로 대체했음을 분석 보고서에 명시하세요.
""")

if hold_m["r2_oos"] > 0:
    print(f"  ★ 결론: 선택 사양이 홀드아웃에서 벤치마크를 이겼습니다.")
    print(f"    단, 18개월 1회 검증이므로 해석에 주의하세요.")
else:
    print(f"  ★ 결론: 선택 사양이 홀드아웃에서 벤치마크를 이기지 못했습니다.")
    print(f"    → 역사적 평균이 더 나은 예측이었습니다. 유효한 발견입니다.")
    print(f"    → 다음 단계 고려:")
    print(f"       · NFCILEVERAGE 대신 NFCI 전체 지수 또는 크레딧 서브인덱스 시도")
    print(f"       · 주간이 아닌 월간 빈도로 RAI 효과 재검증")
    print(f"       · 이진 방향(상승/하락) 분류 문제로 전환")

print("\n" + "=" * 65)
print("분석 완료.")
print("=" * 65)
