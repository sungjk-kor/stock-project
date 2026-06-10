# =============================================================================
# walkforward_5specs.py  —  5개 사양 워크포워드 검증
#
# 테스트 사양 (이 5개만):
#   1) anchor         = [fc_l1]
#   2) anchor_risk    = [fc_l1, vix_l1]
#   3) anchor_credit  = [fc_l1, dHY_l1]
#   4) conditions_mom = [dFC_l1, vix_l1]
#   5) composite      = [RAI_l1]
#   + 벤치마크: 역사적 누적 평균
#
# 변수 정의:
#   fc_l1   = financial_conditions lag-1
#   vix_l1  = vix lag-1
#   dHY_l1  = (HY_spread - HY_spread.shift(4)) lag-1   ← 4주 크레딧 모멘텀
#   dFC_l1  = (financial_conditions - fc.shift(4)) lag-1 ← 4주 금융여건 모멘텀
#   RAI_l1  = [z(-NFCILEVERAGE) - z(HY) - z(VIX)] lag-1  ← 합성 위험선호 지수
#
# 외부 데이터(FRED):
#   NFCILEVERAGE  : 주간 레버리지 서브인덱스
#   BAA10Y        : Moody's Baa 신용스프레드 (일별 → 주간)
#                   ※ BAMLH0A0HYM2(HY OAS)가 FRED API에서 2023~만 제공되어 대체
#
# 필요 라이브러리:
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

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH   = "data/processed/final_dataset.csv"
TARGET_COL = "sp500_log_return"

HOLDOUT_WEEKS   = 78
MIN_TRAIN_WEEKS = 260
REFIT_EVERY     = 4
WINDOW          = "expanding"   # "expanding" 또는 "rolling"

MAX_P = 2   # ARIMA 탐색 범위
MAX_Q = 2

OUTPUT_DIR = Path(__file__).parent

# R²_oos 판정 기준
R2_THRESHOLD_MEANINGFUL = 0.02   # 이 이상이면 의미 있는 개선
R2_THRESHOLD_MARGINAL   = 0.005  # 이 이상이면 미미한 가능성 (노이즈 의심)

# ── 1. FRED 데이터 수신 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 5개 사양 워크포워드 검증")
print("=" * 65)
print("\n■ 1단계: FRED 데이터 조회")

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("[오류] .env 에 FRED_API_KEY 가 없습니다.")
    sys.exit(1)

try:
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
except ImportError:
    print("[오류] fredapi 없음. pip install fredapi 를 실행하세요.")
    sys.exit(1)

# NFCILEVERAGE — Chicago Fed 레버리지 서브인덱스 (주간, 이미 표준화)
try:
    nfci_raw = fred.get_series("NFCILEVERAGE", observation_start="1999-01-01")
    nfci_raw.index = pd.to_datetime(nfci_raw.index)
    print(f"  ✓ NFCILEVERAGE: {len(nfci_raw)}행 "
          f"({nfci_raw.index[0].date()} ~ {nfci_raw.index[-1].date()})")
except Exception as e:
    print(f"  ✗ NFCILEVERAGE 조회 실패: {e}")
    sys.exit(1)

time.sleep(0.5)

# BAA10Y — Moody's Baa 신용스프레드 (일별 → 주간 변환)
# ※ ICE BofA HY OAS(BAMLH0A0HYM2)는 FRED API가 2023~만 반환하므로 대체
try:
    hy_raw = fred.get_series("BAA10Y", observation_start="1999-01-01")
    hy_raw.index = pd.to_datetime(hy_raw.index)
    print(f"  ✓ BAA10Y (HY 대체): {len(hy_raw)}행 "
          f"({hy_raw.index[0].date()} ~ {hy_raw.index[-1].date()}), "
          f"결측={hy_raw.isna().sum()}")
    print("    (Moody's BBB 신용스프레드 — HY OAS와 방향성 동일, 변동폭은 작음)")
except Exception as e:
    print(f"  ✗ BAA10Y 조회 실패: {e}")
    sys.exit(1)

# ── 2. 기존 CSV 로드 ──────────────────────────────────────────────────────────

print("\n■ 2단계: CSV 로드")

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] 파일 없음: {csv_path.resolve()}")
    sys.exit(1)

df_csv = pd.read_csv(csv_path, parse_dates=["date"])
df_csv = df_csv.sort_values("date").reset_index(drop=True)
print(f"  ✓ {len(df_csv):,}행, {df_csv['date'].min().date()} ~ {df_csv['date'].max().date()}")

for col in [TARGET_COL, "vix", "financial_conditions"]:
    if col not in df_csv.columns:
        print(f"  [오류] 컬럼 없음: '{col}'")
        sys.exit(1)

# ── 3. 주간(금요일) 정렬 ─────────────────────────────────────────────────────

print("\n■ 3단계: 주간 정렬 (금요일 기준)")

# 기준 인덱스: CSV 의 날짜 (이미 주간 금요일)
df_work = df_csv[["date", TARGET_COL, "vix", "financial_conditions"]].copy()
df_work["date"] = pd.to_datetime(df_work["date"])
df_work = df_work.set_index("date")

# NFCILEVERAGE: 이미 주간 금요일 → W-FRI 리샘플로 정확히 맞춤
nfci_w = nfci_raw.resample("W-FRI").last().ffill()
nfci_w.name = "nfcileverage"

# BAA10Y: 일별 → 주간 금요일 (마지막 영업일 값, ffill)
hy_w = hy_raw.ffill().resample("W-FRI").last().ffill()
hy_w.name = "hy_spread"

# CSV 기준 날짜에 join
df_work["nfcileverage"] = nfci_w.reindex(df_work.index, method="ffill")
df_work["hy_spread"]    = hy_w.reindex(df_work.index, method="ffill")
df_work = df_work.reset_index()

n_valid = df_work[["nfcileverage", "hy_spread"]].notna().all(axis=1).sum()
print(f"  JOIN 후 유효 행: {n_valid:,} / 전체 {len(df_work):,}행")

# ── 4. 새 변수 생성 (누수 없음) ───────────────────────────────────────────────

print("\n■ 4단계: 새 변수 생성")
print("  (시점 t는 반드시 t-1 이전 정보만 사용 — shift로 보장)")

# ── fc_l1: financial_conditions 1주 lag ──────────────────────────────────────
df_work["fc_l1"] = df_work["financial_conditions"].shift(1)
print("  ✓ fc_l1  = financial_conditions lag-1")

# ── vix_l1: VIX 1주 lag ──────────────────────────────────────────────────────
df_work["vix_l1"] = df_work["vix"].shift(1)
print("  ✓ vix_l1 = vix lag-1")

# ── dHY_l1: 4주 HY 스프레드 모멘텀, 다시 1주 lag ────────────────────────────
# dHY = 현재 스프레드 - 4주 전 스프레드 (크레딧 악화/개선 속도)
# 양수면 스프레드 확대(위험회피 방향 강화), 음수면 축소(위험선호 방향)
df_work["dHY"]    = df_work["hy_spread"] - df_work["hy_spread"].shift(4)
df_work["dHY_l1"] = df_work["dHY"].shift(1)
print("  ✓ dHY_l1 = (hy_spread - hy_spread.shift(4)) lag-1   [4주 크레딧 모멘텀]")

# ── dFC_l1: 4주 금융여건 모멘텀, 다시 1주 lag ───────────────────────────────
# dFC = 현재 금융여건 - 4주 전 (긴축 방향으로 얼마나 움직였나)
df_work["dFC"]    = df_work["financial_conditions"] - df_work["financial_conditions"].shift(4)
df_work["dFC_l1"] = df_work["dFC"].shift(1)
print("  ✓ dFC_l1 = (financial_conditions - fc.shift(4)) lag-1 [4주 금융여건 모멘텀]")

# ── RAI_l1: 합성 위험선호 지수 ────────────────────────────────────────────────
# RAI = z(-NFCILEVERAGE) - z(hy_spread) - z(vix)
# 핵심: 각 성분을 먼저 표준화(z)한 뒤 합성 — 스케일 큰 성분 지배 방지
# z의 통계(평균·표준편차)는 "팽창 윈도우 + shift(1)"로 계산 → 누수 없음
# · expanding().mean().shift(1) = 시점 t에서 0..t-1 구간 평균
# · expanding().std().shift(1)  = 시점 t에서 0..t-1 구간 표준편차
# → expanding window 사용 시 이 값이 정확히 해당 스텝의 훈련 데이터 통계와 일치

def expanding_zscore(s: pd.Series) -> pd.Series:
    """
    시점 t에서 0..t-1 구간의 통계로 z-score를 계산한다.
    shift(1)로 현재 시점을 통계 계산에서 제외 → 데이터 누수 없음.
    """
    exp_mean = s.expanding().mean().shift(1)          # 0..t-1 평균
    exp_std  = s.expanding().std(ddof=1).shift(1)     # 0..t-1 표준편차
    safe_std = exp_std.where(exp_std > 1e-8, np.nan)  # 0 나눗셈 방지
    return (s - exp_mean) / safe_std

neg_nfci = -1 * df_work["nfcileverage"]   # 부호 반전: 값↑ = 레버리지 확대 = 위험선호

z_lev = expanding_zscore(neg_nfci)
z_hy  = expanding_zscore(df_work["hy_spread"])
z_vix = expanding_zscore(df_work["vix"])

df_work["rai_raw"] = z_lev - z_hy - z_vix   # RAI = z(레버리지) - z(HY) - z(VIX)
df_work["RAI_l1"]  = df_work["rai_raw"].shift(1)   # 1주 lag
print("  ✓ RAI_l1 = [z(-NFCILEVERAGE) - z(hy_spread) - z(vix)] lag-1  [합성 위험선호]")
print("    ※ RAI에 vix·HY 이미 포함 → composite 사양에는 단독 vix/HY 넣지 않음")

# ── 5. ADF 단위근 검정 ────────────────────────────────────────────────────────

print("\n■ 5단계: 단위근 검정 (ADF)")
print("  귀무가설: 단위근 있음(비정상). p > 0.05 면 차분 고려.")

NEW_VARS = {
    "fc_l1":   "fc_l1",
    "vix_l1":  "vix_l1",
    "dHY_l1":  "dHY_l1",
    "dFC_l1":  "dFC_l1",
    "RAI_l1":  "RAI_l1",
}
FINAL_COLS: dict[str, str] = {}   # {변수명: 실제 사용할 컬럼}


def run_adf(series: pd.Series, name: str) -> tuple[bool, float]:
    clean = series.dropna()
    if len(clean) < 30:
        print(f"  {name}: 데이터 부족 ({len(clean)}행)")
        return True, 1.0
    try:
        res  = adfuller(clean, maxlag=12, autolag="AIC")
        pval = res[1]
        stat = res[0]
        tag  = "비정상 ⚠" if pval > 0.05 else "정상  ✓"
        print(f"  {name:<10}: ADF={stat:+.3f}, p={pval:.4f} → [{tag}]")
        return pval > 0.05, pval
    except Exception as ex:
        print(f"  {name}: ADF 오류 ({ex})")
        return True, 1.0


for var_name, col_name in NEW_VARS.items():
    if col_name not in df_work.columns:
        continue
    nonstat, _ = run_adf(df_work[col_name], var_name)

    if nonstat:
        # 1차 차분
        diff_col = f"d_{col_name}"
        df_work[diff_col] = df_work[col_name].diff()
        print(f"    → 1차 차분 후 재검정:")
        nonstat2, _ = run_adf(df_work[diff_col], f"d_{var_name}")
        use_col = diff_col
        print(f"    → 처리: 1차 차분 사용 → '{use_col}'")
    else:
        use_col = col_name

    FINAL_COLS[var_name] = use_col

print(f"\n  최종 컬럼 매핑: {FINAL_COLS}")

# ── 6. 상관·VIF 중복 점검 ─────────────────────────────────────────────────────

print("\n■ 6단계: 중복 점검 (상관 + VIF)")
print("  RAI / dHY 가 financial_conditions 와 많이 겹치는지 확인")

check_cols = [
    FINAL_COLS.get("fc_l1",  "fc_l1"),
    FINAL_COLS.get("dHY_l1", "dHY_l1"),
    FINAL_COLS.get("RAI_l1", "RAI_l1"),
]
check_cols = [c for c in check_cols if c in df_work.columns]
df_check = df_work[check_cols].dropna()

if len(df_check) > 10:
    print("\n  [상관계수]")
    print(df_check.corr().round(3).to_string())

    print("\n  [VIF — 다공선성 지수]  VIF>10: 강한 중복, 5~10: 주의, <5: 양호")
    import statsmodels.api as sm
    X_vif = sm.add_constant(df_check.astype(float))
    vif_rows = []
    for i, col in enumerate(df_check.columns):
        try:
            v = variance_inflation_factor(X_vif.values, i + 1)
            vif_rows.append({"변수": col, "VIF": round(v, 2)})
        except Exception:
            vif_rows.append({"변수": col, "VIF": "계산 불가"})
    print(pd.DataFrame(vif_rows).to_string(index=False))

    # 경고 출력
    fc_col = FINAL_COLS.get("fc_l1", "fc_l1")
    for cand in [FINAL_COLS.get("RAI_l1"), FINAL_COLS.get("dHY_l1")]:
        if cand and cand in df_check.columns and fc_col in df_check.columns:
            corr_val = abs(df_check[cand].corr(df_check[fc_col]))
            label = cand.split("_")[0].upper()
            if corr_val > 0.7:
                print(f"\n  ⚠ [{label}] vs fc_l1 상관={corr_val:.3f} → 강한 중복. 재포장 가능성.")
            elif corr_val > 0.4:
                print(f"\n  ⚠ [{label}] vs fc_l1 상관={corr_val:.3f} → 중간 정도 겹침.")
            else:
                print(f"\n  ✓ [{label}] vs fc_l1 상관={corr_val:.3f} → 독립 정보 가능성 있음.")

# ── 7. 사양 정의 ─────────────────────────────────────────────────────────────

print("\n■ 7단계: 사양 정의 (5개 고정)")

SPECS: dict[str, list[str]] = {
    "anchor":         [FINAL_COLS["fc_l1"]],
    "anchor_risk":    [FINAL_COLS["fc_l1"],  FINAL_COLS["vix_l1"]],
    "anchor_credit":  [FINAL_COLS["fc_l1"],  FINAL_COLS["dHY_l1"]],
    "conditions_mom": [FINAL_COLS["dFC_l1"], FINAL_COLS["vix_l1"]],
    "composite":      [FINAL_COLS["RAI_l1"]],
}

# 모든 컬럼이 실제로 존재하는지 확인
for name, cols in list(SPECS.items()):
    missing = [c for c in cols if c not in df_work.columns]
    if missing:
        print(f"  ⚠ '{name}' 사양의 컬럼 없음: {missing} → 제외")
        del SPECS[name]
    else:
        print(f"  {name:<16}: {cols}")

# ── 8. 전처리 + 홀드아웃 분리 ─────────────────────────────────────────────────

print("\n■ 8단계: 전처리 + 홀드아웃 분리")

all_feat = list({c for cols in SPECS.values() for c in cols})
keep     = ["date", TARGET_COL] + all_feat
df_use   = df_work[[c for c in keep if c in df_work.columns]].copy()
df_use   = df_use.apply(
    lambda col: pd.to_numeric(col, errors="coerce") if col.name != "date" else col
)

before   = len(df_use)
df_clean = df_use.dropna().reset_index(drop=True)
after    = len(df_clean)
print(f"  결측 제거: {before:,} → {after:,}행")
print(f"  기간: {df_clean['date'].min().date()} ~ {df_clean['date'].max().date()}")

if after < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 20:
    print(f"[오류] 데이터 부족: 최소 {MIN_TRAIN_WEEKS+HOLDOUT_WEEKS+20}주 필요")
    sys.exit(1)

df_wf   = df_clean.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_full = df_clean.reset_index(drop=True)

print(f"  워크포워드: {len(df_wf):,}주  "
      f"({df_wf['date'].iloc[0].date()} ~ {df_wf['date'].iloc[-1].date()})")
print(f"  홀드아웃  : {HOLDOUT_WEEKS}주  "
      f"({df_clean['date'].iloc[-HOLDOUT_WEEKS].date()} ~ "
      f"{df_clean['date'].iloc[-1].date()})")

# ── 9. 헬퍼 함수 ─────────────────────────────────────────────────────────────


def select_arima_order(y, X=None):
    """AIC 기준으로 ARIMA(p,0,q) 차수 선택. d=0 고정(로그수익률 = 이미 정상)."""
    best_aic, best_ord = np.inf, (1, 0, 1)
    for p, q in itertools.product(range(MAX_P + 1), range(MAX_Q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            Xdf = pd.DataFrame(X) if X is not None else None
            m = SARIMAX(y, exog=Xdf, order=(p, 0, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(disp=False)
            if m.aic < best_aic:
                best_aic, best_ord = m.aic, (p, 0, q)
        except Exception:
            pass
    return best_ord


def compute_metrics(actuals, preds, benchmarks):
    """
    R²_oos = 1 - SSE_모델 / SSE_벤치마크
      양수: 역사적 평균보다 나음 / 음수: 역사적 평균보다 못함
    방향적중률: 예측 부호(상승/하락)가 실제와 맞은 비율
    """
    a, p, b = map(lambda x: np.array(x, float), [actuals, preds, benchmarks])
    sse_m  = np.sum((a - p) ** 2)
    sse_b  = np.sum((a - b) ** 2)
    r2     = float(1 - sse_m / sse_b) if sse_b > 1e-12 else np.nan
    dir_a  = float(np.mean(np.sign(a) == np.sign(p)))
    dir_b  = float(np.mean(np.sign(a) == np.sign(b)))
    return {"r2_oos": r2, "dir_acc": dir_a, "bench_dir": dir_b, "n": len(a)}


def run_walk_forward(df_src, target, specs, orders,
                     min_train, refit_every, window, start_t=None):
    """
    워크포워드 1주 앞 예측 엔진.

    누수 방지 원칙:
      ① StandardScaler 를 훈련 슬라이스에만 fit.
      ② ARIMA 차수는 고정, refit_every 마다 파라미터만 재추정.
      ③ 벤치마크 = 훈련 슬라이스의 타깃 평균.
    """
    if start_t is None:
        start_t = min_train
    n       = len(df_src)
    rec     = {nm: {"actual": [], "pred": [], "bench": []} for nm in specs}
    fitted_ : dict = {}
    scaler_ : dict = {}
    total   = n - start_t
    nxt     = max(50, total // 10)

    for idx, t in enumerate(range(start_t, n)):
        s = slice(0, t) if window == "expanding" else slice(max(0, t - min_train), t)

        y_tr   = df_src[target].iloc[s].values.astype(float)
        y_act  = float(df_src[target].iloc[t])
        y_bm   = float(np.mean(y_tr))   # 벤치마크: 누적 평균
        refit  = (idx % refit_every == 0)

        for nm, cols in specs.items():
            try:
                if refit or nm not in fitted_:
                    if cols:
                        sc   = StandardScaler()
                        Xtr  = sc.fit_transform(df_src[cols].iloc[s].values.astype(float))
                        scaler_[nm] = sc
                    else:
                        Xtr = None; scaler_[nm] = None

                    Xdf = pd.DataFrame(Xtr, columns=cols) if cols else None
                    fitted_[nm] = SARIMAX(
                        y_tr, exog=Xdf, order=orders[nm],
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False)

                if cols and scaler_[nm]:
                    xt = scaler_[nm].transform(df_src[cols].iloc[[t]].values.astype(float))
                    xt = pd.DataFrame(xt, columns=cols)
                else:
                    xt = None

                yp = float(fitted_[nm].forecast(steps=1, exog=xt).iloc[0])
                if not np.isfinite(yp):
                    yp = y_bm
            except Exception:
                yp = y_bm

            rec[nm]["actual"].append(y_act)
            rec[nm]["pred"].append(yp)
            rec[nm]["bench"].append(y_bm)

        if (idx + 1) >= nxt:
            pct = (idx + 1) / total * 100
            print(f"    {idx+1:>4}/{total} ({pct:.0f}%) — {time.time()-_t0:.0f}초")
            nxt += max(50, total // 10)

    return rec


# ── 10. ARIMA 차수 선택 ────────────────────────────────────────────────────────

print("\n■ 9단계: ARIMA(p,0,q) 차수 선택 — 첫 훈련 윈도우 AIC 기준, 이후 고정")

y_init  = df_wf[TARGET_COL].values[:MIN_TRAIN_WEEKS].astype(float)
orders_ : dict[str, tuple] = {}
t0_ord  = time.time()

for nm, cols in SPECS.items():
    Xinit = None
    if cols:
        sc = StandardScaler()
        Xinit = sc.fit_transform(df_wf[cols].values[:MIN_TRAIN_WEEKS].astype(float))
    orders_[nm] = select_arima_order(y_init, Xinit)
    print(f"  {nm:<18}: ARIMA{orders_[nm]}")

print(f"  소요: {time.time()-t0_ord:.1f}초")

# ── 11. 워크포워드 실행 ────────────────────────────────────────────────────────

print(f"\n■ 10단계: 워크포워드 실행 (홀드아웃 제외 구간)")
print(f"  윈도우={WINDOW} | 재적합={REFIT_EVERY}스텝 | 최소학습={MIN_TRAIN_WEEKS}주")
total_steps = len(df_wf) - MIN_TRAIN_WEEKS
print(f"  예측 스텝: {total_steps}주 × {len(SPECS)}사양")
if total_steps > 300:
    print(f"  [안내] 느리면 REFIT_EVERY를 8~12로 높이세요.")

_t0 = time.time()
records_wf = run_walk_forward(
    df_wf, TARGET_COL, SPECS, orders_,
    MIN_TRAIN_WEEKS, REFIT_EVERY, WINDOW,
)
print(f"\n  워크포워드 완료: {time.time()-_t0:.1f}초")

# ── 12. 결과 집계 ─────────────────────────────────────────────────────────────

print("\n■ 11단계: 결과 집계 + 사양 선택")

rows = []

# 벤치마크 행 추가
first_rec = list(records_wf.values())[0]
bm = compute_metrics(first_rec["actual"], first_rec["bench"], first_rec["bench"])
rows.append({"사양": "benchmark", "R2_oos": 0.0,
             "방향적중률": round(bm["bench_dir"], 3), "n": bm["n"]})

for nm, rec in records_wf.items():
    m = compute_metrics(rec["actual"], rec["pred"], rec["bench"])
    rows.append({"사양": nm, "R2_oos": round(m["r2_oos"], 4),
                 "방향적중률": round(m["dir_acc"], 3), "n": m["n"]})

results_df = (
    pd.DataFrame(rows)
    .sort_values("R2_oos", ascending=False)
    .reset_index(drop=True)
)

print(f"\n  R2_oos > 0 = 벤치마크(역사적 평균) 이김  /  R2_oos < 0 = 짐\n")
print(results_df.to_string(index=False))

# ── 판정 ──────────────────────────────────────────────────────────────────────
print("\n  [판정 기준]")
print(f"  R²_oos > {R2_THRESHOLD_MEANINGFUL:.3f} : 의미 있는 개선")
print(f"  R²_oos {R2_THRESHOLD_MARGINAL:.3f}~{R2_THRESHOLD_MEANINGFUL:.3f}: 미미 (노이즈 의심)")
print(f"  R²_oos < {R2_THRESHOLD_MARGINAL:.3f} : 가치 없음")

model_rows   = [r for r in rows if r["사양"] != "benchmark"]
best_row     = max(model_rows, key=lambda x: x["R2_oos"])
best_name    = best_row["사양"]
best_r2_wf   = best_row["R2_oos"]

print(f"\n  → 워크포워드 최고 사양: '{best_name}'  (R2_oos={best_r2_wf:.4f})")

if best_r2_wf >= R2_THRESHOLD_MEANINGFUL:
    print(f"  판정: ✓ 의미 있는 개선 — 홀드아웃 검증 진행")
elif best_r2_wf >= R2_THRESHOLD_MARGINAL:
    print(f"  판정: △ 미미한 개선 — 홀드아웃에서 확인 필요 (노이즈일 수 있음)")
else:
    print(f"  판정: ✗ 어떤 사양도 벤치마크를 뚜렷이 이기지 못함")
    print(f"    → 주간 S&P500 로그수익률은 이 변수들로 예측 가치가 확인되지 않습니다.")
    print(f"       이것도 유효한 발견입니다. 다음 단계: 월간 모델을 고려하세요.")

# ── 13. 최종 홀드아웃 검증 ─────────────────────────────────────────────────────

print(f"\n■ 12단계: 최종 홀드아웃 검증 — 사양 '{best_name}'")
print(f"  ※ 이 숫자가 정직한 표본 외 성적입니다 (사양 선택에 한 번도 쓰지 않은 구간)")

_t0 = time.time()
records_hold = run_walk_forward(
    df_full, TARGET_COL,
    {best_name: SPECS[best_name]},
    {best_name: orders_[best_name]},
    MIN_TRAIN_WEEKS, REFIT_EVERY, WINDOW,
    start_t=len(df_wf),
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
print(f"  예측 스텝 수      : {hold_m['n']}주  (소요: {time.time()-_t0:.1f}초)")

# CSV 저장
hold_df = pd.DataFrame({
    "date":      df_full["date"].iloc[len(df_wf): len(df_wf) + hold_m["n"]].values,
    "actual":    records_hold[best_name]["actual"],
    "predicted": records_hold[best_name]["pred"],
    "benchmark": records_hold[best_name]["bench"],
})
out_path = OUTPUT_DIR / "5specs_holdout_predictions.csv"
hold_df.to_csv(out_path, index=False)
print(f"  저장: {out_path}")

# ── 14. 종합 요약 + 최종 판정 ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약 및 최종 판정")
print("=" * 65)

print(f"\n  [워크포워드 결과 (상위→하위)]")
print(results_df.to_string(index=False))

print(f"\n  [선택 사양: '{best_name}']")
print(f"  · 워크포워드 R2_oos = {best_r2_wf:.4f}")
print(f"  · 홀드아웃  R2_oos = {hold_m['r2_oos']:+.4f}")
print(f"  · 홀드아웃 방향적중률 = {hold_m['dir_acc']:.3f}")

print(f"""
  [주의 사항]
  ① VIX·HY 스프레드는 주가와 동시(contemporaneous) 관계가 강합니다.
     lag-1 적용 시 선행 성분만 남아 예측력이 제한적일 수 있습니다.
  ② RAI가 financial_conditions 와 높은 상관을 보이면, RAI는 새 정보를
     추가하는 게 아니라 같은 정보를 다르게 포장한 것입니다.
  ③ 신용스프레드로 사용한 BAA10Y는 투자적격 등급 기준으로,
     실제 HY OAS(BAMLH0A0HYM2)보다 변동폭이 작습니다.
""")

# 최종 한 줄 결론
any_meaningful = any(
    r["R2_oos"] >= R2_THRESHOLD_MEANINGFUL
    for r in model_rows
)
if hold_m["r2_oos"] >= R2_THRESHOLD_MEANINGFUL:
    print(f"  ★ 최종 결론: '{best_name}' 사양이 홀드아웃에서 벤치마크를 뚜렷이 이겼습니다.")
    print(f"    (R2_oos={hold_m['r2_oos']:+.4f} ≥ {R2_THRESHOLD_MEANINGFUL}) — 다음 사양으로 채택 고려.")
elif hold_m["r2_oos"] > 0:
    print(f"  ★ 최종 결론: 홀드아웃 R2_oos={hold_m['r2_oos']:+.4f} (양수지만 임계값 미달).")
    print(f"    간신히 0을 넘는 수준이므로 노이즈로 간주합니다.")
    print(f"    주간 S&P500은 이 변수들로 예측 가치가 확인되지 않습니다.")
    print(f"    → 유효한 발견. 다음 단계: 월간 빈도로 재검증을 고려하세요.")
else:
    print(f"  ★ 최종 결론: 어떤 사양도 홀드아웃에서 벤치마크를 이기지 못했습니다.")
    print(f"    주간 S&P500 로그수익률은 이 변수들로 예측 가치가 확인되지 않습니다.")
    print(f"    → 유효한 발견. 다음 단계: 월간 모델을 고려하세요.")

print("\n" + "=" * 65)
print("분석 완료.")
print("=" * 65)
