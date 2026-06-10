# =============================================================================
# walkforward_monthly.py  —  월간 빈도 5+1 사양 워크포워드 검증
#
# 목적:
#   주간 모델에서 가치가 확인되지 않은 변수들을
#   월간 빈도로 재검증한다.
#   (월간이 금융여건·신용·레버리지 신호의 본 무대임)
#
# 테스트 사양 (5개 기본 + 1개 보너스):
#   1) anchor         = [fc_m1]
#   2) anchor_risk    = [fc_m1, vix_m1]
#   3) anchor_credit  = [fc_m1, dHY_m3]
#   4) conditions_mom = [dFC_m3, vix_m1]
#   5) composite      = [RAI_m1]
#   6) bonus_invdur   = [fc_m1, inv_dur_m1]   ← 역전 지속 개월, 월간이 본 무대
#   + 벤치마크: 역사적 누적 평균
#
# 변수 정의:
#   fc_m1      = financial_conditions 월말값, 1개월 lag
#   vix_m1     = vix 월평균, 1개월 lag
#   dHY_m3     = (HY_spread - HY_spread.shift(3)) lag-1  [3개월 크레딧 모멘텀]
#   dFC_m3     = (financial_conditions - fc.shift(3)) lag-1 [3개월 금융여건 모멘텀]
#   RAI_m1     = [z(-NFCILEVERAGE) - z(HY) - z(VIX)] lag-1 [합성 위험선호]
#   inv_dur_m1 = 장단기 역전 연속 개월 수, 1개월 lag
#
# 외부 데이터:
#   NFCILEVERAGE : FRED 주간 → 월말 집계
#   BAA10Y       : FRED 일별 → 월말 집계 (BAMLH0A0HYM2 대체)
#   나머지       : 기존 CSV 월간 집계
#
# 필요 라이브러리:
#   pip install pandas numpy statsmodels scikit-learn fredapi python-dotenv
#
# 파라미터 (상단에서 조정):
#   holdout_months=18, min_train_months=60, refit_every=2
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
TARGET_COL = "sp500_log_return"   # 월간 집계 후 같은 이름 유지

HOLDOUT_MONTHS   = 18    # 최종 홀드아웃 개월 수 (~1.5년)
MIN_TRAIN_MONTHS = 60    # 최소 학습 개월 수 (~5년)
REFIT_EVERY      = 2     # N개월마다 모델 재적합

WINDOW = "expanding"     # "expanding" 또는 "rolling"

MAX_P = 2
MAX_Q = 2

# 판정 기준 — 월간은 노이즈가 적으므로 임계값 약간 높게
R2_MEANINGFUL = 0.03    # 이 이상이면 의미 있는 개선
R2_MARGINAL   = 0.01    # 이 이상이면 미미한 가능성

OUTPUT_DIR = Path(__file__).parent

# ── 1. FRED 데이터 조회 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 월간 빈도 워크포워드 검증")
print("=" * 65)
print("\n■ 1단계: FRED 데이터 조회")

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("[오류] .env 파일에 FRED_API_KEY 없음")
    sys.exit(1)
try:
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
except ImportError:
    print("[오류] fredapi 없음. pip install fredapi")
    sys.exit(1)

# NFCILEVERAGE: 주간 → 월말 마지막 값으로 집계
try:
    nfci_raw = fred.get_series("NFCILEVERAGE", observation_start="1999-01-01")
    nfci_raw.index = pd.to_datetime(nfci_raw.index)
    print(f"  ✓ NFCILEVERAGE: {len(nfci_raw)}행")
except Exception as e:
    print(f"  ✗ NFCILEVERAGE 실패: {e}"); sys.exit(1)

time.sleep(0.5)

# BAA10Y: 일별 → 월말 마지막 값으로 집계
try:
    hy_raw = fred.get_series("BAA10Y", observation_start="1999-01-01")
    hy_raw.index = pd.to_datetime(hy_raw.index)
    print(f"  ✓ BAA10Y (HY 대체): {len(hy_raw)}행  (Moody's Baa 신용스프레드)")
except Exception as e:
    print(f"  ✗ BAA10Y 실패: {e}"); sys.exit(1)

# ── 2. CSV 로드 ────────────────────────────────────────────────────────────────

print("\n■ 2단계: CSV 로드 및 월간 집계")

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] 파일 없음: {csv_path.resolve()}"); sys.exit(1)

df_w = pd.read_csv(csv_path, parse_dates=["date"])
df_w = df_w.sort_values("date").set_index("date")

for col in [TARGET_COL, "vix", "financial_conditions", "term_spread_10y_2y"]:
    if col not in df_w.columns:
        print(f"[오류] 컬럼 없음: '{col}'"); sys.exit(1)

# ── 3. 주간 → 월간 집계 ────────────────────────────────────────────────────────
#
# 집계 방법 선택 기준:
#   - 수익률(log return) → 월간 합산  (log 수익률의 합 = 월간 수익률)
#   - 수준 변수(fc, VIX, spread) → 월말 마지막 값  (예측 시점 기준)
#   - VIX → 월평균  (공포 수준은 평균이 대표성 높음)
# ─────────────────────────────────────────────────────────────────────────────

print("  주간 수익률 → 월간 합산 (= 월간 로그수익률)")
print("  FC·스프레드 → 월말 마지막 값 / VIX → 월평균")

df_m = pd.DataFrame({
    TARGET_COL:              df_w[TARGET_COL].resample("ME").sum(),
    "financial_conditions":  df_w["financial_conditions"].resample("ME").last(),
    "vix":                   df_w["vix"].resample("ME").mean(),   # 월 공포 수준
    "term_spread":           df_w["term_spread_10y_2y"].resample("ME").last(),
})

# FRED 시계열 월간 집계 (월말 마지막 값)
nfci_m = nfci_raw.resample("ME").last().ffill()
hy_m   = hy_raw.ffill().resample("ME").last().ffill()

df_m["nfcileverage"] = nfci_m.reindex(df_m.index, method="ffill")
df_m["hy_spread"]    = hy_m.reindex(df_m.index, method="ffill")
df_m = df_m.reset_index()   # date 컬럼 복원

n_valid = df_m.dropna().shape[0]
print(f"  집계 완료: {n_valid}개월  "
      f"({df_m['date'].min().date()} ~ {df_m['date'].max().date()})")
print(f"  월간 S&P500 수익률 — "
      f"평균: {df_m[TARGET_COL].mean():.4f}, "
      f"표준편차: {df_m[TARGET_COL].std():.4f}, "
      f"상승 비율: {(df_m[TARGET_COL]>0).mean()*100:.1f}%")

# ── 4. 새 변수 생성 (누수 없음) ───────────────────────────────────────────────

print("\n■ 3단계: 월간 변수 생성 (lag-1 = 1개월 전 정보만 사용)")

# fc_m1: 금융여건 1개월 lag
df_m["fc_m1"] = df_m["financial_conditions"].shift(1)

# vix_m1: VIX 월평균 1개월 lag
df_m["vix_m1"] = df_m["vix"].shift(1)

# dHY_m3: HY 스프레드 3개월 변화 → 1개월 lag
# 월간에서 3개월 모멘텀 = 주간에서 4주 모멘텀과 유사한 의미
df_m["dHY"]    = df_m["hy_spread"] - df_m["hy_spread"].shift(3)
df_m["dHY_m3"] = df_m["dHY"].shift(1)
print("  ✓ fc_m1, vix_m1, dHY_m3 생성")

# dFC_m3: 금융여건 3개월 변화 → 1개월 lag
df_m["dFC"]    = df_m["financial_conditions"] - df_m["financial_conditions"].shift(3)
df_m["dFC_m3"] = df_m["dFC"].shift(1)
print("  ✓ dFC_m3 생성")

# RAI_m1: 팽창 z-score 합성 위험선호 → 1개월 lag
# 각 성분을 '먼저' 표준화(팽창 z-score, shift(1) → 누수 없음) 후 합성
def expanding_zscore(s: pd.Series) -> pd.Series:
    """시점 t에서 0..t-1 구간 통계로 z-score 계산 (lookahead 없음)."""
    mu  = s.expanding().mean().shift(1)
    sig = s.expanding().std(ddof=1).shift(1)
    sig = sig.where(sig > 1e-8, np.nan)
    return (s - mu) / sig

z_lev = expanding_zscore(-df_m["nfcileverage"])   # 부호 반전: 값↑=위험선호
z_hy  = expanding_zscore(df_m["hy_spread"])
z_vix = expanding_zscore(df_m["vix"])

df_m["rai_raw"] = z_lev - z_hy - z_vix   # 위험선호 높을수록 큼
df_m["RAI_m1"]  = df_m["rai_raw"].shift(1)
print("  ✓ RAI_m1 생성  (z(-NFCI) - z(HY) - z(VIX), lag-1)")
print("    ※ 팽창 z-score: 각 월 t에서 0..t-1 통계 사용 → 누수 없음")

# inv_dur_m1: 장단기 금리 역전 연속 개월 수 (월간 버전) → 1개월 lag
# term_spread < 0 인 달이 연속으로 이어진 수를 카운트
# 역전이 아닌 달은 0으로 리셋
spread_vals = df_m["term_spread"].values
inv_dur = np.zeros(len(spread_vals))
for i in range(1, len(spread_vals)):
    if not np.isnan(spread_vals[i]) and spread_vals[i] < 0:
        inv_dur[i] = inv_dur[i - 1] + 1   # 역전 지속 → 카운터 증가
    else:
        inv_dur[i] = 0                     # 역전 해소 → 리셋
df_m["inv_dur"]    = inv_dur
df_m["inv_dur_m1"] = df_m["inv_dur"].shift(1)
n_inv = int((spread_vals < 0).sum())
print(f"  ✓ inv_dur_m1 생성  (역전 {n_inv}/{len(spread_vals)}개월, "
      f"{100*n_inv/len(spread_vals):.1f}%)")
print("    ← 이전 분석에서 '주간에는 시차가 너무 짧아 월간이 본 무대'로 판단")

# ── 5. ADF 단위근 검정 ────────────────────────────────────────────────────────

print("\n■ 4단계: 단위근 검정 (ADF)")
print("  귀무가설: 단위근 있음(비정상). p > 0.05 면 차분 고려.")

FINAL_COLS: dict[str, str] = {}

def run_adf(series: pd.Series, name: str) -> tuple[bool, float]:
    clean = series.dropna()
    if len(clean) < 20:
        print(f"  {name}: 데이터 부족 ({len(clean)}행)")
        return True, 1.0
    try:
        res  = adfuller(clean, maxlag=6, autolag="AIC")
        pval, stat = res[1], res[0]
        tag = "비정상 ⚠" if pval > 0.05 else "정상  ✓"
        print(f"  {name:<12}: ADF={stat:+.3f}, p={pval:.4f} → [{tag}]")
        return pval > 0.05, pval
    except Exception as ex:
        print(f"  {name}: ADF 오류 ({ex})")
        return True, 1.0

for var, col in [("fc_m1","fc_m1"), ("vix_m1","vix_m1"), ("dHY_m3","dHY_m3"),
                  ("dFC_m3","dFC_m3"), ("RAI_m1","RAI_m1"), ("inv_dur_m1","inv_dur_m1")]:
    nonstat, _ = run_adf(df_m[col], var)
    if nonstat:
        diff_col = f"d_{col}"
        df_m[diff_col] = df_m[col].diff()
        print(f"    → 1차 차분 후:")
        run_adf(df_m[diff_col], f"d_{var}")
        FINAL_COLS[var] = diff_col
        print(f"    → '{diff_col}' 사용")
    else:
        FINAL_COLS[var] = col

print(f"\n  컬럼 매핑: {FINAL_COLS}")

# ── 6. 상관·VIF 중복 점검 ─────────────────────────────────────────────────────

print("\n■ 5단계: 중복 점검 (상관 + VIF)")

fc_c   = FINAL_COLS.get("fc_m1",  "fc_m1")
rai_c  = FINAL_COLS.get("RAI_m1", "RAI_m1")
dhy_c  = FINAL_COLS.get("dHY_m3", "dHY_m3")
inv_c  = FINAL_COLS.get("inv_dur_m1", "inv_dur_m1")
chk    = [c for c in [fc_c, dhy_c, rai_c, inv_c] if c in df_m.columns]
df_chk = df_m[chk].dropna()

if len(df_chk) > 10:
    print("\n  [상관계수]")
    print(df_chk.corr().round(3).to_string())

    import statsmodels.api as sm
    X_vif = sm.add_constant(df_chk.astype(float))
    vif_rows = []
    for i, c in enumerate(df_chk.columns):
        try:
            v = variance_inflation_factor(X_vif.values, i + 1)
            vif_rows.append({"변수": c, "VIF": round(v, 2)})
        except Exception:
            vif_rows.append({"변수": c, "VIF": "계산 불가"})
    print("\n  [VIF]  VIF>10: 강한 중복 / 5~10: 주의 / <5: 양호")
    print(pd.DataFrame(vif_rows).to_string(index=False))

    for cand in [rai_c, dhy_c, inv_c]:
        if cand == fc_c or cand not in df_chk.columns:
            continue
        corr_v = abs(df_chk[cand].corr(df_chk[fc_c]))
        label  = cand.upper()
        msg = ("강한 중복 ⚠" if corr_v > 0.7
               else "중간 겹침 ⚠" if corr_v > 0.4
               else "독립 정보 ✓")
        print(f"\n  [{label}] vs fc_m1 상관={corr_v:.3f} → {msg}")

# ── 7. 사양 정의 ─────────────────────────────────────────────────────────────

print("\n■ 6단계: 사양 정의")

fc  = FINAL_COLS["fc_m1"]
vix = FINAL_COLS["vix_m1"]
dhy = FINAL_COLS["dHY_m3"]
dfc = FINAL_COLS["dFC_m3"]
rai = FINAL_COLS["RAI_m1"]
inv = FINAL_COLS["inv_dur_m1"]

SPECS: dict[str, list[str]] = {
    "anchor":         [fc],
    "anchor_risk":    [fc, vix],
    "anchor_credit":  [fc, dhy],
    "conditions_mom": [dfc, vix],
    "composite":      [rai],
    "bonus_invdur":   [fc, inv],  # 역전 지속 — 월간이 본 무대
}

# 존재하지 않는 컬럼이 포함된 사양 제거
for nm in list(SPECS.keys()):
    missing = [c for c in SPECS[nm] if c not in df_m.columns]
    if missing:
        print(f"  ⚠ '{nm}' 컬럼 없음 {missing} → 제외")
        del SPECS[nm]
    else:
        print(f"  {nm:<16}: {SPECS[nm]}")

# ── 8. 전처리 + 홀드아웃 분리 ─────────────────────────────────────────────────

print("\n■ 7단계: 전처리 + 홀드아웃 분리")

all_feat  = list({c for cols in SPECS.values() for c in cols})
keep_cols = ["date", TARGET_COL] + all_feat
df_use    = df_m[[c for c in keep_cols if c in df_m.columns]].copy()
df_use    = df_use.apply(
    lambda col: pd.to_numeric(col, errors="coerce") if col.name != "date" else col
)

before    = len(df_use)
df_clean  = df_use.dropna().reset_index(drop=True)
after     = len(df_clean)
print(f"  결측 제거: {before} → {after}개월")
print(f"  기간: {df_clean['date'].min().date()} ~ {df_clean['date'].max().date()}")

if after < MIN_TRAIN_MONTHS + HOLDOUT_MONTHS + 10:
    print(f"[오류] 데이터 부족: 최소 {MIN_TRAIN_MONTHS+HOLDOUT_MONTHS+10}개월 필요")
    sys.exit(1)

df_wf   = df_clean.iloc[:-HOLDOUT_MONTHS].reset_index(drop=True)
df_full = df_clean.reset_index(drop=True)

print(f"  워크포워드: {len(df_wf)}개월  "
      f"({df_wf['date'].iloc[0].date()} ~ {df_wf['date'].iloc[-1].date()})")
print(f"  홀드아웃  : {HOLDOUT_MONTHS}개월  "
      f"({df_clean['date'].iloc[-HOLDOUT_MONTHS].date()} ~ "
      f"{df_clean['date'].iloc[-1].date()})")
wf_steps = len(df_wf) - MIN_TRAIN_MONTHS
print(f"  워크포워드 예측 스텝: {wf_steps}개월")

# ── 9. 헬퍼 함수 ─────────────────────────────────────────────────────────────


def select_arima_order(y, X=None):
    """AIC 기준 ARIMA(p,0,q) 차수 선택. d=0 고정 (월간 수익률 = 정상)."""
    best_aic, best_ord = np.inf, (1, 0, 1)
    for p, q in itertools.product(range(MAX_P + 1), range(MAX_Q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            Xdf = pd.DataFrame(X) if X is not None else None
            m   = SARIMAX(y, exog=Xdf, order=(p, 0, q),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            if m.aic < best_aic:
                best_aic, best_ord = m.aic, (p, 0, q)
        except Exception:
            pass
    return best_ord


def compute_metrics(actuals, preds, benchmarks):
    """R²_oos = 1 - SSE모델 / SSE벤치마크, 방향 적중률."""
    a, p, b = map(lambda x: np.array(x, float), [actuals, preds, benchmarks])
    sse_m = np.sum((a - p) ** 2)
    sse_b = np.sum((a - b) ** 2)
    r2    = float(1 - sse_m / sse_b) if sse_b > 1e-12 else np.nan
    dir_a = float(np.mean(np.sign(a) == np.sign(p)))
    dir_b = float(np.mean(np.sign(a) == np.sign(b)))
    return {"r2_oos": r2, "dir_acc": dir_a, "bench_dir": dir_b, "n": len(a)}


def run_walk_forward(df_src, target, specs, orders,
                     min_train, refit_every, window, start_t=None):
    """
    워크포워드 1개월 앞 예측 엔진.
    누수 방지:
      ① StandardScaler → 훈련 구간에만 fit
      ② ARIMA 차수 고정, refit_every 마다 파라미터 재추정
      ③ 벤치마크 = 훈련 구간 누적 평균
    """
    if start_t is None:
        start_t = min_train
    n      = len(df_src)
    rec    = {nm: {"actual": [], "pred": [], "bench": []} for nm in specs}
    fitted_: dict = {}
    scaler_: dict = {}
    total  = n - start_t
    nxt    = max(20, total // 10)

    for idx, t in enumerate(range(start_t, n)):
        s = (slice(0, t) if window == "expanding"
             else slice(max(0, t - min_train), t))

        y_tr  = df_src[target].iloc[s].values.astype(float)
        y_act = float(df_src[target].iloc[t])
        y_bm  = float(np.mean(y_tr))   # 역사적 누적 평균 = 벤치마크
        refit = (idx % refit_every == 0)

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
            print(f"    {idx+1:>3}/{total} ({pct:.0f}%) — {time.time()-_t0:.0f}초")
            nxt += max(20, total // 10)

    return rec


# ── 10. ARIMA 차수 선택 ────────────────────────────────────────────────────────

print("\n■ 8단계: ARIMA(p,0,q) 차수 선택 — 첫 훈련 윈도우 AIC 기준, 이후 고정")

y_init  = df_wf[TARGET_COL].values[:MIN_TRAIN_MONTHS].astype(float)
orders_ : dict[str, tuple] = {}
t0_ord  = time.time()

for nm, cols in SPECS.items():
    Xinit = None
    if cols:
        sc    = StandardScaler()
        Xinit = sc.fit_transform(df_wf[cols].values[:MIN_TRAIN_MONTHS].astype(float))
    orders_[nm] = select_arima_order(y_init, Xinit)
    print(f"  {nm:<16}: ARIMA{orders_[nm]}")

print(f"  소요: {time.time()-t0_ord:.1f}초")

# ── 11. 워크포워드 실행 ────────────────────────────────────────────────────────

print(f"\n■ 9단계: 워크포워드 실행 (홀드아웃 제외)")
print(f"  빈도=월간 | 윈도우={WINDOW} | 재적합={REFIT_EVERY}개월 | "
      f"최소학습={MIN_TRAIN_MONTHS}개월")
print(f"  예측 스텝: {wf_steps}개월 × {len(SPECS)}사양")

_t0 = time.time()
records_wf = run_walk_forward(
    df_wf, TARGET_COL, SPECS, orders_,
    MIN_TRAIN_MONTHS, REFIT_EVERY, WINDOW,
)
print(f"\n  완료: {time.time()-_t0:.1f}초")

# ── 12. 결과 집계 + 사양 선택 ─────────────────────────────────────────────────

print("\n■ 10단계: 결과 집계")

rows = []
first_rec = list(records_wf.values())[0]
bm = compute_metrics(first_rec["actual"], first_rec["bench"], first_rec["bench"])
rows.append({"사양": "benchmark", "R2_oos": 0.0,
             "방향적중률": round(bm["bench_dir"], 3), "n": bm["n"]})

for nm, rec in records_wf.items():
    m = compute_metrics(rec["actual"], rec["pred"], rec["bench"])
    rows.append({"사양": nm, "R2_oos": round(m["r2_oos"], 4),
                 "방향적중률": round(m["dir_acc"], 3), "n": m["n"]})

results_df = (pd.DataFrame(rows)
              .sort_values("R2_oos", ascending=False)
              .reset_index(drop=True))

print(f"\n  R2_oos > 0 = 벤치마크 이김  /  R2_oos < 0 = 짐\n")
print(results_df.to_string(index=False))

# 판정
model_rows = [r for r in rows if r["사양"] != "benchmark"]
best_row   = max(model_rows, key=lambda x: x["R2_oos"])
best_name  = best_row["사양"]
best_r2_wf = best_row["R2_oos"]

print(f"\n  [판정 기준]  >={R2_MEANINGFUL:.2f}: 의미 있음  "
      f"/{R2_MARGINAL:.2f}~{R2_MEANINGFUL:.2f}: 미미  <{R2_MARGINAL:.2f}: 없음")
print(f"\n  → 최고 사양: '{best_name}'  (R2_oos={best_r2_wf:.4f})")

if best_r2_wf >= R2_MEANINGFUL:
    print(f"  판정: ✓ 의미 있는 개선 → 홀드아웃 진행")
elif best_r2_wf >= R2_MARGINAL:
    print(f"  판정: △ 미미한 개선 → 홀드아웃에서 확인 필요")
else:
    print(f"  판정: ✗ 벤치마크를 뚜렷이 이기지 못함")
    print(f"    → 이 변수들은 월간 S&P500 수익률 예측에도 가치가 확인되지 않습니다.")

# ── 13. 홀드아웃 검증 ─────────────────────────────────────────────────────────

print(f"\n■ 11단계: 최종 홀드아웃 검증 — '{best_name}'")
print(f"  ※ 이 숫자가 정직한 표본 외 성적입니다.")

_t0 = time.time()
records_hold = run_walk_forward(
    df_full, TARGET_COL,
    {best_name: SPECS[best_name]},
    {best_name: orders_[best_name]},
    MIN_TRAIN_MONTHS, REFIT_EVERY, WINDOW,
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
print(f"  홀드아웃 스텝     : {hold_m['n']}개월  (소요: {time.time()-_t0:.1f}초)")

# 저장
hold_df = pd.DataFrame({
    "date":      df_full["date"].iloc[len(df_wf):len(df_wf)+hold_m["n"]].values,
    "actual":    records_hold[best_name]["actual"],
    "predicted": records_hold[best_name]["pred"],
    "benchmark": records_hold[best_name]["bench"],
})
out_path = OUTPUT_DIR / "monthly_holdout_predictions.csv"
hold_df.to_csv(out_path, index=False)
print(f"  저장: {out_path}")

# ── 14. 종합 요약 ─────────────────────────────────────────────────────────────

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
  [참고 사항]
  ① 월간 스텝 수({wf_steps}개)는 주간({total_steps if 'total_steps' in dir() else '~967'}개)보다 훨씬 적어
     R²_oos 추정의 표준오차가 큽니다. 결과 해석에 유의하세요.
  ② 홀드아웃 18개월은 관측치가 적어 운(luck)의 영향이 큽니다.
  ③ RAI는 financial_conditions 와 상관이 높아 재포장 가능성이 있습니다.
  ④ inv_dur(역전 지속 개월)은 에피소드 수가 적어 과적합 위험이 있습니다.
""")

if hold_m["r2_oos"] >= R2_MEANINGFUL:
    print(f"  ★ 최종 결론: '{best_name}'이 홀드아웃에서 벤치마크를 뚜렷이 이겼습니다.")
    print(f"    (R2_oos={hold_m['r2_oos']:+.4f} ≥ {R2_MEANINGFUL})")
    print(f"    → 이 사양을 다음 분석의 출발점으로 채택할 수 있습니다.")
elif hold_m["r2_oos"] > 0:
    print(f"  ★ 최종 결론: 홀드아웃 R2_oos={hold_m['r2_oos']:+.4f} (양수지만 임계값 미달).")
    print(f"    노이즈 범위로 간주합니다. 가치가 확인되지 않습니다.")
else:
    print(f"  ★ 최종 결론: 월간에서도 벤치마크를 이기지 못했습니다.")
    print(f"    R2_oos={hold_m['r2_oos']:+.4f}")
    print(f"    → 이 변수들로는 월간 S&P500 수익률 예측 가치가 확인되지 않습니다.")
    print(f"    → 유효한 발견입니다.")

print("\n" + "=" * 65)
print("분석 완료.")
print("=" * 65)
