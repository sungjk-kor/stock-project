# =============================================================================
# regime_full_features.py  —  3-레짐 예측 (확장 특징변수: 기술지표 + 거래량 포함)
#
# 이전 모델(regime_multinomial.py)에서 빠졌던 변수군을 모두 추가:
#   ① 이동평균 / 가격 모멘텀
#   ② 거래량 지표 (vol_ratio, up_vol_ratio)
#   ③ 섹터 상대강도 (나스닥, 산업재)
#   ④ 기존 거시/금융여건 변수
#
# 위기(약세) 주 35주 포함. 강세/약세/횡보 3-클래스 전체 예측.
#
# ★ 누수 주의:
#   레짐 라벨이 전체 표본 마르코프에서 생성됨 → 약한 누수. 결론 해석 시 주의.
#
# 필요 라이브러리:
#   pip install pandas numpy statsmodels scikit-learn fredapi python-dotenv
# =============================================================================

import os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH        = "data/processed/final_dataset.csv"
RET_COL         = "sp500_log_return"
FWD_WEEKS       = 13
HOLDOUT_WEEKS   = 78
MIN_TRAIN_WEEKS = 260
REFIT_EVERY     = 4

REGIME_NAMES = {0: "강세", 1: "약세", 2: "횡보"}
OUTPUT_DIR   = Path(__file__).parent

# ── 1. FRED 데이터 조회 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 3-레짐 예측 — 기술지표·거래량 포함 전체 특징변수")
print("=" * 65)

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("[오류] .env 파일에 FRED_API_KEY 없음"); sys.exit(1)

try:
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
except ImportError:
    print("[오류] fredapi 없음. pip install fredapi"); sys.exit(1)

nfci_raw = hy_raw = None
try:
    nfci_raw = fred.get_series("NFCILEVERAGE", observation_start="1999-01-01")
    nfci_raw.index = pd.to_datetime(nfci_raw.index)
    print(f"  ✓ NFCILEVERAGE: {len(nfci_raw)}행")
except Exception as e:
    print(f"  △ NFCILEVERAGE 실패: {e}")

time.sleep(0.4)
try:
    hy_raw = fred.get_series("BAA10Y", observation_start="1999-01-01")
    hy_raw.index = pd.to_datetime(hy_raw.index)
    print(f"  ✓ BAA10Y (HY 대체): {len(hy_raw)}행")
except Exception as e:
    print(f"  △ BAA10Y 실패: {e}")

# ── 2. CSV 로드 및 정렬 ────────────────────────────────────────────────────────

print("\n■ 2단계: CSV 로드 및 시계열 정렬")

df_csv = pd.read_csv(CSV_PATH, parse_dates=["date"])
df_csv = df_csv.sort_values("date").reset_index(drop=True)

# 필요한 컬럼 선택
base_cols = ["date", RET_COL,
             "sp500", "sp500_volume",
             "nasdaq_log_return", "xli_log_return",
             "vix", "financial_conditions", "term_spread_10y_2y"]
df = df_csv[[c for c in base_cols if c in df_csv.columns]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

# FRED 시계열 JOIN
if nfci_raw is not None:
    df["nfci_leverage"] = nfci_raw.resample("W-FRI").last().ffill().reindex(df.index, method="ffill")
if hy_raw is not None:
    df["hy_spread"] = hy_raw.ffill().resample("W-FRI").last().ffill().reindex(df.index, method="ffill")

df = df.reset_index()
print(f"  {len(df):,}주  ({df['date'].min().date()} ~ {df['date'].max().date()})")

# ── 3. 특징변수 생성 ──────────────────────────────────────────────────────────

print("\n■ 3단계: 특징변수 생성")
print("  (모두 lag-1 적용 전 raw 값 — 미래 정보 없음)")

p = df["sp500"]          # S&P500 가격
v = df["sp500_volume"]   # S&P500 거래량
r = df[RET_COL]          # 주간 로그수익률

# ── [A] 이동평균 / 가격 모멘텀 ───────────────────────────────────────────────
# 52주 이동평균 대비 가격 위치 (추세 방향 판단)
ma13 = p.rolling(13).mean()
ma52 = p.rolling(52).mean()
df["price_to_ma52"]  = p / ma52 - 1          # 52주 이평 대비 위치. 양수=상승추세
df["ma_cross"]       = (ma13 > ma52).astype(float)  # 1=골든크로스(강세), 0=데드크로스

# 13주·26주 가격 모멘텀 (과거 수익률)
df["momentum_13w"]   = p / p.shift(13) - 1   # 3개월 모멘텀
df["momentum_26w"]   = p / p.shift(26) - 1   # 6개월 모멘텀

print("  ✓ 이동평균: price_to_ma52, ma_cross")
print("  ✓ 모멘텀  : momentum_13w, momentum_26w")

# ── [B] 거래량 지표 ───────────────────────────────────────────────────────────
# vol_ratio: 현재 거래량이 평소 대비 몇 배인지 (1 이상이면 활발한 거래)
df["vol_ratio"]      = v / v.rolling(13).mean()

# up_vol_ratio: 13주 중 상승일 거래량 비율 (0.5=중립, >0.6=강세 확인)
up_vol = v.where(r > 0, 0).rolling(13).sum()
df["up_vol_ratio"]   = up_vol / v.rolling(13).sum()

# obv_chg: OBV(누적 방향거래량) 13주 변화를 총 거래량으로 정규화
# 양수=순매수 추세, 음수=순매도 추세
obv = (np.sign(r) * v).cumsum()
df["obv_chg_norm"]   = (obv - obv.shift(13)) / v.rolling(13).sum()

print("  ✓ 거래량  : vol_ratio, up_vol_ratio, obv_chg_norm")

# ── [C] 섹터 상대강도 ─────────────────────────────────────────────────────────
# nasdaq_rel: 나스닥 - S&P500 상대수익률 (기술주 강세 여부)
if "nasdaq_log_return" in df.columns:
    df["nasdaq_rel"] = df["nasdaq_log_return"] - r
    print("  ✓ 섹터강도: nasdaq_rel (나스닥 상대강도)")

# xli_rel: 산업재 - S&P500 상대수익률 (경기 민감 섹터 선행)
if "xli_log_return" in df.columns:
    df["xli_rel"] = df["xli_log_return"] - r
    print("  ✓ 섹터강도: xli_rel (산업재 상대강도)")

# ── [D] 기존 거시/금융여건 변수 ───────────────────────────────────────────────
# inv_duration: 장단기 역전 연속 주 수
sp = df["term_spread_10y_2y"].values
inv_d = np.zeros(len(sp))
for i in range(1, len(sp)):
    if not np.isnan(sp[i]) and sp[i] < 0:
        inv_d[i] = inv_d[i - 1] + 1
    else:
        inv_d[i] = 0
df["inv_duration"] = inv_d

if "hy_spread" in df.columns:
    df["dHY"] = df["hy_spread"] - df["hy_spread"].shift(4)   # HY 4주 변화
df["dFC"] = df["financial_conditions"] - df["financial_conditions"].shift(4)  # FC 4주 변화

print("  ✓ 거시/금융: inv_duration, dHY, dFC")

# ── 4. 마르코프 레짐 전환 모델 ────────────────────────────────────────────────

print("\n■ 4단계: 마르코프 레짐 전환 (전체 표본, 약한 누수 존재)")

ret_s = df.set_index("date")[RET_COL].dropna()
print("  추정 중...")
t0 = time.time()
try:
    mod = MarkovRegression(ret_s, k_regimes=3, trend="c", switching_variance=True)
    res = mod.fit(search_reps=20, search_iter=100, disp=False)
    print(f"  완료 ({time.time()-t0:.0f}초), AIC={res.aic:.1f}")

    smoothed   = res.smoothed_marginal_probabilities
    raw_labels = smoothed.idxmax(axis=1)

    state_stats = {}
    for r_idx in range(3):
        m = raw_labels == r_idx
        state_stats[r_idx] = {
            "mean": ret_s[m].mean() * 100,
            "std":  ret_s[m].std()  * 100,
            "n":    int(m.sum()),
        }
    sorted_m = sorted(range(3), key=lambda r: state_stats[r]["mean"])
    bear_raw, sideways_raw, bull_raw = sorted_m

    state_map = {bull_raw: 0, bear_raw: 1, sideways_raw: 2}
    regime_s  = raw_labels.map(state_map).rename("regime")

    print(f"\n  [레짐 통계]")
    print(f"  {'레짐':<6} {'평균수익률':>10} {'표준편차':>10} {'주 수':>6}")
    for rname, ridx in [("강세", 0), ("약세", 1), ("횡보", 2)]:
        raw = [k for k, v in state_map.items() if v == ridx][0]
        s = state_stats[raw]
        print(f"  {rname:<6} {s['mean']:>+9.3f}%  {s['std']:>9.3f}%  {s['n']:>6}")
    markov_ok = True
except Exception as e:
    print(f"  [오류] {e} → 수익률 분위수 백업")
    q33, q67 = ret_s.quantile(0.33), ret_s.quantile(0.67)
    regime_s = ret_s.apply(lambda r: 1 if r<=q33 else (0 if r>=q67 else 2)).rename("regime")
    markov_ok = False

df = df.merge(regime_s.reset_index(), on="date", how="left")

# ── 5. ADF 단위근 검정 ────────────────────────────────────────────────────────

print("\n■ 5단계: 단위근 검정 (ADF)")

# 검정할 후보 컬럼 목록
CANDIDATES = [
    "term_spread_10y_2y", "inv_duration",
    "hy_spread",          "nfci_leverage",   "financial_conditions",
    "vix",                "dHY",             "dFC",
    "price_to_ma52",      "ma_cross",
    "momentum_13w",       "momentum_26w",
    "vol_ratio",          "up_vol_ratio",    "obv_chg_norm",
    "nasdaq_rel",         "xli_rel",
]

FEAT_COLS: dict[str, str] = {}   # {원천컬럼: 최종사용컬럼}

for col in CANDIDATES:
    if col not in df.columns or df[col].isna().all():
        continue
    clean = df[col].dropna()
    if len(clean) < 30:
        continue
    try:
        p_val = adfuller(clean, maxlag=12, autolag="AIC")[1]
        tag   = "정상 ✓" if p_val <= 0.05 else "비정상 ⚠"
        print(f"  {col:<25}: p={p_val:.4f} [{tag}]")
        if p_val > 0.05:
            diff_col = f"d_{col}"
            df[diff_col] = df[col].diff()
            FEAT_COLS[col] = diff_col
            print(f"    → 1차 차분 → '{diff_col}'")
        else:
            FEAT_COLS[col] = col
    except Exception:
        FEAT_COLS[col] = col

print(f"\n  총 {len(FEAT_COLS)}개 특징변수")

# ── 6. VIF 점검 ───────────────────────────────────────────────────────────────

print("\n■ 6단계: VIF 다공선성 점검")
print("  (로지스틱은 L2 정규화가 있어 VIF 높아도 학습 가능 — 참고용)")

vif_df = df[[v for v in FEAT_COLS.values() if v in df.columns]].dropna()
if len(vif_df) > 10:
    X_vif = sm.add_constant(vif_df.astype(float))
    print(f"  {'변수':<28} {'VIF':>6}")
    for i, col in enumerate(vif_df.columns):
        try:
            v_val = variance_inflation_factor(X_vif.values, i + 1)
            flag  = " ⚠" if v_val > 10 else (" △" if v_val > 5 else "")
            print(f"  {col:<28} {v_val:>6.2f}{flag}")
        except Exception:
            print(f"  {col:<28}  계산 불가")

# ── 7. 타깃 생성 ─────────────────────────────────────────────────────────────

print("\n■ 7단계: 타깃 변수 생성")
print(f"  target_t = t+1~t+{FWD_WEEKS} 구간의 최빈 레짐 (강세=0, 약세=1, 횡보=2)")

regimes = df["regime"].values
target  = np.full(len(regimes), np.nan)
for t in range(len(regimes) - FWD_WEEKS):
    window = regimes[t + 1 : t + 1 + FWD_WEEKS]
    valid  = window[~np.isnan(window)]
    if len(valid) == 0: continue
    counts = np.bincount(valid.astype(int), minlength=3)
    target[t] = int(np.argmax(counts))

df["target"] = target
n_valid = int(pd.notna(df["target"]).sum())
print(f"  유효 타깃: {n_valid}주")
for ridx, rname in REGIME_NAMES.items():
    n = int((df["target"] == ridx).sum())
    print(f"    {rname}({ridx}): {n}주 ({n/n_valid*100:.1f}%)")

# ── 8. lag-1 적용 및 최종 데이터 준비 ────────────────────────────────────────

print("\n■ 8단계: lag-1 적용 및 데이터 정리")

LAG_FEAT_COLS = []
for raw_col, use_col in FEAT_COLS.items():
    lag_col = f"{use_col}_l1"
    df[lag_col] = df[use_col].shift(1)   # 1주 lag → 예측 시점에서 과거 정보만 사용
    LAG_FEAT_COLS.append(lag_col)

print(f"  lag-1 특징변수 {len(LAG_FEAT_COLS)}개:")
for c in LAG_FEAT_COLS:
    print(f"    {c}")

keep     = ["date", RET_COL, "regime", "target"] + LAG_FEAT_COLS
df_model = df[[c for c in keep if c in df.columns]].dropna().reset_index(drop=True)

print(f"\n  결측 제거 후: {len(df_model)}주")
print(f"  기간: {df_model['date'].min().date()} ~ {df_model['date'].max().date()}")

if len(df_model) < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 10:
    print("[오류] 데이터 부족"); sys.exit(1)

df_wf   = df_model.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_full = df_model.reset_index(drop=True)
n_wf    = len(df_wf)
print(f"  워크포워드: {n_wf}주  |  홀드아웃: {HOLDOUT_WEEKS}주")

# ── 9. 헬퍼 함수 ─────────────────────────────────────────────────────────────

def eval_metrics(df_res: pd.DataFrame, label: str = "") -> tuple:
    """3-클래스 log-loss, 정확도, 분류 보고서"""
    if df_res.empty:
        return None, None
    y_true  = df_res["actual"].astype(int).values
    proba   = np.column_stack([df_res["p_bull"], df_res["p_bear"], df_res["p_sideways"]])
    clim    = np.column_stack([df_res["pc_bull"], df_res["pc_bear"], df_res["pc_sideways"]])
    y_pred  = df_res["pred"].astype(int).values

    ll      = log_loss(y_true, proba,  labels=[0, 1, 2])
    ll_clim = log_loss(y_true, clim,   labels=[0, 1, 2])

    print(f"\n  [{label}]")
    print(f"  Log-loss  모델: {ll:.4f}  /  벤치마크(climatology): {ll_clim:.4f}")
    if   ll < ll_clim - 0.02:
        print(f"  → ✓ 의미 있는 개선 (개선폭: {ll_clim-ll:.4f})")
    elif ll < ll_clim - 0.005:
        print(f"  → △ 미미한 개선 ({ll_clim-ll:.4f})")
    elif ll < ll_clim:
        print(f"  → △ 노이즈 수준 ({ll_clim-ll:.4f})")
    else:
        print(f"  → ✗ 벤치마크보다 못함 (차이: {ll-ll_clim:.4f})")

    acc     = (y_pred == y_true).mean() * 100
    maj     = np.bincount(y_true).argmax()
    acc_maj = (y_true == maj).mean() * 100
    print(f"  정확도   모델: {acc:.1f}%  /  다수클래스 기저율: {acc_maj:.1f}%")

    tnames = [REGIME_NAMES[i] for i in sorted(REGIME_NAMES.keys())]
    try:
        print(f"\n  [클래스별 성과]")
        print(classification_report(y_true, y_pred,
                                    target_names=tnames, labels=[0,1,2],
                                    zero_division=0))
    except Exception as e:
        print(f"  분류 보고서 오류: {e}")

    print(f"  [혼동행렬 (행=실제, 열=예측)]")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("  " + "".join(f"  예측{REGIME_NAMES[j]:3}" for j in range(3)))
    for i, row in enumerate(cm):
        print(f"  실제{REGIME_NAMES[i]:3}" + "".join(f"  {v:6}" for v in row))

    return float(ll), float(ll_clim)


# ── 10. 워크포워드 실행 ────────────────────────────────────────────────────────

print(f"\n■ 9단계: 워크포워드 실행")
print(f"  확장 윈도우 | 재적합={REFIT_EVERY}스텝 | 최소학습={MIN_TRAIN_WEEKS}주")
print(f"  모델: LogisticRegression(class_weight='balanced', C=1.0)")

wf_records     = []
last_model     = None
last_scaler    = None
last_fit_step  = -(REFIT_EVERY + 1)

_t0        = time.time()
total_wf   = n_wf - MIN_TRAIN_WEEKS
next_print = max(50, total_wf // 10)

for step_idx, t in enumerate(range(MIN_TRAIN_WEEKS, n_wf)):
    # 훈련 컷오프: 타깃 전방창이 완전히 과거여야 함
    train_end = t - FWD_WEEKS
    if train_end < MIN_TRAIN_WEEKS:
        continue

    train   = df_wf.iloc[:train_end + 1]
    y_train = train["target"].astype(int).values
    if len(np.unique(y_train)) < 2:
        continue

    need_refit = ((step_idx - last_fit_step) >= REFIT_EVERY) or (last_model is None)

    if need_refit:
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(train[LAG_FEAT_COLS].astype(float))
        try:
            clf = LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
            )
            clf.fit(X_tr, y_train)
            last_model    = clf
            last_scaler   = scaler
            last_fit_step = step_idx
        except Exception:
            if last_model is None: continue

    X_test = last_scaler.transform(
        df_wf[LAG_FEAT_COLS].iloc[[t]].astype(float)
    )
    try:
        proba     = last_model.predict_proba(X_test)[0]
        classes   = last_model.classes_
        pd_       = dict(zip(classes, proba))
        p_bull    = pd_.get(0, 0.0)
        p_bear    = pd_.get(1, 0.0)
        p_side    = pd_.get(2, 0.0)
        pred      = int(classes[np.argmax(proba)])
    except Exception:
        continue

    # 벤치마크: 훈련 구간 레짐 빈도
    clim = np.bincount(y_train, minlength=3) / len(y_train)

    wf_records.append({
        "date":       df_wf["date"].iloc[t],
        "actual":     int(df_wf["target"].iloc[t]),
        "pred":       pred,
        "p_bull":     p_bull,
        "p_bear":     p_bear,
        "p_sideways": p_side,
        "pc_bull":    float(clim[0]),
        "pc_bear":    float(clim[1]),
        "pc_sideways":float(clim[2]),
    })

    if (step_idx + 1) >= next_print:
        pct = (step_idx + 1) / total_wf * 100
        print(f"    {step_idx+1:>4}/{total_wf} ({pct:.0f}%) — {time.time()-_t0:.0f}초")
        next_print += max(50, total_wf // 10)

wf_df = pd.DataFrame(wf_records)
print(f"\n  완료: {time.time()-_t0:.1f}초  ({len(wf_df)}스텝)")

# ── 11. 워크포워드 성과 ────────────────────────────────────────────────────────

print("\n■ 10단계: 워크포워드 성과")
wf_ll, wf_ll_clim = eval_metrics(wf_df, label="워크포워드 구간")

# ── 12. 홀드아웃 검증 ─────────────────────────────────────────────────────────

print("\n■ 11단계: 최종 홀드아웃 검증")
print("  ※ 사양 선택에 한 번도 쓰지 않은 구간")

hold_records   = []
last_model_h   = None
last_scaler_h  = None
last_fit_h     = -(REFIT_EVERY + 1)

_t0 = time.time()
for step_h, t in enumerate(range(len(df_wf), len(df_full))):
    train_end = t - FWD_WEEKS
    if train_end < MIN_TRAIN_WEEKS: continue

    train   = df_full.iloc[:train_end + 1]
    y_train = train["target"].astype(int).values
    if len(np.unique(y_train)) < 2: continue

    need_refit = ((step_h - last_fit_h) >= REFIT_EVERY) or (last_model_h is None)
    if need_refit:
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(train[LAG_FEAT_COLS].astype(float))
        try:
            clf = LogisticRegression(class_weight="balanced",
                                     solver="lbfgs", max_iter=1000, C=1.0)
            clf.fit(X_tr, y_train)
            last_model_h  = clf
            last_scaler_h = scaler
            last_fit_h    = step_h
        except Exception:
            if last_model_h is None: continue

    X_test = last_scaler_h.transform(
        df_full[LAG_FEAT_COLS].iloc[[t]].astype(float)
    )
    try:
        proba   = last_model_h.predict_proba(X_test)[0]
        classes = last_model_h.classes_
        pd_     = dict(zip(classes, proba))
        pred    = int(classes[np.argmax(proba)])
    except Exception:
        continue

    clim = np.bincount(y_train, minlength=3) / len(y_train)
    hold_records.append({
        "date":        df_full["date"].iloc[t],
        "actual":      int(df_full["target"].iloc[t]),
        "pred":        pred,
        "p_bull":      pd_.get(0, 0.0),
        "p_bear":      pd_.get(1, 0.0),
        "p_sideways":  pd_.get(2, 0.0),
        "pc_bull":     float(clim[0]),
        "pc_bear":     float(clim[1]),
        "pc_sideways": float(clim[2]),
    })

hold_df = pd.DataFrame(hold_records)
print(f"  홀드아웃 예측 스텝: {len(hold_df)}주  (소요: {time.time()-_t0:.1f}초)")
hold_ll, hold_ll_clim = eval_metrics(hold_df, label=f"홀드아웃 {HOLDOUT_WEEKS}주")

# 저장
if not hold_df.empty:
    hold_df.to_csv(OUTPUT_DIR / "full_features_holdout_probabilities.csv", index=False)
    print(f"\n  저장: full_features_holdout_probabilities.csv")

# ── 13. 확률 시계열 출력 ──────────────────────────────────────────────────────

print("\n■ 최근 20주 확률 시계열 (홀드아웃)")
if not hold_df.empty:
    disp = hold_df.copy()
    disp["실제"]    = disp["actual"].map(REGIME_NAMES)
    disp["예측"]    = disp["pred"].map(REGIME_NAMES)
    disp["P(강세)"] = disp["p_bull"].map("{:.2f}".format)
    disp["P(약세)"] = disp["p_bear"].map("{:.2f}".format)
    disp["P(횡보)"] = disp["p_sideways"].map("{:.2f}".format)
    print(disp[["date","실제","예측","P(강세)","P(약세)","P(횡보)"]].tail(20).to_string(index=False))

# ── 14. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

print(f"\n  [사용 특징변수 전체 목록]")
categories = {
    "이동평균/모멘텀": ["price_to_ma52", "ma_cross", "momentum_13w", "momentum_26w"],
    "거래량":          ["vol_ratio", "up_vol_ratio", "obv_chg_norm"],
    "섹터강도":        ["nasdaq_rel", "xli_rel"],
    "금리/역전":       ["term_spread_10y_2y", "inv_duration"],
    "신용/레버리지":   ["hy_spread", "nfci_leverage", "dHY"],
    "금융여건":        ["financial_conditions", "dFC"],
    "변동성":          ["vix"],
}
for cat, cols in categories.items():
    used = [FEAT_COLS.get(c, c) for c in cols if c in FEAT_COLS or f"d_{c}" in FEAT_COLS.values()]
    if used:
        print(f"  {cat:<14}: {', '.join(used)}")

print()
if wf_ll is not None:
    print(f"  워크포워드  log-loss={wf_ll:.4f}  vs 벤치마크={wf_ll_clim:.4f}")
if hold_ll is not None:
    print(f"  홀드아웃    log-loss={hold_ll:.4f}  vs 벤치마크={hold_ll_clim:.4f}")

print()
# 이전 모델과 비교
print("  [이전 모델 대비 개선 여부]")
print("  이전(거시변수만): 워크포워드 log-loss=1.3703, 홀드아웃=1.0676")
if wf_ll is not None and hold_ll is not None:
    wf_diff  = 1.3703 - wf_ll
    hd_diff  = 1.0676 - hold_ll
    print(f"  이번(기술+거래량): 워크포워드={wf_ll:.4f} ({'+' if wf_diff>0 else ''}{wf_diff:.4f}), "
          f"홀드아웃={hold_ll:.4f} ({'+' if hd_diff>0 else ''}{hd_diff:.4f})")

print(f"""
  [최종 해석]
  ① 이동평균·모멘텀·거래량 변수가 거시변수만 쓸 때 대비
     log-loss를 낮췄다면 → 평시 레짐 구분에 실질적 도움이 됨.
  ② 개선이 없다면 → 이 주간 빈도에서는 레짐 자체가 잘 구분되지 않음.
  ③ 약세 35주는 극단적 사건에 집중. 이진 분류(위기감지)로 별도 모델이 권장됨.
  ④ 레짐 라벨이 전체 표본 마르코프에서 나옴 → 약한 누수 존재.
""")

print("=" * 65)
print("분석 완료.")
print("=" * 65)
