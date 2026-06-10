# =============================================================================
# regime_multinomial.py  —  강세/약세/횡보 3국면 다항 로지스틱 예측
#
# 목적:
#   향후 fwd_weeks(=13주, 약 3개월)의 '지배 레짐'을 예측한다.
#   출력: P(강세) · P(약세) · P(횡보) — 전환 신호 탐지용
#
# 흐름:
#   1) 마르코프 레짐 전환 모델 → 3-상태 라벨 부여
#   2) 타깃 = 향후 13주 구간의 최빈 레짐
#   3) 선행지표 변수 생성 (누수 금지)
#   4) 워크포워드 다항 로지스틱 회귀 (확장 윈도우)
#   5) 벤치마크(climatology)와 log-loss 비교
#
# ★ 누수 주의:
#   레짐 라벨은 전체 표본으로 한 번 추정함 → 미래 정보를 일부 참조하는
#   '약한 누수'가 있음. 1차 검증용으로 허용하되, 결론 해석 시 주의.
#   (엄격 버전: 각 폴드 내 데이터로 마르코프 모델 재추정 필요)
#
# 필요 라이브러리:
#   pip install pandas numpy statsmodels scikit-learn fredapi python-dotenv
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH   = "data/processed/final_dataset.csv"
RET_COL    = "sp500_log_return"   # 주간 로그수익률

FWD_WEEKS       = 13   # 향후 몇 주의 레짐을 예측할 것인가 (≈3개월)
HOLDOUT_WEEKS   = 78   # 최종 홀드아웃 주 수 (사양 선택에 절대 사용 안 함)
MIN_TRAIN_WEEKS = 260  # 최소 학습 주 수
REFIT_EVERY     = 4    # N스텝마다 모델 재적합

# 레짐 이름과 인덱스 (이 스크립트 전체에서 일관되게 사용)
REGIME_NAMES = {0: "강세", 1: "약세", 2: "횡보"}

OUTPUT_DIR = Path(__file__).parent

# ── 1. FRED 데이터 조회 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 3-레짐 다항 로지스틱 예측")
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
    print("[오류] fredapi 없음. pip install fredapi"); sys.exit(1)

nfci_raw = hy_raw = None
try:
    nfci_raw = fred.get_series("NFCILEVERAGE", observation_start="1999-01-01")
    nfci_raw.index = pd.to_datetime(nfci_raw.index)
    print(f"  ✓ NFCILEVERAGE: {len(nfci_raw)}행")
except Exception as e:
    print(f"  ✗ NFCILEVERAGE 실패: {e}")

time.sleep(0.4)
try:
    hy_raw = fred.get_series("BAA10Y", observation_start="1999-01-01")
    hy_raw.index = pd.to_datetime(hy_raw.index)
    print(f"  ✓ BAA10Y (HY 신용스프레드): {len(hy_raw)}행")
except Exception as e:
    print(f"  ✗ BAA10Y 실패: {e}")

# ── 2. CSV 로드 및 시계열 정렬 ────────────────────────────────────────────────

print("\n■ 2단계: CSV 로드 및 정렬")

df_csv = pd.read_csv(CSV_PATH, parse_dates=["date"])
df_csv = df_csv.sort_values("date").reset_index(drop=True)
print(f"  {len(df_csv):,}주  ({df_csv['date'].min().date()} ~ {df_csv['date'].max().date()})")

df = df_csv[["date", RET_COL, "vix", "financial_conditions",
             "term_spread_10y_2y"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

# FRED 시계열 주간 금요일로 정렬 후 JOIN
if nfci_raw is not None:
    nfci_w = nfci_raw.resample("W-FRI").last().ffill()
    df["nfci_leverage"] = nfci_w.reindex(df.index, method="ffill")
else:
    df["nfci_leverage"] = np.nan

if hy_raw is not None:
    hy_w = hy_raw.ffill().resample("W-FRI").last().ffill()
    df["hy_spread"] = hy_w.reindex(df.index, method="ffill")
else:
    df["hy_spread"] = np.nan

df = df.reset_index()

# ── 3. 파생 변수 생성 ─────────────────────────────────────────────────────────

print("\n■ 3단계: 파생 변수 생성 (누수 없음 — shift 적용)")

# inv_duration: term_spread < 0 이 연속으로 지속된 주 수 (역전 없으면 0)
spread = df["term_spread_10y_2y"].values
inv_dur = np.zeros(len(spread))
for i in range(1, len(spread)):
    if not np.isnan(spread[i]) and spread[i] < 0:
        inv_dur[i] = inv_dur[i - 1] + 1
    else:
        inv_dur[i] = 0
df["inv_duration"] = inv_dur

# 4주 모멘텀 변수
df["dHY"]  = df["hy_spread"]           - df["hy_spread"].shift(4)
df["dFC"]  = df["financial_conditions"] - df["financial_conditions"].shift(4)

print("  ✓ inv_duration, dHY(4주 HY 변화), dFC(4주 FC 변화) 생성")

# ── 4. ADF 단위근 검정 ────────────────────────────────────────────────────────

print("\n■ 4단계: 단위근 검정 (ADF)")
print("  p > 0.05 이면 차분 처리")

RAW_FEATURES = {
    "term_spread_10y_2y": "term_spread_10y_2y",
    "inv_duration":       "inv_duration",
    "hy_spread":          "hy_spread",
    "nfci_leverage":      "nfci_leverage",
    "financial_conditions":"financial_conditions",
    "vix":                "vix",
    "dHY":                "dHY",
    "dFC":                "dFC",
}
FEAT_COLS: dict[str, str] = {}   # {이름: 실제 컬럼명}

for name, col in RAW_FEATURES.items():
    if col not in df.columns or df[col].isna().all():
        print(f"  {name}: 컬럼 없음 → 건너뜀")
        continue
    clean = df[col].dropna()
    if len(clean) < 30:
        continue
    try:
        p = adfuller(clean, maxlag=12, autolag="AIC")[1]
        tag = "정상  ✓" if p <= 0.05 else "비정상 ⚠"
        print(f"  {name:<25}: p={p:.4f} [{tag}]")
        if p > 0.05:
            diff_col = f"d_{col}"
            df[diff_col] = df[col].diff()
            FEAT_COLS[name] = diff_col
            print(f"    → 1차 차분 → '{diff_col}' 사용")
        else:
            FEAT_COLS[name] = col
    except Exception:
        FEAT_COLS[name] = col

print(f"\n  최종 특징변수 {len(FEAT_COLS)}개: {list(FEAT_COLS.values())}")

# ── 5. 마르코프 레짐 전환 모델 → 레짐 라벨 부여 ──────────────────────────────

print("\n■ 5단계: 마르코프 레짐 전환 모델 (3-상태, 전체 표본)")
print("  ★ 전체 표본으로 추정 → 약한 누수 존재. 결론 해석 시 주의.")

# 수익률 시계열 준비 (결측 제거)
ret_series = df.set_index("date")[RET_COL].dropna()

print("  MarkovRegression 추정 중 (시간이 걸릴 수 있습니다)...")
t0 = time.time()

try:
    mod = MarkovRegression(
        ret_series,
        k_regimes=3,
        trend="c",               # 레짐별 평균(상수항)
        switching_variance=True, # 레짐별 분산 허용
    )
    res_markov = mod.fit(
        search_reps=20,     # 20번의 랜덤 초기값으로 전역 최적해 탐색
        search_iter=100,
        disp=False,
    )
    print(f"  완료: {time.time()-t0:.0f}초, AIC={res_markov.aic:.1f}")
    markov_ok = True
except Exception as e:
    print(f"  [오류] MarkovRegression 실패: {e}")
    print("  → 백업: 수익률 분위수로 레짐 수동 라벨링")
    markov_ok = False

# 레짐 라벨 생성
if markov_ok:
    # 각 시점의 최빈 상태 (smoothed probabilities 기반)
    smoothed = res_markov.smoothed_marginal_probabilities   # DataFrame: columns=0,1,2
    raw_labels = smoothed.idxmax(axis=1)                   # 각 시점의 최빈 상태 번호

    # 상태별 통계 계산 → 강세/약세/횡보 매핑
    state_stats = {}
    for r in range(3):
        mask = raw_labels == r
        vals = ret_series[mask]
        state_stats[r] = {
            "mean": vals.mean() * 100,   # %
            "std":  vals.std()  * 100,
            "n":    mask.sum(),
        }

    # 평균 수익률 기준으로 정렬: 높은 순 → 강세(0), 낮은 순 → 약세(1), 중간 → 횡보(2)
    sorted_by_mean = sorted(state_stats.keys(), key=lambda r: state_stats[r]["mean"])
    bear_raw     = sorted_by_mean[0]   # 평균 가장 낮음 → 약세
    bull_raw     = sorted_by_mean[2]   # 평균 가장 높음 → 강세
    sideways_raw = sorted_by_mean[1]   # 중간 → 횡보

    state_map = {bull_raw: 0, bear_raw: 1, sideways_raw: 2}
    regime_series = raw_labels.map(state_map).rename("regime")

    print("\n  [레짐별 추정 통계]")
    print(f"  {'레짐':<8} {'추정평균(%)':<14} {'추정표준편차(%)':<16} {'주 수'}")
    for r_idx, r_name in REGIME_NAMES.items():
        raw_r = [k for k, v in state_map.items() if v == r_idx][0]
        s = state_stats[raw_r]
        print(f"  {r_name:<8} {s['mean']:>+10.3f}%    {s['std']:>12.3f}%    {s['n']}")

    print(f"\n  [참고] 사용자 제시 기준: 강세+0.30% / 약세-0.45% / 횡보+0.02%")

else:
    # 백업: 수익률 분위수로 라벨 (q33=약세, q67=강세, 나머지=횡보)
    q33 = ret_series.quantile(0.33)
    q67 = ret_series.quantile(0.67)
    def manual_regime(r):
        if r <= q33: return 1   # 약세
        elif r >= q67: return 0  # 강세
        else: return 2            # 횡보
    regime_series = ret_series.apply(manual_regime).rename("regime")
    print("  수동 라벨 사용 (분위수 기반)")

# 레짐 시리즈를 메인 데이터프레임에 병합
df = df.merge(
    regime_series.reset_index().rename(columns={"date": "date", "regime": "regime"}),
    on="date", how="left"
)

print(f"\n  레짐 분포:")
vc = df["regime"].value_counts().sort_index()
for r, n in vc.items():
    print(f"    {REGIME_NAMES.get(int(r),'?')}({int(r)}): {n}주 ({n/len(df)*100:.1f}%)")

# ── 6. VIF 점검 ────────────────────────────────────────────────────────────────

print("\n■ 6단계: VIF 다공선성 점검")

vif_cols = [v for v in FEAT_COLS.values() if v in df.columns]
df_vif = df[vif_cols].dropna()
if len(df_vif) > 10:
    X_vif = sm.add_constant(df_vif.astype(float))
    print(f"  {'변수':<30} VIF")
    for i, col in enumerate(df_vif.columns):
        try:
            v = variance_inflation_factor(X_vif.values, i + 1)
            flag = " ⚠ 중복 주의" if v > 10 else (" △" if v > 5 else "")
            print(f"  {col:<30} {v:.2f}{flag}")
        except Exception:
            print(f"  {col:<30} 계산 불가")

# ── 7. 타깃 변수 생성 ─────────────────────────────────────────────────────────

print("\n■ 7단계: 타깃 변수 생성")
print(f"  target_t = t+1 ~ t+{FWD_WEEKS} 구간의 최빈 레짐 (mode)")
print(f"  마지막 {FWD_WEEKS}주는 전방창 부족 → 타깃 결측 (제거)")
print("  [대안 주석] t+fwd_weeks 시점 레짐만 사용하려면:")
print("    target_t = regime.shift(-fwd_weeks) 로 교체 가능")

regime_arr = df["regime"].values

def rolling_mode(arr, fwd):
    """각 시점 t에서 t+1..t+fwd 구간의 최빈값을 반환"""
    n = len(arr)
    result = np.full(n, np.nan)
    for t in range(n - fwd):
        window = arr[t + 1 : t + 1 + fwd]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        counts = np.bincount(valid.astype(int), minlength=3)
        result[t] = int(np.argmax(counts))
    return result

df["target"] = rolling_mode(regime_arr, FWD_WEEKS)
n_valid_target = df["target"].notna().sum()
print(f"  타깃 유효 행: {n_valid_target}주")

# ── 8. 모델 데이터 준비 ────────────────────────────────────────────────────────

print("\n■ 8단계: 특징변수 lag-1 적용 및 데이터 정리")

# 모든 특징변수에 lag-1 적용 (시점 t에서 t-1 값만 사용 → 누수 없음)
LAG_FEAT_COLS = []
for name, raw_col in FEAT_COLS.items():
    lag_col = f"{raw_col}_l1"
    df[lag_col] = df[raw_col].shift(1)   # 1주 lag
    LAG_FEAT_COLS.append(lag_col)

print(f"  lag-1 적용 특징변수 ({len(LAG_FEAT_COLS)}개):")
for c in LAG_FEAT_COLS:
    print(f"    {c}")

keep = ["date", RET_COL, "regime", "target"] + LAG_FEAT_COLS
df_model = df[keep].dropna().reset_index(drop=True)

print(f"\n  결측 제거 후: {len(df_model)}주")
print(f"  기간: {df_model['date'].min().date()} ~ {df_model['date'].max().date()}")

if len(df_model) < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + FWD_WEEKS + 10:
    print("[오류] 데이터 부족"); sys.exit(1)

# 홀드아웃 분리
df_wf   = df_model.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_full = df_model.reset_index(drop=True)
print(f"  워크포워드: {len(df_wf)}주  |  홀드아웃: {HOLDOUT_WEEKS}주")

# ── 9. 워크포워드 다항 로지스틱 회귀 ─────────────────────────────────────────

print("\n■ 9단계: 워크포워드 실행")
print(f"  확장 윈도우 | 재적합={REFIT_EVERY}스텝 | 최소학습={MIN_TRAIN_WEEKS}주")
print(f"  모델: LogisticRegression(multinomial, class_weight='balanced')")
print(f"  벤치마크: 훈련 구간 레짐 빈도(climatology)")

# 워크포워드에서 training cutoff = t - FWD_WEEKS (타깃이 완전히 확인된 마지막 행)
# 즉 t 시점의 예측에 쓰는 훈련 데이터는 0..t-FWD_WEEKS 까지

wf_records = []   # {date, actual, pred, p_bull, p_bear, p_sideways, p_clim_bull, ...}
last_model  = None
last_scaler = None

_t0 = time.time()
n_wf = len(df_wf)
total_steps = n_wf - MIN_TRAIN_WEEKS
next_print = max(50, total_steps // 10)

for idx, t in enumerate(range(MIN_TRAIN_WEEKS, n_wf)):
    # 훈련 구간: 타깃이 완전히 확인된 마지막 행까지
    train_end = t - FWD_WEEKS
    if train_end < MIN_TRAIN_WEEKS:
        continue

    y_actual = df_wf["target"].iloc[t]
    if np.isnan(y_actual):
        continue
    y_actual = int(y_actual)

    train = df_wf.iloc[:train_end + 1]
    y_train = train["target"].dropna()
    train = train.loc[y_train.index]

    if len(np.unique(y_train)) < 2 or len(y_train) < 50:
        continue

    # 재적합 여부 결정
    need_refit = (idx % REFIT_EVERY == 0) or (last_model is None)

    if need_refit:
        # 표준화: 훈련 구간에만 fit (누수 금지)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[LAG_FEAT_COLS].astype(float))
        y_tr = y_train.values.astype(int)

        # 다항 로지스틱 회귀 (class_weight='balanced' → 약세 클래스 보정)
        try:
            # sklearn 1.7+ 에서 multi_class 파라미터 제거됨 → 자동으로 다중분류 처리
            clf = LogisticRegression(
                class_weight="balanced",   # 약세처럼 드문 클래스 보정
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
            )
            clf.fit(X_tr, y_tr)
            last_model  = clf
            last_scaler = scaler
        except Exception as e:
            if last_model is None:
                continue

    # 현재 시점 예측
    X_test = df_wf[LAG_FEAT_COLS].iloc[[t]].astype(float)
    X_test_scaled = last_scaler.transform(X_test)

    try:
        proba = last_model.predict_proba(X_test_scaled)[0]
        classes = last_model.classes_
        # 클래스 순서가 다를 수 있으므로 인덱스로 접근
        prob_dict = dict(zip(classes, proba))
        p_bull     = prob_dict.get(0, 0.0)
        p_bear     = prob_dict.get(1, 0.0)
        p_sideways = prob_dict.get(2, 0.0)
        pred_class = int(classes[np.argmax(proba)])
    except Exception:
        continue

    # 벤치마크: 훈련 구간 레짐 빈도 (climatology)
    clim_counts = np.bincount(y_train.astype(int), minlength=3)
    clim_proba  = clim_counts / clim_counts.sum()

    wf_records.append({
        "date":        df_wf["date"].iloc[t],
        "actual":      y_actual,
        "pred":        pred_class,
        "p_bull":      p_bull,
        "p_bear":      p_bear,
        "p_sideways":  p_sideways,
        "p_clim_bull":     clim_proba[0],
        "p_clim_bear":     clim_proba[1],
        "p_clim_sideways": clim_proba[2],
    })

    if (idx + 1) >= next_print:
        pct = (idx + 1) / total_steps * 100
        print(f"    {idx+1:>4}/{total_steps} ({pct:.0f}%) — {time.time()-_t0:.0f}초")
        next_print += max(50, total_steps // 10)

wf_df = pd.DataFrame(wf_records)
print(f"\n  워크포워드 완료: {time.time()-_t0:.1f}초  ({len(wf_df)}스텝)")

# ── 10. 워크포워드 구간 성과 측정 ─────────────────────────────────────────────

print("\n■ 10단계: 워크포워드 성과")

def eval_metrics(df_res, label=""):
    """다중클래스 log-loss, 정확도, 약세 재현율 계산"""
    if df_res.empty:
        return None, None
    y_true = df_res["actual"].astype(int).values
    proba_cols = [df_res["p_bull"].values, df_res["p_bear"].values, df_res["p_sideways"].values]
    proba_mat  = np.column_stack(proba_cols)

    # log-loss
    ll = log_loss(y_true, proba_mat, labels=[0, 1, 2])

    # 벤치마크 log-loss
    clim_mat = np.column_stack([df_res["p_clim_bull"], df_res["p_clim_bear"], df_res["p_clim_sideways"]])
    ll_clim  = log_loss(y_true, clim_mat, labels=[0, 1, 2])

    y_pred = df_res["pred"].astype(int).values

    print(f"\n  [{label}]")
    print(f"  Log-loss — 모델: {ll:.4f}  /  벤치마크(climatology): {ll_clim:.4f}")
    if ll < ll_clim:
        print(f"  → ✓ 모델이 벤치마크보다 나음 (차이: {ll_clim-ll:.4f})")
    else:
        print(f"  → ✗ 모델이 벤치마크보다 못함 (차이: {ll-ll_clim:.4f})")

    print(f"\n  전체 정확도: {(y_pred == y_true).mean()*100:.1f}%")
    print(f"  (※ 정확도만 보지 말 것 — 약세 클래스가 적어 불균형)")

    print(f"\n  [분류 보고서]")
    target_names = [REGIME_NAMES[i] for i in sorted(REGIME_NAMES.keys())]
    try:
        print(classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0,
        ))
    except Exception as e:
        print(f"  분류 보고서 오류: {e}")

    print(f"  [혼동행렬 (행=실제, 열=예측)]")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    header = f"  {'':10}" + "".join(f"예측{REGIME_NAMES[j]:4}" for j in range(3))
    print(header)
    for i, row in enumerate(cm):
        print(f"  실제{REGIME_NAMES[i]:4}  " + "  ".join(f"{v:6}" for v in row))

    return float(ll), float(ll_clim)

wf_ll, wf_ll_clim = eval_metrics(wf_df, label="워크포워드 구간")

# ── 11. 최종 홀드아웃 검증 ────────────────────────────────────────────────────

print("\n■ 11단계: 최종 홀드아웃 검증")
print(f"  ※ 사양 선택에 한 번도 쓰지 않은 구간 — 정직한 성적")

hold_records = []
last_model_h  = None
last_scaler_h = None
holdout_start = len(df_wf)

_t0 = time.time()
for idx_h, t in enumerate(range(holdout_start, len(df_full))):
    train_end = t - FWD_WEEKS
    if train_end < MIN_TRAIN_WEEKS:
        continue

    y_actual = df_full["target"].iloc[t]
    if np.isnan(y_actual):
        continue
    y_actual = int(y_actual)

    train = df_full.iloc[:train_end + 1]
    y_train = train["target"].dropna()
    train   = train.loc[y_train.index]
    if len(np.unique(y_train)) < 2 or len(y_train) < 50:
        continue

    need_refit = (idx_h % REFIT_EVERY == 0) or (last_model_h is None)
    if need_refit:
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(train[LAG_FEAT_COLS].astype(float))
        y_tr   = y_train.values.astype(int)
        try:
            clf = LogisticRegression(
                class_weight="balanced",
                solver="lbfgs", max_iter=1000, C=1.0,
            )
            clf.fit(X_tr, y_tr)
            last_model_h  = clf
            last_scaler_h = scaler
        except Exception:
            if last_model_h is None: continue

    X_test = df_full[LAG_FEAT_COLS].iloc[[t]].astype(float)
    X_test_s = last_scaler_h.transform(X_test)

    try:
        proba      = last_model_h.predict_proba(X_test_s)[0]
        classes    = last_model_h.classes_
        prob_dict  = dict(zip(classes, proba))
        p_bull     = prob_dict.get(0, 0.0)
        p_bear     = prob_dict.get(1, 0.0)
        p_sideways = prob_dict.get(2, 0.0)
        pred_class = int(classes[np.argmax(proba)])
    except Exception:
        continue

    clim_counts = np.bincount(y_train.astype(int), minlength=3)
    clim_proba  = clim_counts / clim_counts.sum()

    hold_records.append({
        "date":        df_full["date"].iloc[t],
        "actual":      y_actual,
        "pred":        pred_class,
        "p_bull":      p_bull,
        "p_bear":      p_bear,
        "p_sideways":  p_sideways,
        "p_clim_bull":     clim_proba[0],
        "p_clim_bear":     clim_proba[1],
        "p_clim_sideways": clim_proba[2],
    })

hold_df = pd.DataFrame(hold_records)
print(f"  홀드아웃 예측 스텝: {len(hold_df)}주  (소요: {time.time()-_t0:.1f}초)")

if not hold_df.empty:
    hold_result = eval_metrics(hold_df, label=f"홀드아웃 {HOLDOUT_WEEKS}주")

    # 확률 시계열 저장
    hold_df.to_csv(OUTPUT_DIR / "regime_holdout_probabilities.csv", index=False)
    print(f"\n  저장: regime_holdout_probabilities.csv")

# ── 12. 확률 시계열 샘플 출력 ─────────────────────────────────────────────────

print("\n■ 12단계: 최근 확률 시계열 (홀드아웃 구간)")
if not hold_df.empty:
    disp = hold_df[["date","actual","pred","p_bull","p_bear","p_sideways"]].copy()
    disp["실제"]  = disp["actual"].map(REGIME_NAMES)
    disp["예측"]  = disp["pred"].map(REGIME_NAMES)
    disp["P(강세)"] = disp["p_bull"].map("{:.2f}".format)
    disp["P(약세)"] = disp["p_bear"].map("{:.2f}".format)
    disp["P(횡보)"] = disp["p_sideways"].map("{:.2f}".format)
    print(disp[["date","실제","예측","P(강세)","P(약세)","P(횡보)"]].tail(20).to_string(index=False))

# ── 13. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

if wf_ll is not None:
    print(f"\n  [워크포워드]  log-loss={wf_ll:.4f}  vs 벤치마크={wf_ll_clim:.4f}")

if not hold_df.empty and len(hold_df) > 5:
    y_h = hold_df["actual"].astype(int).values
    proba_h = np.column_stack([hold_df["p_bull"], hold_df["p_bear"], hold_df["p_sideways"]])
    clim_h  = np.column_stack([hold_df["p_clim_bull"], hold_df["p_clim_bear"], hold_df["p_clim_sideways"]])
    ll_h    = log_loss(y_h, proba_h, labels=[0, 1, 2])
    ll_h_c  = log_loss(y_h, clim_h,  labels=[0, 1, 2])
    print(f"  [홀드아웃]    log-loss={ll_h:.4f}  vs 벤치마크={ll_h_c:.4f}")

    # 약세 재현율
    bear_mask   = y_h == 1
    if bear_mask.sum() > 0:
        bear_pred   = (hold_df["pred"].astype(int).values == 1)
        bear_recall = (bear_pred & bear_mask).sum() / bear_mask.sum()
        print(f"\n  약세 실제 발생: {bear_mask.sum()}주 / 예측 적중: {(bear_pred & bear_mask).sum()}주")
        print(f"  약세 재현율(recall): {bear_recall*100:.1f}%")
    else:
        print(f"\n  홀드아웃 기간 중 약세 레짐 없음")

print(f"""
  [주요 주의 사항]
  ① 레짐 라벨은 전체 표본으로 추정 → 미래 정보를 일부 참조하는 '약한 누수'.
     완전 누수 제거 버전은 각 폴드에서 마르코프 모델 재추정 필요 (훨씬 느림).
  ② 약세 클래스는 표본이 적어 통계적 증거가 약할 수 있음. 재현율을 우선 확인.
  ③ log-loss > 벤치마크이면 "이 선행지표들은 레짐 예측에 가치를 더하지 못함"
     → 그것도 유효한 발견.
  ④ 향후 개선: PCA 합성 특징, 비선형 모델(GBM), 또는 폴드 내 마르코프 재추정.
""")

print("=" * 65)
print("분석 완료.")
print("=" * 65)
