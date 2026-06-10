# =============================================================================
# regime_binary_peacetime.py  —  평시(위기 제외) 강세 vs 횡보 이진 예측
#
# ★ 이 모델은 하락/위기를 예측하지 않습니다 ★
#   위기(약세) 구간을 학습·평가 모두에서 제외하고,
#   '평시'(강세+횡보) 안에서 향후 3개월이 강세인지 횡보인지만 예측합니다.
#
# 흐름:
#   1) 마르코프 레짐 전환 → 약세 주 식별
#   2) 약세 주 + 전방창에 약세가 포함된 주 제외
#   3) 남은 강세/횡보 2클래스로 이진 로지스틱 회귀
#   4) 워크포워드 검증 (확장 윈도우, 날짜 기준 컷오프)
#   5) 벤치마크(평시 사전확률)와 log-loss 비교
#
# ★ 누수 주의:
#   레짐 라벨이 전체 표본 마르코프에서 생성됨 → 약한 누수 존재.
#   결론 해석 시 주의. (엄격판: 폴드 내 마르코프 재추정 필요)
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
FWD_WEEKS       = 13    # 향후 몇 주의 레짐을 예측할 것인가
HOLDOUT_WEEKS   = 78
MIN_TRAIN_WEEKS = 260
REFIT_EVERY     = 4

# 이진 라벨 (약세 제외 후)
LABEL_BULL     = 0   # 강세
LABEL_SIDEWAYS = 1   # 횡보 (원래 2에서 리매핑)
LABEL_NAMES    = {LABEL_BULL: "강세", LABEL_SIDEWAYS: "횡보"}

OUTPUT_DIR = Path(__file__).parent

# ── 1. FRED 데이터 조회 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 평시 이진 레짐 예측 (강세 vs 횡보)")
print("  ※ 이 모델은 위기(약세)를 예측하지 않습니다")
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
    print(f"  △ NFCILEVERAGE 실패: {e} — financial_conditions 단독 사용")

time.sleep(0.4)
try:
    hy_raw = fred.get_series("BAA10Y", observation_start="1999-01-01")
    hy_raw.index = pd.to_datetime(hy_raw.index)
    print(f"  ✓ BAA10Y (HY 대체): {len(hy_raw)}행")
except Exception as e:
    print(f"  △ BAA10Y 실패: {e} — hy_spread 없이 진행")

# ── 2. CSV 로드 및 정렬 ────────────────────────────────────────────────────────

print("\n■ 2단계: CSV 로드 및 시계열 정렬")

df_csv = pd.read_csv(CSV_PATH, parse_dates=["date"])
df_csv = df_csv.sort_values("date").reset_index(drop=True)

df = df_csv[["date", RET_COL, "vix", "financial_conditions",
             "term_spread_10y_2y"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

if nfci_raw is not None:
    nfci_w = nfci_raw.resample("W-FRI").last().ffill()
    df["nfci_leverage"] = nfci_w.reindex(df.index, method="ffill")

if hy_raw is not None:
    hy_w = hy_raw.ffill().resample("W-FRI").last().ffill()
    df["hy_spread"] = hy_w.reindex(df.index, method="ffill")

df = df.reset_index()
print(f"  {len(df):,}주  ({df['date'].min().date()} ~ {df['date'].max().date()})")

# ── 3. 파생 변수 생성 ─────────────────────────────────────────────────────────

# inv_duration: 장단기 역전이 연속으로 지속된 주 수 (역전 없으면 0)
sp = df["term_spread_10y_2y"].values
inv_d = np.zeros(len(sp))
for i in range(1, len(sp)):
    if not np.isnan(sp[i]) and sp[i] < 0:
        inv_d[i] = inv_d[i - 1] + 1
    else:
        inv_d[i] = 0
df["inv_duration"] = inv_d

# 4주 모멘텀
if "hy_spread" in df.columns:
    df["dHY"] = df["hy_spread"] - df["hy_spread"].shift(4)
df["dFC"] = df["financial_conditions"] - df["financial_conditions"].shift(4)

# ── 4. 마르코프 레짐 전환 모델 ────────────────────────────────────────────────

print("\n■ 3단계: 마르코프 레짐 전환 (전체 표본, 약한 누수 존재)")

ret_s = df.set_index("date")[RET_COL].dropna()
print("  추정 중 (10~30초 소요)...")
t0 = time.time()
try:
    mod = MarkovRegression(ret_s, k_regimes=3, trend="c", switching_variance=True)
    res = mod.fit(search_reps=20, search_iter=100, disp=False)
    print(f"  완료 ({time.time()-t0:.0f}초), AIC={res.aic:.1f}")

    smoothed   = res.smoothed_marginal_probabilities
    raw_labels = smoothed.idxmax(axis=1)  # 각 주의 최빈 상태

    # 상태 통계 → 강세/약세/횡보 매핑
    state_stats = {}
    for r in range(3):
        m = raw_labels == r
        state_stats[r] = {"mean": ret_s[m].mean() * 100, "std": ret_s[m].std() * 100, "n": int(m.sum())}

    sorted_m = sorted(range(3), key=lambda r: state_stats[r]["mean"])
    bear_raw, sideways_raw, bull_raw = sorted_m   # 낮은→높은 평균 순

    # 원래 레짐 번호 → 이름 라벨: 강세=0, 약세=1, 횡보=2
    state_map = {bull_raw: 0, bear_raw: 1, sideways_raw: 2}
    regime_s  = raw_labels.map(state_map).rename("regime_3")

    print(f"\n  [레짐 통계]  (★ 전체 표본 추정 → 약한 누수)")
    for rname, ridx in [("강세", 0), ("약세", 1), ("횡보", 2)]:
        raw = [k for k, v in state_map.items() if v == ridx][0]
        s = state_stats[raw]
        print(f"    {rname}({ridx}): 평균={s['mean']:+.3f}%, 표준편차={s['std']:.3f}%, {s['n']}주")

    markov_ok = True
except Exception as e:
    print(f"  [오류] MarkovRegression 실패: {e}")
    # 백업: 수익률 분위수 라벨
    q33, q67 = ret_s.quantile(0.33), ret_s.quantile(0.67)
    regime_s = ret_s.apply(lambda r: 1 if r <= q33 else (0 if r >= q67 else 2)).rename("regime_3")
    markov_ok = False

# 메인 데이터프레임에 병합
df = df.merge(regime_s.reset_index(), on="date", how="left")

# ── 5. 위기 주 식별 및 제외 ────────────────────────────────────────────────────

print("\n■ 4단계: 위기(약세) 주 식별 및 평시 필터링")

# ① 현재 시점 t 자체가 약세 → 제외
# ② t+1..t+FWD_WEEKS 창에 약세가 하나라도 있으면 → 타깃 오염 → 제외
regimes = df["regime_3"].values

is_bear     = (regimes == 1).astype(float)                          # 약세 여부
window_bear = np.zeros(len(regimes))                                # 전방창 오염 여부

for t in range(len(regimes) - FWD_WEEKS):
    window = regimes[t + 1 : t + 1 + FWD_WEEKS]
    if np.any(window == 1):
        window_bear[t] = 1

peacetime_mask = (is_bear == 0) & (window_bear == 0)               # 둘 다 아닌 경우만 True
df["peacetime"] = peacetime_mask

total_bear    = int(is_bear.sum())
total_window  = int(((is_bear == 0) & (window_bear == 1)).sum())
total_peace   = int(peacetime_mask.sum())

print(f"  전체: {len(df)}주")
print(f"  제외 — 약세 주 자체: {total_bear}주, 전방창 오염: {total_window}주")
print(f"  평시 잔여: {total_peace}주  ({total_peace/len(df)*100:.1f}%)")

# ── 6. ADF 단위근 검정 ────────────────────────────────────────────────────────

print("\n■ 5단계: 단위근 검정 (ADF)")

RAW_FEATS = {}
cands = [
    ("term_spread_10y_2y", "term_spread"),
    ("inv_duration",       "inv_duration"),
    ("nfci_leverage",      "nfci_leverage"),
    ("financial_conditions","fin_cond"),
    ("vix",                "vix"),
]
if "hy_spread" in df.columns:
    cands += [("hy_spread", "hy_spread"), ("dHY", "dHY")]
cands.append(("dFC", "dFC"))

FEAT_COLS: dict[str, str] = {}   # {단축명: 실제 컬럼명}

for col, name in cands:
    if col not in df.columns or df[col].isna().all():
        continue
    clean = df[col].dropna()
    if len(clean) < 30:
        continue
    try:
        p = adfuller(clean, maxlag=12, autolag="AIC")[1]
        tag = "정상 ✓" if p <= 0.05 else "비정상 ⚠"
        print(f"  {name:<20}: p={p:.4f} [{tag}]")
        if p > 0.05:
            diff_col = f"d_{col}"
            df[diff_col] = df[col].diff()
            FEAT_COLS[name] = diff_col
            print(f"    → 1차 차분 → '{diff_col}'")
        else:
            FEAT_COLS[name] = col
    except Exception:
        FEAT_COLS[name] = col

print(f"\n  사용 컬럼 {len(FEAT_COLS)}개: {list(FEAT_COLS.values())}")

# ── 7. VIF 점검 ───────────────────────────────────────────────────────────────

print("\n■ 6단계: VIF 다공선성 점검")

vif_cols = [v for v in FEAT_COLS.values() if v in df.columns]
df_vif   = df[vif_cols].dropna()
if len(df_vif) > 10:
    X_vif = sm.add_constant(df_vif.astype(float))
    print(f"  {'변수':<28} VIF")
    for i, col in enumerate(df_vif.columns):
        try:
            v = variance_inflation_factor(X_vif.values, i + 1)
            flag = " ⚠" if v > 10 else (" △" if v > 5 else "")
            print(f"  {col:<28} {v:.2f}{flag}")
        except Exception:
            print(f"  {col:<28} 계산 불가")

# ── 8. 평시 전방창 이진 타깃 생성 ────────────────────────────────────────────

print("\n■ 7단계: 이진 타깃 생성 (평시 전방창 최빈 레짐)")
print(f"  target_t = t+1..t+{FWD_WEEKS} 평시 창의 최빈 레짐 (강세=0, 횡보=1)")

target = np.full(len(df), np.nan)

for t in range(len(df) - FWD_WEEKS):
    if not peacetime_mask[t]:
        continue   # 현재 시점이 평시가 아니면 스킵

    window_reg = regimes[t + 1 : t + 1 + FWD_WEEKS]
    # 전방창에 약세 있으면 이미 peacetime_mask=False → 여기 도달 안 함

    # 약세 제외 후 이진: 원래 0(강세)→0, 원래 2(횡보)→1
    peace_only = np.where(window_reg == 2, 1, window_reg)  # 2→1
    peace_only = peace_only[peace_only != 1 + 1]           # (약세=1은 제외 — 없어야 함)

    # 강세(0) / 횡보(1) 중 최빈값
    counts = np.bincount(peace_only.astype(int), minlength=2)
    target[t] = int(np.argmax(counts))

df["binary_target"] = target
n_valid = int(pd.notna(df["binary_target"]).sum())
bull_n   = int((df["binary_target"] == 0).sum())
side_n   = int((df["binary_target"] == 1).sum())
print(f"  유효 타깃: {n_valid}주  |  강세={bull_n}주 ({bull_n/n_valid*100:.1f}%)"
      f"  횡보={side_n}주 ({side_n/n_valid*100:.1f}%)")

# ── 9. lag-1 피처 생성 및 모델 데이터 준비 ────────────────────────────────────

LAG_FEAT_COLS = []
for name, raw_col in FEAT_COLS.items():
    lag_col = f"{raw_col}_l1"
    df[lag_col] = df[raw_col].shift(1)
    LAG_FEAT_COLS.append(lag_col)

keep = ["date", RET_COL, "regime_3", "peacetime", "binary_target"] + LAG_FEAT_COLS
df_model = df[[c for c in keep if c in df.columns]].copy()

# 평시이면서 타깃도 유효한 행만
df_peace = (
    df_model[df_model["peacetime"] & df_model["binary_target"].notna()]
    .dropna(subset=LAG_FEAT_COLS)
    .reset_index(drop=True)
)
df_peace["binary_target"] = df_peace["binary_target"].astype(int)

print(f"\n  lag-1 적용 후 평시 유효 행: {len(df_peace)}주")
print(f"  기간: {df_peace['date'].min().date()} ~ {df_peace['date'].max().date()}")

if len(df_peace) < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 10:
    print("[오류] 데이터 부족"); sys.exit(1)

# 홀드아웃: 마지막 HOLDOUT_WEEKS 평시 행
df_peace_wf   = df_peace.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_peace_full = df_peace.reset_index(drop=True)
print(f"  워크포워드: {len(df_peace_wf)}주  |  홀드아웃: {HOLDOUT_WEEKS}주")

# ── 10. 헬퍼 함수 ─────────────────────────────────────────────────────────────

def eval_binary(df_res: pd.DataFrame, label: str = "") -> tuple:
    """이진 분류 지표: log-loss, 정확도, 클래스별 성과"""
    if df_res.empty:
        return None, None
    y_true = df_res["actual"].values
    proba  = np.column_stack([df_res["p_bull"], df_res["p_sideways"]])
    clim   = np.column_stack([df_res["p_clim_bull"], df_res["p_clim_sideways"]])

    ll      = log_loss(y_true, proba,  labels=[0, 1])
    ll_clim = log_loss(y_true, clim,   labels=[0, 1])
    y_pred  = df_res["pred"].values
    acc     = (y_pred == y_true).mean() * 100
    # 다수 클래스 기저율 (항상 다수 클래스 예측)
    majority = np.bincount(y_true).argmax()
    acc_maj  = (y_true == majority).mean() * 100

    print(f"\n  [{label}]")
    print(f"  Log-loss  모델: {ll:.4f}  /  벤치마크(climatology): {ll_clim:.4f}")
    if ll < ll_clim - 0.01:
        print(f"  → ✓ 모델이 벤치마크보다 나음 (개선폭: {ll_clim-ll:.4f})")
    elif ll < ll_clim:
        print(f"  → △ 미미하게 나음 (개선폭 0.01 미만 — 노이즈 수준)")
    else:
        print(f"  → ✗ 모델이 벤치마크보다 못함 (차이: {ll-ll_clim:.4f})")

    print(f"  정확도   모델: {acc:.1f}%  /  다수클래스 기저율: {acc_maj:.1f}%")

    try:
        print(f"\n  [클래스별 성과]")
        print(classification_report(
            y_true, y_pred,
            target_names=["강세", "횡보"],
            labels=[0, 1],
            zero_division=0,
        ))
    except Exception as e:
        print(f"  분류 보고서 오류: {e}")

    print(f"  [혼동행렬 (행=실제, 열=예측)]")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"  {'':10}  예측강세  예측횡보")
    for i, row in enumerate(cm):
        print(f"  실제{'강세' if i==0 else '횡보'}     {row[0]:6}    {row[1]:6}")

    return ll, ll_clim


# ── 11. 워크포워드 실행 ────────────────────────────────────────────────────────

print("\n■ 8단계: 워크포워드 실행 (평시 행만, 날짜 기준 컷오프)")
print(f"  확장 윈도우 | 재적합={REFIT_EVERY}스텝 | 최소학습={MIN_TRAIN_WEEKS}주(평시)")
print(f"  날짜 컷오프: train ← date ≤ current_date − {FWD_WEEKS}주")

wf_records: list[dict] = []
last_model  = None
last_scaler = None
last_fit_step = -(REFIT_EVERY + 1)   # 첫 스텝에서 반드시 재적합

_t0 = time.time()
n_wf = len(df_peace_wf)
total_steps = n_wf - MIN_TRAIN_WEEKS
next_print  = max(30, total_steps // 10)

for step_idx, t in enumerate(range(MIN_TRAIN_WEEKS, n_wf)):
    current_date = df_peace_wf["date"].iloc[t]
    # 훈련 컷오프: 타깃 전방창이 완전히 과거여야 함
    cutoff_date  = current_date - pd.Timedelta(weeks=FWD_WEEKS)

    # 날짜 기준으로 훈련 데이터 선택
    train = df_peace_wf[df_peace_wf["date"] <= cutoff_date]
    if len(train) < MIN_TRAIN_WEEKS:
        continue

    y_train = train["binary_target"].values.astype(int)
    if len(np.unique(y_train)) < 2:   # 한 클래스만 있으면 스킵
        continue

    need_refit = ((step_idx - last_fit_step) >= REFIT_EVERY) or (last_model is None)

    if need_refit:
        scaler = StandardScaler()
        # 표준화는 훈련 구간에만 fit (누수 금지)
        X_tr = scaler.fit_transform(train[LAG_FEAT_COLS].astype(float))
        try:
            clf = LogisticRegression(
                class_weight="balanced",  # 강세/횡보 비율 차이 보정
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
            )
            clf.fit(X_tr, y_train)
            last_model     = clf
            last_scaler    = scaler
            last_fit_step  = step_idx
        except Exception:
            if last_model is None:
                continue

    # 현재 시점 예측
    X_test = last_scaler.transform(
        df_peace_wf[LAG_FEAT_COLS].iloc[[t]].astype(float)
    )
    try:
        proba      = last_model.predict_proba(X_test)[0]
        classes    = last_model.classes_
        prob_dict  = dict(zip(classes, proba))
        p_bull     = prob_dict.get(0, 0.0)
        p_sideways = prob_dict.get(1, 0.0)
        pred       = int(classes[np.argmax(proba)])
    except Exception:
        continue

    # 벤치마크: 훈련 구간 평시 사전확률 (climatology)
    clim = np.bincount(y_train, minlength=2) / len(y_train)

    wf_records.append({
        "date":          current_date,
        "actual":        int(df_peace_wf["binary_target"].iloc[t]),
        "pred":          pred,
        "p_bull":        p_bull,
        "p_sideways":    p_sideways,
        "p_clim_bull":   float(clim[0]),
        "p_clim_sideways": float(clim[1]),
    })

    if (step_idx + 1) >= next_print:
        pct = (step_idx + 1) / total_steps * 100
        print(f"    {step_idx+1:>4}/{total_steps} ({pct:.0f}%) — {time.time()-_t0:.0f}초")
        next_print += max(30, total_steps // 10)

wf_df = pd.DataFrame(wf_records)
print(f"\n  완료: {time.time()-_t0:.1f}초  ({len(wf_df)}스텝)")

# ── 12. 워크포워드 성과 ────────────────────────────────────────────────────────

print("\n■ 9단계: 워크포워드 성과")
wf_ll, wf_ll_clim = eval_binary(wf_df, label="워크포워드 구간")

# ── 13. 홀드아웃 검증 ─────────────────────────────────────────────────────────

print("\n■ 10단계: 최종 홀드아웃 검증")
print("  ※ 사양 선택에 한 번도 쓰지 않은 구간 — 정직한 성적")

hold_records: list[dict] = []
last_model_h  = None
last_scaler_h = None
last_fit_h    = -(REFIT_EVERY + 1)
holdout_start = len(df_peace_wf)

_t0 = time.time()
for step_h, t in enumerate(range(holdout_start, len(df_peace_full))):
    current_date = df_peace_full["date"].iloc[t]
    cutoff_date  = current_date - pd.Timedelta(weeks=FWD_WEEKS)
    train = df_peace_full[df_peace_full["date"] <= cutoff_date]
    if len(train) < MIN_TRAIN_WEEKS: continue
    y_train = train["binary_target"].values.astype(int)
    if len(np.unique(y_train)) < 2: continue

    need_refit = ((step_h - last_fit_h) >= REFIT_EVERY) or (last_model_h is None)
    if need_refit:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train[LAG_FEAT_COLS].astype(float))
        try:
            clf = LogisticRegression(class_weight="balanced", solver="lbfgs",
                                     max_iter=1000, C=1.0)
            clf.fit(X_tr, y_train)
            last_model_h  = clf
            last_scaler_h = scaler
            last_fit_h    = step_h
        except Exception:
            if last_model_h is None: continue

    X_test = last_scaler_h.transform(
        df_peace_full[LAG_FEAT_COLS].iloc[[t]].astype(float)
    )
    try:
        proba      = last_model_h.predict_proba(X_test)[0]
        classes    = last_model_h.classes_
        prob_dict  = dict(zip(classes, proba))
        p_bull     = prob_dict.get(0, 0.0)
        p_sideways = prob_dict.get(1, 0.0)
        pred       = int(classes[np.argmax(proba)])
    except Exception:
        continue

    clim = np.bincount(y_train, minlength=2) / len(y_train)
    hold_records.append({
        "date":          current_date,
        "actual":        int(df_peace_full["binary_target"].iloc[t]),
        "pred":          pred,
        "p_bull":        p_bull,
        "p_sideways":    p_sideways,
        "p_clim_bull":   float(clim[0]),
        "p_clim_sideways": float(clim[1]),
    })

hold_df = pd.DataFrame(hold_records)
print(f"  홀드아웃 예측 스텝: {len(hold_df)}주  (소요: {time.time()-_t0:.1f}초)")
hold_ll, hold_ll_clim = eval_binary(hold_df, label=f"홀드아웃 {HOLDOUT_WEEKS}주(평시)")

# 저장
if not hold_df.empty:
    hold_df.to_csv(OUTPUT_DIR / "peacetime_holdout_probabilities.csv", index=False)
    print(f"\n  저장: peacetime_holdout_probabilities.csv")

# ── 14. 확률 시계열 (최근 20행) ────────────────────────────────────────────────

print("\n■ 최근 확률 시계열 (홀드아웃)")
if not hold_df.empty:
    disp = hold_df.copy()
    disp["실제"]    = disp["actual"].map(LABEL_NAMES)
    disp["예측"]    = disp["pred"].map(LABEL_NAMES)
    disp["P(강세)"] = disp["p_bull"].map("{:.2f}".format)
    disp["P(횡보)"] = disp["p_sideways"].map("{:.2f}".format)
    print(disp[["date","실제","예측","P(강세)","P(횡보)"]].tail(20).to_string(index=False))

# ── 15. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

print(f"""
  ★★ 이 모델은 위기(약세) 구간을 예측하지 않습니다 ★★
     약세 주({total_bear}주)와 전방창 오염 주({total_window}주)를 완전히 제외했습니다.
     위기가 발생하는 시점을 예고하는 신호를 제공하지 않으며,
     평시(강세+횡보) 안에서의 레짐 구분만 시도합니다.
""")

if wf_ll is not None:
    print(f"  [워크포워드]  log-loss={wf_ll:.4f}  벤치마크={wf_ll_clim:.4f}")
if hold_ll is not None:
    print(f"  [홀드아웃]    log-loss={hold_ll:.4f}  벤치마크={hold_ll_clim:.4f}")

print()
# 최종 판정
if hold_ll is not None:
    if hold_ll < hold_ll_clim - 0.01:
        print("  ★ 결론: 선행지표가 평시 강세 vs 횡보를 climatology보다 잘 가립니다.")
        print("    → 단, '약한 누수'(전체 표본 레짐 라벨)를 감안하면 신중한 해석이 필요합니다.")
    elif hold_ll < hold_ll_clim:
        print("  ★ 결론: 모델이 벤치마크보다 아주 미미하게 나으나 노이즈 범위입니다.")
        print("    → 선행지표가 평시 레짐 구분에 뚜렷한 가치를 더한다고 보기 어렵습니다.")
    else:
        print("  ★ 결론: 선행지표가 평시 강세 vs 횡보를 climatology보다 잘 가리지 못합니다.")
        print("    → '평시에도 이 변수들로 레짐 구분 가치 확인 안 됨' — 유효한 발견입니다.")

print(f"""
  [주요 주의 사항]
  ① 레짐 라벨: 전체 표본 마르코프 → 약한 누수. 엄격판은 폴드 내 재추정 필요.
  ② 약세 제외 후 남은 평시 표본({len(df_peace)}주)에서 강세가 다수파이므로
     '모두 강세 예측'이 이미 높은 정확도를 줍니다. 정확도보다 log-loss를 봐야 합니다.
  ③ 평시 분류 모델과 위기 탐지 모델을 별도로 운용하는 것이 권장됩니다.
""")

print("=" * 65)
print("분석 완료.")
print("=" * 65)
