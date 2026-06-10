# =============================================================================
# multihorizon_regression.py  —  다중 기간 S&P500 수익률 회귀 예측
#
# 목표: 향후 4·8·13·26주 수익률을 3개 모델로 예측
#
# 모델: LightGBM  /  XGBoost  /  Random Forest
#
# 입력 변수 (12개, 모두 lag-1):
#   거시·금융여건: NFCI(financial_conditions), NFCILEVERAGE(Margin Debt 프록시)
#   신용·위험  : HY Spread(BAA10Y), VIX, VIX 4주 변화율(Term Structure 프록시)
#   환율·금리  : DXY, 10Y 국채수익률
#   가격·모멘텀: SPX 13주 수익률, SPX 26주 수익률, MA200 괴리율(40주 이평)
#   거래량     : 상승일 거래량 비율(up_vol_ratio)
#   AR항       : 직전 주 수익률
#
# ★ VIX Term Structure: VXVCLS(3M VIX)는 2007~부터 존재.
#   2007 이전은 vix/vix.shift(4)-1(4주 VIX 변화율)로 백필.
# ★ Margin Debt: FINRA 데이터 FRED에 없음 → NFCILEVERAGE 레버리지 서브인덱스 대체.
#
# 검증: 워크포워드 확장 윈도우 (홀드아웃=78주, 최소학습=260주)
# 지표: R²_oos (vs 역사적 평균), MAE, RMSE, 방향 적중률
#
# 필요 라이브러리:
#   pip install pandas numpy scikit-learn lightgbm xgboost fredapi python-dotenv
# =============================================================================

import os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH        = "data/processed/final_dataset.csv"
HORIZONS        = [4, 8, 13, 26]    # 예측 기간 (주)
HOLDOUT_WEEKS   = 78
MIN_TRAIN_WEEKS = 260
REFIT_EVERY     = 8                 # 트리모델은 8스텝마다 재적합 (속도/정확도 균형)

OUTPUT_DIR = Path(__file__).parent

# ── 1. FRED 데이터 조회 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 다중 기간 수익률 회귀 예측 (4·8·13·26주)")
print("  모델: LightGBM / XGBoost / Random Forest")
print("=" * 65)

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("[오류] FRED_API_KEY 없음"); sys.exit(1)

try:
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)
except ImportError:
    print("[오류] fredapi 없음. pip install fredapi"); sys.exit(1)

print("\n■ 1단계: FRED 데이터 조회")
fred_data = {}
fetch_list = {
    "NFCILEVERAGE": "NFCILEVERAGE (Margin Debt 프록시, 주간)",
    "BAA10Y":       "BAA10Y (HY Spread 대체, 일별)",
    "VXVCLS":       "VXVCLS (VIX 3M, 일별, 2007~)",
    "DGS10":        "DGS10 (10Y 국채수익률, 일별)",
}
for sid, desc in fetch_list.items():
    try:
        s = fred.get_series(sid, observation_start="1999-01-01")
        s.index = pd.to_datetime(s.index)
        fred_data[sid] = s
        print(f"  ✓ {sid} [{desc}]: {len(s)}행, ~ {s.index[-1].date()}")
    except Exception as e:
        print(f"  △ {sid}: 실패 ({e})")
    time.sleep(0.3)

# ── 2. CSV 로드 ────────────────────────────────────────────────────────────────

print("\n■ 2단계: CSV 로드")

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"  {len(df):,}주  ({df['date'].min().date()} ~ {df['date'].max().date()})")

# 필요 컬럼 확인
for col in ["sp500", "sp500_log_return", "vix", "financial_conditions",
            "dxy", "treasury_10y", "sp500_volume"]:
    if col not in df.columns:
        print(f"  [오류] '{col}' 없음"); sys.exit(1)

df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

# FRED 시계열 주간 정렬 후 JOIN
def to_weekly(s):
    return s.ffill().resample("W-FRI").last().ffill()

if "NFCILEVERAGE" in fred_data:
    df["nfci_leverage"] = to_weekly(fred_data["NFCILEVERAGE"]).reindex(df.index, method="ffill")
if "BAA10Y" in fred_data:
    df["hy_spread"]     = to_weekly(fred_data["BAA10Y"]).reindex(df.index, method="ffill")
if "VXVCLS" in fred_data:
    df["vix3m"]         = to_weekly(fred_data["VXVCLS"]).reindex(df.index, method="ffill")
if "DGS10" in fred_data:
    df["treasury_10y_fred"] = to_weekly(fred_data["DGS10"]).reindex(df.index, method="ffill")

df = df.reset_index()

# ── 3. 특징변수 생성 ──────────────────────────────────────────────────────────

print("\n■ 3단계: 특징변수 생성")

p = df["sp500"]
v = df["sp500_volume"]
r = df["sp500_log_return"]

# ① NFCI (거시·금융여건)
df["nfci"]         = df["financial_conditions"]

# ② VIX term structure = VIX / VIX3M
# VXVCLS가 없는 2007 이전은 VIX 4주 변화율로 백필
if "vix3m" in df.columns:
    df["vix_ts"] = df["vix"] / df["vix3m"]           # > 1 = 백워데이션(공포)
    vix_ts_proxy = df["vix"] / df["vix"].shift(4)     # 백필용 프록시
    df["vix_ts"]  = df["vix_ts"].fillna(vix_ts_proxy) # NaN을 프록시로 대체
    print("  ✓ vix_ts: VXVCLS 기반 (2007~), 이전은 VIX 4주 변화율 백필")
else:
    df["vix_ts"]  = df["vix"] / df["vix"].shift(4)    # VIX 4주 변화율만
    print("  ✓ vix_ts: VIX 4주 변화율 (VXVCLS 없음)")

# ③ SPX 13주·26주 수익률 (모멘텀)
df["ret_13w"] = p / p.shift(13) - 1
df["ret_26w"] = p / p.shift(26) - 1

# ④ MA200 괴리율 (200일 ≈ 40주)
df["ma200_dev"] = p / p.rolling(40).mean() - 1

# ⑤ HY Spread: hy_spread (BAA10Y) or treasury_10y from CSV as fallback
if "hy_spread" not in df.columns:
    df["hy_spread"] = df["treasury_10y"]  # 신용스프레드 없을 때 단기 대체

# ⑥ 10Y 금리: FRED 값 우선, 없으면 CSV
if "treasury_10y_fred" in df.columns:
    df["y10"] = df["treasury_10y_fred"].fillna(df["treasury_10y"])
else:
    df["y10"] = df["treasury_10y"]

# ⑦ 상승일 거래량 비율 (거래량 확인 지표)
df["up_vol"] = (v.where(r > 0, 0).rolling(13).sum()) / v.rolling(13).sum()

# ⑧ AR항: 직전 주 수익률
df["ar1"] = r.shift(0)   # 현재 주 값, lag-1 적용 후 t-1 수익률

print("  ✓ 전체 특징변수 생성 완료")

# ── 4. 특징변수 목록 및 lag-1 적용 ────────────────────────────────────────────

# 사용할 원천 컬럼 목록
RAW_FEATS = {
    "nfci":         "NFCI (거시 금융여건)",
    "nfci_leverage":"NFCILEVERAGE (레버리지/Margin 프록시)",
    "vix":          "VIX (공포지수)",
    "vix_ts":       "VIX Term Structure (VIX/VIX3M)",
    "hy_spread":    "HY Spread (BAA10Y)",
    "dxy":          "DXY (달러인덱스)",
    "y10":          "10Y 국채수익률",
    "ret_13w":      "SPX 13주 수익률",
    "ret_26w":      "SPX 26주 수익률",
    "ma200_dev":    "MA200 괴리율 (40주 이평)",
    "up_vol":       "상승일 거래량 비율",
    "ar1":          "직전주 수익률 (AR항)",
}

# 없는 컬럼 제거
RAW_FEATS = {k: v for k, v in RAW_FEATS.items() if k in df.columns}

LAG_FEATS = [f"{c}_l1" for c in RAW_FEATS]   # 실제 모델 투입 컬럼명

print("\n  [특징변수 최종 목록]")
for raw, desc in RAW_FEATS.items():
    lag = f"{raw}_l1"
    df[lag] = df[raw].shift(1)   # ← 1주 lag 적용 (누수 없음)
    print(f"  {lag:<28} {desc}")

print(f"\n  총 {len(LAG_FEATS)}개 특징변수 (모두 lag-1)")

# ── 5. 전진(Forward) 수익률 타깃 생성 ────────────────────────────────────────

print("\n■ 4단계: 전진 수익률 타깃 생성")
print("  fwd_h_t = log(sp500_{t+h} / sp500_t)")

for h in HORIZONS:
    # h주 후 가격 대비 현재 → 합산 로그수익률
    df[f"fwd_{h}w"] = np.log(p.shift(-h) / p)
    n_v = df[f"fwd_{h}w"].notna().sum()
    print(f"  fwd_{h}w: {n_v}개 유효  "
          f"(평균={df[f'fwd_{h}w'].mean()*100:.2f}%, "
          f"표준편차={df[f'fwd_{h}w'].std()*100:.2f}%)")

# ── 6. 데이터 정리 및 홀드아웃 분리 ─────────────────────────────────────────

print("\n■ 5단계: 데이터 정리 및 홀드아웃 분리")

tgt_cols  = [f"fwd_{h}w" for h in HORIZONS]
keep      = ["date", "sp500_log_return"] + LAG_FEATS + tgt_cols
df_model  = df[[c for c in keep if c in df.columns]].copy()

# 특징변수 결측만 제거 (타깃은 horizon별로 별도 dropna)
df_model  = df_model.dropna(subset=LAG_FEATS).reset_index(drop=True)

print(f"  특징변수 결측 제거 후: {len(df_model)}주")
print(f"  기간: {df_model['date'].min().date()} ~ {df_model['date'].max().date()}")

df_wf   = df_model.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_full = df_model.reset_index(drop=True)
n_wf    = len(df_wf)
print(f"  워크포워드: {n_wf}주  |  홀드아웃: {HOLDOUT_WEEKS}주")

# ── 7. 모델 팩토리 ───────────────────────────────────────────────────────────

def make_model(name: str):
    """모델 이름으로 인스턴스 반환. 과적합 방지 파라미터 설정."""
    if name == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, num_leaves=15,
            min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
    elif name == "xgb":
        return xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, verbosity=0,
        )
    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=200, max_depth=5,
            min_samples_leaf=10, max_features=0.7,
            random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")

MODEL_NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random Forest"}

# ── 8. 워크포워드 함수 ────────────────────────────────────────────────────────

def run_wf_regression(df_src, df_train_src, horizon, model_name, start_t):
    """
    단일 (horizon, model) 조합 워크포워드 실행.
    반환: DataFrame(date, actual, pred, benchmark)
    """
    tgt_col  = f"fwd_{horizon}w"
    records  = []
    last_clf = None
    last_sc  = None
    last_fit = -(REFIT_EVERY + 1)
    n        = len(df_src)

    for idx, t in enumerate(range(start_t, n)):
        # 훈련 컷오프: fwd_h_i가 완전히 과거여야 함 (i + h ≤ t → i ≤ t - h)
        train_end = t - horizon
        if train_end < MIN_TRAIN_WEEKS:
            continue

        # 현재 시점 타깃 (알려진 미래값 — 평가용)
        y_actual = df_src[tgt_col].iloc[t]
        if np.isnan(y_actual):
            continue

        train = df_train_src.iloc[:train_end + 1]
        # 훈련 데이터에서 타깃이 유효한 행만
        valid_mask = train[tgt_col].notna()
        train_v = train[valid_mask]
        if len(train_v) < 50:
            continue

        y_tr = train_v[tgt_col].values
        X_tr_raw = train_v[LAG_FEATS].astype(float).values

        need_refit = ((idx - last_fit) >= REFIT_EVERY) or (last_clf is None)

        if need_refit:
            sc     = StandardScaler()
            X_tr   = sc.fit_transform(X_tr_raw)     # 훈련 구간에만 fit
            try:
                clf = make_model(model_name)
                clf.fit(X_tr, y_tr)
                last_clf = clf
                last_sc  = sc
                last_fit = idx
            except Exception:
                if last_clf is None:
                    continue

        # 현재 시점 예측
        X_test = last_sc.transform(
            df_src[LAG_FEATS].iloc[[t]].astype(float).values
        )
        try:
            pred = float(last_clf.predict(X_test)[0])
        except Exception:
            pred = float(y_tr.mean())

        bench = float(y_tr.mean())   # 역사적 평균 벤치마크

        records.append({
            "date":      df_src["date"].iloc[t],
            "actual":    float(y_actual),
            "pred":      pred,
            "benchmark": bench,
        })

    return pd.DataFrame(records)


def calc_metrics(df_r: pd.DataFrame) -> dict:
    """R²_oos, MAE, RMSE, 방향 적중률 계산"""
    if df_r.empty or len(df_r) < 5:
        return {"r2_oos": np.nan, "mae": np.nan, "rmse": np.nan, "dir_acc": np.nan, "n": 0}

    a = df_r["actual"].values
    p = df_r["pred"].values
    b = df_r["benchmark"].values

    sse_m  = np.sum((a - p) ** 2)
    sse_b  = np.sum((a - b) ** 2)
    r2_oos = float(1 - sse_m / sse_b) if sse_b > 1e-12 else np.nan
    mae    = float(mean_absolute_error(a, p))
    rmse   = float(np.sqrt(np.mean((a - p) ** 2)))
    dir_a  = float(np.mean(np.sign(a) == np.sign(p)))

    return {"r2_oos": r2_oos, "mae": mae, "rmse": rmse, "dir_acc": dir_a, "n": len(df_r)}


# ── 9. 전체 실행 ─────────────────────────────────────────────────────────────

print(f"\n■ 6단계: 워크포워드 실행")
print(f"  최소학습={MIN_TRAIN_WEEKS}주 | 재적합={REFIT_EVERY}스텝 | 홀드아웃={HOLDOUT_WEEKS}주")
print(f"  조합: {len(HORIZONS)}기간 × {len(MODEL_NAMES)}모델 = {len(HORIZONS)*len(MODEL_NAMES)}개")

wf_results  = {}   # {(horizon, model_name): df_results}
hold_results= {}

t0_total = time.time()

for horizon in HORIZONS:
    for mname in MODEL_NAMES:
        key = (horizon, mname)
        label = f"{MODEL_NAMES[mname]} / fwd_{horizon}w"
        print(f"\n  [{label}]", end="", flush=True)

        t0 = time.time()
        df_wf_res   = run_wf_regression(df_wf, df_wf, horizon, mname, MIN_TRAIN_WEEKS)
        df_hold_res = run_wf_regression(df_full, df_full, horizon, mname, len(df_wf))

        wf_results[key]   = df_wf_res
        hold_results[key] = df_hold_res
        elapsed = time.time() - t0
        wf_m = calc_metrics(df_wf_res)
        print(f"  WF R²={wf_m['r2_oos']:.4f}  DirAcc={wf_m['dir_acc']:.3f}"
              f"  ({elapsed:.0f}초, {wf_m['n']}스텝)")

print(f"\n  총 소요: {time.time()-t0_total:.0f}초")

# ── 10. 결과 집계 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 7단계: 워크포워드 결과 요약")
print("=" * 65)

rows_wf = []
for horizon in HORIZONS:
    for mname, mfull in MODEL_NAMES.items():
        m = calc_metrics(wf_results[(horizon, mname)])
        rows_wf.append({
            "기간(주)": horizon,
            "모델":    mfull,
            "R²_oos":  round(m["r2_oos"], 4) if not np.isnan(m["r2_oos"]) else "N/A",
            "방향적중률": round(m["dir_acc"], 3),
            "MAE":     round(m["mae"] * 100, 3),    # %
            "RMSE":    round(m["rmse"] * 100, 3),   # %
            "n":       m["n"],
        })

wf_df_summary = pd.DataFrame(rows_wf)
print("\n  WF 결과 (MAE·RMSE 단위: %)")
print(wf_df_summary.to_string(index=False))

print("\n" + "=" * 65)
print("■ 8단계: 홀드아웃 결과 요약")
print("=" * 65)

rows_hold = []
for horizon in HORIZONS:
    for mname, mfull in MODEL_NAMES.items():
        m = calc_metrics(hold_results[(horizon, mname)])
        rows_hold.append({
            "기간(주)": horizon,
            "모델":    mfull,
            "R²_oos":  round(m["r2_oos"], 4) if not np.isnan(m["r2_oos"]) else "N/A",
            "방향적중률": round(m["dir_acc"], 3),
            "MAE":     round(m["mae"] * 100, 3),
            "RMSE":    round(m["rmse"] * 100, 3),
            "n":       m["n"],
        })

hold_df_summary = pd.DataFrame(rows_hold)
print("\n  홀드아웃 결과 (MAE·RMSE 단위: %)")
print(hold_df_summary.to_string(index=False))

# ── 11. 특징변수 중요도 (최장 기간 모델 기준) ─────────────────────────────────

print("\n" + "=" * 65)
print("■ 9단계: 특징변수 중요도 (fwd_26w 기준, 마지막 학습 모델)")
print("=" * 65)

for mname, mfull in MODEL_NAMES.items():
    # 마지막 재적합된 모델에서 중요도 추출
    horizon = 26
    key     = (horizon, mname)
    df_res  = wf_results[key]
    if df_res.empty:
        continue

    # 마지막 시점으로 모델 재학습하여 중요도 추출
    train_end = len(df_wf) - horizon
    train = df_wf.iloc[:train_end + 1]
    valid = train[f"fwd_{horizon}w"].notna()
    train_v = train[valid]
    if len(train_v) < 50:
        continue

    sc = StandardScaler()
    X  = sc.fit_transform(train_v[LAG_FEATS].astype(float))
    y  = train_v[f"fwd_{horizon}w"].values

    try:
        clf = make_model(mname)
        clf.fit(X, y)

        # 중요도 추출
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            feat_imp = pd.DataFrame({
                "특징변수": LAG_FEATS,
                "설명":     [RAW_FEATS.get(c.replace("_l1",""), c) for c in LAG_FEATS],
                "중요도":   imp,
            }).sort_values("중요도", ascending=False)

            print(f"\n  [{mfull}]  상위 10개:")
            print(feat_imp.head(10)[["특징변수","중요도"]].to_string(index=False))
    except Exception as e:
        print(f"  [{mfull}] 중요도 추출 실패: {e}")

# ── 12. 결과 저장 ─────────────────────────────────────────────────────────────

# 최고 성능 홀드아웃 예측값 저장 (R²_oos 기준 상위 모델)
best_hold = hold_df_summary.copy()
best_hold["R²_oos_num"] = pd.to_numeric(best_hold["R²_oos"], errors="coerce")
best_row = best_hold.loc[best_hold["R²_oos_num"].idxmax()]
best_h   = int(best_row["기간(주)"])
best_m   = {v: k for k, v in MODEL_NAMES.items()}[best_row["모델"]]

best_df = hold_results[(best_h, best_m)].copy()
best_df.to_csv(OUTPUT_DIR / "multihorizon_best_holdout.csv", index=False)
print(f"\n  최고 성능 저장: fwd_{best_h}w × {MODEL_NAMES[best_m]}  "
      f"(R²_oos={best_row['R²_oos']})")

# ── 13. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

print(f"\n  [사용 변수 {len(LAG_FEATS)}개]")
for c in LAG_FEATS:
    raw = c.replace("_l1", "")
    print(f"  {c:<28}  {RAW_FEATS.get(raw,'')}")

print()

# 기간별 최고 R²_oos 모델
print("  [기간별 최고 WF R²_oos]")
for horizon in HORIZONS:
    sub = wf_df_summary[wf_df_summary["기간(주)"] == horizon].copy()
    sub["r2_num"] = pd.to_numeric(sub["R²_oos"], errors="coerce")
    best = sub.loc[sub["r2_num"].idxmax()]
    r2v  = best["R²_oos"]
    flag = "✓" if isinstance(r2v, float) and r2v > 0.01 else (
           "△" if isinstance(r2v, float) and r2v > 0 else "✗")
    print(f"  fwd_{horizon:2}w: {best['모델']:<14}  R²={r2v}  {flag}")

print("""
  [판정 기준]
  R²_oos > 0.02 : 의미 있는 예측력
  R²_oos 0~0.02 : 미미 (노이즈 경계)
  R²_oos < 0    : 역사적 평균보다 못함

  [주요 주의]
  ① 레짐 라벨 누수는 없음. 단, 특징변수 표준화는 훈련 구간에만 fit.
  ② 트리모델은 비선형 관계를 잡지만 과적합 경향이 있음.
     홀드아웃 R² > WF R² 이면 기간 특수성 주의.
  ③ 장기(26주) 수익률은 단기보다 예측이 어렵지만,
     평균회귀 구조(VIX·금융여건 과잉에서 회복)가 더 강하게 나타날 수 있음.
  ④ Margin Debt: FINRA 공개 데이터 FRED 미수록 → NFCILEVERAGE 대체.
""")
print("=" * 65)
print("분석 완료.")
print("=" * 65)
