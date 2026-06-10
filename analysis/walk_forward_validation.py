# =============================================================================
# 워크포워드(Walk-Forward) SARIMAX 사양 비교 & 홀드아웃 검증
#
# 목적:
#   CCF로 추린 후보 lag으로 만든 소수의 SARIMAX 사양을 비교해서,
#   "역사적 평균 벤치마크를 표본 밖에서 이기는 사양이 있는지" 정직하게 확인한다.
#   → 못 이겨도 유효한 발견이다. 과대 포장하지 않는다.
#
# 필요한 라이브러리 설치 (처음 한 번만):
#   pip install pandas numpy statsmodels scikit-learn
# =============================================================================

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── 0. 설정 — 여기만 수정하면 됩니다 ─────────────────────────────────────────

CSV_PATH   = "data/processed/final_dataset.csv"   # CSV 파일 경로
TARGET_COL = "sp500_log_return"                    # 타깃 컬럼 (이미 정상성)

# CCF 분석(lag_candidate_selector.py)에서 선정한 후보 lag
CANDIDATE_LAGS = {
    "financial_conditions": [1],       # 앵커 변수 — 모든 사양에 포함
    "vix_log_return":       [1, 2],    # 경계 후보 — 대표 lag 1개 선택
    "cpi_change":           [4, 10],   # 경계 후보 — 대표 lag 1개 선택
}

# 각 경계 변수의 대표 lag (lag_candidate_selector.py 결과의 1순위)
REPR_LAGS = {
    "vix_log_return": 1,   # CCF ratio 1순위 lag
    "cpi_change":     4,   # CCF ratio 1순위 lag
}

ANCHOR_VAR = "financial_conditions"
ANCHOR_LAG = 1

HOLDOUT_WEEKS   = 78     # 최종 홀드아웃 주 수 (~18개월). 사양 선택에 절대 사용 안 함
MIN_TRAIN_WEEKS = 260    # 최소 학습 주 수 (~5년). 여기 미달이면 예측 안 함
REFIT_EVERY     = 4      # N 스텝마다 모델 재적합 (크게 하면 빠르지만 덜 정확)
WINDOW          = "expanding"   # "expanding"(전체 확장) 또는 "rolling"(고정 길이)
# ↑ 느리면 REFIT_EVERY를 8~12로 올리세요

OUTPUT_DIR = Path(__file__).parent

# ── 1. 데이터 불러오기 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 1단계: 데이터 불러오기")
print("=" * 65)

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] 파일을 찾을 수 없습니다: {csv_path.resolve()}")
    print("  CSV_PATH를 실제 경로로 바꿔주세요.")
    sys.exit(1)

try:
    df_raw = pd.read_csv(csv_path, parse_dates=["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    print(f"  로드 완료: {len(df_raw):,}행 × {len(df_raw.columns)}열")
except Exception as e:
    print(f"[오류] CSV 읽기 실패: {e}")
    sys.exit(1)

# ── 2. 컬럼 확인 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 2단계: 컬럼 확인")
print("=" * 65)

if TARGET_COL not in df_raw.columns:
    print(f"[오류] 타깃 컬럼 '{TARGET_COL}'이 없습니다.")
    sys.exit(1)

raw_vars = [ANCHOR_VAR] + list(REPR_LAGS.keys())
missing = [v for v in raw_vars if v not in df_raw.columns]
if missing:
    print(f"[오류] 아래 컬럼이 CSV에 없습니다: {missing}")
    print("  CANDIDATE_LAGS 또는 REPR_LAGS를 확인하세요.")
    sys.exit(1)

print(f"  타깃: {TARGET_COL}")
print(f"  원본 변수: {raw_vars}  → lag 피처로 변환 예정")

# ── 3. Lag 피처 생성 ──────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 3단계: Lag 피처 생성")
print("=" * 65)
print("  shift(k) = 변수를 k주 뒤로 밀어서, 현재 시점에서 k주 전 값을 사용")

df = df_raw.copy()

# 앵커 변수 lag 피처
anchor_feat = f"{ANCHOR_VAR}_lag{ANCHOR_LAG}"
df[anchor_feat] = df[ANCHOR_VAR].shift(ANCHOR_LAG)
print(f"  {anchor_feat} 생성 (앵커)")

# 대표 lag 피처 (경계 후보 변수)
repr_feats: dict[str, str] = {}   # {원본변수명: 피처컬럼명}
for var, lag in REPR_LAGS.items():
    feat = f"{var}_lag{lag}"
    df[feat] = df[var].shift(lag)
    repr_feats[var] = feat
    print(f"  {feat} 생성")

# ── 4. 사양 정의 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 4단계: 비교 사양 정의")
print("=" * 65)

# 각 사양은 "어떤 외생변수를 쓸 것인가"를 정의
SPECS: dict[str, list[str]] = {
    "anchor_only":    [anchor_feat],
    "anchor_vix":     [anchor_feat, repr_feats["vix_log_return"]],
    "anchor_cpi":     [anchor_feat, repr_feats["cpi_change"]],
    "anchor_all":     [anchor_feat, repr_feats["vix_log_return"], repr_feats["cpi_change"]],
}

for name, cols in SPECS.items():
    print(f"  {name:<16}: {cols}")

# ── 5. 전처리 + 홀드아웃 분리 ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 5단계: 전처리 + 홀드아웃 분리")
print("=" * 65)

# 모든 사양에 필요한 피처 컬럼을 한 번에 수집
all_feat_cols = sorted({col for cols in SPECS.values() for col in cols})
keep_cols     = ["date", TARGET_COL] + all_feat_cols if "date" in df.columns else [TARGET_COL] + all_feat_cols

df_clean = df[keep_cols].copy()
df_clean = df_clean.apply(
    lambda col: pd.to_numeric(col, errors="coerce") if col.name != "date" else col
)
before = len(df_clean)
df_clean = df_clean.dropna().reset_index(drop=True)
after   = len(df_clean)

print(f"  결측치 제거 전: {before:,}행  →  제거 후: {after:,}행")

if after < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 20:
    print(f"[오류] 데이터가 너무 짧습니다. 최소 {MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 20}주 필요.")
    sys.exit(1)

# 홀드아웃을 맨 끝에서 떼어낸다 — 이후 사양 선택 과정에서 절대 보지 않음
df_wf   = df_clean.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)   # 워크포워드용
df_hold_raw = df_clean.iloc[-(HOLDOUT_WEEKS + 10):].reset_index(drop=True)  # 홀드아웃 평가용 (앞 10행은 모델 초기화용)

print(f"  워크포워드 구간: {len(df_wf):,}주  |  홀드아웃 구간: {HOLDOUT_WEEKS}주")
if "date" in df_clean.columns:
    print(f"  워크포워드: {df_wf['date'].iloc[0].date()} ~ {df_wf['date'].iloc[-1].date()}")
    print(f"  홀드아웃  : {df_clean['date'].iloc[-HOLDOUT_WEEKS].date()} ~ {df_clean['date'].iloc[-1].date()}")

# ── 6. 헬퍼 함수 ─────────────────────────────────────────────────────────────

def select_arima_order(
    y: np.ndarray,
    X: np.ndarray | None = None,
    max_p: int = 2,
    max_q: int = 2,
) -> tuple[int, int, int]:
    """
    AIC 기준으로 ARIMA(p, 0, q) 차수를 고릅니다.
    d=0 고정: 타깃이 이미 정상성(로그수익률)이기 때문입니다.
    """
    best_aic   = np.inf
    best_order = (1, 0, 1)   # 기본값

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue   # 상수 모델은 의미 없음
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
) -> dict[str, float]:
    """
    표본외 R²(R2_oos)와 방향 적중률을 계산합니다.

    R2_oos = 1 - SSE_모델 / SSE_벤치마크
      · 양수: 모델이 역사적 평균보다 나음
      · 음수: 역사적 평균보다 못함 (흔한 결과 — 솔직히 받아들여야 함)

    방향 적중률 = 예측 부호(상승/하락)가 실제와 일치하는 비율
    """
    a = np.array(actuals,    dtype=float)
    p = np.array(preds,      dtype=float)
    b = np.array(benchmarks, dtype=float)

    sse_model = float(np.sum((a - p) ** 2))
    sse_bench = float(np.sum((a - b) ** 2))
    r2_oos    = float(1 - sse_model / sse_bench) if sse_bench > 1e-12 else np.nan

    dir_acc   = float(np.mean(np.sign(a) == np.sign(p)))
    bench_dir = float(np.mean(np.sign(a) == np.sign(b)))   # 벤치마크 자체의 방향 적중률

    return {"r2_oos": r2_oos, "dir_acc": dir_acc, "bench_dir": bench_dir,
            "n_steps": len(a), "sse_model": sse_model, "sse_bench": sse_bench}


def run_walk_forward(
    df_source:      pd.DataFrame,
    target_col:     str,
    specs:          dict[str, list[str]],
    orders:         dict[str, tuple],
    min_train:      int,
    refit_every:    int,
    window:         str,
    start_t:        int | None = None,   # None이면 min_train에서 시작
    verbose:        bool = True,
) -> dict[str, dict]:
    """
    워크포워드 1주 앞 예측을 실행합니다.

    [핵심 규칙 — 데이터 누수 방지]
    ① 표준화(StandardScaler)를 항상 '훈련 구간에만' fit합니다.
       전체 데이터로 fit 후 transform하면 미래 정보가 새어들어갑니다.
    ② 모델은 refit_every 스텝마다 재적합하되, ARIMA 차수(p,q)는 고정합니다.
    ③ 벤치마크 = 훈련 구간의 타깃 평균 (그 시점까지의 역사적 평균)
    """
    if start_t is None:
        start_t = min_train

    n = len(df_source)
    records = {name: {"actual": [], "pred": [], "bench": []} for name in specs}
    fitted_  = {}   # {사양명: 마지막으로 적합된 모델}
    scaler_  = {}   # {사양명: 마지막으로 적합된 스케일러}

    total_steps = n - start_t
    next_print  = 50

    for step_idx, t in enumerate(range(start_t, n)):
        # 훈련 구간 결정
        if window == "expanding":
            s = slice(0, t)                      # 시작부터 t 직전까지 전부 사용
        else:
            s = slice(max(0, t - min_train), t)  # 최근 min_train 주만 사용

        y_train  = df_source[target_col].iloc[s].values.astype(float)
        y_actual = float(df_source[target_col].iloc[t])
        y_bench  = float(np.mean(y_train))       # 벤치마크: 훈련 구간 평균

        need_refit = (step_idx % refit_every == 0)

        for spec_name, feat_cols in specs.items():
            try:
                if need_refit or spec_name not in fitted_:
                    # 스케일러를 훈련 구간에만 fit
                    if feat_cols:
                        scaler = StandardScaler()
                        X_tr   = scaler.fit_transform(
                            df_source[feat_cols].iloc[s].values.astype(float)
                        )
                        scaler_[spec_name] = scaler
                    else:
                        X_tr = None
                        scaler_[spec_name] = None

                    # SARIMAX 모델 재적합
                    X_tr_df = (pd.DataFrame(X_tr, columns=feat_cols)
                               if feat_cols else None)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted = SARIMAX(
                            y_train,
                            exog=X_tr_df,
                            order=orders[spec_name],
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False)
                    fitted_[spec_name] = fitted

                # 예측: 훈련 때 fit한 스케일러로 테스트 포인트를 transform
                if feat_cols and scaler_[spec_name] is not None:
                    x_test = scaler_[spec_name].transform(
                        df_source[feat_cols].iloc[[t]].values.astype(float)
                    )
                    x_test_df = pd.DataFrame(x_test, columns=feat_cols)
                else:
                    x_test_df = None

                # 1주 앞 예측
                y_pred = float(
                    fitted_[spec_name].forecast(steps=1, exog=x_test_df).iloc[0]
                )

                # 예측값이 비정상(NaN, ±inf)이면 벤치마크로 대체
                if not np.isfinite(y_pred):
                    y_pred = y_bench

            except Exception:
                y_pred = y_bench   # 모델 실패 시 벤치마크로 대체

            records[spec_name]["actual"].append(y_actual)
            records[spec_name]["pred"].append(y_pred)
            records[spec_name]["bench"].append(y_bench)

        # 진행 상황 출력
        if verbose and (step_idx + 1) >= next_print:
            elapsed = time.time() - _wf_start
            pct = (step_idx + 1) / total_steps * 100
            print(f"    {step_idx+1:>4}/{total_steps} 스텝 ({pct:.0f}%) — {elapsed:.0f}초 경과")
            next_print += 50

    return records


# ── 7. ARIMA 차수 선택 ────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 6단계: ARIMA(p,0,q) 차수 선택 — 첫 훈련 윈도우에서 한 번")
print("=" * 65)
print("  (이후 refit_every 스텝마다 파라미터만 재추정, 차수는 고정)")

# 첫 훈련 윈도우: 처음 MIN_TRAIN_WEEKS 행
y_init = df_wf[TARGET_COL].values[:MIN_TRAIN_WEEKS].astype(float)
orders_: dict[str, tuple] = {}

t0_order = time.time()
for spec_name, feat_cols in SPECS.items():
    if feat_cols:
        sc_init = StandardScaler()
        X_init  = sc_init.fit_transform(
            df_wf[feat_cols].values[:MIN_TRAIN_WEEKS].astype(float)
        )
    else:
        X_init = None

    order = select_arima_order(y_init, X_init)
    orders_[spec_name] = order
    print(f"  {spec_name:<16}: ARIMA{order}  선택")

print(f"  차수 선택 소요 시간: {time.time()-t0_order:.1f}초")

# ── 8. 워크포워드 — 홀드아웃 제외 구간 ───────────────────────────────────────

print("\n" + "=" * 65)
print("■ 7단계: 워크포워드 실행 (홀드아웃 제외 구간)")
print(f"  윈도우: {WINDOW} | 재적합 주기: {REFIT_EVERY}스텝 | 최소학습: {MIN_TRAIN_WEEKS}주")
total_wf = len(df_wf) - MIN_TRAIN_WEEKS
print(f"  예측 스텝 수: {total_wf}주")
if total_wf > 300:
    print(f"  [안내] 스텝이 많아 시간이 걸립니다. REFIT_EVERY를 8~12로 올리면 빨라집니다.")
print("=" * 65)

_wf_start = time.time()
records_wf = run_walk_forward(
    df_source=df_wf,
    target_col=TARGET_COL,
    specs=SPECS,
    orders=orders_,
    min_train=MIN_TRAIN_WEEKS,
    refit_every=REFIT_EVERY,
    window=WINDOW,
)
wf_elapsed = time.time() - _wf_start
print(f"\n  워크포워드 완료: {wf_elapsed:.1f}초")

# ── 9. 워크포워드 결과 + 사양 선택 ───────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 8단계: 워크포워드 결과 비교 + 사양 선택")
print("=" * 65)

rows = []
for spec_name, rec in records_wf.items():
    m = compute_metrics(rec["actual"], rec["pred"], rec["bench"])
    rows.append({
        "사양":        spec_name,
        "외생변수":    " | ".join(SPECS[spec_name]),
        "R2_oos":     round(m["r2_oos"], 4),
        "방향적중률":  round(m["dir_acc"], 3),
        "n_steps":    m["n_steps"],
    })

results_df = (
    pd.DataFrame(rows)
    .sort_values("R2_oos", ascending=False)
    .reset_index(drop=True)
)

# 벤치마크(역사적 평균) 방향 적중률 참고값
bench_metrics = compute_metrics(
    records_wf["anchor_only"]["actual"],
    records_wf["anchor_only"]["bench"],   # 벤치마크 예측 = 훈련 평균
    records_wf["anchor_only"]["bench"],
)
print(f"  [참고] 역사적 평균 벤치마크 방향 적중률: {bench_metrics['bench_dir']:.3f}")
print(f"  [참고] R2_oos > 0 = 벤치마크 이김, R2_oos < 0 = 벤치마크에 짐\n")

# 표 출력
print(results_df.to_string(index=False))

# 가장 높은 R2_oos 사양 선택
best_spec = str(results_df.iloc[0]["사양"])
best_r2   = float(results_df.iloc[0]["R2_oos"])
best_dir  = float(results_df.iloc[0]["방향적중률"])
print(f"\n  → 워크포워드 R2_oos 기준 선택 사양: '{best_spec}'")
print(f"     (R2_oos={best_r2:.4f}, 방향 적중률={best_dir:.3f})")

# ── 10. 홀드아웃 최종 검증 (딱 한 번) ─────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 9단계: 최종 홀드아웃 검증 — 선택 사양을 한 번만 적용")
print(f"  사양: '{best_spec}'  |  홀드아웃: {HOLDOUT_WEEKS}주")
print("  ※ 이 숫자가 보고서에 쓸 수 있는 정직한 표본 외 성적입니다.")
print("=" * 65)

# 홀드아웃 walk-forward:
# df_clean 전체에서 워크포워드 시작점을 len(df_wf)로 설정
# → 처음 len(df_wf)행이 학습 데이터, 이후가 홀드아웃
df_full  = df_clean.reset_index(drop=True)
hold_specs  = {best_spec: SPECS[best_spec]}
hold_orders = {best_spec: orders_[best_spec]}

_wf_start = time.time()
records_hold = run_walk_forward(
    df_source=df_full,
    target_col=TARGET_COL,
    specs=hold_specs,
    orders=hold_orders,
    min_train=MIN_TRAIN_WEEKS,
    refit_every=REFIT_EVERY,
    window=WINDOW,
    start_t=len(df_wf),          # 홀드아웃 첫 번째 주부터 예측 시작
)
hold_elapsed = time.time() - _wf_start

hold_m = compute_metrics(
    records_hold[best_spec]["actual"],
    records_hold[best_spec]["pred"],
    records_hold[best_spec]["bench"],
)

print(f"\n  홀드아웃 R2_oos   : {hold_m['r2_oos']:+.4f}  "
      f"({'벤치마크보다 나음 ✓' if hold_m['r2_oos'] > 0 else '벤치마크보다 못함 ✗'})")
print(f"  홀드아웃 방향 적중률: {hold_m['dir_acc']:.3f}  "
      f"(벤치마크 기준: {hold_m['bench_dir']:.3f})")
print(f"  홀드아웃 예측 스텝 수: {hold_m['n_steps']}주  (소요: {hold_elapsed:.1f}초)")

# 홀드아웃 예측 결과를 CSV로 저장
hold_df = pd.DataFrame({
    "actual":    records_hold[best_spec]["actual"],
    "predicted": records_hold[best_spec]["pred"],
    "benchmark": records_hold[best_spec]["bench"],
})
if "date" in df_full.columns:
    hold_df.insert(0, "date", df_full["date"].iloc[len(df_wf):len(df_wf)+len(hold_df)].values)

hold_csv = OUTPUT_DIR / "holdout_predictions.csv"
hold_df.to_csv(hold_csv, index=False)
print(f"\n  홀드아웃 예측값 저장 완료: {hold_csv}")

# ── 11. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

# 워크포워드에서 벤치마크를 이긴 사양 수
beat_bench_wf = [r["사양"] for _, r in results_df.iterrows() if r["R2_oos"] > 0]

print(f"\n  [워크포워드 구간 ({len(df_wf) - MIN_TRAIN_WEEKS}주)]")
if beat_bench_wf:
    print(f"  벤치마크(역사적 평균)를 이긴 사양: {beat_bench_wf}")
    best_row = results_df.iloc[0]
    print(f"  → 최고 사양 '{best_row['사양']}': R2_oos={best_row['R2_oos']:.4f}, 방향 적중률={best_row['방향적중률']:.3f}")
else:
    print("  벤치마크를 이긴 사양이 없습니다. (모든 사양의 R2_oos < 0)")

# 경계 후보 기여 확인
anchor_r2  = float(results_df.loc[results_df["사양"] == "anchor_only",  "R2_oos"].values[0])
vix_r2     = float(results_df.loc[results_df["사양"] == "anchor_vix",   "R2_oos"].values[0])
cpi_r2     = float(results_df.loc[results_df["사양"] == "anchor_cpi",   "R2_oos"].values[0])
all_r2     = float(results_df.loc[results_df["사양"] == "anchor_all",   "R2_oos"].values[0])

print(f"\n  [경계 후보 기여 여부]")
vix_added = vix_r2 - anchor_r2
cpi_added = cpi_r2 - anchor_r2
all_added = all_r2 - anchor_r2

def _delta_str(d):
    return f"{d:+.4f} ({'기여' if d > 0 else '손해'})"

print(f"  vix 추가 시 R2_oos 변화: {_delta_str(vix_added)}")
print(f"  cpi 추가 시 R2_oos 변화: {_delta_str(cpi_added)}")
print(f"  둘 다 추가 시 R2_oos 변화: {_delta_str(all_added)}")

print(f"\n  [최종 홀드아웃 ({HOLDOUT_WEEKS}주) — 보고서용 정직한 성적]")
if hold_m["r2_oos"] > 0:
    print(f"  사양 '{best_spec}'이 홀드아웃에서 벤치마크를 R2_oos={hold_m['r2_oos']:+.4f}로 이겼습니다.")
    print(f"  방향 적중률 {hold_m['dir_acc']:.3f} (벤치마크: {hold_m['bench_dir']:.3f})")
    print(f"  단, 18개월 홀드아웃 1회 검증이므로 우연일 가능성도 있습니다. 해석에 주의하세요.")
else:
    print(f"  사양 '{best_spec}'이 홀드아웃에서 벤치마크를 이기지 못했습니다.")
    print(f"  R2_oos={hold_m['r2_oos']:+.4f}  → 역사적 평균이 더 나은 예측이었습니다.")
    print()
    print("  ★ 결론: 이 외생변수 조합으로는 주간 S&P500 로그수익률 예측 가치가")
    print("    표본 밖에서 확인되지 않았습니다. 이것도 유효한 발견입니다.")
    print("    다음 단계로 고려할 수 있는 것:")
    print("    1) 다른 외생변수 또는 더 긴 lag 탐색")
    print("    2) 선형 모델 대신 비선형/앙상블 모델 시도")
    print("    3) 타깃을 방향(상승/하락 이진)으로 바꿔 분류 문제로 접근")

print("\n" + "=" * 65)
print("분석 완료.")
print("=" * 65)
