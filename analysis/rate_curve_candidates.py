# =============================================================================
# rate_curve_candidates.py  —  금리곡선 후보 변수 워크포워드 검증
#
# 목적:
#   기존 워크포워드(walk_forward_validation.py)에서 채택된 사양(best_spec)을
#   기준선으로 두고, 금리곡선 관련 두 변수를 새 후보로 추가했을 때
#   표본 밖에서 실제로 가치를 더하는지 정직하게 검증한다.
#
#   새 후보:
#     ① inv_duration   — 장단기 금리 역전이 연속으로 지속된 주 수
#     ② rate_level_hybrid — 단기·장기·정책 금리 수준의 합성 지표
#
# 필요 라이브러리 설치 (처음 한 번만):
#   pip install pandas numpy statsmodels scikit-learn
#
# 느리면 REFIT_EVERY 를 8~12로 높이세요.
# =============================================================================

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# ── 0. 설정 — 여기만 수정하면 됩니다 ─────────────────────────────────────────

CSV_PATH   = "data/processed/final_dataset.csv"
TARGET_COL = "sp500_log_return"   # 타깃: 이미 정상성(d=0)

# ★ 직전 워크포워드(walk_forward_validation.py)에서 채택된 최고 사양으로 교체하세요.
#   walk_forward_validation.py 가 만드는 컬럼명은 예) "financial_conditions_lag1"
#   → 해당 스크립트에서 anchor_feat / repr_feats 로 만든 컬럼명 그대로 적으세요.
#   아직 실행 전이라면 아래 기본값(앵커만)을 그대로 두면 됩니다.
BEST_SPEC: list[str] = [
    "financial_conditions_lag1",   # 앵커 변수
    # "vix_log_return_lag1",       # vix가 채택됐으면 주석 해제
    # "cpi_change_lag4",           # cpi가 채택됐으면 주석 해제
]

# 금리 역전 판단 기준 컬럼
SPREAD_COL = "term_spread_10y_2y"

# rate_level_hybrid 계산에 쓸 금리 레벨 컬럼 목록 (없는 컬럼은 자동 제외)
# ※ 분석 의도가 다르면 이 리스트를 직접 교체하세요.
RATE_LEVEL_COLS = ["fed_funds", "treasury_2y", "treasury_10y"]

# 워크포워드 파라미터
HOLDOUT_WEEKS   = 78     # 최종 홀드아웃 주 수 (~18개월). 사양 선택에 절대 사용 금지
MIN_TRAIN_WEEKS = 260    # 최소 학습 주 수 (~5년)
REFIT_EVERY     = 4      # N 스텝마다 모델 재적합 (클수록 빠르나 덜 정확)
WINDOW          = "expanding"   # "expanding" 또는 "rolling"

# ARIMA 탐색 범위 (첫 훈련 윈도우에서 한 번만)
MAX_P = 2
MAX_Q = 2

OUTPUT_DIR = Path(__file__).parent

# ── 1. 데이터 불러오기 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 금리곡선 후보 변수 워크포워드 검증")
print("=" * 65)

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] CSV를 찾을 수 없습니다: {csv_path.resolve()}")
    print("  CSV_PATH를 올바른 경로로 수정하세요.")
    sys.exit(1)

try:
    df_raw = pd.read_csv(csv_path, parse_dates=["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    print(f"\n  로드 완료: {len(df_raw):,}행 × {len(df_raw.columns)}열")
    print(f"  기간: {df_raw['date'].min().date()} ~ {df_raw['date'].max().date()}")
except Exception as e:
    print(f"[오류] CSV 읽기 실패: {e}")
    sys.exit(1)

# 숫자형 강제 변환
for col in [TARGET_COL, SPREAD_COL] + RATE_LEVEL_COLS:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# ── 2. BEST_SPEC 컬럼 재생성
#       walk_forward_validation.py 는 raw 컬럼에서 shift()로 lag 피처를 만들었다.
#       CSV 에는 이미 _lag_1 형식 컬럼이 있으나, 이전 스크립트는 lag1 (언더바 없이)
#       이름을 사용하므로 없는 컬럼은 여기서 다시 만든다.
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 1단계: BEST_SPEC 컬럼 확인 및 생성")
print("=" * 65)

df = df_raw.copy()

for feat in BEST_SPEC:
    if feat in df.columns:
        print(f"  ✓ '{feat}' — 이미 존재")
        continue
    # "basevar_lagN" 형식 분해 시도
    import re as _re
    m = _re.match(r"^(.+)_lag(\d+)$", feat)
    if m:
        base, lag_n = m.group(1), int(m.group(2))
        if base in df.columns:
            df[feat] = df[base].shift(lag_n)
            df[feat] = pd.to_numeric(df[feat], errors="coerce")
            print(f"  ✓ '{feat}' 생성 완료 (shift({lag_n}) of '{base}')")
        else:
            print(f"  ✗ '{feat}' 생성 실패 — '{base}' 컬럼 없음. BEST_SPEC을 확인하세요.")
    else:
        print(f"  ✗ '{feat}' — 규칙 파악 불가. BEST_SPEC에서 직접 컬럼명을 확인하세요.")

# 실제로 존재하는 BEST_SPEC 컬럼만 남긴다
valid_best = [c for c in BEST_SPEC if c in df.columns]
missing_best = [c for c in BEST_SPEC if c not in df.columns]
if missing_best:
    print(f"\n  ⚠ 다음 BEST_SPEC 컬럼을 찾을 수 없어 제외합니다: {missing_best}")
print(f"  → 실제 사용할 BEST_SPEC: {valid_best}")

# ── 3. 새 변수 생성 (누수 없음 — 시점 t는 t 이전 정보만 사용) ──────────────────

print("\n" + "=" * 65)
print("■ 2단계: 새 후보 변수 생성")
print("=" * 65)

# ---- ① inv_duration: 역전 지속 주 수 ----------------------------------------
# term_spread_10y_2y < 0 인 주(week)가 연속으로 이어진 수를 카운트한다.
# 역전이 아니면 0 으로 리셋. 오직 과거 정보만 사용하므로 누수 없음.
if SPREAD_COL in df.columns:
    spread = df[SPREAD_COL].values.copy()
    inv_dur = np.zeros(len(spread))
    for i in range(1, len(spread)):                     # i=0은 과거 없으므로 0 유지
        if not np.isnan(spread[i]) and spread[i] < 0:
            inv_dur[i] = inv_dur[i - 1] + 1            # 역전 지속 → 카운터 증가
        else:
            inv_dur[i] = 0                              # 역전 해소 → 리셋

    df["inv_duration"] = inv_dur

    n_inv_weeks = int((spread < 0).sum())
    total_weeks = int((~np.isnan(spread)).sum())
    print(f"  ✓ inv_duration 생성 완료")
    print(f"    역전 주(weeks) 수: {n_inv_weeks}/{total_weeks}주 "
          f"({100*n_inv_weeks/total_weeks:.1f}%)")
    print(f"    ⚠ [주의] 역전 에피소드가 표본에 적어 통계적 증거가 빈약할 수 있습니다.")
    print(f"    ⚠ [주의] 역전→경기침체→주가 하락 경로는 수개월 시차.")
    print(f"             주간 '다음 주' 예측에서는 신호가 약하게 나올 수 있습니다.")
    print(f"             이 변수의 본 무대는 월간 하락확률 모델임을 기억하세요.")
else:
    df["inv_duration"] = np.nan
    print(f"  ✗ inv_duration 생성 실패 — '{SPREAD_COL}' 컬럼 없음")

# ---- ② rate_level_hybrid: 금리 수준 합성 지표 --------------------------------
# 사용 가능한 금리 레벨 컬럼(fed_funds, treasury_2y, treasury_10y)을 각각
# 단순 평균하여 합성 지표를 만든다.
# · 정상화(StandardScaler)는 워크포워드 루프 안에서 훈련 데이터만으로 수행한다.
# · 여기서는 원시 수준값 평균만 계산 — 전체 샘플 통계를 쓰지 않으므로 누수 없음.
# ※ 다른 금리 지표를 사용하고 싶으면 RATE_LEVEL_COLS 리스트를 교체하세요.
# ※ PCA 를 원한다면 아래 주석 코드를 참조하세요 (워크포워드 내부에서 fit 필요).
avail_rate_cols = [c for c in RATE_LEVEL_COLS if c in df.columns]
if avail_rate_cols:
    df["rate_level_hybrid"] = df[avail_rate_cols].mean(axis=1)
    print(f"\n  ✓ rate_level_hybrid 생성 완료")
    print(f"    사용 컬럼: {avail_rate_cols}")
    print(f"    정의: 각 금리 수준의 단순 평균 (스케일링은 워크포워드 내부에서)")
    # [PCA 변형 — 워크포워드 내에서 fit 필요]
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=1)
    # df["rate_level_hybrid"] = pca.fit_transform(
    #     df[avail_rate_cols].fillna(method="ffill")
    # )[:, 0]
else:
    df["rate_level_hybrid"] = np.nan
    print(f"\n  ✗ rate_level_hybrid 생성 실패 — 금리 레벨 컬럼 없음")

# ── 4. 단위근 검정 (ADF) ──────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 3단계: 새 변수 단위근 검정 (ADF)")
print("  귀무가설: 단위근 있음(비정상). p > 0.05 이면 차분 고려.")
print("=" * 65)


def run_adf(series: pd.Series, name: str) -> tuple[bool, float]:
    """ADF 검정을 실행하고 (비정상여부, p값) 을 반환한다."""
    clean = series.dropna()
    if len(clean) < 30:
        print(f"  {name}: 데이터 부족({len(clean)}행) — 건너뜀")
        return True, 1.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = adfuller(clean, maxlag=12, autolag="AIC")
        stat, pval = res[0], res[1]
        status = "비정상 ⚠" if pval > 0.05 else "정상  ✓"
        print(f"  {name:<22}: ADF={stat:+.3f}, p={pval:.4f} → [{status}]")
        return pval > 0.05, pval
    except Exception as ex:
        print(f"  {name}: ADF 실패 ({ex})")
        return True, 1.0


NEW_FEAT_COLS: list[str] = []   # 최종 모델에 투입할 새 변수 컬럼명 (lag-1 버전)

for raw_col in ["inv_duration", "rate_level_hybrid"]:
    if df[raw_col].isna().all():
        print(f"  {raw_col}: 유효 데이터 없음 — 건너뜀")
        continue

    nonstat, pval = run_adf(df[raw_col], raw_col)

    if nonstat:
        # 1차 차분 후 재검정
        diff_col = f"d_{raw_col}"
        df[diff_col] = df[raw_col].diff()
        print(f"    → 1차 차분 적용 후 재검정:")
        nonstat2, _ = run_adf(df[diff_col], diff_col)
        use_col  = diff_col
        action   = "1차 차분 후 lag-1 사용"
        if nonstat2:
            print(f"    ⚠ 1차 차분 후에도 비정상 가능성. 분석을 계속하되 해석에 주의.")
    else:
        use_col = raw_col
        action  = "수준값 그대로 lag-1 사용"

    # lag-1 생성: 예측 시점 t 에서 t-1 의 값이 알려진 정보이므로 누수 없음
    lag_col = f"{use_col}_lag1"
    df[lag_col] = df[use_col].shift(1)
    NEW_FEAT_COLS.append(lag_col)
    print(f"    → 처리: {action}  →  컬럼 '{lag_col}' 생성")

print(f"\n  최종 후보 컬럼: {NEW_FEAT_COLS}")

# ── 5. 전처리 + 홀드아웃 분리 ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 4단계: 전처리 + 홀드아웃 분리")
print("=" * 65)

# 모든 사양에서 쓸 컬럼 수집
all_feat_cols = list({c for c in valid_best + NEW_FEAT_COLS})
keep_cols = ["date", TARGET_COL] + all_feat_cols

df_use = df[[c for c in keep_cols if c in df.columns]].copy()
df_use = df_use.apply(
    lambda col: pd.to_numeric(col, errors="coerce") if col.name != "date" else col
)

before = len(df_use)
df_clean = df_use.dropna().reset_index(drop=True)
after = len(df_clean)
print(f"  결측치 제거: {before:,}행 → {after:,}행")

if after < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 20:
    print(f"[오류] 데이터가 너무 짧습니다. "
          f"최소 {MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 20}주가 필요합니다.")
    sys.exit(1)

# 홀드아웃은 맨 끝에서 잘라낸다 — 사양 선택 과정에서 절대 보지 않음
df_wf   = df_clean.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_full = df_clean.reset_index(drop=True)   # 홀드아웃 평가용 전체 데이터

print(f"  워크포워드 구간: {len(df_wf):,}주  "
      f"({df_wf['date'].iloc[0].date()} ~ {df_wf['date'].iloc[-1].date()})")
print(f"  홀드아웃 구간 : {HOLDOUT_WEEKS}주  "
      f"({df_clean['date'].iloc[-HOLDOUT_WEEKS].date()} "
      f"~ {df_clean['date'].iloc[-1].date()})")

# ── 6. 사양 정의 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 5단계: 검증할 사양 정의")
print("=" * 65)

inv_col  = next((c for c in NEW_FEAT_COLS if "inv_duration" in c), None)
rate_col = next((c for c in NEW_FEAT_COLS if "rate_level" in c), None)

SPECS: dict[str, list[str]] = {"1_base": valid_best}
if inv_col:
    SPECS["2_base+inv"]  = valid_best + [inv_col]
if rate_col:
    SPECS["3_base+rate"] = valid_best + [rate_col]
if inv_col and rate_col:
    SPECS["4_base+both"] = valid_best + [inv_col, rate_col]

for name, cols in SPECS.items():
    print(f"  {name:<18}: {cols}")

# ── 7. 헬퍼 함수 ─────────────────────────────────────────────────────────────


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
        "n_steps": len(a), "sse_model": sse_model, "sse_bench": sse_bench,
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
    워크포워드 1주 앞 예측 엔진 (기존 walk_forward_validation.py 와 동일 구조).

    [데이터 누수 방지 원칙]
    ① StandardScaler 는 항상 훈련 구간에만 fit 한다.
    ② ARIMA 차수(p, q)는 고정, refit_every 마다 파라미터만 재추정한다.
    ③ 벤치마크 = 훈련 구간의 타깃 누적 평균 (역사적 평균).

    예측 구조:
      - t 행의 exog(x_t) 로 t 행의 y_t 를 예측한다.
      - lag-1 변수를 사용하므로 x_t[lag1] = 원시값_{t-1}: 시점 t-1 에 알려진 값.
      - 즉 의사결정 시점 t-1 에 t의 y 를 예측하는 것과 동일하다.
    """
    if start_t is None:
        start_t = min_train

    n       = len(df_source)
    records = {name: {"actual": [], "pred": [], "bench": []} for name in specs}
    fitted_ : dict[str, object]           = {}
    scaler_ : dict[str, StandardScaler | None] = {}

    total_steps = n - start_t
    next_print  = max(50, total_steps // 10)

    for step_idx, t in enumerate(range(start_t, n)):
        # ---- 훈련 슬라이스 결정 ----
        if window == "expanding":
            s = slice(0, t)
        else:
            s = slice(max(0, t - min_train), t)

        y_train  = df_source[target_col].iloc[s].values.astype(float)
        y_actual = float(df_source[target_col].iloc[t])
        y_bench  = float(np.mean(y_train))   # 벤치마크: 그 시점까지의 누적 평균

        need_refit = (step_idx % refit_every == 0)

        for spec_name, feat_cols in specs.items():
            try:
                if need_refit or spec_name not in fitted_:
                    # ---- 스케일러: 훈련 구간만으로 fit ----
                    if feat_cols:
                        scaler = StandardScaler()
                        X_tr   = scaler.fit_transform(
                            df_source[feat_cols].iloc[s].values.astype(float)
                        )
                        scaler_[spec_name] = scaler
                    else:
                        X_tr = None
                        scaler_[spec_name] = None

                    # ---- SARIMAX 재적합 ----
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

                # ---- 예측: 훈련 스케일러로 test 포인트 transform ----
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
                    y_pred = y_bench   # 비정상 예측값 → 벤치마크로 대체

            except Exception:
                y_pred = y_bench       # 적합·예측 실패 → 벤치마크로 대체

            records[spec_name]["actual"].append(y_actual)
            records[spec_name]["pred"].append(y_pred)
            records[spec_name]["bench"].append(y_bench)

        # ---- 진행 상황 출력 ----
        if verbose and (step_idx + 1) >= next_print:
            pct = (step_idx + 1) / total_steps * 100
            elapsed = time.time() - _wf_start
            print(f"    {step_idx+1:>4}/{total_steps} 스텝 ({pct:.0f}%) — {elapsed:.0f}초 경과")
            next_print += max(50, total_steps // 10)

    return records


# ── 8. ARIMA 차수 선택 (첫 훈련 윈도우에서 한 번, 이후 고정) ────────────────────

print("\n" + "=" * 65)
print("■ 6단계: ARIMA(p,0,q) 차수 선택 — 첫 훈련 윈도우에서 한 번")
print("  (이후 refit_every 스텝마다 파라미터만 재추정, 차수는 고정)")
print("=" * 65)

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
    print(f"  {spec_name:<20}: ARIMA{order}")

print(f"  차수 선택 소요: {time.time()-t0_order:.1f}초")

# ── 9. 워크포워드 실행 (홀드아웃 제외 구간) ─────────────────────────────────────

print("\n" + "=" * 65)
print("■ 7단계: 워크포워드 실행 (홀드아웃 제외 구간)")
print(f"  윈도우: {WINDOW} | 재적합 주기: {REFIT_EVERY}스텝 | 최소학습: {MIN_TRAIN_WEEKS}주")
total_wf = len(df_wf) - MIN_TRAIN_WEEKS
print(f"  예측 스텝 수: {total_wf}주 × {len(SPECS)}사양")
if total_wf > 300:
    print(f"  [안내] 스텝이 많아 시간이 걸립니다. 느리면 REFIT_EVERY를 8~12로 높이세요.")
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
wf_elapsed = time.time() - _wf_start
print(f"\n  워크포워드 완료: {wf_elapsed:.1f}초")

# ── 10. 결과 집계 + 사양 선택 ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 8단계: 워크포워드 결과 비교 + 사양 선택")
print("=" * 65)

rows = []

# 벤치마크(역사적 평균) 성과
first_rec = list(records_wf.values())[0]
bm_metrics = compute_metrics(
    first_rec["actual"], first_rec["bench"], first_rec["bench"]
)
rows.append({
    "사양":       "0_benchmark",
    "변수 구성":  "역사적 누적 평균",
    "R2_oos":    0.0,           # 정의상 0
    "방향적중률": round(bm_metrics["dir_acc"], 3),
    "n_steps":   bm_metrics["n_steps"],
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

print(f"\n  [참고] R2_oos > 0 = 벤치마크(역사적 평균) 이김,  R2_oos < 0 = 짐\n")
print(results_df[["사양", "R2_oos", "방향적중률", "n_steps"]].to_string(index=False))

# 새 후보 기여도 분석
print("\n  [새 변수 기여도]")
base_r2 = next(
    (r["R2_oos"] for r in rows if r["사양"] == "1_base"), np.nan
)

for spec_name, spec_cols in SPECS.items():
    if spec_name == "1_base":
        continue
    r = next((r["R2_oos"] for r in rows if r["사양"] == spec_name), np.nan)
    if np.isnan(r) or np.isnan(base_r2):
        continue
    delta = r - base_r2
    added  = spec_name.replace("2_base+", "").replace("3_base+", "").replace("4_base+", "")
    if delta > 0.005:
        verdict = f"개선  (+{delta:.4f})"
    elif delta > 0:
        verdict = f"미미한 개선  (+{delta:.4f})"
    elif delta > -0.005:
        verdict = f"거의 차이 없음  ({delta:+.4f})"
    else:
        verdict = f"오히려 악화  ({delta:+.4f})"
    print(f"  [{added}] base 대비 ΔR2_oos: {verdict}")

# 최고 사양 선택 (모델 사양 중 R2_oos 최대)
model_rows = [r for r in rows if r["사양"] != "0_benchmark"]
best_row   = max(model_rows, key=lambda x: x["R2_oos"])
best_name  = best_row["사양"]
best_cols  = SPECS[best_name]
best_r2_wf = best_row["R2_oos"]

print(f"\n  → 워크포워드 R2_oos 기준 선택 사양: '{best_name}'")
print(f"     (R2_oos={best_r2_wf:.4f}, 방향 적중률={best_row['방향적중률']:.3f})")

# ── 11. 최종 홀드아웃 검증 (딱 한 번) ─────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 9단계: 최종 홀드아웃 검증 — 선택 사양을 한 번만 적용")
print(f"  사양: '{best_name}'  |  홀드아웃: {HOLDOUT_WEEKS}주")
print("  ※ 이 숫자가 보고서에 쓸 수 있는 정직한 표본 외 성적입니다.")
print("=" * 65)

# 차수는 워크포워드와 동일하게 고정
hold_specs  = {best_name: best_cols}
hold_orders = {best_name: orders_[best_name]}
hold_start  = len(df_wf)   # 홀드아웃 첫 번째 주부터 예측

_wf_start = time.time()
records_hold = run_walk_forward(
    df_source   = df_full,
    target_col  = TARGET_COL,
    specs       = hold_specs,
    orders      = hold_orders,
    min_train   = MIN_TRAIN_WEEKS,
    refit_every = REFIT_EVERY,
    window      = WINDOW,
    start_t     = hold_start,
)
hold_elapsed = time.time() - _wf_start

hold_m = compute_metrics(
    records_hold[best_name]["actual"],
    records_hold[best_name]["pred"],
    records_hold[best_name]["bench"],
)

print(f"\n  홀드아웃 R2_oos    : {hold_m['r2_oos']:+.4f}  "
      f"({'벤치마크보다 나음 ✓' if hold_m['r2_oos'] > 0 else '벤치마크보다 못함 ✗'})")
print(f"  홀드아웃 방향 적중률: {hold_m['dir_acc']:.3f}  "
      f"(벤치마크 방향 적중률: {hold_m['bench_dir']:.3f})")
print(f"  홀드아웃 예측 스텝 : {hold_m['n_steps']}주  (소요: {hold_elapsed:.1f}초)")

# 예측 결과를 CSV 로 저장
hold_df = pd.DataFrame({
    "date":      df_full["date"].iloc[hold_start:hold_start + hold_m["n_steps"]].values,
    "actual":    records_hold[best_name]["actual"],
    "predicted": records_hold[best_name]["pred"],
    "benchmark": records_hold[best_name]["bench"],
})
out_csv = OUTPUT_DIR / "rate_curve_holdout_predictions.csv"
hold_df.to_csv(out_csv, index=False)
print(f"\n  홀드아웃 예측값 저장: {out_csv}")

# ── 12. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

print(f"\n  [워크포워드 결과 표 (상위 → 하위)]")
print(results_df[["사양", "변수 구성", "R2_oos", "방향적중률"]].to_string(index=False))

print(f"\n  [선택 사양: '{best_name}']")
print(f"  · 워크포워드 R2_oos = {best_r2_wf:.4f}")
print(f"  · 홀드아웃 R2_oos   = {hold_m['r2_oos']:+.4f}")
print(f"  · 홀드아웃 방향 적중률 = {hold_m['dir_acc']:.3f}")

print("\n  [새 변수 해석]")

# inv_duration 결론
inv_spec_r2  = next((r["R2_oos"] for r in rows if "inv" in r["사양"]), None)
if inv_spec_r2 is not None and not np.isnan(base_r2):
    delta_inv = inv_spec_r2 - base_r2
    if delta_inv > 0.005:
        print(f"  inv_duration: base 대비 +{delta_inv:.4f} 개선. 표본 내 기여 확인.")
    else:
        print(f"  inv_duration: base 대비 {delta_inv:+.4f} — 주간 예측에서 기여 미미.")
    print(f"    (예상된 결과: 역전→침체 경로는 수개월 시차, 주간 모델에선 약한 신호)")
    print(f"    이 변수의 진가는 월간 하락확률 모델에서 발휘됩니다.")

# rate_level_hybrid 결론
rate_spec_r2 = next((r["R2_oos"] for r in rows if "rate" in r["사양"]), None)
if rate_spec_r2 is not None and not np.isnan(base_r2):
    delta_rate = rate_spec_r2 - base_r2
    if delta_rate > 0.005:
        print(f"\n  rate_level_hybrid: base 대비 +{delta_rate:.4f} 개선.")
    else:
        print(f"\n  rate_level_hybrid: base 대비 {delta_rate:+.4f} — 기여 미미 또는 없음.")
    print(f"    (금리 수준은 할인율 채널로 주가에 영향하나, 주간 고주파에서는")
    print(f"     수준보다 변화율(Δ금리)이 더 중요한 경우가 많습니다.)")

print()
if hold_m["r2_oos"] > 0:
    print(f"  ★ 최종 결론: 선택 사양이 홀드아웃에서 벤치마크를 이겼습니다.")
    print(f"    단, 18개월 1회 검증이므로 우연일 가능성도 배제할 수 없습니다.")
else:
    print(f"  ★ 최종 결론: 선택 사양이 홀드아웃에서 벤치마크를 이기지 못했습니다.")
    print(f"    → 역사적 평균이 더 나은 예측이었습니다. 이것도 유효한 발견입니다.")
    print(f"    → 다음 단계 고려 사항:")
    print(f"       1) 금리 변화율(Δ금리) 또는 더 긴 lag 탐색")
    print(f"       2) inv_duration 은 월간 모델로 이전하여 재검토")
    print(f"       3) 비선형/앙상블 모델 또는 이진 방향 분류로 접근")

print("\n" + "=" * 65)
print("분석 완료.")
print("=" * 65)
