# =============================================================================
# 외생변수 후보 lag 선정 스크립트 — 사전백색화 CCF 기반
#
# 이 코드의 역할:
#   CCF로 각 외생변수가 "몇 주 앞서는지" 진단해서
#   워크포워드(walk-forward)에서 검증할 후보 lag 목록을 추린다.
#   최종 채택은 이후 워크포워드 결과가 결정한다.
#
# 필요한 라이브러리 설치 (처음 한 번만):
#   pip install pandas numpy statsmodels
#   pip install pmdarima    ← 있으면 자동 최적 ARIMA 차수, 없어도 동작
# =============================================================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import ccf as sm_ccf

# pmdarima 가 설치되어 있으면 사용, 없으면 AIC 격자탐색으로 대체
try:
    from pmdarima import auto_arima as _auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

# ── 0. 설정 — 여기만 수정하면 됩니다 ─────────────────────────────────────────

CSV_PATH   = "data/processed/final_dataset.csv"   # CSV 파일 경로
TARGET_COL = "sp500_log_return"                    # 예측하려는 타깃 컬럼

# 분석할 외생변수 목록 (없는 컬럼은 자동으로 제외)
EXOG_COLUMNS = [
    "term_spread_10y_2y",
    "financial_conditions",
    "vix_log_return",
    "dxy_log_return",
    "oil_log_return",
    "cpi_change",
]

# 변수별 경제적 lag 창 (단위: 주)
# 이 범위 안의 lag만 "경제적으로 말이 되는 후보"로 본다
ECON_WINDOWS = {
    "term_spread_10y_2y":   (4, 12),
    "financial_conditions": (1, 8),
    "vix_log_return":       (1, 4),
    "dxy_log_return":       (1, 4),
    "oil_log_return":       (1, 4),
    "cpi_change":           (4, 12),
}

MAX_LAG               = 16    # CCF 계산 최대 lag (주)
STRONG_MULT           = 2.5   # ratio >= 이 값이면 "강한 후보"
MAX_CANDIDATES_PER_VAR = 2    # 변수당 최대 후보 lag 개수

OUTPUT_DIR = Path(__file__).parent   # 결과 저장 폴더

# ── 1. 데이터 불러오기 ────────────────────────────────────────────────────────

print("=" * 65)
print("■ 1단계: 데이터 불러오기")
print("=" * 65)

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] 파일을 찾을 수 없습니다: {csv_path.resolve()}")
    print("  CSV_PATH를 실제 경로로 수정해 주세요.")
    sys.exit(1)

try:
    df_raw = pd.read_csv(csv_path, parse_dates=["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    print(f"  로드 완료: {len(df_raw):,}행 × {len(df_raw.columns)}열")
except Exception as e:
    print(f"[오류] CSV 읽기 실패: {e}")
    sys.exit(1)

# ── 2. 컬럼 존재 확인 ─────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 2단계: 컬럼 확인")
print("=" * 65)

if TARGET_COL not in df_raw.columns:
    print(f"[오류] 타깃 컬럼 '{TARGET_COL}'이 없습니다.")
    sys.exit(1)

available_exog = []
for col in EXOG_COLUMNS:
    if col in df_raw.columns:
        available_exog.append(col)
    else:
        print(f"  [경고] '{col}'이 CSV에 없어 제외합니다.")

if not available_exog:
    print("[오류] 분석할 외생변수가 없습니다.")
    sys.exit(1)

print(f"  타깃 변수  : {TARGET_COL}")
print(f"  외생변수   : {available_exog}")

# ── 3. 전처리 ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 3단계: 전처리")
print("=" * 65)

cols_needed = [TARGET_COL] + available_exog
df_work = df_raw[cols_needed].copy()
df_work = df_work.apply(pd.to_numeric, errors="coerce")

before = len(df_work)
df_work = df_work.dropna()
after  = len(df_work)

print(f"  결측치 제거 전: {before:,}행  →  제거 후: {after:,}행")

if after < 50:
    print(f"  [경고] 표본 {after}개는 너무 적습니다. 결과를 신중하게 해석하세요.")
else:
    print(f"  최종 표본 수: {after:,}주  ← N값으로 유의구간을 계산합니다.")

# ── 4. ADF 단위근 검정 ────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 4단계: ADF 단위근 검정  (p<0.05 = 정상, p≥0.05 = 비정상)")
print("=" * 65)

def _is_stationary(series: pd.Series) -> bool:
    """ADF 검정: 정상이면 True."""
    _, p_val, *_ = adfuller(series.dropna(), autolag="AIC")
    return p_val < 0.05

for col in cols_needed:
    s = df_work[col]
    stat_orig = _is_stationary(s)
    if stat_orig:
        print(f"  {col:35s} → 정상")
    else:
        stat_diff = _is_stationary(s.diff().dropna())
        diff_note = "1차 차분 후 정상" if stat_diff else "차분 후에도 비정상"
        print(f"  {col:35s} → 비정상  ({diff_note}, ARIMA d=1 권장)")

print()
print("  ※ 비정상 변수는 ARIMA의 d=1 옵션으로 자동 처리됩니다.")
print("     여기서 직접 차분하지 않아도 됩니다.")

# ── 5. 다중검정 경고 (반드시 확인) ───────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 다중검정 경고 — 반드시 읽어 주세요")
print("=" * 65)

expected_false = round(0.05 * MAX_LAG, 1)
print(
    f"  lag 0~{MAX_LAG}까지 {MAX_LAG+1}개를 5% 유의수준으로 검정하면, "
    f"아무 관계 없어도 순전히 우연으로 약 {expected_false}개가 유의해 보입니다."
)
print(
    "  → band 바로 위(ratio 1.0~1.5 수준)의 단발 신호는 노이즈일 가능성이 높습니다."
)
print(
    "  → 경계 후보(marginal)는 반드시 워크포워드에서 실제 성능을 검증하세요."
)

# ── 6. 헬퍼 함수 ─────────────────────────────────────────────────────────────

def _select_arima_order(x_arr: np.ndarray) -> tuple[int, int, int]:
    """
    ARIMA 차수를 결정합니다.
    pmdarima가 있으면 auto_arima(AIC 기준), 없으면 격자탐색.
    """
    if HAS_PMDARIMA:
        try:
            m = _auto_arima(
                x_arr,
                start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
                seasonal=False, stepwise=True,
                suppress_warnings=True, error_action="ignore",
                information_criterion="aic", trace=False,
            )
            return m.order
        except Exception:
            pass

    # AIC 격자탐색 (pmdarima 없거나 실패 시)
    best_aic   = np.inf
    best_order = (1, 0, 0)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(x_arr, order=(p, d, q), trend="n").fit(disp=False)
                        if m.aic < best_aic:
                            best_aic   = m.aic
                            best_order = (p, d, q)
                except Exception:
                    pass
    return best_order


def prewhiten_ccf(
    x: pd.Series,
    y: pd.Series,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray, int, tuple, bool]:
    """
    사전백색화(pre-whitening) CCF를 계산합니다.

    [원리]
    x와 y가 각자 자기상관을 가지면 실제 관계가 없어도 상관이 높게 나옵니다.
    이를 막으려면:
      1) x에 ARIMA를 적합 → 잔차 alpha (자기상관이 제거된 순수 노이즈)
      2) 같은 ARIMA 필터를 y에도 적용 → 잔차 beta (y에서도 같은 패턴 제거)
      3) alpha와 beta의 교차상관을 계산 → 가짜 상관이 제거된 순수 관계

    [부호 규약]
    양의 lag k = x가 타깃 y를 k주 앞선다  (x[t-k] ↔ y[t])
    """
    # 공통 구간 정렬 + 결측치 제거
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    x_arr = pair["x"].values.astype(float)
    y_arr = pair["y"].values.astype(float)

    # 평균 제거: 두 시계열의 평균 차이가 필터를 왜곡하지 않도록
    x_arr -= x_arr.mean()
    y_arr -= y_arr.mean()

    fallback = False
    order    = (1, 0, 0)

    try:
        order = _select_arima_order(x_arr)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ① x에 ARIMA 적합 → alpha = 백색화된 x 잔차
            res_x = SARIMAX(x_arr, order=order, trend="n").fit(disp=False)
            alpha = res_x.resid.copy()

            # ② 같은 파라미터로 y 필터링 → beta = 필터링된 y 잔차
            #    .filter(params): ARIMA 계수를 고정한 채 칼만 필터만 실행
            res_y = SARIMAX(y_arr, order=order, trend="n").filter(res_x.params)
            beta  = res_y.resid.copy()

    except Exception as e:
        # 실패 시 원시 시계열로 CCF (가짜 상관 위험 있음)
        print(f"    [주의] ARIMA 필터 실패, 원시 CCF 사용: {e}")
        alpha    = x_arr
        beta     = y_arr
        fallback = True

    # 번인(burn-in) 구간 제거: 초기화 과정의 불안정한 잔차를 버림
    burnin     = sum(order) + 5
    alpha_trim = np.array(alpha[burnin:], dtype=float)
    beta_trim  = np.array(beta[burnin:],  dtype=float)

    # 길이 맞추기 + NaN 제거
    n = min(len(alpha_trim), len(beta_trim))
    alpha_trim, beta_trim = alpha_trim[-n:], beta_trim[-n:]
    mask       = ~(np.isnan(alpha_trim) | np.isnan(beta_trim))
    alpha_trim = alpha_trim[mask]
    beta_trim  = beta_trim[mask]
    n_eff      = len(alpha_trim)

    # CCF: sm_ccf(beta, alpha) → corr(beta[t], alpha[t-k])
    # → 양의 k = alpha(x)가 k주 앞섬 ✓
    ccf_vals = sm_ccf(beta_trim, alpha_trim, nlags=max_lag, adjusted=False)
    lags     = np.arange(0, max_lag + 1)

    return lags, ccf_vals, n_eff, order, fallback


# ── 7. CCF 계산 + 후보 lag 선정 ──────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 5단계: 사전백색화 CCF 계산 및 후보 lag 선정")
print(f"  pmdarima 사용 여부: {'예 (auto_arima)' if HAS_PMDARIMA else '아니오 (AIC 격자탐색)'}")
print(f"  강한 후보 기준    : ratio >= {STRONG_MULT}  (= |CCF| / band)")
print(f"  경계 후보 기준    : 1.0 <= ratio < {STRONG_MULT}")
print("=" * 65)

# 결과를 모을 구조
# all_results[exog] = list of dict {lag, ccf, ratio, grade}
all_results: dict[str, list[dict]] = {}

for exog in available_exog:
    lags, ccf_vals, n_eff, order, fallback = prewhiten_ccf(
        df_work[exog], df_work[TARGET_COL], max_lag=MAX_LAG
    )

    # 유의구간: ±1.96 / sqrt(유효 표본 수)
    band = 1.96 / np.sqrt(n_eff)

    # 경제적 창 + 양의 lag + |CCF| > band 인 lag만 추출
    econ_lo, econ_hi = ECON_WINDOWS.get(exog, (1, MAX_LAG))
    candidates = []
    for k, cv in zip(lags, ccf_vals):
        k = int(k)
        if k == 0:
            continue                         # lag=0 은 동시 발생이므로 제외
        if not (econ_lo <= k <= econ_hi):
            continue                         # 경제적 창 밖 제외
        ratio = abs(cv) / band
        if ratio < 1.0:
            continue                         # 유의구간 못 넘는 것 제외
        grade = "strong" if ratio >= STRONG_MULT else "marginal"
        candidates.append({
            "lag":   k,
            "ccf":   float(cv),
            "ratio": round(ratio, 2),
            "grade": grade,
        })

    # |CCF| 내림차순 정렬 후 상위 max_candidates_per_var 개만 보관
    candidates.sort(key=lambda d: abs(d["ccf"]), reverse=True)
    candidates = candidates[:MAX_CANDIDATES_PER_VAR]

    all_results[exog] = candidates

# ── 8. 결과 출력 — 변수별 표 ─────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 6단계: 결과 — 변수별 후보 lag 표")
print("=" * 65)

# 유의구간 값을 한 번 더 계산해서 표 옆에 참고 출력
_, _, n_eff_ref, _, _ = prewhiten_ccf(
    df_work[available_exog[0]], df_work[TARGET_COL], max_lag=MAX_LAG
)
band_ref = 1.96 / np.sqrt(n_eff_ref)
print(f"  유의구간 band ≈ ±{band_ref:.4f}  (1.96 / √{n_eff_ref})")
print(f"  강한 후보 기준: |CCF| ≥ {round(band_ref * STRONG_MULT, 4)}  (band × {STRONG_MULT})\n")

# 헤더
header = f"  {'변수':<28} {'후보lag':>6} {'CCF값':>8} {'부호':>5} {'ratio':>7} {'분류':>10}"
print(header)
print("  " + "-" * (len(header) - 2))

for exog, candidates in all_results.items():
    econ_lo, econ_hi = ECON_WINDOWS.get(exog, (1, MAX_LAG))
    if not candidates:
        # 유의한 후보가 없는 변수
        print(f"  {exog:<28} {'(없음)':>6}   선행 신호 없음 → 제외 권장")
    else:
        for i, c in enumerate(candidates):
            sign = "양(+)" if c["ccf"] > 0 else "음(-)"
            var_label = exog if i == 0 else ""
            print(
                f"  {var_label:<28} "
                f"lag {c['lag']:>2d}주 "
                f"{c['ccf']:>+8.4f} "
                f"{sign:>5} "
                f"{c['ratio']:>6.2f}x "
                f"  {c['grade']:>10}"
            )

# ── 9. 워크포워드 입력용 dict 출력 ───────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 7단계: 워크포워드 입력용 dict")
print("=" * 65)
print("  # 아래 dict를 복사해서 워크포워드 코드에 바로 붙여넣으세요.")
print("  # strong = 강한 선행 신호, marginal = 경계값(반드시 검증 필요)\n")

# candidate_lags: 모든 후보 (strong + marginal)
candidate_lags: dict[str, list[int]] = {}
strong_lags:    dict[str, list[int]] = {}
marginal_lags:  dict[str, list[int]] = {}

for exog, candidates in all_results.items():
    all_c     = [c["lag"] for c in candidates]
    strong_c  = [c["lag"] for c in candidates if c["grade"] == "strong"]
    marg_c    = [c["lag"] for c in candidates if c["grade"] == "marginal"]
    if all_c:
        candidate_lags[exog] = all_c
    if strong_c:
        strong_lags[exog]   = strong_c
    if marg_c:
        marginal_lags[exog] = marg_c

# 예쁘게 dict 형태로 출력
def _fmt_dict(d: dict, name: str) -> str:
    lines = [f"{name} = {{"]
    for k, v in d.items():
        lines.append(f'    "{k}": {v},')
    lines.append("}")
    return "\n".join(lines)

print(_fmt_dict(candidate_lags, "candidate_lags"))
print()
print(_fmt_dict(strong_lags,   "strong_lags    # strong만 (ratio >= " + str(STRONG_MULT) + ")"))
print()
print(_fmt_dict(marginal_lags, "marginal_lags  # marginal만 (반드시 워크포워드 검증)"))

# Python 파일로도 저장
output_py = OUTPUT_DIR / "candidate_lags.py"
with open(output_py, "w", encoding="utf-8") as f:
    f.write("# 사전백색화 CCF 기반 후보 lag — 워크포워드 입력용\n")
    f.write(f"# 생성 기준: MAX_LAG={MAX_LAG}, STRONG_MULT={STRONG_MULT}, "
            f"MAX_CANDIDATES_PER_VAR={MAX_CANDIDATES_PER_VAR}\n\n")
    f.write(_fmt_dict(candidate_lags, "candidate_lags") + "\n\n")
    f.write(_fmt_dict(strong_lags,   "strong_lags    # strong만") + "\n\n")
    f.write(_fmt_dict(marginal_lags, "marginal_lags  # marginal만") + "\n")
print(f"\n  dict 파일 저장 완료: {output_py}")

# ── 10. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

strong_vars   = [e for e, c in all_results.items() if any(x["grade"] == "strong"   for x in c)]
marginal_vars = [e for e, c in all_results.items() if any(x["grade"] == "marginal" for x in c) and
                                                       not any(x["grade"] == "strong" for x in c)]
no_signal_vars = [e for e, c in all_results.items() if not c]

if strong_vars:
    print(f"\n  ★ 강한 선행 후보 (ratio ≥ {STRONG_MULT}, 워크포워드 우선 포함 권장):")
    for v in strong_vars:
        lags_str = ", ".join(f"lag_{c['lag']}" for c in all_results[v] if c["grade"] == "strong")
        print(f"    - {v}: {lags_str}")

if marginal_vars:
    print(f"\n  △ 경계 후보 (ratio 1.0~{STRONG_MULT}, 단독 노이즈 가능성 있음 → 워크포워드 검증 필수):")
    for v in marginal_vars:
        lags_str = ", ".join(f"lag_{c['lag']}" for c in all_results[v])
        print(f"    - {v}: {lags_str}")

if no_signal_vars:
    print(f"\n  ✗ 선행 신호 없음 (제외 권장):")
    for v in no_signal_vars:
        print(f"    - {v}")

print()
print("  [해석 안내]")
print("  · 강한 후보는 경제적 창 안에서 노이즈 수준의 2.5배 이상 상관이 확인된 변수입니다.")
print("  · 경계 후보는 유의하긴 하지만 CCF 하나만으로 채택하기에는 근거가 약합니다.")
print("  · 어떤 lag을 실제로 쓸지는 워크포워드 성능(RMSE, 방향성 등)으로 최종 결정하세요.")

print("\n" + "=" * 65)
print("분석 완료.")
print("=" * 65)
