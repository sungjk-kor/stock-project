# =============================================================================
# 외생변수 선행 lag 진단 — 사전백색화(pre-whitening) CCF
#
# 필요한 라이브러리 설치 (처음 한 번만):
#   pip install pandas numpy matplotlib statsmodels
#   pip install pmdarima          ← 있으면 자동으로 최적 ARIMA 차수를 찾아줌
#                                    없어도 AIC 격자탐색으로 대체됨
# =============================================================================

import sys
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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

# CSV 파일 경로
CSV_PATH = "data/processed/final_dataset.csv"

# 분석할 타깃 컬럼 (정상성 시계열이어야 함 — 예: 주간 로그수익률)
TARGET_COL = "sp500_log_return"

# 외생변수 후보 목록 (없는 컬럼은 자동 제외)
EXOG_COLUMNS = [
    "term_spread_10y_2y",
    "financial_conditions",
    "vix_log_return",
    "dxy_log_return",
    "oil_log_return",
    "cpi_change",
]

# 변수별 경제적 lag 창 (이 범위 안의 유의한 lag을 후보로 강조)
ECON_WINDOWS = {
    "term_spread_10y_2y":   (4, 12),
    "financial_conditions": (1, 8),
    "vix_log_return":       (1, 4),
    "dxy_log_return":       (1, 4),
    "oil_log_return":       (1, 4),
    "cpi_change":           (4, 12),
}

# CCF 계산 최대 lag (주 단위)
MAX_LAG = 16

# 결과 PNG 저장 폴더
OUTPUT_DIR = Path(__file__).parent

# ── 1. 데이터 불러오기 ────────────────────────────────────────────────────────

print("=" * 60)
print("■ 1단계: 데이터 불러오기")
print("=" * 60)

csv_path = Path(CSV_PATH)
if not csv_path.exists():
    print(f"[오류] 파일을 찾을 수 없습니다: {csv_path.resolve()}")
    print("  CSV_PATH를 실제 파일 경로로 바꿔주세요.")
    sys.exit(1)

try:
    df_raw = pd.read_csv(csv_path, parse_dates=["date"])
    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    print(f"  로드 완료: {len(df_raw):,}행 × {len(df_raw.columns)}열")
except Exception as e:
    print(f"[오류] CSV 읽기 실패: {e}")
    sys.exit(1)

# ── 2. 컬럼 존재 확인 ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 2단계: 컬럼 확인")
print("=" * 60)

# 타깃 컬럼 확인
if TARGET_COL not in df_raw.columns:
    print(f"[오류] 타깃 컬럼 '{TARGET_COL}'이 CSV에 없습니다.")
    print(f"  사용 가능한 컬럼 일부: {list(df_raw.columns[:10])}")
    sys.exit(1)

# 외생변수 컬럼 확인
available_exog = []
for col in EXOG_COLUMNS:
    if col in df_raw.columns:
        available_exog.append(col)
    else:
        print(f"  [경고] '{col}'이 CSV에 없어 분석에서 제외합니다.")

if not available_exog:
    print("[오류] 분석할 외생변수가 없습니다.")
    sys.exit(1)

print(f"\n  타깃 변수: {TARGET_COL}")
print(f"  외생변수 ({len(available_exog)}개): {available_exog}")

# ── 3. 전처리 ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 3단계: 전처리")
print("=" * 60)

# 분석에 쓸 컬럼만 선택하고 숫자형으로 변환
cols_needed = [TARGET_COL] + available_exog
df_work = df_raw[cols_needed].copy()
df_work = df_work.apply(pd.to_numeric, errors="coerce")  # 문자가 섞여 있어도 숫자로 변환

before = len(df_work)
df_work = df_work.dropna()  # 결측치(NaN)가 있는 행 제거
after = len(df_work)

print(f"  결측치 제거 전: {before:,}행  →  제거 후: {after:,}행")

if after < 50:
    print(f"  [경고] 표본이 {after}개로 너무 적습니다. CCF 결과를 신중하게 해석하세요.")
else:
    print(f"  최종 표본 수: {after:,}주")

# ── 4. ADF 단위근 검정 (정상성 확인) ─────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 4단계: ADF 단위근 검정")
print("=" * 60)
print("  (p값 < 0.05 이면 '정상', p값 ≥ 0.05 이면 '비정상 → 1차 차분 권장')")

def adf_report(series: pd.Series, name: str) -> bool:
    """ADF 검정 결과를 출력하고, 정상이면 True 반환."""
    stat, p_val, _, _, _, _ = adfuller(series.dropna(), autolag="AIC")
    is_stationary = p_val < 0.05
    mark = "✓ 정상" if is_stationary else "△ 비정상"
    print(f"  {name:35s}  ADF통계={stat:7.3f}  p값={p_val:.4f}  → {mark}")
    return is_stationary

for col in cols_needed:
    adf_report(df_work[col], col)

print()
print("  ※ 비정상 변수는 ARIMA 내부에서 d=1 차분을 선택해 자동 처리됩니다.")
print("     사전백색화 전처리에서 직접 차분하지 않아도 됩니다.")

# ── 5. 헬퍼 함수 ─────────────────────────────────────────────────────────────

def _select_arima_order(x_arr: np.ndarray) -> tuple[int, int, int]:
    """
    입력 시계열 x에 적합할 ARIMA(p,d,q) 차수를 결정합니다.
    pmdarima가 있으면 auto_arima(AIC 기준)를 쓰고,
    없으면 간단한 격자탐색(p∈{0,1,2}, d∈{0,1}, q∈{0,1,2})으로 대체합니다.
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
            return m.order   # (p, d, q)
        except Exception:
            pass

    # AIC 격자탐색 (pmdarima가 없거나 실패한 경우)
    best_aic = np.inf
    best_order = (1, 0, 0)
    for p in range(3):
        for d in range(2):
            for q in range(3):
                if p == 0 and d == 0 and q == 0:
                    continue   # ARIMA(0,0,0)은 그냥 상수 모델 → 의미 없음
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(x_arr, order=(p, d, q), trend="n").fit(disp=False)
                        if m.aic < best_aic:
                            best_aic = m.aic
                            best_order = (p, d, q)
                except Exception:
                    pass
    return best_order


def prewhiten_ccf(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = MAX_LAG,
) -> tuple[np.ndarray, np.ndarray, int, tuple, bool]:
    """
    사전백색화(pre-whitening) 교차상관함수(CCF)를 계산합니다.

    [순서]
    1) x에 ARIMA를 적합해 백색잡음 α (pre-whitened x)를 얻습니다.
    2) 같은 ARIMA 파라미터(계수)를 y에 그대로 적용해 필터링된 β를 얻습니다.
       ※ y를 y 자신의 모델이 아닌, x의 모델로 필터링하는 것이 핵심입니다.
          이렇게 해야 두 변수 사이의 공통 추세/자기상관에 의한 '가짜 상관'이 제거됩니다.
    3) α와 β의 교차상관을 계산합니다.
       [부호 규약] 양의 lag k = x가 y를 k주 앞선다 (x[t-k] ↔ y[t])

    Returns
    -------
    lags       : lag 배열 (0, 1, ..., max_lag)
    ccf_values : 각 lag에서의 교차상관 계수
    n_eff      : 유효 표본 수 (번인 제거 후)
    order      : 사용된 ARIMA 차수 (p, d, q)
    fallback   : True이면 필터 적용 실패로 원시 CCF를 사용한 것
    """
    # 결측치 없는 공통 구간으로 정렬
    df_pair = pd.DataFrame({"x": x, "y": y}).dropna()
    x_arr = df_pair["x"].values.astype(float)
    y_arr = df_pair["y"].values.astype(float)

    # 평균 제거 (두 시계열의 평균 차이가 ARIMA 필터를 왜곡하지 않도록)
    x_arr = x_arr - x_arr.mean()
    y_arr = y_arr - y_arr.mean()

    fallback = False

    try:
        # ① x의 ARIMA 차수 결정
        order = _select_arima_order(x_arr)
        p, d, q = order

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ② x에 ARIMA 적합 → 잔차(α) = 백색화된 x
            mod_x = SARIMAX(x_arr, order=order, trend="n")
            res_x = mod_x.fit(disp=False)
            alpha = res_x.resid.copy()

            # ③ 같은 파라미터를 y에 적용 → 잔차(β) = 필터링된 y
            #    mod_y.filter(params): ARIMA 계수를 고정한 채 칼만 필터만 돌림
            mod_y = SARIMAX(y_arr, order=order, trend="n")
            res_y = mod_y.filter(res_x.params)
            beta = res_y.resid.copy()

    except Exception as e:
        # 모든 시도가 실패하면 원시 시계열로 CCF 계산 (가짜 상관이 섞일 수 있음)
        print(f"    [주의] ARIMA 필터 적용 실패 → 원시 CCF 사용: {e}")
        alpha = x_arr
        beta = y_arr
        order = (0, 0, 0)
        fallback = True

    # 초반 번인(burn-in) 제거: ARIMA 초기화 구간은 잔차가 불안정할 수 있음
    burnin = (order[0] + order[1] + order[2]) + 5
    alpha_trim = np.array(alpha[burnin:], dtype=float)
    beta_trim  = np.array(beta[burnin:],  dtype=float)

    # NaN 제거 후 길이 맞추기
    n = min(len(alpha_trim), len(beta_trim))
    alpha_trim = alpha_trim[-n:]
    beta_trim  = beta_trim[-n:]
    mask = ~(np.isnan(alpha_trim) | np.isnan(beta_trim))
    alpha_trim = alpha_trim[mask]
    beta_trim  = beta_trim[mask]
    n_eff = len(alpha_trim)

    # CCF 계산
    # sm_ccf(beta, alpha, nlags) → corr(beta[t], alpha[t-k]), k=0,...,max_lag
    # → 양의 k = alpha(x의 백색화) 가 k주 앞서므로, 원래 x가 y를 k주 앞선다는 뜻
    ccf_vals = sm_ccf(beta_trim, alpha_trim, nlags=max_lag, adjusted=False)
    lags = np.arange(0, max_lag + 1)

    return lags, ccf_vals, n_eff, order, fallback


# ── 6. 사전백색화 CCF 계산 (변수별) ──────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 5단계: 사전백색화 CCF 계산")
print(f"  pmdarima 사용 가능 여부: {'예 (auto_arima)' if HAS_PMDARIMA else '아니오 (AIC 격자탐색 사용)'}")
print("=" * 60)

results = {}   # {변수명: (lags, ccf_vals, n_eff, order, fallback)}

for exog in available_exog:
    print(f"\n  처리 중: {exog}")
    lags, ccf_vals, n_eff, order, fallback = prewhiten_ccf(
        df_work[exog], df_work[TARGET_COL], max_lag=MAX_LAG
    )
    results[exog] = (lags, ccf_vals, n_eff, order, fallback)

    flag = " [원시CCF]" if fallback else ""
    print(f"    ARIMA 차수: {order}  |  유효 표본: {n_eff}주{flag}")

# ── 7. 시각화 — 변수별 CCF stem 그래프 ───────────────────────────────────────

print("\n" + "=" * 60)
print("■ 6단계: 시각화")
print("=" * 60)

plt.rcParams["axes.unicode_minus"] = False
try:
    plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 한글 글꼴
except Exception:
    pass

n_vars = len(available_exog)
ncols = 2
nrows = (n_vars + 1) // 2   # 2열 기준 행 수

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4.5))
axes = np.array(axes).flatten()   # 1차원 배열로 통일

for idx, exog in enumerate(available_exog):
    ax = axes[idx]
    lags, ccf_vals, n_eff, order, fallback = results[exog]

    # 유의구간: ±1.96 / sqrt(유효 표본 수)
    sig_bound = 1.96 / np.sqrt(n_eff)

    # ─ 경제적 lag 창 음영 ─
    econ_lo, econ_hi = ECON_WINDOWS.get(exog, (1, MAX_LAG))
    ax.axvspan(econ_lo, econ_hi, alpha=0.10, color="steelblue", label=f"경제적 창 ({econ_lo}~{econ_hi}주)")

    # ─ 유의구간 선 ─
    ax.axhline( sig_bound, color="tomato",  linestyle="--", linewidth=1.2, label=f"±1.96/√N ({sig_bound:.3f})")
    ax.axhline(-sig_bound, color="tomato",  linestyle="--", linewidth=1.2)
    ax.axhline(0,          color="dimgray", linestyle="-",  linewidth=0.7)

    # ─ stem 그래프 (막대 + 점) ─
    # 유의한 lag은 색을 달리 표시
    for k, cv in zip(lags, ccf_vals):
        in_econ = econ_lo <= k <= econ_hi
        is_sig  = abs(cv) > sig_bound
        if is_sig and in_econ:
            color = "#e74c3c"   # 빨강 — 경제적 창 안, 유의
        elif is_sig:
            color = "#f39c12"   # 주황 — 경제적 창 밖, 유의
        else:
            color = "#aab7c4"   # 회색 — 비유의
        ax.vlines(k, 0, cv, colors=color, linewidth=1.8)
        ax.plot(k, cv, "o", color=color, markersize=4)

    # ─ 축 및 레이블 ─
    order_str = f"ARIMA{order}"
    fallback_str = " [원시CCF]" if fallback else ""
    ax.set_title(
        f"{exog}\n({order_str}, N={n_eff}){fallback_str}",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("lag (주)  ←  양의 lag = x가 y를 앞섬", fontsize=8)
    ax.set_ylabel("교차상관계수", fontsize=8)
    ax.set_xlim(-0.5, MAX_LAG + 0.5)
    ax.set_xticks(range(0, MAX_LAG + 1, 2))
    ax.tick_params(labelsize=8)

    # 범례 (첫 번째 서브플롯에만 상세히)
    if idx == 0:
        legend_handles = [
            mpatches.Patch(color="#e74c3c", label="유의 (창 안)"),
            mpatches.Patch(color="#f39c12", label="유의 (창 밖)"),
            mpatches.Patch(color="#aab7c4", label="비유의"),
            mpatches.Patch(color="steelblue", alpha=0.3, label="경제적 창"),
        ]
        ax.legend(handles=legend_handles, fontsize=7, loc="upper right")

# 남은 빈 서브플롯 숨기기
for i in range(n_vars, len(axes)):
    axes[i].set_visible(False)

fig.suptitle(
    f"사전백색화 CCF — 외생변수별 선행 lag 진단\n"
    f"(타깃: {TARGET_COL}  |  양의 lag = 외생변수가 타깃을 앞섬)",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()

ccf_path = OUTPUT_DIR / "ccf_prewhitened.png"
fig.savefig(ccf_path, dpi=150, bbox_inches="tight")
print(f"  CCF 그래프 저장 완료: {ccf_path}")
plt.show()

# ── 8. 텍스트 해석 출력 ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("■ 7단계: 해석 — 변수별 후보 lag 제안")
print(f"  (기준: 경제적 창 안에서 |CCF| > 1.96/√N 인 lag)")
print("=" * 60)

for exog in available_exog:
    lags, ccf_vals, n_eff, order, fallback = results[exog]
    sig_bound = 1.96 / np.sqrt(n_eff)
    econ_lo, econ_hi = ECON_WINDOWS.get(exog, (1, MAX_LAG))

    # 경제적 창 안에서 유의한 lag 추출 (|CCF| 내림차순)
    sig_in_window = [
        (int(k), float(cv))
        for k, cv in zip(lags, ccf_vals)
        if econ_lo <= k <= econ_hi and abs(cv) > sig_bound
    ]
    sig_in_window.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n◆ {exog}  (경제적 창: lag {econ_lo}~{econ_hi}주)")

    if not sig_in_window:
        print(f"  → 경제적 창 안에서 유의한 선행 lag 없음  (유의구간 ±{sig_bound:.3f})")
        print(f"  → 선행 신호 약함 — 후보 보류 검토 (모델에서 제외하거나 lag=1만 시험)")
    else:
        print(f"  유의한 lag (|CCF| 내림차순):")
        for k, cv in sig_in_window:
            direction = "양(+)" if cv > 0 else "음(-)"
            print(f"    lag {k:2d}주  CCF={cv:+.3f}  ({direction}의 선행 상관)")

        # 후보 lag 제안: 절댓값 상위 1~2개
        top = sig_in_window[:2]
        top_lags = [k for k, _ in top]
        if len(top_lags) == 1:
            print(f"  → 후보 lag 제안: lag_{top_lags[0]}")
        else:
            print(f"  → 후보 lag 제안: lag_{top_lags[0]}  (1순위),  lag_{top_lags[1]}  (2순위)")
        print(f"     두 lag을 모두 넣으면 다중공선성 확인 필요.")

print("\n" + "=" * 60)
print("분석 완료.")
print("=" * 60)
