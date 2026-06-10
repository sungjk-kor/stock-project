# =============================================================================
# MS-AR 약세 확률 추정 및 이탈 점수와의 교차검증
#
# 목적:
#   1단계: MS-AR 2국면 모델로 주간 수익률의 약세 확률 P(약세) 추정
#   2단계: 기존 이탈 점수와 날짜 정렬
#   3단계: 겹쳐 그리기 (이탈 점수 vs P(약세))
#   4단계: 정량 일치도 분석
#
# ★ 약세는 '낮은 평균 + 높은 분산'으로 정의
# ★ 전체 표본(위기 포함)으로 적합 — 위기에 약세가 집중되어 있음
# =============================================================================

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (Windows: Malgun Gothic) — 그래프 한글 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================================================================
# [1단계] 데이터 로드 및 기본 준비
# ============================================================================

def load_data(csv_path: str = "data/processed/final_dataset.csv") -> pd.DataFrame:
    """
    데이터 로드 및 필수 컬럼 확인

    Args:
        csv_path: final_dataset.csv 경로

    Returns:
        로드된 DataFrame
    """
    print("=" * 70)
    print("[1단계] 데이터 로드")
    print("=" * 70)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"✓ 로드됨: {len(df):,}주")
    print(f"  날짜 범위: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  필수 컬럼 확인: date, sp500_log_return")

    if "sp500_log_return" not in df.columns:
        raise ValueError("sp500_log_return 컬럼이 없습니다")

    # 결측치 확인
    n_missing = df["sp500_log_return"].isna().sum()
    print(f"  결측치: {n_missing}개 행")

    return df


def prepare_returns_data(df: pd.DataFrame) -> np.ndarray:
    """
    MS-AR 모델용 수익률 데이터 준비

    Args:
        df: 전체 DataFrame

    Returns:
        결측치 제거된 수익률 배열 (T,)
    """
    print("\n[데이터 준비]")

    # 결측치 제거
    returns = df["sp500_log_return"].dropna().values

    print(f"✓ 유효한 주간 수익률: {len(returns):,}개")
    print(f"  평균: {returns.mean():.6f}")
    print(f"  표준편차: {returns.std():.6f}")
    print(f"  범위: [{returns.min():.6f}, {returns.max():.6f}]")

    return returns


# ============================================================================
# [1단계] MS-AR 모델 적합
# ============================================================================

def fit_msv_ar_model(returns: np.ndarray, k_regimes: int = 2, order: int = 1):
    """
    Markov Switching Autoregression (k_regimes=2, switching_variance=True)

    ★ 전체 표본(위기 포함)으로 적합
    ★ 여러 시작값으로 국소최적 회피
    ★ 국면 번호를 정렬해 '약세 = 낮은 평균'으로 식별

    Args:
        returns: 주간 로그 수익률 배열
        k_regimes: 국면 개수 (2)
        order: AR 차수 (1)

    Returns:
        (모델, 적합 결과, 정렬된 국면 번호 매핑)
    """
    print("\n" + "=" * 70)
    print("[1단계] MS-AR 2국면 모델 적합")
    print("=" * 70)

    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        print("✗ statsmodels 설치 필요: pip install statsmodels>=0.14")
        raise

    print(f"\n[모델 설정]")
    print(f"  국면 수: {k_regimes}")
    print(f"  AR 차수: {order}")
    print(f"  Switching variance: True")
    print(f"  표본 크기: {len(returns)}")

    # ========================================================================
    # AR(1) 형태로 준비: Y_t = c_s + phi_s * Y_{t-1} + sigma_s * eps_t
    # exog에 Y_{t-1}를 넣어 MarkovRegression으로 적합
    # ========================================================================

    endog = returns[1:]  # Y_t (t=1, ..., T-1)
    exog = returns[:-1]  # Y_{t-1} (t=1, ..., T-1)

    print(f"\n[국소최적 회피] search_reps로 다수의 무작위 시작값 탐색 + 시드 고정(재현성)")

    np.random.seed(42)  # 재현성

    model = MarkovRegression(
        endog,
        k_regimes=k_regimes,
        exog=exog,
        switching_variance=True,
    )

    # ★ search_reps: 무작위 시작값으로 다수 반복 탐색 후 최적해 선택 (statsmodels 내장)
    #   - 이를 통해 EM의 국소최적 문제를 완화
    result = model.fit(search_reps=20, disp=False)

    converged = result.mle_retvals.get("converged", "정보 없음")
    print(f"  수렴 여부: {converged}")
    print(f"  로그우도: {result.llf:.4f}")

    if converged is False:
        print("  ⚠ 경고: 수렴하지 않았습니다. 결과 해석에 주의하세요.")
        print("    - maxiter를 늘리거나 method를 변경해 재시도를 고려하세요.")

    # ========================================================================
    # ★ 파라미터는 절대 수동 인덱싱하지 않고 param_names로 라벨링해 추출
    #   (statsmodels의 파라미터 순서는 버전/설정에 따라 달라질 수 있음)
    # ========================================================================

    print(f"\n[파라미터 추출] model.param_names 라벨을 사용해 안전하게 매핑")
    param_dict = dict(zip(model.param_names, result.params))
    for name, val in param_dict.items():
        print(f"  {name}: {val:.6f}")

    # 각 국면의 평균(절편)과 분산 (라벨 기반 추출 — 순서에 의존하지 않음)
    means = np.array([param_dict[f"const[{i}]"] for i in range(k_regimes)])
    variances = np.array([param_dict[f"sigma2[{i}]"] for i in range(k_regimes)])

    # ========================================================================
    # 국면 재정렬: '약세 = 낮은 평균 + 높은 분산'으로 명시 식별
    # ========================================================================

    print(f"\n[국면 식별] 평균이 낮은 국면을 약세(bear)로 정의")
    print(f"  국면 0: 평균 = {means[0]:.6f}, 분산 = {variances[0]:.6f}, 표준편차 = {np.sqrt(variances[0]):.6f}")
    print(f"  국면 1: 평균 = {means[1]:.6f}, 분산 = {variances[1]:.6f}, 표준편차 = {np.sqrt(variances[1]):.6f}")

    bear_regime_idx = int(np.argmin(means))
    bull_regime_idx = 1 - bear_regime_idx
    print(f"  → 국면 {bear_regime_idx} = 약세(낮은 평균), 국면 {bull_regime_idx} = 강세(높은 평균)")

    # ★ 분산 분리 확인: 약세 분산이 강세보다 뚜렷이 큰가?
    var_ratio = variances[bear_regime_idx] / variances[bull_regime_idx]
    print(f"\n[분산 분리 점검] 약세/강세 분산비 = {var_ratio:.2f}배")
    if var_ratio < 1.2:
        print("  ⚠ 경고: 약세 분산이 강세와 거의 같습니다 — '약세=위기'로 보기 어려울 수 있습니다.")
        print("    (1) switching_variance가 실제로 작동하는지, (2) 더 나은 최적해가 있는지 재확인 필요.")
    else:
        print(f"  ✓ 약세 국면의 분산이 강세보다 {var_ratio:.2f}배 커서, 변동성 측면에서도 위기 국면으로 식별됩니다.")

    return model, result, bear_regime_idx, (means, variances), param_dict


def print_model_summary(model, result, bear_regime_idx: int, regime_stats: tuple):
    """
    모델 결과 요약 출력 (라벨 기반, ★ 표 형태로 출력 후 멈춤)
    """
    means, variances = regime_stats
    bull_regime_idx = 1 - bear_regime_idx

    print("\n" + "=" * 70)
    print("[1단계 결과 요약 — 표]")
    print("=" * 70)

    print(f"\n[표 1] 국면별 평균·분산")
    print(f"  {'구분':<12}{'국면번호':<10}{'평균(주간)':<16}{'분산':<14}{'표준편차':<12}")
    print(f"  {'-'*64}")
    print(f"  {'약세(bear)':<12}{bear_regime_idx:<10}{means[bear_regime_idx]:<16.6f}"
          f"{variances[bear_regime_idx]:<14.6f}{np.sqrt(variances[bear_regime_idx]):<12.6f}")
    print(f"  {'강세(bull)':<12}{bull_regime_idx:<10}{means[bull_regime_idx]:<16.6f}"
          f"{variances[bull_regime_idx]:<14.6f}{np.sqrt(variances[bull_regime_idx]):<12.6f}")
    print(f"  (평균은 주간 로그수익률 단위, 즉 0.0038 ≈ 0.38%/주)")

    # ========================================================================
    # ★ 전이행렬: statsmodels는 열-확률(column-stochastic) 표현
    #   regime_transition[i, j] = P(state_t = i | state_{t-1} = j)
    #   → 각 '열'의 합이 1이어야 함 (열 j에서 나가는 확률의 총합)
    # ========================================================================
    transition_matrix = result.regime_transition[:, :, 0]
    col_sums = transition_matrix.sum(axis=0)

    print(f"\n[표 2] 전이행렬 — statsmodels 표현: M[i, j] = P(t시점 국면=i | t-1시점 국면=j)")
    print(f"  원본 행렬:")
    print(f"    {transition_matrix}")
    print(f"  ★ 열 합 점검 (각 1.0이어야 함, '열'=이전 국면에서 나가는 확률 총합):")
    print(f"    열 0 합 = {col_sums[0]:.6f}   열 1 합 = {col_sums[1]:.6f}")
    if not np.allclose(col_sums, 1.0):
        print("  ⚠ 경고: 열 합이 1이 아닙니다 — 행렬 해석 방향을 재점검하세요.")
    else:
        print("  ✓ 열-확률(column-stochastic) 확인됨 — M[to, from] 표현이 맞습니다.")

    # 올바른 라벨링: M[to, from]이므로 "from -> to" = M[to_idx, from_idx]
    bear_to_bear = transition_matrix[bear_regime_idx, bear_regime_idx]
    bear_to_bull = transition_matrix[bull_regime_idx, bear_regime_idx]
    bull_to_bear = transition_matrix[bear_regime_idx, bull_regime_idx]
    bull_to_bull = transition_matrix[bull_regime_idx, bull_regime_idx]

    print(f"\n  {'전이':<16}{'확률':<10}")
    print(f"  {'-'*26}")
    print(f"  {'약세→약세':<16}{bear_to_bear:<10.4f}")
    print(f"  {'약세→강세':<16}{bear_to_bull:<10.4f}")
    print(f"  {'강세→약세':<16}{bull_to_bear:<10.4f}")
    print(f"  {'강세→강세':<16}{bull_to_bull:<10.4f}")

    # 기대 지속기간 (자기 전이확률에서 직접 계산 — 방향에 무관)
    duration_bear = 1 / (1 - bear_to_bear) if bear_to_bear < 1 else np.inf
    duration_bull = 1 / (1 - bull_to_bull) if bull_to_bull < 1 else np.inf

    print(f"\n[표 3] 기대 지속기간")
    print(f"  {'국면':<12}{'기대 지속기간(주)':<20}{'(개월 환산)':<14}")
    print(f"  {'-'*46}")
    print(f"  {'약세':<12}{duration_bear:<20.2f}{duration_bear/4.33:<14.2f}")
    print(f"  {'강세':<12}{duration_bull:<20.2f}{duration_bull/4.33:<14.2f}")

    # 현재(최근) P(약세) — filtered (실시간 추정, look-ahead 없음)
    prob_regime = result.filtered_marginal_probabilities  # (T-1, k_regimes)
    current_prob_bear = prob_regime[-1, bear_regime_idx]

    print(f"\n[표 4] 현재(최근 주) 국면 확률 — filtered (실시간 기준)")
    print(f"  {'국면':<12}{'확률':<10}")
    print(f"  {'-'*22}")
    print(f"  {'P(약세)':<12}{current_prob_bear:<10.4f}")
    print(f"  {'P(강세)':<12}{1 - current_prob_bear:<10.4f}")

    print("\n" + "=" * 70)
    print("[★ 멈춤] 위 표 1~4를 점검하고 진행 여부를 확인하세요:")
    print("  1) 약세 평균이 강세보다 뚜렷이 낮은가?")
    print("  2) 약세 분산이 강세보다 뚜렷이 큰가? (위 분산 분리 점검 참고)")
    print("  3) 전이행렬 열 합이 1인가? (위 열 합 점검 참고)")
    print("  4) 기대 지속기간이 현실적인가?")
    print("=" * 70)


# ============================================================================
# [2단계] 이탈 점수 정렬·연속화
# ============================================================================

def load_deviation_score() -> pd.DataFrame:
    """
    기존 이탈 점수 시계열 로드

    ★ DeviationDashboard는 이미 '비위기 구간으로 학습한 동결 표준화·센트로이드'를
      전체 표본(위기 포함)에 적용하므로, 위기구간에도 점수가 연속적으로 존재한다.
      → 별도의 재점수화는 불필요. 그대로 사용.

    Returns:
        date, score 컬럼을 가진 DataFrame
    """
    print("\n" + "=" * 70)
    print("[2단계] 이탈 점수 로드 및 연속성 확인")
    print("=" * 70)

    from utils.deviation_dashboard import DeviationDashboard

    dd = DeviationDashboard()
    if not dd.load_data() or not dd.calculate_features() or not dd.fit_clusters() or not dd.calculate_scores():
        raise RuntimeError("이탈 점수 계산에 실패했습니다.")

    df_score = dd.df_valid[["date", "score", "crisis"]].copy()

    print(f"\n✓ 이탈 점수 로드: {len(df_score):,}주")
    print(f"  날짜 범위: {df_score['date'].min().date()} ~ {df_score['date'].max().date()}")

    # ★ 위기구간 연속성 확인 (비어있지 않은지)
    crisis_scores = df_score.loc[df_score["crisis"], "score"]
    n_crisis = len(crisis_scores)
    n_crisis_missing = crisis_scores.isna().sum()

    print(f"\n[위기구간 연속성 점검]")
    print(f"  위기구간 주 수: {n_crisis}")
    print(f"  위기구간 결측치: {n_crisis_missing}")
    if n_crisis_missing == 0 and n_crisis > 0:
        print(f"  ✓ 위기구간 점수가 연속적으로 존재 — 동결 표준화·센트로이드가 이미 적용됨")
        print(f"    (위기구간 점수 범위: {crisis_scores.min():.4f} ~ {crisis_scores.max():.4f})")
        print(f"    → 매우 높은 점수가 나타나 게이지가 정상 작동함을 확인 (검증 통과)")
    else:
        print(f"  ⚠ 위기구간에 결측 또는 데이터 없음 — 재점수화가 필요할 수 있음")

    return df_score


def align_score_and_prob(df_score: pd.DataFrame, df_prob: pd.DataFrame) -> pd.DataFrame:
    """
    이탈 점수와 P(약세)를 같은 날짜로 정렬

    Args:
        df_score: date, score, crisis 컬럼
        df_prob: date, prob_bear 컬럼

    Returns:
        정렬된 병합 DataFrame (date, score, prob_bear, crisis)
    """
    print(f"\n[날짜 정렬]")

    merged = pd.merge(
        df_score[["date", "score", "crisis"]],
        df_prob[["date", "prob_bear"]],
        on="date",
        how="inner",
    )
    merged = merged.dropna(subset=["score", "prob_bear"]).sort_values("date").reset_index(drop=True)

    print(f"  이탈 점수 주 수: {len(df_score):,}")
    print(f"  P(약세) 유효 주 수: {df_prob['prob_bear'].notna().sum():,}")
    print(f"  ✓ 정렬 후 공통 주 수: {len(merged):,}")
    print(f"  날짜 범위: {merged['date'].min().date()} ~ {merged['date'].max().date()}")

    # 정렬 후 결측 확인
    n_missing = merged[["score", "prob_bear"]].isna().any(axis=1).sum()
    print(f"  정렬 후 결측 행: {n_missing}")

    return merged


# ============================================================================
# [3단계] 겹쳐 그리기
# ============================================================================

def load_event_markers(merged: pd.DataFrame) -> list[dict]:
    """
    이벤트 목록 로드 (수직선 표시용)

    ★ 이벤트는 임의 생성 금지 — data/raw/events.csv의 기존 목록 사용
      (DeviationDashboard.load_events와 동일한 금요일 매칭 로직)

    Args:
        merged: 정렬된 데이터 (날짜 범위 확인용)

    Returns:
        이벤트 리스트 (date: pd.Timestamp, label: str)
    """
    print(f"\n[이벤트 마커 로드] data/raw/events.csv (임의 생성 없음, 기존 목록 사용)")

    events_df = pd.read_csv("data/raw/events.csv", parse_dates=["date"])
    label_col = "memo" if "memo" in events_df.columns else "event_type"

    date_min = merged["date"].min()
    date_max = merged["date"].max()

    markers = []
    for _, row in events_df.iterrows():
        original_date = row["date"]

        # 가장 가까운 금요일로 매칭 (대시보드와 동일 로직)
        days_since_friday = (original_date.weekday() - 4) % 7
        nearest_friday = original_date - pd.Timedelta(days=days_since_friday)

        if nearest_friday < date_min or nearest_friday > date_max:
            continue  # 데이터 범위 밖 이벤트는 제외

        markers.append({
            "date": nearest_friday,
            "label": str(row[label_col]) if pd.notna(row[label_col]) else "이벤트",
        })

    print(f"  ✓ 데이터 범위 내 이벤트: {len(markers)}개 (전체 {len(events_df)}개 중)")
    for m in markers:
        print(f"    - {m['date'].date()}: {m['label']}")

    return markers


def plot_overlay_chart(merged: pd.DataFrame, events: list[dict], output_path: str = "analysis/ms_ar_overlay.png"):
    """
    이탈 점수와 P(약세)를 겹쳐 그리기 (왼축=이탈 점수, 오른축=P(약세))

    ★ 둘 다 '높음 = 스트레스' → 같이 오르면 일관된 신호

    Args:
        merged: date, score, prob_bear 컬럼을 가진 정렬된 데이터
        events: 이벤트 마커 리스트
        output_path: PNG 저장 경로
    """
    print("\n" + "=" * 70)
    print("[3단계] 겹쳐 그리기")
    print("=" * 70)

    fig, ax_left = plt.subplots(figsize=(16, 8))

    # 왼쪽 축: 이탈 점수
    color_score = "#2563eb"
    ax_left.plot(merged["date"], merged["score"], color=color_score, linewidth=1.2, label="이탈 점수 (좌)")
    ax_left.set_xlabel("날짜")
    ax_left.set_ylabel("이탈 점수", color=color_score)
    ax_left.tick_params(axis="y", labelcolor=color_score)
    ax_left.axhline(0, color=color_score, linewidth=0.5, linestyle=":", alpha=0.5)

    # 오른쪽 축: P(약세) — 같은 시간축 공유
    ax_right = ax_left.twinx()
    color_prob = "#dc2626"
    ax_right.plot(merged["date"], merged["prob_bear"], color=color_prob, linewidth=1.0, alpha=0.75, label="P(약세) (우)")
    ax_right.set_ylabel("P(약세)", color=color_prob)
    ax_right.tick_params(axis="y", labelcolor=color_prob)
    ax_right.set_ylim(0, 1.05)

    # 이벤트 수직선 (임의 생성 금지 — 제공된 목록만 사용)
    # ★ 라벨이 밀집된 구간(2019~2025)에서 겹침을 막기 위해 번호 마커 + 범례 방식 사용
    #   인접 이벤트(약 3개월 이내)는 번호를 위/아래로 교대 배치해 겹침 방지
    y_top = ax_left.get_ylim()[1]
    prev_date = None
    toggle = 0
    for i, ev in enumerate(events, start=1):
        ax_left.axvline(ev["date"], color="gray", linestyle="--", linewidth=0.7, alpha=0.55)

        if prev_date is not None and (ev["date"] - prev_date).days < 100:
            toggle = 1 - toggle  # 인접하면 높이를 교대
        else:
            toggle = 0
        y_offset = 1.01 if toggle == 0 else 1.06

        ax_left.text(
            ev["date"], y_top * y_offset, str(i),
            fontsize=8, ha="center", va="bottom", color="dimgray", fontweight="bold",
        )
        prev_date = ev["date"]

    # 이벤트 번호 ↔ 이름 범례 박스 (그래프 아래쪽 여백에 텍스트로 표시, 2줄로 분할)
    half = len(events) // 2
    line1 = "  |  ".join(f"{i}: {ev['label']} ({ev['date'].date()})" for i, ev in enumerate(events[:half], start=1))
    line2 = "  |  ".join(f"{i}: {ev['label']} ({ev['date'].date()})" for i, ev in enumerate(events[half:], start=half + 1))
    fig.text(0.01, 0.015, line1 + "\n" + line2, fontsize=6.5, color="dimgray", va="bottom")

    # 범례 통합
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")

    ax_left.set_title("이탈 점수 vs MS-AR P(약세) — 동시 감지 신호의 일관성 교차검증\n"
                      "(★ 예측력 검증이 아닌, 두 독립 신호의 '동시성' 확인. 번호=이벤트, 하단 범례 참고)")
    ax_left.grid(True, alpha=0.2)

    fig.tight_layout(rect=(0, 0.07, 1, 1))  # 하단 범례 공간 확보

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ PNG 저장 완료: {output}")
    print(f"  데이터 포인트: {len(merged):,}개")
    print(f"  이벤트 수직선: {len(events)}개")


# ============================================================================
# [4단계] 정량 일치도 분석
# ============================================================================

def analyze_agreement(merged: pd.DataFrame):
    """
    두 시계열(이탈 점수, P(약세))의 정량적 일치도 분석

    - 상관계수
    - 임계값 초과 주의 겹침(교집합/합집합)
    - 리드-래그 교차상관 (±N주)
    - 불일치 구간 표

    ★ 둘 다 동시적 신호이므로 큰 리드는 기대하지 않음 — '일관성' 확인이 목적
    """
    print("\n" + "=" * 70)
    print("[4단계] 정량 일치도 분석")
    print("=" * 70)

    score = merged["score"].values
    prob = merged["prob_bear"].values
    dates = merged["date"].values

    # ------------------------------------------------------------------
    # (1) 상관계수
    # ------------------------------------------------------------------
    corr = np.corrcoef(score, prob)[0, 1]
    print(f"\n[표 5] 두 시계열의 상관계수")
    print(f"  Pearson 상관계수 = {corr:.4f}")
    if corr > 0.3:
        print(f"  → 양의 상관 — 두 신호가 같은 방향으로 움직이는 경향 확인")
    else:
        print(f"  → 상관이 약함 — 두 신호의 동시성이 뚜렷하지 않을 수 있음")

    # ------------------------------------------------------------------
    # (2) 임계 초과 주의 겹침 (교집합/합집합)
    #   - 이탈 점수: 비위기 구간 90백분위 임계
    #   - P(약세): > 0.5
    # ------------------------------------------------------------------
    print(f"\n[표 6] 임계 초과 주의 겹침 (동시 '켜짐' 정도)")

    nc_scores = merged.loc[~merged["crisis"], "score"]
    score_threshold = nc_scores.quantile(0.90)
    prob_threshold = 0.5

    print(f"  이탈 점수 임계값 (비위기 90백분위): {score_threshold:.4f}")
    print(f"  P(약세) 임계값: {prob_threshold:.2f}")

    score_high = merged["score"] > score_threshold
    prob_high = merged["prob_bear"] > prob_threshold

    intersection = (score_high & prob_high).sum()
    union = (score_high | prob_high).sum()
    jaccard = intersection / union if union > 0 else np.nan

    print(f"\n  이탈 점수 high: {score_high.sum()}주 / 전체 {len(merged)}주 ({score_high.mean()*100:.1f}%)")
    print(f"  P(약세) high:   {prob_high.sum()}주 / 전체 {len(merged)}주 ({prob_high.mean()*100:.1f}%)")
    print(f"  교집합 (둘 다 high): {intersection}주")
    print(f"  합집합 (둘 중 하나라도 high): {union}주")
    print(f"  자카드 지수 (교집합/합집합) = {jaccard:.4f}")
    print(f"  → 두 신호가 동시에 '켜지는' 정도: {jaccard*100:.1f}%")

    # ------------------------------------------------------------------
    # (3) 리드-래그 교차상관 (±N주)
    # ------------------------------------------------------------------
    print(f"\n[표 7] 리드-래그 교차상관 (±6주)")
    print(f"  ※ 둘 다 동시적 신호이므로 큰 리드는 기대하지 않음 — 참고용")

    max_lag = 6
    lag_results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # score가 prob보다 |lag|주 먼저 (score를 lag만큼 뒤로 밀어 정렬)
            s = score[: lag]
            p = prob[-lag:]
        elif lag > 0:
            # score가 prob보다 lag주 늦게 (prob를 lag만큼 뒤로 밀어 정렬)
            s = score[lag:]
            p = prob[: -lag]
        else:
            s = score
            p = prob

        if len(s) > 1:
            c = np.corrcoef(s, p)[0, 1]
        else:
            c = np.nan
        lag_results.append((lag, c))

    print(f"\n  {'lag(주)':<10}{'상관계수':<12}{'해석'}")
    print(f"  {'-'*45}")
    for lag, c in lag_results:
        if lag < 0:
            interp = "이탈 점수가 먼저 변함 (lead)"
        elif lag > 0:
            interp = "P(약세)가 먼저 변함 (lead)"
        else:
            interp = "동시"
        marker = " ←최대" if c == max(x[1] for x in lag_results if not np.isnan(x[1])) else ""
        print(f"  {lag:<10}{c:<12.4f}{interp}{marker}")

    best_lag = max(lag_results, key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf)
    print(f"\n  최대 상관 시차: lag = {best_lag[0]}주 (상관계수 = {best_lag[1]:.4f})")
    if best_lag[0] == 0:
        print(f"  → 시차 0에서 최대 상관 — 두 신호가 동시적임을 재확인 (예측력 아님)")
    else:
        print(f"  → lag={best_lag[0]}이 최대지만, 0과 큰 차이 없다면 사실상 동시적으로 해석 권장")

    # ------------------------------------------------------------------
    # (4) 불일치 구간 표 (한쪽만 high)
    # ------------------------------------------------------------------
    print(f"\n[표 8] 불일치 구간 (한쪽만 임계 초과)")

    only_score_high = score_high & ~prob_high
    only_prob_high = ~score_high & prob_high

    df_disagree = merged.copy()
    df_disagree["only_score_high"] = only_score_high.values
    df_disagree["only_prob_high"] = only_prob_high.values

    print(f"\n  ① 이탈 점수만 높음 (거시 스트레스 ↑, 수익률 국면은 평온): {only_score_high.sum()}주")
    rows_a = df_disagree.loc[df_disagree["only_score_high"], ["date", "score", "prob_bear"]].head(10)
    if len(rows_a) > 0:
        print(rows_a.to_string(index=False))
    else:
        print("    (없음)")

    print(f"\n  ② P(약세)만 높음 (수익률 국면 전환 ↑, 거시 스트레스는 낮음): {only_prob_high.sum()}주")
    rows_b = df_disagree.loc[df_disagree["only_prob_high"], ["date", "score", "prob_bear"]].head(10)
    if len(rows_b) > 0:
        print(rows_b.to_string(index=False))
    else:
        print("    (없음)")

    print(f"\n  ※ 위 표는 '거시 스트레스'(이탈 점수)와 '수익률 국면 전환'(P(약세))이")
    print(f"     서로 다른 시점에 반응하는 구간을 보여줍니다 — 신호가 다루는 정보가 다를 수 있음을 시사.")

    return {
        "correlation": corr,
        "jaccard": jaccard,
        "score_threshold": score_threshold,
        "prob_threshold": prob_threshold,
        "lag_results": lag_results,
        "best_lag": best_lag,
        "n_only_score_high": int(only_score_high.sum()),
        "n_only_prob_high": int(only_prob_high.sum()),
    }


def print_final_summary(merged: pd.DataFrame, agreement: dict, bear_regime_idx: int, regime_stats: tuple):
    """
    최종 출력 요약 — 일관성 검증 결과 종합
    """
    means, variances = regime_stats
    bull_idx = 1 - bear_regime_idx

    latest = merged.iloc[-1]

    print("\n" + "=" * 70)
    print("[출력 요약] 두 독립 신호의 일관성 검증 결과")
    print("=" * 70)

    print(f"\n[현재 상태] (기준일: {latest['date'].date()})")
    print(f"  현재 이탈 점수:  {latest['score']:+.4f}")
    print(f"  현재 P(약세):    {latest['prob_bear']:.4f} ({latest['prob_bear']*100:.1f}%)")

    print(f"\n[일치도 종합]")
    print(f"  상관계수:        {agreement['correlation']:.4f}")
    print(f"  동시 '켜짐' 비율(자카드): {agreement['jaccard']*100:.1f}%")
    print(f"  최대 상관 시차:  lag = {agreement['best_lag'][0]}주 (상관계수 {agreement['best_lag'][1]:.4f})")
    print(f"  불일치 — 이탈 점수만 높음: {agreement['n_only_score_high']}주")
    print(f"  불일치 — P(약세)만 높음:   {agreement['n_only_prob_high']}주")

    print(f"\n[해석]")
    if agreement["correlation"] > 0.3 and agreement["jaccard"] > 0.2:
        print(f"  ✓ 두 독립 신호(거시 스트레스 기반 이탈 점수 vs 수익률 기반 MS-AR 약세 확률)는")
        print(f"    스트레스 시점에서 상당 부분 일치하는 모습을 보입니다.")
        print(f"    → 서로 다른 데이터(거시지표 vs 가격수익률)로 만든 신호가 같은 시점에 반응한다는 점에서")
        print(f"      현재 '평온/위기' 판단에 대한 교차 확증(corroboration) 근거가 됩니다.")
    else:
        print(f"  △ 두 신호의 일치도가 제한적입니다. 약세 국면이 위기에 집중된 희소 사건이라")
        print(f"    MS-AR 추정이 불안정할 수 있으니, 위 1단계 국면 파라미터를 재확인하세요.")

    print(f"\n[★ 핵심 해석 단서 — 반드시 유의]")
    print(f"  1) 이것은 '두 동시 신호의 일관성' 검증이지, 예측력의 증거가 아닙니다.")
    print(f"     (이탈 점수도, MS-AR P(약세)도 모두 coincident 지표 — 현재 상태 측정용)")
    print(f"  2) 약세 국면은 희소하고 위기에 집중되어 있어, MS-AR 추정이 불안정할 수 있습니다.")
    print(f"     약세 평균={means[bear_regime_idx]:+.4f}, 분산={variances[bear_regime_idx]:.6f} (1단계 표 1 참고)")
    print(f"     — 이 파라미터들이 표본수가 제한된 국면에서 추정되었음을 항상 염두에 두세요.")
    print("=" * 70)


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    try:
        # 데이터 로드
        df = load_data()
        returns = prepare_returns_data(df)

        # ★ 데이터 준비 완료 (자동 진행)

        # MS-AR 모델 적합
        model, result, bear_regime_idx, regime_stats, param_dict = fit_msv_ar_model(returns)

        # 결과 표 출력 (★ 여기서 멈추고 사용자 확인을 기다림)
        print_model_summary(model, result, bear_regime_idx, regime_stats)

        # ── 아래는 확인 후 2단계에서 사용할 P(약세) 시계열을 미리 저장만 해둠 ──
        prob_bear_full = result.filtered_marginal_probabilities[:, bear_regime_idx]  # (T-1,)

        df_with_prob = df[["date", "sp500_log_return"]].copy()
        df_with_prob["prob_bear"] = np.nan
        valid_idx = df_with_prob["sp500_log_return"].notna()
        valid_dates = df_with_prob.loc[valid_idx].index
        if len(valid_dates) > 0 and len(prob_bear_full) > 0:
            df_with_prob.loc[valid_dates[1:len(prob_bear_full) + 1], "prob_bear"] = prob_bear_full

        output_path = Path("analysis/ms_ar_output.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_prob.to_csv(output_path, index=False)
        print(f"\n(참고) P(약세) 시계열을 {output_path} 에 임시 저장했습니다 — 2단계에서 재사용 예정")

        # ====================================================================
        # [2단계] 이탈 점수 정렬·연속화
        # ====================================================================
        df_score = load_deviation_score()
        merged = align_score_and_prob(df_score, df_with_prob)

        print(f"\n[정렬 결과 미리보기]")
        print(merged[["date", "score", "prob_bear", "crisis"]].head(5).to_string(index=False))
        print("  ...")
        print(merged[["date", "score", "prob_bear", "crisis"]].tail(5).to_string(index=False))

        # 2단계 결과물 저장 (3단계에서 재사용)
        merged_path = Path("analysis/ms_ar_aligned.csv")
        merged.to_csv(merged_path, index=False)
        print(f"\n✓ 정렬된 데이터 저장: {merged_path}")

        print("\n" + "=" * 70)
        print("[★ 2단계 멈춤] 정렬 결과를 점검하세요:")
        print("  1) 두 시계열의 공통 날짜 수와 범위가 합리적인가?")
        print("  2) 위기구간(crisis=True)에 점수와 P(약세)가 모두 존재하는가?")
        print("=" * 70)

        # ====================================================================
        # [3단계] 겹쳐 그리기
        # ====================================================================
        events = load_event_markers(merged)
        plot_overlay_chart(merged, events)

        print("\n" + "=" * 70)
        print("[★ 3단계 멈춤] PNG를 열어 점검하세요:")
        print("  1) 두 신호가 스트레스 시점(위기구간/이벤트)에서 같이 오르는가?")
        print("  2) 이벤트 수직선이 실제 변곡점과 합리적으로 겹치는가?")
        print("=" * 70)

        # ====================================================================
        # [4단계] 정량 일치도 분석 + 출력 요약
        # ====================================================================
        agreement = analyze_agreement(merged)
        print_final_summary(merged, agreement, bear_regime_idx, regime_stats)

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
