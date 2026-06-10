# =============================================================================
# within_regime_validation.py  —  평온강세(군집2) 국면 내 표본외 검증
#
# 목적:
#   현재 국면(평온강세) '안에서' 선행지표가 향후 8주 수익률을
#   '국면 내 드리프트' 보다 잘 예측하는지 에피소드 워크포워드로 검증.
#
# ★ 벤치마크:
#   국면 내 평균수익률(≈+1.52%)  ← 이것이 R²_oos 기준
#   국면 내 상승률(≈71%)         ← 이것이 방향 적중률 기준
#   (전체평균 / 50% 기준 일절 사용 안 함)
#
# ★ 약한 누수 경고:
#   국면 라벨이 전체표본 마르코프로 생성됨.
#   엄격 버전: 각 학습 폴드에서 마르코프 재추정 필요.
#
# 검증 방식:
#   시간순 에피소드 교차 — 과거 에피소드 학습 → 다음 에피소드 예측.
#   창 중첩 완화: 각 에피소드에서 8주마다 1개씩(비중첩) 표본 추출.
#
# 필요 라이브러리:
#   pip install pandas numpy scikit-learn statsmodels matplotlib
# =============================================================================

import warnings, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH        = "data/processed/final_dataset.csv"
FWD_WEEKS       = 8      # 예측 기간 (주). 26주는 창 중첩 과다 → 8주 사용
CHANGE_WINDOW   = 4      # 선행지표 변화율 계산 기간 (주)
CLUSTER_CHG_W   = 13     # 클러스터링용 변화 기간 (주, regime_clustering.py와 동일)
MIN_TRAIN_EPS   = 5      # 최소 학습 에피소드 수
MIN_TRAIN_WEEKS = 40     # 최소 학습 주 수

CRISIS_PERIODS = [
    ("2008-09-01", "2009-06-30"),
    ("2020-02-15", "2020-06-30"),
]
OUTPUT_DIR = Path(__file__).parent

print("=" * 65)
print("■ 평온강세(군집2) 국면 내 표본외 검증")
print(f"  예측 기간: fwd_{FWD_WEEKS}w  /  에피소드 워크포워드")
print("  ★ 약한 누수: 국면 라벨이 전체표본 마르코프에서 나옴")
print("=" * 65)

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"\n  원본: {len(df):,}주  ({df['date'].min().date()} ~ {df['date'].max().date()})")

# ── 2. 클러스터링 (regime_clustering.py와 동일) ───────────────────────────────

print("\n■ 1단계: 군집화 (K=4, 위기 제외)")

# 위기 구간 마스크
crisis_mask = pd.Series(False, index=df.index)
for s, e in CRISIS_PERIODS:
    crisis_mask |= (df["date"] >= s) & (df["date"] <= e)

# 클러스터링 정의 변수
df["dxy_chg_13w"] = np.log(df["dxy"] / df["dxy"].shift(CLUSTER_CHG_W))
df["y10_chg_13w"] = df["treasury_10y"] - df["treasury_10y"].shift(CLUSTER_CHG_W)

CLUSTER_FEATS = ["dxy_chg_13w", "y10_chg_13w",
                 "financial_conditions", "credit_spread", "vix"]

dw = df[~crisis_mask].dropna(subset=CLUSTER_FEATS).copy()

sc_cl   = StandardScaler()
X_cl    = sc_cl.fit_transform(dw[CLUSTER_FEATS])
km      = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
dw["cl4"] = km.fit_predict(X_cl)

# 평온강세 = VIX 가장 낮은 클러스터 (robust to label permutation)
vix_by_cl = {c: dw.loc[dw["cl4"]==c, "vix"].mean() for c in range(4)}
PB_LABEL  = min(vix_by_cl, key=vix_by_cl.get)
print(f"  평온강세 클러스터: {PB_LABEL}번  (VIX 평균={vix_by_cl[PB_LABEL]:.2f})")

# ── 3. 군집2 필터 + 선행지표 변수 생성 ───────────────────────────────────────

print("\n■ 2단계: 군집2 필터 + 선행지표 생성")

# 선행지표 변수를 원본 df 전체에 생성 (lag-1 포함)
df["hy_spread"]  = df["credit_spread"]
df["nfci"]       = df["financial_conditions"]

for col in ["hy_spread", "vix", "nfci"]:
    df[f"d{col}_{CHANGE_WINDOW}w"] = df[col] - df[col].shift(CHANGE_WINDOW)

# dxy_chg, y10_chg는 클러스터링과 동일한 13주 변화를 사용
# (이미 생성됨: dxy_chg_13w, y10_chg_13w)

# 예측에 쓸 특징변수 목록 (모두 lag-1 적용 → t-1 값으로 t 예측)
RAW_FEAT_COLS = [
    "hy_spread",             # HY 스프레드 수준
    f"dhy_spread_{CHANGE_WINDOW}w",  # HY 4주 변화
    "vix",                   # VIX 수준
    f"dvix_{CHANGE_WINDOW}w",        # VIX 4주 변화
    "nfci",                  # NFCI 수준
    f"dnfci_{CHANGE_WINDOW}w",       # NFCI 4주 변화
    "dxy_chg_13w",           # DXY 13주 변화
    "y10_chg_13w",           # 10Y 금리 13주 변화
]

# 8주 전진 수익률 타깃 (원본 df 기준으로 계산)
df["fwd8w"] = np.log(df["sp500"].shift(-FWD_WEEKS) / df["sp500"])

# 군집2 주만 필터 (dw에서 cl4 라벨 가져오기)
dw2 = dw[dw["cl4"] == PB_LABEL][["date"]].copy()
# 원본 df에서 필요한 모든 컬럼 join
feat_target_cols = RAW_FEAT_COLS + ["fwd8w", "sp500"]
c2 = dw2.merge(df[["date"] + feat_target_cols], on="date", how="left")
c2 = c2.sort_values("date").reset_index(drop=True)

# lag-1: 모든 특징변수를 1주 뒤로 밀기 (시점 t에서 t-1 값으로 예측 → 누수 없음)
LAG_FEAT_COLS = [f"{c}_l1" for c in RAW_FEAT_COLS]
for raw, lag in zip(RAW_FEAT_COLS, LAG_FEAT_COLS):
    c2[lag] = c2[raw].shift(1)

print(f"  군집2 행: {len(c2)}주")
print(f"  사용 특징변수 {len(LAG_FEAT_COLS)}개 (모두 lag-1):")
for c in LAG_FEAT_COLS:
    print(f"    {c}")

# ── 4. 에피소드 식별 ──────────────────────────────────────────────────────────

def identify_episodes(dates_list: list, gap_days: int = 10) -> list[list]:
    """날짜 목록에서 연속 구간(에피소드)을 식별한다."""
    if not dates_list:
        return []
    episodes = []
    ep = [dates_list[0]]
    for i in range(1, len(dates_list)):
        if (dates_list[i] - dates_list[i-1]).days <= gap_days:
            ep.append(dates_list[i])
        else:
            episodes.append(ep[:])   # 반드시 복사해서 저장
            ep = [dates_list[i]]
    episodes.append(ep[:])
    return episodes

all_dates = list(c2["date"])
episodes  = identify_episodes(all_dates)
lens      = [len(e) for e in episodes]

print(f"\n  에피소드 수: {len(episodes)}")
print(f"  에피소드 길이: 최소={min(lens)}주, 최대={max(lens)}주, 평균={np.mean(lens):.1f}주")

# ── 5. 국면 내 드리프트 전체 통계 (참고용 — 사후 계산) ───────────────────────

fwd_all = c2["fwd8w"].dropna()
REGIME_MEAN = fwd_all.mean()
REGIME_UP   = (fwd_all > 0).mean()
REGIME_STD  = fwd_all.std()

print(f"\n■ 3단계: 국면 내 드리프트 통계 (전체 참고용)")
print(f"  fwd_{FWD_WEEKS}w 평균     : {REGIME_MEAN*100:+.3f}%")
print(f"  fwd_{FWD_WEEKS}w 상승률   : {REGIME_UP*100:.1f}%  ← 방향 벤치마크")
print(f"  fwd_{FWD_WEEKS}w 표준편차 : {REGIME_STD*100:.3f}%")
print(f"  유효 표본        : {len(fwd_all)}주")
print(f"  ★ 적중률이 {REGIME_UP*100:.0f}%여도 그건 '국면 드리프트'일 뿐 — 초과분만 실력")

# ── 6. 에피소드 워크포워드 ────────────────────────────────────────────────────

print(f"\n■ 4단계: 에피소드 워크포워드 (학습→예측)")
print(f"  최소 학습: {MIN_TRAIN_EPS}에피소드 or {MIN_TRAIN_WEEKS}주")
print(f"  비중첩 평가: 에피소드당 {FWD_WEEKS}주 간격으로 1개씩 표본 추출")
print(f"  ★ 약한 누수: 국면 라벨이 전체표본 마르코프에서 나옴")

# date → row 매핑을 위한 인덱스
c2_idx = c2.set_index("date")


def get_rows_by_dates(dates):
    """날짜 목록으로 c2 데이터 서브셋 반환 (결측 제거)."""
    rows = c2_idx.reindex(dates).dropna(subset=LAG_FEAT_COLS + ["fwd8w"])
    return rows


records   = []          # 예측 기록
fold_info = []          # 폴드별 정보

for test_ep_idx in range(MIN_TRAIN_EPS, len(episodes)):
    # ── 학습 데이터: 0 ~ test_ep_idx-1 에피소드 ─────────────────────────────
    train_dates = [d for ep in episodes[:test_ep_idx] for d in ep]
    train_df    = get_rows_by_dates(train_dates)

    if len(train_df) < MIN_TRAIN_WEEKS:
        continue
    if train_df["fwd8w"].notna().sum() < 20:
        continue

    y_tr_valid = train_df["fwd8w"].dropna()
    X_tr       = train_df.loc[y_tr_valid.index, LAG_FEAT_COLS].astype(float)

    # 추가 결측 제거
    valid_mask = X_tr.notna().all(axis=1)
    X_tr       = X_tr[valid_mask]
    y_tr       = y_tr_valid[valid_mask]

    if len(y_tr) < 20:
        continue

    # ── 폴드 내 벤치마크 (훈련 구간 통계 — 누수 없음) ──────────────────────
    fold_mean   = float(y_tr.mean())       # 이 폴드의 국면 내 평균
    fold_up     = float((y_tr > 0).mean()) # 이 폴드의 국면 내 상승률
    fold_n      = len(y_tr)

    # ── 표준화: 훈련 구간에만 fit ────────────────────────────────────────────
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)

    dir_tr = (y_tr > 0).astype(int)

    # ── 모델 학습 ────────────────────────────────────────────────────────────
    # 수익률 예측: Ridge 회귀 (작은 표본에서 안정적)
    ridge = Ridge(alpha=1.0)
    try:
        ridge.fit(X_tr_sc, y_tr)
        ridge_ok = True
    except Exception:
        ridge_ok = False

    # 방향 예측: 로지스틱 (class_weight='balanced' — 음수 케이스 보정)
    logit = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0)
    try:
        logit.fit(X_tr_sc, dir_tr)
        logit_ok = True
    except Exception:
        logit_ok = False

    # ── 테스트: 현재 에피소드 → 비중첩 표본 ─────────────────────────────────
    test_ep   = episodes[test_ep_idx]
    # FWD_WEEKS 간격마다 1개씩 (비중첩 평가)
    test_sample = test_ep[::FWD_WEEKS]
    test_df  = get_rows_by_dates(test_sample)

    if test_df.empty:
        # 에피소드가 너무 짧아 비중첩 표본 없음 → 학습 에피소드로는 누적
        fold_info.append({
            "fold": test_ep_idx,
            "train_eps": test_ep_idx,
            "train_n": fold_n,
            "test_n": 0,
            "fold_mean%": round(fold_mean*100, 3),
            "fold_up%": round(fold_up*100, 1),
        })
        continue

    X_te_raw = test_df[LAG_FEAT_COLS].astype(float)
    te_valid = X_te_raw.notna().all(axis=1)
    X_te_raw = X_te_raw[te_valid]
    y_te     = test_df.loc[te_valid, "fwd8w"]

    if len(y_te) == 0:
        continue

    X_te_sc = sc.transform(X_te_raw)

    # ── 예측 기록 ─────────────────────────────────────────────────────────────
    for date, y_actual in zip(X_te_raw.index, y_te):
        rec = {
            "date":       date,
            "actual":     float(y_actual),
            "actual_dir": int(y_actual > 0),
            "fold":       test_ep_idx,
            "fold_mean":  fold_mean,
            "fold_up":    fold_up,
            # "국면 내 평균을 하회"하면 조기경보 라벨 = 1
            "alert_label_mean": int(y_actual < fold_mean),
            "alert_label_zero": int(y_actual < 0),
        }

        row_idx = list(X_te_raw.index).index(date)
        x_one   = X_te_sc[[row_idx]]

        if ridge_ok:
            rec["pred_ret"] = float(ridge.predict(x_one)[0])
        else:
            rec["pred_ret"] = fold_mean

        if logit_ok:
            proba = logit.predict_proba(x_one)[0]
            classes = list(logit.classes_)
            rec["pred_dir"]      = int(logit.predict(x_one)[0])
            rec["pred_prob_up"]  = float(proba[classes.index(1)] if 1 in classes else 0.5)
        else:
            rec["pred_dir"]     = 1
            rec["pred_prob_up"] = fold_up

        # 예측 수익률이 폴드 평균보다 낮으면 조기경보 트리거
        rec["alert_trigger"] = int(rec["pred_ret"] < fold_mean)

        records.append(rec)

    fold_info.append({
        "fold": test_ep_idx,
        "train_eps": test_ep_idx,
        "train_n": fold_n,
        "test_n": len(y_te),
        "fold_mean%": round(fold_mean*100, 3),
        "fold_up%": round(fold_up*100, 1),
    })

res_df   = pd.DataFrame(records)
fold_df  = pd.DataFrame(fold_info)

n_total  = len(res_df)
n_valid  = res_df["actual"].notna().sum()
print(f"\n  완료: {len(fold_info)}폴드, 총 예측 {n_total}개 (비중첩 표본)")

# ── 7. 지표 계산 ──────────────────────────────────────────────────────────────

print("\n■ 5단계: 성과 지표 계산")

if res_df.empty:
    print("  [경고] 유효한 예측 기록 없음 — 에피소드 수나 최소학습 설정 확인")
    sys.exit(0)

a   = res_df["actual"].values
p_r = res_df["pred_ret"].values
b   = res_df["fold_mean"].values     # 폴드별 국면 내 평균 (누수 없는 벤치마크)

# ── R²_oos (국면 내 평균 대비) ─────────────────────────────────────────────────
sse_model = np.sum((a - p_r) ** 2)
sse_bench = np.sum((a - b)   ** 2)
r2_oos    = float(1 - sse_model / sse_bench) if sse_bench > 1e-12 else np.nan

# ── 방향 적중률 (vs 국면 내 상승률) ──────────────────────────────────────────
actual_dir = res_df["actual_dir"].values
pred_dir   = res_df["pred_dir"].values
dir_acc    = float(np.mean(actual_dir == pred_dir))
fold_up_mean = float(res_df["fold_up"].mean())   # 각 폴드 상승률의 평균

# ── 조기경보: 국면 내 평균 하회 감지 ──────────────────────────────────────────
alert_label   = res_df["alert_label_mean"].values
alert_trigger = res_df["alert_trigger"].values

n_alert = int(alert_label.sum())
n_total_test = len(alert_label)

if n_alert >= 2 and (n_total_test - n_alert) >= 2:
    prec_mean = float(precision_score(alert_label, alert_trigger, zero_division=0))
    rec_mean  = float(recall_score(alert_label, alert_trigger, zero_division=0))
    cm_mean   = confusion_matrix(alert_label, alert_trigger, labels=[0,1])
else:
    prec_mean = rec_mean = float("nan")
    cm_mean   = None

# 음수 수익률 감지
alert_zero  = res_df["alert_label_zero"].values
n_neg = int(alert_zero.sum())
if n_neg >= 2 and (n_total_test - n_neg) >= 2:
    prec_zero = float(precision_score(alert_zero, alert_trigger, zero_division=0))
    rec_zero  = float(recall_score(alert_zero, alert_trigger, zero_division=0))
else:
    prec_zero = rec_zero = float("nan")

# ── 8. 결과 출력 ──────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 검증 결과")
print("=" * 65)

print(f"""
  [유효 표본]
  · 비중첩 예측 포인트: {n_total}개
  · 학습 폴드 수      : {len(fold_info)}개
  · 국면 내 상승률 벤치마크: {fold_up_mean*100:.1f}%  (각 폴드 평균)
  · 국면 내 평균수익률 벤치마크: {np.mean(b)*100:+.3f}%  (각 폴드 평균)
""")

print("  [수익률 예측 — Ridge 회귀]")
print(f"  R²_oos (국면내평균 대비)  : {r2_oos:+.4f}")
if r2_oos > 0.02:
    print(f"  → ✓ 의미 있는 개선 (국면 내 드리프트보다 나음)")
elif r2_oos > 0:
    print(f"  → △ 미미한 개선 (노이즈 수준)")
else:
    print(f"  → ✗ 국면 내 평균수익률 단순 사용보다 못함")
    print(f"     '평온강세 안에서도 추가 예측력 없음 = 국면 정보가 사실상 전부'")

print(f"\n  [방향 예측 — 로지스틱]")
print(f"  방향 적중률 (모델) : {dir_acc*100:.1f}%")
print(f"  국면 드리프트 기준 : {fold_up_mean*100:.1f}%  (이 숫자가 진짜 벤치마크)")
delta_dir = (dir_acc - fold_up_mean) * 100
if delta_dir > 3:
    print(f"  → ✓ 벤치마크 초과 +{delta_dir:.1f}%p — 방향 예측에 정보 있음")
elif delta_dir > 0:
    print(f"  → △ 미미한 초과 +{delta_dir:.1f}%p — 노이즈 수준")
else:
    print(f"  → ✗ 벤치마크 미달 {delta_dir:.1f}%p")
    print(f"     높은 적중률({dir_acc*100:.1f}%)은 국면 드리프트({fold_up_mean*100:.1f}%)에서 오는 것")

print(f"\n  [조기경보 — '국면 내 평균 하회' 감지]")
print(f"  실제 평균하회 사례: {n_alert}개 / {n_total_test}개 ({n_alert/n_total_test*100:.1f}%)")
if not np.isnan(prec_mean):
    print(f"  정밀도(Precision) : {prec_mean:.3f}  (경보 중 실제 하회 비율)")
    print(f"  재현율(Recall)    : {rec_mean:.3f}   (실제 하회 중 경보 발령 비율)")
    if prec_mean > 0.5 and rec_mean > 0.3:
        print(f"  → △ 일부 유용한 조기경보 가능성")
    else:
        print(f"  → ✗ 조기경보로서 신뢰도 부족")
else:
    print(f"  → 사례 수 부족으로 계산 불가")

print(f"\n  [조기경보 — '음수 수익률' 감지]")
print(f"  실제 음수 사례: {n_neg}개 / {n_total_test}개 ({n_neg/n_total_test*100:.1f}%)")
if not np.isnan(prec_zero):
    print(f"  정밀도: {prec_zero:.3f}  재현율: {rec_zero:.3f}")
else:
    print(f"  → 사례 수 부족으로 계산 불가")

# 혼동 행렬
if cm_mean is not None:
    print(f"\n  [혼동행렬 — 평균하회 감지]  행=실제, 열=예측")
    print(f"           예측정상  예측경보")
    print(f"  실제정상   {cm_mean[0,0]:>6}    {cm_mean[0,1]:>6}")
    print(f"  실제경보   {cm_mean[1,0]:>6}    {cm_mean[1,1]:>6}")

# ── 9. 폴드별 요약표 ──────────────────────────────────────────────────────────

print(f"\n■ 폴드별 상세 (테스트 표본 있는 것만)")
fold_with_test = fold_df[fold_df["test_n"] > 0].copy()
if not fold_with_test.empty:
    print(fold_with_test[["fold","train_eps","train_n","test_n",
                            "fold_mean%","fold_up%"]].to_string(index=False))

# ── 10. 시각화 ────────────────────────────────────────────────────────────────

print("\n■ 시각화 저장 중...")

if not res_df.empty:
    res_df["date"] = pd.to_datetime(res_df["date"])
    res_df = res_df.sort_values("date")

    fig, axes = plt.subplots(3, 1, figsize=(15, 11),
                             gridspec_kw={"height_ratios": [2, 1.5, 1.5]})
    fig.suptitle(
        f"평온강세(군집2) 국면 내 표본외 검증  —  fwd_{FWD_WEEKS}w\n"
        f"★ 방향 벤치마크: 국면 내 상승률 {fold_up_mean*100:.1f}%  "
        f"(전체평균·50% 기준 아님)",
        fontsize=11, y=0.99,
    )

    # ── 패널1: 실제 vs 예측 수익률 ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(res_df["date"], res_df["actual"]*100, color="#1f77b4",
             linewidth=1.0, label="실제 fwd_8w 수익률")
    ax1.plot(res_df["date"], res_df["pred_ret"]*100, color="#ff7f0e",
             linewidth=1.0, linestyle="--", label="Ridge 예측")
    ax1.axhline(y=np.mean(b)*100, color="green", linewidth=1,
                linestyle=":", label=f"국면내 평균 {np.mean(b)*100:+.2f}%")
    ax1.axhline(y=0, color="gray", linewidth=0.7, linestyle="-")
    ax1.fill_between(res_df["date"], res_df["actual"]*100, np.mean(b)*100,
                     where=(res_df["actual"] < np.mean(b)),
                     alpha=0.15, color="red", label="평균 하회 구간")
    ax1.set_ylabel("8주 누적 수익률 (%)", fontsize=9)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title(f"수익률 예측  R²_oos={r2_oos:+.4f} (국면내평균 대비)", fontsize=9)

    # ── 패널2: 방향 예측 ─────────────────────────────────────────────────────
    ax2 = axes[1]
    correct = (res_df["actual_dir"] == res_df["pred_dir"])
    wrong   = ~correct
    ax2.scatter(res_df.loc[correct, "date"], res_df.loc[correct, "actual_dir"],
                color="green", s=20, alpha=0.6, label=f"적중 ({correct.sum()})")
    ax2.scatter(res_df.loc[wrong, "date"], res_df.loc[wrong, "actual_dir"],
                color="red", marker="x", s=30, alpha=0.8, label=f"오류 ({wrong.sum()})")
    ax2.axhline(y=0.5, color="gray", linewidth=0.7, linestyle=":")
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["하락", "상승"])
    ax2.set_ylabel("실제 방향", fontsize=9)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title(
        f"방향 예측  적중={dir_acc*100:.1f}%  vs  국면드리프트={fold_up_mean*100:.1f}%"
        f"  (초과={delta_dir:+.1f}%p)", fontsize=9,
    )

    # ── 패널3: 조기경보 ──────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.bar(res_df["date"], res_df["actual"]*100,
            color=["#d62728" if a < m*100 else "#2ca02c"
                   for a, m in zip(res_df["actual"]*100, res_df["fold_mean"]*100)],
            width=5, alpha=0.7, label="실제수익률 (빨강=평균하회)")
    # 조기경보 발령 시점 표시
    alerted = res_df[res_df["alert_trigger"] == 1]
    ax3.scatter(alerted["date"], [np.mean(b)*100 + 1.5]*len(alerted),
                marker="v", color="orange", s=50, zorder=5, label="경보 발령")
    ax3.axhline(y=np.mean(b)*100, color="green", linewidth=1, linestyle=":",
                label=f"국면내 평균 {np.mean(b)*100:+.2f}%")
    ax3.set_ylabel("8주 수익률 (%)", fontsize=9)
    ax3.legend(fontsize=8, loc="upper left")
    prec_str = f"{prec_mean:.2f}" if not np.isnan(prec_mean) else "N/A"
    rec_str  = f"{rec_mean:.2f}"  if not np.isnan(rec_mean) else "N/A"
    ax3.set_title(
        f"조기경보(평균하회)  정밀도={prec_str}  재현율={rec_str}", fontsize=9
    )

    for ax in axes:
        ax.set_xlim(res_df["date"].min(), res_df["date"].max())

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUTPUT_DIR / "within_regime_validation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {out_path.name}")

# ── 11. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)
print(f"""
  [데이터]
  · 군집2(평온강세): {len(c2)}주, {len(episodes)}에피소드
  · 비중첩 검증 표본: {n_total}개 포인트
  · ★ 약한 누수: 국면 라벨이 전체표본 마르코프에서 나옴

  [국면 내 드리프트 — 진짜 벤치마크]
  · 8주 평균수익률 : {REGIME_MEAN*100:+.3f}%
  · 상승률         : {REGIME_UP*100:.1f}%
  ✦ 이 숫자를 이기지 못하면 선행지표는 추가 정보가 없음

  [선행지표 기여 판정]
  · 수익률 R²_oos : {r2_oos:+.4f}  {'✓ 기여' if r2_oos > 0.02 else ('△ 미미' if r2_oos > 0 else '✗ 없음')}
  · 방향 초과분    : {delta_dir:+.1f}%p  {'✓ 기여' if delta_dir > 3 else ('△ 미미' if delta_dir > 0 else '✗ 없음')}
  · 조기경보 정밀도: {prec_str}
""")

if r2_oos <= 0 and delta_dir <= 0:
    print("  ★ 결론: '평온강세 안에서도 추가 예측력 없음'")
    print("     = 군집 정보(국면 자체)가 사실상 전부.")
    print("     이 결과 자체가 유효한 발견 — 다음 단계는 국면 전환 조기 감지 모델.")
elif r2_oos > 0.02 or delta_dir > 3:
    print("  ★ 결론: 선행지표가 국면 드리프트를 넘어 추가 정보를 제공합니다.")
    print("     → 하지만 표본 수가 적어 과적합 위험 존재. 신중한 해석 필요.")
else:
    print("  ★ 결론: 미미한 추가 정보 가능성 — 노이즈와 구분하기 어렵습니다.")

print("\n" + "=" * 65)
print("분석 완료.")
print("=" * 65)
