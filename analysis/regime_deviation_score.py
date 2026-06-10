# =============================================================================
# regime_deviation_score.py  —  평온강세 이탈 점수 진단 (coincident)
#
# 목적:
#   현재 시장이 '평온강세 국면'에서 얼마나 '위험주의 국면' 쪽으로 이탈하는지를
#   연속 점수(deviation score)로 계산하고, 과거와 비교하며 지표별 기여를 분해.
#   ★ 예측이 아니라 현재 상태의 정량 진단(coincident indicator).
#
# 이탈 점수 정의:
#   d = normalize(위험주의 센트로이드 − 평온강세 센트로이드)  ← 이탈 방향 벡터
#   score_t = (x_t_표준화 − 평온강세_센트로이드) · d
#   · score > 0: 위험주의 방향으로 이탈
#   · score = 0: 정확히 평온강세 중심
#   · score < 0: 평온강세보다도 더 완화된 상태
#
# ★ 약한 누수:
#   표준화(StandardScaler)를 전체 비위기 표본으로 fit → 미래 정보 포함.
#   라이브 운용 시에는 과거 구간에만 fit해야 함. 여기서는 진단용으로 허용.
#
# 필요 라이브러리:
#   pip install pandas numpy scikit-learn matplotlib
# =============================================================================

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH = "data/processed/final_dataset.csv"
CRISIS_PERIODS = [
    ("2008-09-01", "2009-06-30"),
    ("2020-02-15", "2020-06-30"),
]
OUTPUT_DIR = Path(__file__).parent

print("=" * 65)
print("■ 평온강세 이탈 점수 진단 (coincident indicator)")
print("  ★ 예측이 아니라 현재 상태 측정")
print("=" * 65)

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"\n  로드: {len(df):,}주  ({df['date'].min().date()} ~ {df['date'].max().date()})")

# ── 2. 군집화 변수 생성 (regime_clustering.py와 동일 로직) ────────────────────

df["dxy_chg"]  = np.log(df["dxy"] / df["dxy"].shift(13))   # DXY 13주 변화율
df["y10_chg"]  = df["treasury_10y"] - df["treasury_10y"].shift(13)  # 10Y 금리 13주 변화(pp)
# NFCI, HY 스프레드(credit_spread), VIX는 수준값 그대로 사용

# 표준화에 쓸 5개 변수 (순서 고정 — 이후 기여 분해에 사용)
FEATS = ["dxy_chg", "y10_chg", "financial_conditions", "credit_spread", "vix"]
FEAT_KR = {   # 한글 이름
    "dxy_chg":             "DXY 13주 변화율",
    "y10_chg":             "10Y 금리 변화(pp)",
    "financial_conditions":"NFCI 금융여건",
    "credit_spread":       "HY 신용스프레드",
    "vix":                 "VIX",
}

# ── 3. 위기 구간 제거 → 표준화 파라미터 학습 ──────────────────────────────────

crisis_mask = pd.Series(False, index=df.index)
for s, e in CRISIS_PERIODS:
    crisis_mask |= (df["date"] >= s) & (df["date"] <= e)

# 비위기 구간으로 StandardScaler fit (★ 약한 누수 — 전체표본 사용)
df_nc = df[~crisis_mask].dropna(subset=FEATS)
sc    = StandardScaler()
sc.fit(df_nc[FEATS])   # 비위기 전체로 fit

# 원 데이터 전체(위기 포함)에 transform — 과거 비교용
df_valid = df.dropna(subset=FEATS).copy()
X_std    = sc.transform(df_valid[FEATS])    # (n_valid, 5)
df_valid[FEATS + ["__std_" + f for f in FEATS]] = pd.concat([
    df_valid[FEATS],
    pd.DataFrame(X_std, columns=["__std_" + f for f in FEATS], index=df_valid.index)
], axis=1)

# 표준화된 컬럼명
STD_FEATS = ["__std_" + f for f in FEATS]

# ── 4. KMeans 군집화 (비위기 표본) → 센트로이드 추출 ──────────────────────────

print("\n■ 1단계: 군집화 → 센트로이드 추출")

X_nc = sc.transform(df_nc[FEATS])   # 비위기 표준화값
km   = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
km.fit(X_nc)

# 각 클러스터 원지표 평균으로 정체성 파악
df_nc2 = df_nc.copy()
df_nc2["cl4"] = km.predict(X_nc)

vix_by_cl  = {c: df_nc2.loc[df_nc2["cl4"]==c, "vix"].mean()               for c in range(4)}
nfci_by_cl = {c: df_nc2.loc[df_nc2["cl4"]==c, "financial_conditions"].mean() for c in range(4)}

PB_LABEL     = min(vix_by_cl,  key=vix_by_cl.get)   # 평온강세 = VIX 최저
DANGER_LABEL = max(nfci_by_cl, key=nfci_by_cl.get)  # 위험주의 = NFCI 최대(가장 긴축)

C_PEACE  = km.cluster_centers_[PB_LABEL]     # 평온강세 센트로이드 (표준화 공간)
C_DANGER = km.cluster_centers_[DANGER_LABEL] # 위험주의 센트로이드 (표준화 공간)

print(f"  평온강세 클러스터: {PB_LABEL}번  (VIX={vix_by_cl[PB_LABEL]:.2f})")
print(f"  위험주의 클러스터: {DANGER_LABEL}번  (NFCI={nfci_by_cl[DANGER_LABEL]:.3f})")

# 원지표 기준 두 센트로이드 출력
pc_raw = sc.inverse_transform(C_PEACE.reshape(1,-1))[0]
dc_raw = sc.inverse_transform(C_DANGER.reshape(1,-1))[0]
print(f"\n  {'변수':<22} {'평온강세 중심':>14} {'위험주의 중심':>14}")
print("  " + "-"*52)
for i, feat in enumerate(FEATS):
    print(f"  {FEAT_KR[feat]:<22} {pc_raw[i]:>+14.4f} {dc_raw[i]:>+14.4f}")

# ── 5. 이탈 방향 벡터 d 계산 ──────────────────────────────────────────────────

print("\n■ 2단계: 이탈 방향 벡터(d) 계산")

d_raw  = C_DANGER - C_PEACE                      # 위험주의 − 평온강세
d_norm = d_raw / np.linalg.norm(d_raw)           # 단위 벡터로 정규화
d_pct  = np.abs(d_norm) / np.abs(d_norm).sum() * 100  # 각 축의 방향 기여 %

print(f"  {'변수':<22} {'d 방향 성분':>12} {'방향 기여%':>10}")
print("  " + "-"*46)
for i, feat in enumerate(FEATS):
    print(f"  {FEAT_KR[feat]:<22} {d_norm[i]:>+12.4f} {d_pct[i]:>9.1f}%")

# ── 6. 전 기간 이탈 점수 계산 ─────────────────────────────────────────────────

print("\n■ 3단계: 전 기간 이탈 점수 계산")
print("  score_t = (x_t_표준화 − 평온강세_센트로이드) · d")

# 전체 유효 행에 대해 계산
X_all     = df_valid[STD_FEATS].values    # 이미 표준화됨
dev_vecs  = X_all - C_PEACE               # 평온강세 센트로이드에서의 편차 벡터
scores    = dev_vecs @ d_norm             # 이탈 점수 (스칼라)
contribs  = dev_vecs * d_norm             # 지표별 기여 (행렬, 각 행의 합 = score)

df_valid["score"]    = scores
df_valid["crisis"]   = crisis_mask.reindex(df_valid.index, fill_value=False).values

# 기여도 컬럼
for i, feat in enumerate(FEATS):
    df_valid[f"contrib_{feat}"] = contribs[:, i]

print(f"  계산 완료: {len(df_valid)}주")

# ── 7. GMM 소속확률 P(평온강세) 계산 ─────────────────────────────────────────

print("\n■ 4단계: GMM P(평온강세) 계산 (보조 게이지)")

gm = GaussianMixture(n_components=4, covariance_type="full",
                     random_state=42, n_init=5)
gm.fit(X_nc)   # 비위기 표준화 데이터로 학습

# GMM 컴포넌트와 KMeans 클러스터 매칭 — 센트로이드 거리로 대응
gm_means      = gm.means_   # (4, 5)
peace_comp    = int(np.argmin(np.linalg.norm(gm_means - C_PEACE, axis=1)))
danger_comp   = int(np.argmin(np.linalg.norm(gm_means - C_DANGER, axis=1)))
p_all         = gm.predict_proba(X_all)    # (n_valid, 4)

df_valid["p_peace"]  = p_all[:, peace_comp]
df_valid["p_danger"] = p_all[:, danger_comp]
print(f"  GMM 완료. 평온강세 ↔ 컴포넌트{peace_comp}, 위험주의 ↔ 컴포넌트{danger_comp}")

# ── 8. 역대 통계 (비위기 구간 기준) ─────────────────────────────────────────

print("\n■ 5단계: 역대 통계 (비위기 구간 기준)")

nc_scores = df_valid.loc[~df_valid["crisis"], "score"]
all_scores = df_valid["score"]

max_score     = float(nc_scores.max())
max_date      = df_valid.loc[nc_scores.idxmax(), "date"].date()
min_score     = float(nc_scores.min())
min_date      = df_valid.loc[nc_scores.idxmin(), "date"].date()
mean_score    = float(nc_scores.mean())
std_score     = float(nc_scores.std())
median_score  = float(nc_scores.median())

# 현재값: 마지막 비결측 행
latest_row    = df_valid.dropna(subset=FEATS).iloc[-1]
current_score = float(latest_row["score"])
current_date  = latest_row["date"].date()
current_pct   = float((nc_scores < current_score).mean() * 100)
current_peace = float(latest_row["p_peace"])

print(f"  현재 ({current_date})")
print(f"    이탈 점수: {current_score:+.4f}")
print(f"    P(평온강세): {current_peace:.3f}")
print(f"    비위기 백분위: {current_pct:.1f}%  (낮을수록 평온강세에 가까움)")
print(f"\n  역대 최고 (비위기): {max_score:+.4f}  ({max_date})")
print(f"  역대 최저 (비위기): {min_score:+.4f}  ({min_date})")
print(f"  평균: {mean_score:+.4f}  표준편차: {std_score:.4f}  중앙값: {median_score:+.4f}")

# ── 9. 현재 주 — 지표별 기여 분해 ────────────────────────────────────────────

print("\n■ 6단계: 현재 주 지표별 기여 분해")
print("  (각 지표가 이탈 점수를 얼마나 끌어올리거나 누르는지)")

contrib_cols = [f"contrib_{f}" for f in FEATS]
latest_contribs = {f: float(latest_row[f"contrib_{f}"]) for f in FEATS}

# 현재 표준화값 (평온강세 센트로이드와 비교)
latest_std = {f: float(X_all[-1, i]) for i, f in enumerate(FEATS)}
peace_std  = {f: float(C_PEACE[i])   for i, f in enumerate(FEATS)}

# 원지표 현재값 vs 평온강세 평균
latest_raw = {f: float(latest_row[f]) for f in FEATS}
peace_raw  = {f: float(pc_raw[i])     for i, f in enumerate(FEATS)}

total_abs = sum(abs(v) for v in latest_contribs.values())

print(f"\n  {'지표':<22} {'현재 원값':>10} {'평온강세 평균':>12} "
      f"{'기여도':>8}  {'비중%':>6}  방향")
print("  " + "-"*72)

sorted_contribs = sorted(latest_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
for feat, contrib in sorted_contribs:
    raw_now = latest_raw[feat]
    raw_pb  = peace_raw[feat]
    pct     = abs(contrib) / total_abs * 100 if total_abs > 1e-8 else 0
    direction = "↑ 이탈↑" if contrib > 0.01 else ("↓ 완화↓" if contrib < -0.01 else "→ 중립")
    sign    = "+" if contrib >= 0 else ""
    print(f"  {FEAT_KR[feat]:<22} {raw_now:>+10.4f} {raw_pb:>+12.4f} "
          f"  {sign}{contrib:.4f}  {pct:>5.1f}%  {direction}")

print(f"\n  이탈 점수 합계: {current_score:+.4f}")
print(f"  P(평온강세):    {current_peace:.3f}  (1에 가까울수록 평온강세 내부)")

# ── 10. 시각화 ────────────────────────────────────────────────────────────────

print("\n■ 7단계: 차트 생성")

fig = plt.figure(figsize=(15, 11))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                         height_ratios=[2.5, 1.5, 1.5])

fig.suptitle(
    f"평온강세(군집{PB_LABEL}) 이탈 점수 진단  —  {current_date}\n"
    "★ 현재 상태 측정(coincident indicator) — 미래 예측 아님  "
    "★ 표준화 전체표본 fit → 약한 누수",
    fontsize=10, y=0.99,
)

# ── 패널1: 이탈 점수 시계열 (좌측 상단, 넓게) ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])   # 전체 첫 행

ax1.plot(df_valid["date"], df_valid["score"],
         color="#444", linewidth=0.9, label="이탈 점수")

# 위기 구간 회색 음영
for s, e in CRISIS_PERIODS:
    ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                color="gray", alpha=0.35, zorder=0, label="_nolegend_")
ax1.text(pd.Timestamp("2008-12-01"), ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] else 3,
         "위기(제외)", color="gray", fontsize=8, ha="center")

# 현재값 마커
ax1.axhline(y=current_score, color="#e74c3c", linewidth=1.2, linestyle="--",
            label=f"현재 {current_score:+.3f}")
ax1.scatter([latest_row["date"]], [current_score],
            color="#e74c3c", s=80, zorder=5)

# 역대 최고
ax1.axhline(y=max_score, color="#f39c12", linewidth=1, linestyle=":",
            label=f"역대최대 {max_score:+.3f} ({max_date})")

# 0 기준선 (평온강세 정중앙)
ax1.axhline(y=0, color="#2ecc71", linewidth=1.2, linestyle="-",
            label="0 = 평온강세 중심")

# 분포 음영 (±1σ)
ax1.axhspan(mean_score - std_score, mean_score + std_score,
            color="#3498db", alpha=0.08, label=f"비위기 평균±1σ")

ax1.set_ylabel("이탈 점수 (↑ = 위험주의 쪽)", fontsize=9)
ax1.set_title("전 기간 이탈 점수 시계열  "
              f"(현재 {current_pct:.0f} 백분위 · P(평온강세)={current_peace:.2f})",
              fontsize=9)
ax1.legend(fontsize=8, loc="upper left", ncol=3)
ax1.set_xlim(df_valid["date"].min(), df_valid["date"].max())

# ── 패널2: 현재 기여도 수평 막대 (하단 좌측) ──────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

feat_names = [FEAT_KR[f] for f in FEATS]
vals       = [latest_contribs[f] for f in FEATS]
bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]
bars = ax2.barh(feat_names, vals, color=bar_colors, alpha=0.8, height=0.5)

# 값 라벨
for bar, val in zip(bars, vals):
    ax2.text(val + 0.002 * np.sign(val), bar.get_y() + bar.get_height()/2,
             f"{val:+.4f}", va="center", ha="left" if val >= 0 else "right",
             fontsize=8)

ax2.axvline(x=0, color="black", linewidth=0.8)
ax2.set_xlabel("기여도 (빨강=이탈↑, 초록=완화↓)", fontsize=8)
ax2.set_title(f"현재 기여 분해  (합={current_score:+.3f})", fontsize=9)

# ── 패널3: 현재값 vs 평온강세 평균 (하단 우측) ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])

# 표준화 값으로 비교
curr_std_vals = [latest_std[f] - peace_std[f] for f in FEATS]  # 중심 대비 편차
bar2_colors   = ["#e74c3c" if v > 0 else "#2ecc71" for v in curr_std_vals]
bars2 = ax3.barh(feat_names, curr_std_vals, color=bar2_colors, alpha=0.8, height=0.5)

ax3.axvline(x=0, color="black", linewidth=0.8)
ax3.set_xlabel("현재값 − 평온강세 센트로이드 (표준화 단위)", fontsize=8)
ax3.set_title("현재 vs 평온강세 중심 편차", fontsize=9)

# ── 패널4: 이탈 점수 히스토그램 (하단 좌측) ───────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])

ax4.hist(nc_scores, bins=40, color="#95a5a6", alpha=0.7, edgecolor="white",
         label="비위기 분포")
ax4.axvline(x=current_score, color="#e74c3c", linewidth=2,
            label=f"현재 {current_score:+.3f}")
ax4.axvline(x=max_score, color="#f39c12", linewidth=1.5, linestyle=":",
            label=f"역대최대 {max_score:+.3f}")
ax4.axvline(x=0, color="#2ecc71", linewidth=1.2, label="평온강세 중심")
ax4.set_xlabel("이탈 점수", fontsize=8)
ax4.set_ylabel("빈도", fontsize=8)
ax4.set_title(f"비위기 이탈 점수 분포  (현재 {current_pct:.0f}백분위)", fontsize=9)
ax4.legend(fontsize=7)

# ── 패널5: P(평온강세) 시계열 ──────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])

ax5.fill_between(df_valid["date"], df_valid["p_peace"],
                 color="#2ecc71", alpha=0.4, label="P(평온강세)")
ax5.plot(df_valid["date"], df_valid["p_peace"],
         color="#27ae60", linewidth=0.7)
ax5.scatter([latest_row["date"]], [current_peace],
            color="#e74c3c", s=60, zorder=5, label=f"현재 {current_peace:.2f}")
ax5.axhline(y=0.5, color="gray", linewidth=0.7, linestyle="--", label="0.5 기준")
for s, e in CRISIS_PERIODS:
    ax5.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="gray", alpha=0.3, zorder=0)
ax5.set_ylabel("P(평온강세)", fontsize=8)
ax5.set_ylim(0, 1.05)
ax5.set_title("GMM P(평온강세) 시계열", fontsize=9)
ax5.legend(fontsize=7)
ax5.set_xlim(df_valid["date"].min(), df_valid["date"].max())

# 저장
save_path = OUTPUT_DIR / "regime_deviation_score.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  저장: {save_path.name}")

# ── 11. 평이한 한 문단 요약 ───────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 현재 국면 진단 요약")
print("=" * 65)

# 주요 기여 분류
pushers  = [(f, v) for f, v in sorted_contribs if v > 0.002]  # 이탈 가중
pullers  = [(f, v) for f, v in sorted_contribs if v < -0.002] # 완화 가중
neutral  = [(f, v) for f, v in sorted_contribs if abs(v) <= 0.002]

pusher_str = "·".join([f"{FEAT_KR[f]}↑" for f, _ in pushers[:3]]) if pushers else "없음"
puller_str = "·".join([f"{FEAT_KR[f]}" for f, _ in pullers[:3]]) if pullers else "없음"

# 점수 해석
if current_score < 0:
    score_interp = "평온강세보다도 더 완화적인 상태"
elif current_score < std_score * 0.5:
    score_interp = "평온강세 중심에 근접 — 매우 안정적"
elif current_score < std_score:
    score_interp = "평온강세 내 정상 범위"
elif current_score < std_score * 2:
    score_interp = "평온강세 경계 — 일부 긴장 감지"
else:
    score_interp = "위험주의 방향으로 주목할 만한 이탈"

print(f"""
  이탈 점수: 현재 {current_score:+.4f}  ({score_interp})
  역대 최대(비위기): {max_score:+.4f}  [{max_date}]
  비위기 구간 백분위: {current_pct:.1f}%  (낮을수록 평온강세 중심에 가까움)
  P(평온강세, GMM): {current_peace:.3f}  (1=완전 평온강세, 0=완전 이탈)

  이탈을 끌어올리는 지표: {pusher_str}
  이탈을 눌러주는 지표 : {puller_str}
""")

# 원지표 현재값 vs 평온강세 평균 테이블
print("  [원지표 현재값 vs 평온강세 중심]")
print(f"  {'지표':<22} {'현재값':>10}  {'평온강세 평균':>12}  {'차이':>8}")
print("  " + "-"*56)
for i, feat in enumerate(FEATS):
    diff = latest_raw[feat] - peace_raw[feat]
    print(f"  {FEAT_KR[feat]:<22} {latest_raw[feat]:>+10.4f}  "
          f"{peace_raw[feat]:>+12.4f}  {diff:>+8.4f}")

print(f"""
  ─────────────────────────────────────────────────────────
  ★ 이 진단은 현재 상태 측정(coincident indicator)이며
    미래 수익률·국면 전환 예측이 아닙니다.
  ★ 표준화에 전체 비위기 표본 사용 → 약한 누수 존재.
    라이브 운용 시 과거 구간 고정 스케일러 사용 권장.
  ─────────────────────────────────────────────────────────
""")
print("=" * 65)
print("분석 완료.")
print("=" * 65)
