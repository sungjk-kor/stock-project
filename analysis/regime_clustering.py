# =============================================================================
# regime_clustering.py  —  거시변수 기반 시장 국면(레짐) 군집화
#
# 목적:
#   예측이 아니라 '탐색적 국면 지도'를 만드는 단계.
#   사전관측 가능한 거시변수 5개로 재현되는 국면 유형을 군집화해,
#   다음 단계(조건부 백테스트) 설계의 입력으로 쓴다.
#
# 국면 정의 변수 (결과 변수 제외 — 순환 방지):
#   dxy_chg    DXY 13주 변화율
#   y10_chg    10Y 국채수익률 13주 변화(bp)
#   nfci       NFCI 금융여건 수준
#   hy_spread  신용스프레드 수준
#   vix        VIX 수준
#
# ★ SPX 수익률·실현변동성은 국면 정의에 넣지 않는다(순환 정의 방지).
#    사후 묘사(post-hoc)로만 표시.
#
# 필요 라이브러리:
#   pip install pandas numpy scikit-learn matplotlib
# =============================================================================

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 화면 표시 없이 파일 저장
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH      = "data/processed/final_dataset.csv"
CHANGE_WINDOW = 13        # DXY·10Y 변화 계산 기간 (주)
K_CANDIDATES  = [3, 4, 5] # 군집 수 후보
FWD_WEEKS     = 26        # 사후 묘사용 전진 수익률 기간

# 위기 제외 구간 (조정 가능)
CRISIS_PERIODS = [
    ("2008-09-01", "2009-06-30"),  # 금융위기
    ("2020-02-15", "2020-06-30"),  # 코로나 충격
]

OUTPUT_DIR = Path(__file__).parent

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────

print("=" * 65)
print("■ 거시변수 기반 시장 국면 군집화 (탐색적 분석)")
print("=" * 65)

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"\n  원본: {len(df):,}주  ({df['date'].min().date()} ~ {df['date'].max().date()})")

# ── 2. 국면 정의 변수 생성 ────────────────────────────────────────────────────

# dxy_chg: DXY 13주 변화율 (로그변화 = 수익률과 동일한 척도)
df["dxy_chg"]  = np.log(df["dxy"] / df["dxy"].shift(CHANGE_WINDOW))

# y10_chg: 10Y 수익률 13주 변화 (% 포인트, 수준 차이)
df["y10_chg"]  = df["treasury_10y"] - df["treasury_10y"].shift(CHANGE_WINDOW)

# 수준값 그대로 (이미 CSV에 있음)
# nfci    = financial_conditions
# hy_spread = credit_spread (BAA10Y)
# vix     = vix

# 국면 정의 변수 목록 (5개)
FEAT_COLS = ["dxy_chg", "y10_chg", "financial_conditions", "credit_spread", "vix"]
FEAT_LABELS = {
    "dxy_chg":             "DXY 13주 변화율",
    "y10_chg":             "10Y 금리 변화(pp)",
    "financial_conditions":"NFCI 금융여건",
    "credit_spread":       "HY 신용스프레드",
    "vix":                 "VIX",
}

# ── 3. 위기 구간 제거 ─────────────────────────────────────────────────────────

print("\n■ 1단계: 위기 구간 제거")

crisis_mask = pd.Series(False, index=df.index)
for start, end in CRISIS_PERIODS:
    m = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    crisis_mask |= m
    print(f"  제외: {start} ~ {end}  ({m.sum()}주)")

df["is_crisis"] = crisis_mask
df_clean = df[~crisis_mask].copy()

# 전진 수익률 (사후 묘사용 — 국면 정의와 무관)
# sp500 로그수익률 향후 26주 합산
df_clean["fwd_26w"] = np.log(
    df_clean["sp500"].shift(-FWD_WEEKS) / df_clean["sp500"]
)
# 위기 구간이 전방창에 포함되면 NaN 처리
for start, end in CRISIS_PERIODS:
    for i in range(df_clean.index.size):
        idx    = df_clean.index[i]
        t_date = df_clean.loc[idx, "date"]
        t_fwd  = t_date + pd.Timedelta(weeks=FWD_WEEKS)
        if (t_fwd >= pd.Timestamp(start)) and (t_date <= pd.Timestamp(end)):
            df_clean.loc[idx, "fwd_26w"] = np.nan

# 결측 제거
df_clean = df_clean.dropna(subset=FEAT_COLS).reset_index(drop=True)
print(f"\n  비위기 유효 행: {len(df_clean)}주")
print(f"  기간: {df_clean['date'].min().date()} ~ {df_clean['date'].max().date()}")

# ── 4. 표준화 및 상관·VIF 점검 ────────────────────────────────────────────────

print("\n■ 2단계: 표준화 + 상관·VIF 점검")
print("  (탐색적 군집화 → 전체 비위기 표본 표준화 허용)")

scaler = StandardScaler()
X_raw  = df_clean[FEAT_COLS].values.astype(float)
X_std  = scaler.fit_transform(X_raw)   # 표준화된 행렬

# 상관행렬
corr = pd.DataFrame(X_std, columns=FEAT_COLS).corr()
print("\n  [상관행렬]")
print(corr.round(2).to_string())

# VIF
print("\n  [VIF]  >10: 강한 중복, 5~10: 주의, <5: 양호")
X_vif = sm.add_constant(X_std)
for i, col in enumerate(FEAT_COLS):
    try:
        v = variance_inflation_factor(X_vif, i + 1)
        flag = " ⚠" if v > 10 else (" △" if v > 5 else "")
        print(f"  {col:<28}: {v:.2f}{flag}")
    except Exception:
        print(f"  {col:<28}: 계산 불가")

# ── 5. PCA ────────────────────────────────────────────────────────────────────

print("\n■ 3단계: PCA (누적 분산 ~90% 주성분 사용)")

pca = PCA(random_state=42)
pca.fit(X_std)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_comp  = int(np.searchsorted(cum_var, 0.90) + 1)   # 90% 설명하는 최소 주성분 수

print("  주성분별 설명분산비:")
for i, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cum_var)):
    marker = " ← 여기까지 사용 (~90%)" if i + 1 == n_comp else ""
    print(f"  PC{i+1}: {ev*100:.1f}%  누적={cv*100:.1f}%{marker}")

pca_final = PCA(n_components=n_comp, random_state=42)
X_pca = pca_final.fit_transform(X_std)
print(f"\n  PCA 입력: 5개 변수 → {n_comp}개 주성분  (누적 분산 {cum_var[n_comp-1]*100:.1f}%)")

# ── 6. 군집화 (KMeans + GaussianMixture, K=3,4,5) ─────────────────────────────

print("\n■ 4단계: 군집화 (KMeans + GaussianMixture, 원변수 vs PCA)")

results = {}   # {(method, use_pca, k): {'labels':..., 'sil':..., 'bic':...}}

print(f"\n  {'방법':<22} {'K':>2}  {'실루엣':>8}  {'BIC(GMM)':>12}")
print("  " + "-"*50)

for use_pca in [False, True]:
    X_in    = X_pca if use_pca else X_std
    pca_tag = "PCA" if use_pca else "원변수"

    for k in K_CANDIDATES:
        # ---- KMeans ----
        km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        km_labels = km.fit_predict(X_in)
        km_sil = silhouette_score(X_in, km_labels) if len(set(km_labels)) > 1 else -1
        results[("KMeans", use_pca, k)] = {
            "labels": km_labels, "sil": km_sil, "model": km
        }
        print(f"  KMeans({pca_tag}){'':<10} {k:>2}  {km_sil:>8.4f}")

        # ---- GaussianMixture ----
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=42, n_init=5)
        gm.fit(X_in)
        gm_labels = gm.predict(X_in)
        gm_sil = silhouette_score(X_in, gm_labels) if len(set(gm_labels)) > 1 else -1
        gm_bic = gm.bic(X_in)
        results[("GMM", use_pca, k)] = {
            "labels": gm_labels, "sil": gm_sil, "bic": gm_bic, "model": gm
        }
        print(f"  GMM({pca_tag}){'':<13} {k:>2}  {gm_sil:>8.4f}  {gm_bic:>12.1f}")

# 추천: 실루엣 최고 조합
best_key = max(
    [(k, v["sil"]) for k, v in results.items()],
    key=lambda x: x[1]
)[0]
best_method, best_pca, best_k = best_key
print(f"\n  → 실루엣 점수 최고: {best_method}({'PCA' if best_pca else '원변수'}), K={best_k}")
print(f"     (최종 K 선택은 본인이 아래 결과를 보고 결정)")

# ── 7. K별 군집 특성 출력 ─────────────────────────────────────────────────────

print("\n■ 5단계: K별 군집 특성 (원변수 KMeans 기준)")
print("  ★ SPX 향후 26주 수익률 = 사후 묘사 (국면 정의 아님!)")

for k in K_CANDIDATES:
    labels = results[("KMeans", False, k)]["labels"]
    df_clean[f"cluster_k{k}"] = labels

    print(f"\n  {'='*60}")
    print(f"  K={k}  (KMeans, 원변수)")
    print(f"  {'='*60}")

    rows = []
    for cl in sorted(np.unique(labels)):
        mask = labels == cl
        row  = {}
        for col in FEAT_COLS:
            row[col] = df_clean.loc[mask, col].mean()
        # 사후 묘사: 전진 수익률 (국면 정의와 무관)
        fwd_vals = df_clean.loc[mask, "fwd_26w"].dropna()
        row["fwd_26w_avg"]  = fwd_vals.mean() * 100 if len(fwd_vals) else np.nan
        row["fwd_26w_up%"]  = (fwd_vals > 0).mean() * 100 if len(fwd_vals) else np.nan
        row["n_weeks"]      = int(mask.sum())
        row["cluster"]      = int(cl)
        rows.append(row)

    tbl = pd.DataFrame(rows).set_index("cluster")
    # 컬럼 이름을 짧게
    tbl.columns = [
        "DXY변화율", "10Y변화pp", "NFCI", "HY스프레드", "VIX",
        "사후fwd26w%", "사후상승%", "주수"
    ]
    print(tbl.round(3).to_string())
    print(f"\n  ★ 사후 묘사(fwd_26w)는 군집 정의에 사용하지 않았음")

# ── 8. 현재 국면 식별 ─────────────────────────────────────────────────────────

print("\n■ 6단계: 현재 국면 (가장 최근 주)")

# 가장 최근 비위기 행
latest = df_clean.iloc[-1]
latest_date = latest["date"].date()
print(f"  기준 날짜: {latest_date}")
print(f"  변수값:")
for col in FEAT_COLS:
    print(f"    {col:<28}: {latest[col]:.4f}")

print()
for k in K_CANDIDATES:
    cl = int(latest[f"cluster_k{k}"])
    sil = results[("KMeans", False, k)]["sil"]
    print(f"  K={k}: 군집 {cl}번  (실루엣={sil:.4f})")

# ── 9. 에피소드 분석 ──────────────────────────────────────────────────────────

print("\n■ 7단계: 에피소드 분석 (연속 구간 = 한 국면이 이어진 기간)")
print("  ★ 이 숫자가 다음 단계(조건부 백테스트) 가능 여부를 좌우함")


def count_episodes(dates, labels):
    """연속 같은 레이블 구간(에피소드)을 추출. 반환: {cl: [(start, end, weeks), ...]}"""
    episodes = {}
    if len(labels) == 0:
        return episodes
    current_cl   = labels[0]
    episode_start = dates[0]
    episode_len   = 1

    for i in range(1, len(labels)):
        if labels[i] == current_cl:
            episode_len += 1
        else:
            episodes.setdefault(int(current_cl), []).append(
                (episode_start, dates[i - 1], episode_len)
            )
            current_cl    = labels[i]
            episode_start = dates[i]
            episode_len   = 1
    episodes.setdefault(int(current_cl), []).append(
        (episode_start, dates[-1], episode_len)
    )
    return episodes


for k in K_CANDIDATES:
    labels = results[("KMeans", False, k)]["labels"]
    dates  = df_clean["date"].values
    ep     = count_episodes(dates, labels)

    print(f"\n  [K={k}]")
    print(f"  {'군집':>4}  {'에피소드 수':>10}  {'총 주수':>8}  {'평균 지속(주)':>13}  {'현재?':>6}")

    latest_cl = int(df_clean.iloc[-1][f"cluster_k{k}"])
    for cl in sorted(ep.keys()):
        ep_list  = ep[cl]
        n_ep     = len(ep_list)
        n_weeks  = sum(e[2] for e in ep_list)
        avg_dur  = n_weeks / n_ep
        is_now   = "← 현재" if cl == latest_cl else ""
        print(f"  {cl:>4}  {n_ep:>10}  {n_weeks:>8}  {avg_dur:>13.1f}  {is_now}")

    # 현재 에피소드 시작일
    cur_ep = ep.get(latest_cl, [])
    if cur_ep:
        last_ep = cur_ep[-1]
        print(f"\n  현재 에피소드: {pd.Timestamp(last_ep[0]).date()} ~ "
              f"{pd.Timestamp(last_ep[1]).date()}  ({last_ep[2]}주 지속)")
        print(f"  이 유형(군집 {latest_cl})은 과거 {len(cur_ep)-1}번 끝난 에피소드 존재")

# ── 10. 시각화 ────────────────────────────────────────────────────────────────

print("\n■ 8단계: 시각화 (타임라인 국면 지도)")

COLORS_MAP = {
    3: ["#4C72B0", "#DD8452", "#55A868"],
    4: ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    5: ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9B59B6"],
}

for k in K_CANDIDATES:
    labels = results[("KMeans", False, k)]["labels"]
    colors = COLORS_MAP[k]
    sil    = results[("KMeans", False, k)]["sil"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        f"시장 국면 지도 (K={k}, KMeans·원변수, 실루엣={sil:.3f})\n"
        f"※ 국면 정의: DXY변화·10Y변화·NFCI·HY스프레드·VIX  /  SPX 미포함",
        fontsize=11, y=0.98
    )

    # ── 패널 1: S&P500 가격 + 국면 색 배경 ──
    ax1 = axes[0]
    sp_dates  = df_clean["date"].values
    sp_prices = df_clean["sp500"].values

    # 국면별 배경색 채우기
    prev_cl = labels[0]
    seg_start = sp_dates[0]
    for i in range(1, len(labels)):
        if labels[i] != prev_cl or i == len(labels) - 1:
            seg_end = sp_dates[i]
            ax1.axvspan(pd.Timestamp(seg_start), pd.Timestamp(seg_end),
                        alpha=0.25, color=colors[prev_cl], linewidth=0)
            prev_cl   = labels[i]
            seg_start = sp_dates[i]

    ax1.plot(sp_dates, sp_prices, color="black", linewidth=0.8, zorder=3)

    # 위기 구간 회색 사선
    for cs, ce in CRISIS_PERIODS:
        ax1.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                    color="gray", alpha=0.5, hatch="///", linewidth=0, zorder=4)

    ax1.set_yscale("log")
    ax1.set_ylabel("S&P 500 (로그)", fontsize=9)
    ax1.set_xlim(df["date"].min(), df["date"].max())

    # 범례
    patches = [mpatches.Patch(color=colors[c], alpha=0.7, label=f"군집 {c}")
               for c in range(k)]
    patches.append(mpatches.Patch(color="gray", alpha=0.5, hatch="///", label="위기(제외)"))
    ax1.legend(handles=patches, loc="upper left", fontsize=8, ncol=k + 1)

    # ── 패널 2: VIX ──
    ax2 = axes[1]
    # 전체 데이터(위기 포함) 사용
    df_all = df.set_index("date")
    ax2.plot(df_all.index, df_all["vix"], color="#C44E52", linewidth=0.7)
    for cs, ce in CRISIS_PERIODS:
        ax2.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                    color="gray", alpha=0.4, linewidth=0)
    ax2.set_ylabel("VIX", fontsize=9)
    ax2.set_xlim(df["date"].min(), df["date"].max())

    # ── 패널 3: HY 스프레드 ──
    ax3 = axes[2]
    ax3.plot(df_all.index, df_all["credit_spread"], color="#DD8452", linewidth=0.7)
    for cs, ce in CRISIS_PERIODS:
        ax3.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                    color="gray", alpha=0.4, linewidth=0)
    ax3.set_ylabel("HY Spread", fontsize=9)
    ax3.set_xlim(df["date"].min(), df["date"].max())
    ax3.set_xlabel("날짜", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = OUTPUT_DIR / f"regime_map_k{k}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {save_path.name}")

# ── 11. 종합 요약 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ 종합 요약")
print("=" * 65)

print(f"""
  [국면 정의 변수 5개]
  ① dxy_chg      DXY {CHANGE_WINDOW}주 변화율  (달러 강세/약세 방향)
  ② y10_chg      10Y 금리 {CHANGE_WINDOW}주 변화(pp)  (금리 방향)
  ③ financial_conditions  NFCI 금융여건  (긴축/완화 수준)
  ④ credit_spread  HY 신용스프레드  (크레딧 리스크 수준)
  ⑤ vix          VIX 변동성  (공포 수준)
""")

# 실루엣 점수 요약
print("  [K별 실루엣 점수 (KMeans·원변수)]")
for k in K_CANDIDATES:
    sil = results[("KMeans", False, k)]["sil"]
    best_mark = " ← 최고" if k == best_k and not best_pca and best_method == "KMeans" else ""
    print(f"  K={k}: {sil:.4f}{best_mark}")

# 현재 국면 요약
print(f"\n  [현재 국면 ({latest_date})]")
for k in K_CANDIDATES:
    cl = int(latest[f"cluster_k{k}"])
    ep = count_episodes(df_clean["date"].values,
                        results[("KMeans", False, k)]["labels"])
    ep_list = ep.get(cl, [])
    n_past  = len(ep_list) - 1   # 현재 에피소드 제외
    print(f"  K={k}: 군집 {cl}번  (과거 {n_past}번의 동일 유형 에피소드)")

print(f"""
  [해석 가이드]
  ① 실루엣 > 0.2이면 군집이 의미 있게 분리됨. 낮으면 국면 경계가 불명확.
  ② 군집별 에피소드가 최소 5~10번 이상이어야 다음 단계(조건부 백테스트)가 통계적으로 유효.
  ③ 사후 묘사(fwd_26w)는 참고용 — 낮은 n 군집은 과적합 위험 있음.
  ④ PCA 버전과 원변수 버전의 실루엣을 비교해 더 나은 쪽 선택 가능.
  ⑤ 위기 제외 구간은 조정 가능 — CRISIS_PERIODS 리스트를 수정하세요.
""")
print("=" * 65)
print("분석 완료. 저장된 차트:", OUTPUT_DIR)
print("=" * 65)
