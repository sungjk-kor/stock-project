# =============================================================================
# regime_v3.py  —  3-레짐 예측 v3 (변수 6개, VIF < 5 검증 완료)
#
# 이전 문제:
#   v1(거시 8개): 워크포워드 log-loss=1.37, 홀드아웃=1.07
#   v2(전체 17개): 과적합 → 워크포워드 1.60으로 악화
#
# v3 변수 선택 원칙:
#   ① 차원별 대표 1개씩  ② VIF < 5  ③ 서로 다른 정보 제공
#
#   변수             VIF   역할
#   momentum_13w    3.47  가격 추세 방향 (핵심 레짐 신호)
#   price_to_ma52   3.44  52주 이평 대비 위치 (장기 추세)
#   vix             2.66  공포·변동성
#   financial_cond  2.20  거시·금융여건 종합
#   up_vol_ratio    2.27  거래량이 추세를 확인하는가
#   nasdaq_rel      1.02  기술주 상대강도 (섹터 브레드스)
#
# ★ 누수 주의: 레짐 라벨이 전체 표본 마르코프 → 약한 누수
# =============================================================================

import os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

# ── 0. 설정 ──────────────────────────────────────────────────────────────────

CSV_PATH        = "data/processed/final_dataset.csv"
RET_COL         = "sp500_log_return"
FWD_WEEKS       = 13
HOLDOUT_WEEKS   = 78
MIN_TRAIN_WEEKS = 260
REFIT_EVERY     = 4
REGIME_NAMES    = {0: "강세", 1: "약세", 2: "횡보"}
OUTPUT_DIR      = Path(__file__).parent

# ★ 최종 선택 변수 6개 (ADF·VIF 검증 완료)
SELECTED_VARS = [
    "momentum_13w",       # 3개월 가격 모멘텀
    "price_to_ma52",      # 52주 이평 대비 위치
    "vix",                # 공포지수
    "financial_conditions",# 거시·금융여건
    "up_vol_ratio",       # 상승일 거래량 비율 (거래량 확인)
    "nasdaq_rel",         # 나스닥 상대강도 (섹터 브레드스)
]
# 모두 lag-1 적용 → 실제 투입 컬럼명은 {var}_l1

# ── 1. 데이터 로드 및 변수 생성 ────────────────────────────────────────────────

print("=" * 65)
print("■ 3-레짐 예측 v3 — 선택 변수 6개 (VIF < 5)")
print("=" * 65)

df = pd.read_csv(CSV_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
print(f"  {len(df):,}주  ({df['date'].min().date()} ~ {df['date'].max().date()})")

p = df["sp500"]
v = df["sp500_volume"]
r = df[RET_COL]

# 1) 13주 가격 모멘텀
df["momentum_13w"]   = p / p.shift(13) - 1

# 2) 52주 이동평균 대비 위치
df["price_to_ma52"]  = p / p.rolling(52).mean() - 1

# 3) vix: 이미 있음

# 4) financial_conditions: 이미 있음

# 5) 상승일 거래량 비율 (13주)
df["up_vol_ratio"]   = (v.where(r > 0, 0).rolling(13).sum()) / v.rolling(13).sum()

# 6) 나스닥 상대강도 = 나스닥 수익률 - S&P500 수익률
if "nasdaq_log_return" in df.columns:
    df["nasdaq_rel"] = df["nasdaq_log_return"] - r
else:
    print("  ⚠ nasdaq_log_return 없음 → nasdaq_rel 계산 불가")
    SELECTED_VARS.remove("nasdaq_rel")

print(f"  변수 생성 완료: {SELECTED_VARS}")

# ── 2. 마르코프 레짐 전환 ─────────────────────────────────────────────────────

print("\n■ 2단계: 마르코프 레짐 전환 (전체 표본, 약한 누수)")

ret_s = df.set_index("date")[RET_COL].dropna()
print("  추정 중...")
t0 = time.time()
try:
    res = MarkovRegression(
        ret_s, k_regimes=3, trend="c", switching_variance=True
    ).fit(search_reps=20, search_iter=100, disp=False)
    print(f"  완료 ({time.time()-t0:.0f}초)  AIC={res.aic:.1f}")

    raw_labels = res.smoothed_marginal_probabilities.idxmax(axis=1)
    stats = {r_i: {"mean": ret_s[raw_labels==r_i].mean()*100,
                   "std":  ret_s[raw_labels==r_i].std()*100,
                   "n":    int((raw_labels==r_i).sum())}
             for r_i in range(3)}
    srt   = sorted(range(3), key=lambda x: stats[x]["mean"])
    smap  = {srt[2]: 0, srt[0]: 1, srt[1]: 2}  # 높은=강세, 낮은=약세, 중간=횡보
    regime_s = raw_labels.map(smap).rename("regime")

    print(f"\n  {'레짐':<6} {'평균':>9} {'표준편차':>9} {'주수':>6}")
    for idx, name in REGIME_NAMES.items():
        raw = [k for k, v_ in smap.items() if v_ == idx][0]
        s = stats[raw]
        print(f"  {name:<6} {s['mean']:>+8.3f}%  {s['std']:>8.3f}%  {s['n']:>6}")
except Exception as e:
    print(f"  [오류] {e} → 분위수 백업")
    q33, q67 = ret_s.quantile(0.33), ret_s.quantile(0.67)
    regime_s = ret_s.apply(
        lambda x: 1 if x <= q33 else (0 if x >= q67 else 2)
    ).rename("regime")

df = df.merge(regime_s.reset_index(), on="date", how="left")

# ── 3. 타깃 생성 ─────────────────────────────────────────────────────────────

print("\n■ 3단계: 타깃 생성 (향후 13주 최빈 레짐)")

reg = df["regime"].values
tgt = np.full(len(reg), np.nan)
for t_i in range(len(reg) - FWD_WEEKS):
    w = reg[t_i + 1 : t_i + 1 + FWD_WEEKS]
    valid = w[~np.isnan(w)]
    if len(valid):
        tgt[t_i] = int(np.argmax(np.bincount(valid.astype(int), minlength=3)))
df["target"] = tgt

n_v = int(pd.notna(df["target"]).sum())
print(f"  유효: {n_v}주  |  " +
      " / ".join(f"{REGIME_NAMES[i]}={int((df['target']==i).sum())}주"
                 for i in range(3)))

# ── 4. lag-1 적용 및 데이터 준비 ─────────────────────────────────────────────

LAG_COLS = [f"{c}_l1" for c in SELECTED_VARS]
for c, lc in zip(SELECTED_VARS, LAG_COLS):
    df[lc] = df[c].shift(1)

keep     = ["date", RET_COL, "regime", "target"] + LAG_COLS
df_model = df[keep].dropna().reset_index(drop=True)

print(f"\n■ 4단계: 최종 데이터  {len(df_model)}주")
print(f"  투입 컬럼: {LAG_COLS}")
print(f"  기간: {df_model['date'].min().date()} ~ {df_model['date'].max().date()}")

if len(df_model) < MIN_TRAIN_WEEKS + HOLDOUT_WEEKS + 10:
    print("[오류] 데이터 부족"); sys.exit(1)

df_wf   = df_model.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
df_full = df_model.reset_index(drop=True)
print(f"  워크포워드: {len(df_wf)}주  |  홀드아웃: {HOLDOUT_WEEKS}주")

# ── 5. 워크포워드 함수 ────────────────────────────────────────────────────────

def run_wf(df_src, df_train_src, start_t, label):
    records = []
    last_clf = last_sc = None
    last_fit = -(REFIT_EVERY + 1)
    n = len(df_src)
    total = n - start_t
    nxt   = max(30, total // 10)
    t0_   = time.time()

    for idx, t in enumerate(range(start_t, n)):
        train_end = t - FWD_WEEKS
        if train_end < MIN_TRAIN_WEEKS:
            continue

        train   = df_train_src.iloc[:train_end + 1]
        y_tr    = train["target"].astype(int).values
        if len(np.unique(y_tr)) < 2:
            continue

        if (idx - last_fit) >= REFIT_EVERY or last_clf is None:
            sc  = StandardScaler()
            Xtr = sc.fit_transform(train[LAG_COLS].astype(float))
            try:
                clf = LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs", max_iter=1000, C=1.0,
                )
                clf.fit(Xtr, y_tr)
                last_clf, last_sc, last_fit = clf, sc, idx
            except Exception:
                if last_clf is None: continue

        Xt = last_sc.transform(df_src[LAG_COLS].iloc[[t]].astype(float))
        try:
            prb  = last_clf.predict_proba(Xt)[0]
            cls  = last_clf.classes_
            pd_  = dict(zip(cls, prb))
            pred = int(cls[np.argmax(prb)])
        except Exception:
            continue

        clim = np.bincount(y_tr, minlength=3) / len(y_tr)
        records.append({
            "date":       df_src["date"].iloc[t],
            "actual":     int(df_src["target"].iloc[t]),
            "pred":       pred,
            "p_bull":     pd_.get(0, 0.0),
            "p_bear":     pd_.get(1, 0.0),
            "p_sideways": pd_.get(2, 0.0),
            "pc_bull":    float(clim[0]),
            "pc_bear":    float(clim[1]),
            "pc_side":    float(clim[2]),
        })

        if (idx + 1) >= nxt:
            pct = (idx + 1) / total * 100
            print(f"    {idx+1:>4}/{total} ({pct:.0f}%) — {time.time()-t0_:.0f}초")
            nxt += max(30, total // 10)

    return pd.DataFrame(records)


def print_metrics(df_r, label):
    if df_r.empty:
        return None, None
    y_t  = df_r["actual"].astype(int).values
    prb  = np.column_stack([df_r["p_bull"], df_r["p_bear"], df_r["p_sideways"]])
    clm  = np.column_stack([df_r["pc_bull"], df_r["pc_bear"], df_r["pc_side"]])
    y_p  = df_r["pred"].astype(int).values
    ll   = log_loss(y_t, prb,  labels=[0, 1, 2])
    llc  = log_loss(y_t, clm,  labels=[0, 1, 2])
    acc  = (y_p == y_t).mean() * 100
    maj  = (y_t == np.bincount(y_t).argmax()).mean() * 100

    print(f"\n  [{label}]")
    print(f"  Log-loss  모델={ll:.4f}  벤치마크={llc:.4f}  "
          f"({'✓ 개선' if ll < llc - 0.02 else ('△ 미미' if ll < llc else '✗ 악화')})")
    print(f"  정확도    모델={acc:.1f}%  다수클래스 기저율={maj:.1f}%")

    tnames = [REGIME_NAMES[i] for i in range(3)]
    print(f"\n  [클래스별]")
    try:
        print(classification_report(y_t, y_p, target_names=tnames,
                                    labels=[0,1,2], zero_division=0))
    except Exception as e:
        print(f"  오류: {e}")

    print(f"  [혼동행렬]  행=실제, 열=예측")
    cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2])
    print("  " + "".join(f"  예측{REGIME_NAMES[j]:3}" for j in range(3)))
    for i, row in enumerate(cm):
        print(f"  실제{REGIME_NAMES[i]:3}" + "".join(f"  {v:6}" for v in row))

    return float(ll), float(llc)


# ── 6. 실행 ───────────────────────────────────────────────────────────────────

print(f"\n■ 5단계: 워크포워드 실행 (최소학습={MIN_TRAIN_WEEKS}주, 재적합={REFIT_EVERY}스텝)")
wf_df = run_wf(df_wf, df_wf, MIN_TRAIN_WEEKS, "워크포워드")
print(f"\n  완료 ({len(wf_df)}스텝)")

print("\n■ 6단계: 워크포워드 성과")
wf_ll, wf_llc = print_metrics(wf_df, "워크포워드 구간")

print(f"\n■ 7단계: 홀드아웃 검증")
hold_df = run_wf(df_full, df_full, len(df_wf), "홀드아웃")
print(f"\n  완료 ({len(hold_df)}스텝)")
hold_ll, hold_llc = print_metrics(hold_df, f"홀드아웃 {HOLDOUT_WEEKS}주")

# 저장
if not hold_df.empty:
    hold_df.to_csv(OUTPUT_DIR / "regime_v3_holdout.csv", index=False)

# ── 7. 확률 시계열 출력 ────────────────────────────────────────────────────────

print("\n■ 최근 20주 확률 (홀드아웃)")
if not hold_df.empty:
    disp = hold_df.copy()
    disp["실제"]    = disp["actual"].map(REGIME_NAMES)
    disp["예측"]    = disp["pred"].map(REGIME_NAMES)
    disp["P(강세)"] = disp["p_bull"].map("{:.2f}".format)
    disp["P(약세)"] = disp["p_bear"].map("{:.2f}".format)
    disp["P(횡보)"] = disp["p_sideways"].map("{:.2f}".format)
    print(disp[["date","실제","예측","P(강세)","P(약세)","P(횡보)"]].tail(20).to_string(index=False))

# ── 8. 종합 비교 ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("■ v1·v2·v3 모델 성과 비교")
print("=" * 65)

rows = [
    ("v1 거시변수 8개",  1.3703, 1.2956, 1.0676, 1.0676),
    ("v2 전체변수 17개", 1.6013, 1.2956, 0.9459, 0.6480),
]
if wf_ll and hold_ll:
    rows.append((f"v3 선택변수 6개", wf_ll, wf_llc, hold_ll, hold_llc))

print(f"\n  {'모델':<18}  {'WF log-loss':>11}  {'WF 벤치마크':>11}  "
      f"{'Hold log-loss':>13}  {'Hold 벤치마크':>13}")
for r in rows:
    wf_mark  = "✓" if r[1] < r[2] - 0.01 else ("△" if r[1] < r[2] else "✗")
    hd_mark  = "✓" if r[3] < r[4] - 0.01 else ("△" if r[3] < r[4] else "✗")
    print(f"  {r[0]:<18}  {r[1]:>11.4f}  {r[2]:>11.4f}  "
          f"{r[3]:>13.4f}  {r[4]:>13.4f}  WF:{wf_mark} HD:{hd_mark}")

print(f"""
  [해석]
  ① log-loss < 벤치마크 = 선행지표가 레짐 분포를 더 잘 예측함.
  ② 약세(bear) 35주는 극단 사건에 몰려 있어 재현율이 낮을 수밖에 없음.
  ③ 레짐 라벨이 전체 표본 마르코프 → 약한 누수. 엄격 검증은 폴드 내 재추정 필요.
  ④ 홀드아웃 벤치마크가 WF보다 낮은 건 홀드아웃 기간(2024~2026)에
     강세 비율이 높아 사전확률이 낮게 유지됐기 때문.
""")
print("=" * 65)
print("분석 완료.")
print("=" * 65)
