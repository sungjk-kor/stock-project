# =============================================================================
# validation/step5_charts.py — 5단계(4/4-b): 차트 3종 (PNG)
#   차트1 국면 확률 적층 시계열 + 사후 실제 국면 띠 (전체 2005~2026 / 최근 확대)
#   차트2 전환확률 캘리브레이션 (h=1·4·13, ② 재적합 vs ① 당시고정)
#   차트3 4주 전환확률 시계열 vs 실제 전환 발생
# 팔레트: dataviz 기본 (blue #2a78d6 = ② 재적합, orange #eb6834 = ① 당시고정, 검증 PASS)
# =============================================================================
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "outputs"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_LIGHT = "#cde2fb"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BAND = "#898781"
PRIOR_END = pd.Timestamp("2026-06-05")

avail = {f.name for f in font_manager.fontManager.ttflist}
for fam in ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic"]:
    if fam in avail:
        plt.rcParams["font.family"] = fam; break
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def bands(ax, dates, L, y0, y1, label=None):
    """사후 실제 약세(L=1) 구간을 회색 띠로"""
    on = False; start = None; first = True
    for d, l in zip(dates, L):
        if l == 1 and not on:
            on, start = True, d
        elif l == 0 and on:
            ax.axvspan(start, d, ymin=y0, ymax=y1, color=BAND, alpha=0.35, linewidth=0, label=label if first else None); first = False; on = False
    if on:
        ax.axvspan(start, dates[-1], ymin=y0, ymax=y1, color=BAND, alpha=0.35, linewidth=0, label=label if first else None)


ts = pd.read_csv(OUT / "국면확률_시계열.csv", parse_dates=["날짜"])
dfB = pd.read_csv(OUT / "전환확률_예측vs실제.csv", parse_dates=["예측시점", "목표시점"])
cal = pd.read_csv(OUT / "전환확률_캘리브레이션.csv")

# ── 차트1 ─────────────────────────────────────────────────────────────────
ev = ts[ts["평가구간"] == 1].reset_index(drop=True)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35})
d = ev["날짜"]; p2 = ev["P약세_②재적합"].values; L = ev["실제국면_θT"].values
ax1.fill_between(d, 0, p2, color=BLUE, alpha=0.9, linewidth=0, label="P(약세) — ② 매주 재적합 필터")
ax1.fill_between(d, p2, 1, color=BLUE_LIGHT, linewidth=0, label="P(강세) = 1 - P(약세)")
bands(ax1, d.values, L, 1.0, 1.0)  # placeholder (띠는 아래 strip 에)
# 사후 띠: 축 위쪽 1.03~1.10
on = False; start = None; lab = True
for dd, l in zip(d.values, L):
    if l == 1 and not on:
        on, start = True, dd
    elif l == 0 and on:
        ax1.fill_between([start, dd], 1.03, 1.10, color=BAND, alpha=0.8, linewidth=0, label="사후 실제 약세 국면 (θT 평활 > 0.5)" if lab else None); lab = False; on = False
if on:
    ax1.fill_between([start, d.values[-1]], 1.03, 1.10, color=BAND, alpha=0.8, linewidth=0)
ax1.axhline(1.03, color=GRID, linewidth=0.6); ax1.axhline(1.10, color=GRID, linewidth=0.6)
ax1.axvline(PRIOR_END, color=INK2, linewidth=0.8)
ax1.text(PRIOR_END, 1.12, " 당시 데이터 끝 2026-06-05", color=INK2, fontsize=9, va="bottom")
ax1.set_ylim(0, 1.16); ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax1.set_ylabel("확률")
ax1.set_title("국면 확률 적층 시계열(② 매주 재적합)과 사후 실제 국면 띠 — 2013-01 ~ 2026-10")
ax1.legend(loc="upper left", ncol=3, fontsize=9, bbox_to_anchor=(0, -0.08))
ax1.xaxis.set_major_locator(mdates.YearLocator(2)); ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
style(ax1)
# 확대
z = ts[(ts["날짜"] >= "2024-01-05")].reset_index(drop=True)
ax2.plot(z["날짜"], z["P약세_②재적합"], color=BLUE, linewidth=2, label="② 매주 재적합 필터")
ax2.plot(z["날짜"], z["P약세_①당시고정"], color=ORANGE, linewidth=2, label="① 당시 모델 고정 (06-05까지 당시 발표값)")
ax2.plot(z["날짜"], z["P약세_θT평활"], color=MUTED, linewidth=1.5, linestyle=(0, (2, 2)), label="사후 평활 P(약세), θT")
Lz = z["실제국면_θT"].values
on = False; start = None
for dd, l in zip(z["날짜"].values, Lz):
    if l == 1 and not on:
        on, start = True, dd
    elif l == 0 and on:
        ax2.fill_between([start, dd], 1.03, 1.10, color=BAND, alpha=0.8, linewidth=0); on = False
if on:
    ax2.fill_between([start, z["날짜"].values[-1]], 1.03, 1.10, color=BAND, alpha=0.8, linewidth=0)
ax2.axhline(1.03, color=GRID, linewidth=0.6); ax2.axhline(1.10, color=GRID, linewidth=0.6)
ax2.axhline(0.5, color=AXIS, linewidth=0.8, linestyle=(0, (4, 3))); ax2.text(z["날짜"].iloc[0], 0.52, "판정 임계 0.5", color=MUTED, fontsize=8)
ax2.axvline(PRIOR_END, color=INK2, linewidth=0.8); ax2.text(PRIOR_END, 1.12, " 당시 데이터 끝", color=INK2, fontsize=9, va="bottom")
last = z.iloc[-1]
ax2.annotate(f"② {last['P약세_②재적합']*100:.1f}%", (last["날짜"], last["P약세_②재적합"]), xytext=(6, 8), textcoords="offset points", color=INK, fontsize=9)
ax2.annotate(f"① {last['P약세_①당시고정']*100:.1f}%", (last["날짜"], last["P약세_①당시고정"]), xytext=(6, -12), textcoords="offset points", color=INK, fontsize=9)
pk = z.loc[z["날짜"] == PRIOR_END].iloc[0]
ax2.annotate(f"06-05 완결주 ② {pk['P약세_②재적합']*100:.1f}%\n당시 발표(부분주) {pk['P약세_당시발표']*100:.1f}%", (PRIOR_END, pk["P약세_②재적합"]),
             xytext=(-150, 25), textcoords="offset points", color=INK2, fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
ax2.set_ylim(0, 1.16); ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax2.set_ylabel("P(약세)")
ax2.set_title("최근 확대 — 2024-01 ~ 2026-10: 필터(실시간) ①·② 와 사후 평활, 띠 = 사후 실제 약세")
ax2.legend(loc="upper left", ncol=3, fontsize=9, bbox_to_anchor=(0, -0.08))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3)); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
style(ax2)
fig.savefig(OUT / "차트1_국면확률_적층_실제국면띠.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 차트2 캘리브레이션 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, h in zip(axes, [1, 4, 13]):
    c = cal[(cal["h"] == h) & (cal["종류"] == "전환")]
    lim = 0.0
    for ver, color, filled in [("②재적합", BLUE, True), ("①당시고정", ORANGE, False)]:
        g = c[(c["버전"] == ver) & (c["n"] > 0)]
        if g.empty:
            continue
        lim = max(lim, float(g["평균예측확률"].max()), float(g["실제빈도"].max()))
        sizes = 20 + 160 * np.sqrt(g["n"].values / max(1, c["n"].max()))
        if filled:
            ax.plot(g["평균예측확률"], g["실제빈도"], color=color, linewidth=1.2, zorder=2)
            ax.scatter(g["평균예측확률"], g["실제빈도"], s=sizes, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=f"{ver} (크기 = 건수)")
            for x, y, n in zip(g["평균예측확률"], g["실제빈도"], g["n"]):
                ax.annotate(f"{int(n)}", (x, y), xytext=(7, 4), textcoords="offset points", fontsize=7.5, color=INK2)
        else:
            ax.scatter(g["평균예측확률"], g["실제빈도"], s=sizes, facecolor="none", edgecolor=color, linewidth=1.5, zorder=3, label=ver)
    lim = min(1.0, float(np.ceil((lim + 0.05) * 10) / 10)); lim = max(lim, 0.3)
    ax.plot([0, lim], [0, lim], color=AXIS, linewidth=0.8, zorder=1)
    rec = dfB[(dfB["h"] == h) & (dfB["버전"] == "②재적합") & (dfB["최근구간"] == 1)]
    if not rec.empty:
        ax.scatter(rec["q_전환"], np.full(len(rec), -0.02), marker="|", s=60, color=INK, linewidth=1, zorder=4,
                   label=f"최근 구간 ② 예측 {len(rec)}건 (실제 전환 {int(rec['y_전환'].sum())}건)")
    ax.set_xlim(-0.03, lim); ax.set_ylim(-0.05, lim)
    ax.set_xlabel("예측 전환확률 (구간 평균)"); ax.set_ylabel("실제 전환 빈도")
    ax.set_title(f"h = {h}주: 전환확률 캘리브레이션 (2013-01~2026-10)")
    ax.legend(loc="upper left", fontsize=8)
    style(ax)
fig.tight_layout()
fig.savefig(OUT / "차트2_전환확률_캘리브레이션.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ── 차트3 4주 전환확률 시계열 ──────────────────────────────────────────────
g = dfB[(dfB["h"] == 4) & (dfB["버전"] == "②재적합") & (dfB["이월의존구간"] == 0)].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(g["예측시점"], g["q_전환"], color=BLUE, linewidth=1.6, label="② 4주 내 국면 전환확률 q (예측시점 기준)")
ax.plot(g["예측시점"], g["B2_q_전환"], color=MUTED, linewidth=1.0, label="기준선 ② 장기 평균 전환빈도 (확장 윈도우)")
sw = g[g["y_전환"] == 1]
ax.scatter(sw["예측시점"], np.full(len(sw), 1.02), marker="|", s=50, color=INK, linewidth=0.8, label=f"실제: 4주 안에 국면이 바뀐 예측시점 ({len(sw)}건)")
ax.axvspan(PRIOR_END, g["예측시점"].max(), color=ORANGE, alpha=0.08, linewidth=0, label="최근 구간 (2026-06-05~)")
ax.axvline(PRIOR_END, color=INK2, linewidth=0.8)
lastv = g.iloc[-1]
ax.annotate(f"{lastv['예측시점']:%Y-%m-%d}: {lastv['q_전환']*100:.1f}%", (lastv["예측시점"], lastv["q_전환"]), xytext=(-110, 20), textcoords="offset points", color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
ax.set_ylim(0, 1.08); ax.set_ylabel("확률"); ax.set_title("4주 전환확률(② 매주 재적합) vs 실제 전환 발생 — 2013-01 ~ 2026-09")
ax.legend(loc="upper left", ncol=2, fontsize=9, bbox_to_anchor=(0, -0.12))
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
style(ax)
fig.savefig(OUT / "차트3_전환확률_시계열_h4.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("charts saved:", [p.name for p in OUT.glob("차트*.png")])
