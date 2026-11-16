# =============================================================================
# validation/step5_build_xlsx.py — 5단계(4/4-a): 전환확률_검증.xlsx 작성 (엑셀 수식으로 재계산 가능)
#
# 시트: 설명 / 원데이터 / 국면확률 / 전이행렬 / 예측vs실제 / 지표 / 캘리브레이션 / 탐지지연 / 걸어서재적합현황
#   · 값은 CSV(step5_evaluate.py 산출)에서 가져오고, Brier·log loss·skill·캘리브레이션·전이행렬 거듭제곱·손검산은 전부 수식.
#   · 글꼴 Arial. 파란 글씨 = 하드코딩 입력(전이행렬 값), 초록 = 다른 시트 참조, 검정 = 수식.
# 실행: .venv-stock/bin/python validation/step5_build_xlsx.py [출력경로] [--max-rows N]  (N: 검증용 축소본)
# =============================================================================
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter as L

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "outputs"
ARIAL = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
BLUE = Font(name="Arial", size=10, color="0000FF")
GREEN = Font(name="Arial", size=10, color="008000")
HEAD_FILL = PatternFill("solid", fgColor="F2F2F2")
CLIP_LO, CLIP_HI = 0.01, 0.99


def build(path: Path, max_rows: int | None = None) -> dict:
    ts = pd.read_csv(OUT / "국면확률_시계열.csv", parse_dates=["날짜"])
    dfB = pd.read_csv(OUT / "전환확률_예측vs실제.csv", parse_dates=["예측시점", "목표시점"])
    dfM = pd.read_csv(OUT / "전환확률_지표.csv")
    dfA = pd.read_csv(OUT / "국면진단_지표.csv")
    dfLag = pd.read_csv(OUT / "국면전환_탐지지연.csv")
    dfW = pd.read_csv(OUT / "걸어서재적합_현황.csv")
    summ = json.load(open(OUT / "step5_summary.json", encoding="utf-8"))
    if max_rows:   # 검증용 축소본: 국면확률은 뒤에서 max_rows주, 예측표는 버전×h별 뒤에서 max_rows행
        ts = ts.tail(max_rows).reset_index(drop=True)
        dfB = dfB.groupby(["버전", "h"], sort=False).tail(max_rows).reset_index(drop=True)

    wb = Workbook()
    info = {}

    def style_header(ws, ncol, row=1):
        for c in range(1, ncol + 1):
            cell = ws.cell(row=row, column=c)
            cell.font = BOLD; cell.fill = HEAD_FILL; cell.alignment = Alignment(wrap_text=True, vertical="center")

    def set_font(ws):
        for row in ws.iter_rows():
            for cell in row:
                if cell.font == Font():
                    cell.font = ARIAL

    # ── 설명 ────────────────────────────────────────────────────────────────
    ws = wb.active; ws.title = "설명"
    lines = [
        ("국면 전환확률 모델 검증 — 계산 워크북", BOLD),
        (f"작성 {datetime.now():%Y-%m-%d %H:%M} · 데이터 2000-12-08 ~ 2026-10-02 (S&P500 주간 로그수익률, W-FRI) · 당시 데이터 종료 2026-06-05", ARIAL),
        ("", ARIAL),
        ("[정의]", BOLD),
        ("모델: 2국면 Markov-Switching AR(1), switching_variance=True (레포 utils/ms_ar_model.py의 fit_ms_ar_model 사양). 약세 = 상수항이 낮은 국면.", ARIAL),
        ("θ0 = 당시(2026-06-05 종료) 데이터 적합 파라미터, θT = 2026-10-02 종료 데이터 적합 파라미터.", ARIAL),
        ("사후 실제 국면 L_t = IF(θT 평활 P(약세) > 0.5, 1, 0). 임계값 0.5는 당시 코드(compute_agreement)의 값. 마지막 8주(2026-08-14~)는 잠정.", ARIAL),
        ("버전 ① 당시고정: 2026-06-05까지 당시 발표 필터값(analysis/ms_ar_output.csv), 2026-06-12부터 θ0 고정 필터. 전이행렬 P0 고정.", ARIAL),
        ("버전 ② 재적합: 매주 t까지 데이터로 원본 함수 재적합(2005-01-07~). 예외·수렴실패 주는 직전 수렴 파라미터를 이월해 필터만 적용. 전이행렬 P_t.", ARIAL),
        ("h주 후 약세확률 q_약세 = (1-p_t)·P^h[강→약] + p_t·P^h[약→약].  전환확률 q_전환 = (1-p_t)·(1-P^h[강→강]) + p_t·(1-P^h[약→약]).", ARIAL),
        ("목표 y_약세 = L_{t+h}.  y_전환 = IF(L_{t+h} ≠ L_t, 1, 0).", ARIAL),
        ("기준선 B1(현재 국면 유지): q_약세 = p_t, q_전환 = 0.  기준선 B2(장기 평균): t까지 사후 라벨의 약세 비중 / h주 전환 빈도(확장 윈도우).", ARIAL),
        (f"Brier = (q-y)^2 의 평균.  log loss = -[y·ln(q') + (1-y)·ln(1-q')], q' = MIN(MAX(q, {CLIP_LO}), {CLIP_HI}).  skill = 1 - Brier_모델/Brier_기준선.", ARIAL),
        ("구간: 전체 = 2013-01-04 ~ 2026-10-02 (매주 수렴 적합이 연속 성립하는 구간), 최근 = 예측시점 2026-06-05 이후(당시 데이터 종료 이후), ②가용 = 2011-08-05 ~ (첫 수렴 적합 이후, 2011-08-12~2012-12-28은 단일 적합 파라미터 75주 이월).", ARIAL),
        ("원본 환경(statsmodels 0.14.6, 시드 42·탐색 20회)에서 2005-01~2011-07 주간 적합은 전부 예외 또는 수렴 실패(파라미터 퇴화)여서 그 구간은 걸어서 검증이 불가.", ARIAL),
        ("", ARIAL),
        ("[시트]", BOLD),
        ("원데이터: 날짜·S&P500·주간 로그수익률.  국면확률: 시점별 P(약세)와 라벨, (A) 국면 진단용 Brier·LL 수식.", ARIAL),
        ("전이행렬: θ0·θT 전이행렬(입력값, 파란 글씨)과 P^2·P^4·P^8·P^12·P^13 수식, 정상상태 약세비중, 2026-10-02 손검산.", ARIAL),
        ("예측vs실제: 모든 (버전, h, 예측시점) 행. q 검산·Brier·LL 은 수식.  지표: 구간별 평균(AVERAGEIFS) 수식과 파이썬 계산값 대조.", ARIAL),
        ("캘리브레이션: 예측확률 10%p 구간별 건수·평균 예측·실제 빈도(COUNTIFS/AVERAGEIFS).  탐지지연·걸어서재적합현황: 값 표.", ARIAL),
        ("", ARIAL),
        (f"[환경] 계산: Python {summ.get('python')} · statsmodels {summ.get('statsmodels')} (원본 환경 버전). 걸어서 재적합 시점 {summ['walkforward']['cutoffs']}개 중 예외 {summ['walkforward']['errors']}, 수렴실패 {summ['walkforward']['nonconverged']}, 채택 {summ['walkforward']['adopted']}, 이월 주 {summ['walkforward']['carried_weeks']} (최대 경과 {summ['walkforward']['max_param_age_weeks']}주).", ARIAL),
        ("[주의] 당시 2026-06-05 행은 수요일(06-03)까지만 수집된 부분주라 당시 발표 P(약세) 2.92%는 부분주 값. 확장본의 06-05 행은 완결주(수익률 -2.62%).", ARIAL),
    ]
    for i, (txt, f) in enumerate(lines, 1):
        c = ws.cell(row=i, column=1, value=txt); c.font = f
    ws.column_dimensions["A"].width = 150

    # ── 원데이터 ───────────────────────────────────────────────────────────
    ws = wb.create_sheet("원데이터")
    ws.append(["날짜", "S&P500 종가", "주간 로그수익률"])
    for _, r in ts.iterrows():
        ws.append([r["날짜"].to_pydatetime(), float(r["S&P500"]), float(r["주간로그수익률"])])
    style_header(ws, 3)
    for row in ws.iter_rows(min_row=2):
        row[0].number_format = "yyyy-mm-dd"; row[1].number_format = "#,##0.00"; row[2].number_format = "0.00000"
    ws.freeze_panes = "A2"; ws.column_dimensions["A"].width = 12

    # ── 국면확률 ───────────────────────────────────────────────────────────
    ws = wb.create_sheet("국면확률")
    cols = ["날짜", "주간로그수익률", "P약세_당시발표", "P약세_θ0고정필터", "P약세_①당시고정", "P약세_②재적합", "②이월여부", "②파라미터기준일",
            "②강→강", "②강→약", "②약→강", "②약→약", "P약세_θT필터", "P약세_θT평활", "실제국면_θT", "잠정", "최근구간", "평가구간", "②가용구간",
            "①Brier", "①p절단", "①LL", "①정확", "②Brier", "②p절단", "②LL", "②정확", "θT필터Brier", "θT필터p절단", "θT필터LL"]
    ws.append(cols)
    ci = {c: i + 1 for i, c in enumerate(cols)}
    n_ts = len(ts)
    for k, r in enumerate(ts.itertuples(index=False), start=2):
        d = dict(zip(ts.columns, r))
        def v(name):
            x = d[name]
            return None if pd.isna(x) else (float(x) if not isinstance(x, (pd.Timestamp, str)) else x)
        ws.append([
            d["날짜"].to_pydatetime(), v("주간로그수익률"), v("P약세_당시발표"), v("P약세_θ0고정필터"), v("P약세_①당시고정"), v("P약세_②재적합"),
            v("②이월여부"), (pd.Timestamp(d["②파라미터기준일"]).to_pydatetime() if not pd.isna(d["②파라미터기준일"]) else None),
            v("②강→강"), v("②강→약"), v("②약→강"), v("②약→약"), v("P약세_θT필터"), v("P약세_θT평활"),
            None, int(d["잠정"]), int(d["최근구간"]), int(d["평가구간"]), int(d["②가용구간"]),
        ])
        M, N = L(ci["P약세_θT평활"]), L(ci["실제국면_θT"])
        ws.cell(row=k, column=ci["실제국면_θT"], value=f"=IF({M}{k}>0.5,1,0)")
        for tag, pcol in [("①", "P약세_①당시고정"), ("②", "P약세_②재적합"), ("θT필터", "P약세_θT필터")]:
            P = L(ci[pcol]); Bc, Cc, Lc = L(ci[f"{tag}Brier"]), L(ci[f"{tag}p절단"]), L(ci[f"{tag}LL"])
            ws.cell(row=k, column=ci[f"{tag}Brier"], value=f'=IF({P}{k}="","",({P}{k}-{N}{k})^2)')
            ws.cell(row=k, column=ci[f"{tag}p절단"], value=f'=IF({P}{k}="","",MIN(MAX({P}{k},{CLIP_LO}),{CLIP_HI}))')
            ws.cell(row=k, column=ci[f"{tag}LL"], value=f'=IF({Cc}{k}="","",-({N}{k}*LN({Cc}{k})+(1-{N}{k})*LN(1-{Cc}{k})))')
            if tag != "θT필터":
                ws.cell(row=k, column=ci[f"{tag}정확"], value=f'=IF({P}{k}="","",IF(({P}{k}>0.5)=({N}{k}=1),1,0))')
    style_header(ws, len(cols))
    for row in ws.iter_rows(min_row=2):
        row[0].number_format = "yyyy-mm-dd"; row[ci["②파라미터기준일"] - 1].number_format = "yyyy-mm-dd"
        for j in range(1, len(cols)):
            if j not in (ci["②파라미터기준일"] - 1,):
                row[j].number_format = "0.00000"
    ws.freeze_panes = "B2"; ws.column_dimensions["A"].width = 12
    last_ts = n_ts + 1
    info["ts_rows"] = n_ts

    # ── 전이행렬 ───────────────────────────────────────────────────────────
    ws = wb.create_sheet("전이행렬")
    ws["A1"] = "전이행렬 (행 = 현재 국면, 열 = 다음 국면). 파란 글씨 = 적합 결과 입력값, 검정 = 수식"; ws["A1"].font = BOLD

    def matrix_block(top: int, title: str, P: list, is_formula_source: str | None = None):
        """2×2 행렬을 (top, B..C)에 쓰고, P^2·P^4·P^8·P^12·P^13 을 수식으로 오른쪽에 전개. 반환: {'P':(r,c)...} 좌표"""
        ws.cell(row=top, column=1, value=title).font = BOLD
        hdr = ["", "→강세", "→약세"]
        blocks = {}
        names = ["P", "P^2", "P^4", "P^8", "P^12", "P^13"]
        col0 = 2
        for bi, nm in enumerate(names):
            c = col0 + bi * 4
            ws.cell(row=top + 1, column=c - 1, value=nm).font = BOLD
            ws.cell(row=top + 1, column=c, value="→강세").font = BOLD; ws.cell(row=top + 1, column=c + 1, value="→약세").font = BOLD
            ws.cell(row=top + 2, column=c - 1, value="강세").font = BOLD; ws.cell(row=top + 3, column=c - 1, value="약세").font = BOLD
            blocks[nm] = (top + 2, c)
        r, c = blocks["P"]
        if is_formula_source:
            for i in range(2):
                for j in range(2):
                    cell = ws.cell(row=r + i, column=c + j, value=is_formula_source[i][j]); cell.font = GREEN; cell.number_format = "0.00000"
        else:
            for i in range(2):
                for j in range(2):
                    cell = ws.cell(row=r + i, column=c + j, value=float(P[i][j])); cell.font = BLUE; cell.number_format = "0.00000"

        def mult(dst, A, B):
            (ra, ca), (rb, cb), (rd, cd) = A, B, dst
            for i in range(2):
                for j in range(2):
                    f = f"={L(ca)}{ra+i}*{L(cb+j)}{rb}+{L(ca+1)}{ra+i}*{L(cb+j)}{rb+1}"
                    cell = ws.cell(row=rd + i, column=cd + j, value=f); cell.number_format = "0.00000"
        mult(blocks["P^2"], blocks["P"], blocks["P"])
        mult(blocks["P^4"], blocks["P^2"], blocks["P^2"])
        mult(blocks["P^8"], blocks["P^4"], blocks["P^4"])
        mult(blocks["P^12"], blocks["P^8"], blocks["P^4"])
        mult(blocks["P^13"], blocks["P^12"], blocks["P"])
        rp, cp = blocks["P"]
        ws.cell(row=top + 4, column=1, value="정상상태 약세비중 = P[강→약]/(P[강→약]+P[약→강])")
        cell = ws.cell(row=top + 4, column=2, value=f"={L(cp+1)}{rp}/({L(cp+1)}{rp}+{L(cp)}{rp+1})"); cell.number_format = "0.0000"
        ws.cell(row=top + 5, column=1, value="기대지속기간(주): 강세 = 1/(1-P[강→강]), 약세 = 1/(1-P[약→약])")
        ws.cell(row=top + 5, column=2, value=f"=1/(1-{L(cp)}{rp})").number_format = "0.0"
        ws.cell(row=top + 5, column=3, value=f"=1/(1-{L(cp+1)}{rp+1})").number_format = "0.0"
        return blocks

    P0 = summ["theta0"]["P"]; PT = summ["thetaT"]["P"]
    b0 = matrix_block(3, "θ0 (당시 2026-06-05 적합) — 버전 ① 고정 전이행렬", P0)
    bT = matrix_block(11, "θT (2026-10-02 적합) — 사후 라벨용(참고)", PT)
    # ② 2026-10-02 재적합 전이행렬: 국면확률 시트 마지막 평가행에서 참조
    last_date = ts["날짜"].iloc[-1]
    dd = f"DATE({last_date.year},{last_date.month},{last_date.day})"
    def idx(colname):   # 마지막(=최신) 행 직접 참조
        return f"='국면확률'!${L(ci[colname])}${last_ts}"
    b2 = matrix_block(19, f"② {last_date:%Y-%m-%d} 재적합 전이행렬 (국면확률 시트 참조)", None,
                      is_formula_source=[[idx("②강→강"), idx("②강→약")], [idx("②약→강"), idx("②약→약")]])

    # 손검산
    top = 27
    ws.cell(row=top, column=1, value=f"손검산 — 예측시점 {last_date:%Y-%m-%d} (국면확률 시트 참조값 × 위 전이행렬 거듭제곱)").font = BOLD
    hc = summ["hand_check"]
    ws.cell(row=top + 1, column=1, value="항목").font = BOLD; ws.cell(row=top + 1, column=2, value="① 수식").font = BOLD
    ws.cell(row=top + 1, column=3, value="① 파이썬").font = BOLD; ws.cell(row=top + 1, column=4, value="② 수식").font = BOLD; ws.cell(row=top + 1, column=5, value="② 파이썬").font = BOLD
    p1 = f"'국면확률'!${L(ci['P약세_①당시고정'])}${last_ts}"
    p2 = f"'국면확률'!${L(ci['P약세_②재적합'])}${last_ts}"
    ws.cell(row=top + 2, column=1, value="p_t = P(약세) 필터값")
    ws.cell(row=top + 2, column=2, value=f"={p1}").font = GREEN; ws.cell(row=top + 2, column=3, value=hc["①당시고정"]["p_t"])
    ws.cell(row=top + 2, column=4, value=f"={p2}").font = GREEN; ws.cell(row=top + 2, column=5, value=hc["②재적합"]["p_t"])
    rr = top + 3
    for h, key in [(1, "P"), (4, "P^4"), (13, "P^13")]:
        for ver, blk, pcell, py in [("①", b0, f"B{top+2}", hc["①당시고정"]), ("②", b2, f"D{top+2}", hc["②재적합"])]:
            r0, c0 = blk[key]
            ub, bb, uu = f"{L(c0+1)}{r0}", f"{L(c0+1)}{r0+1}", f"{L(c0)}{r0}"
            col = 2 if ver == "①" else 4
            ws.cell(row=rr, column=1, value=f"h={h}: q_약세 = (1-p)·{key}[강→약] + p·{key}[약→약]")
            ws.cell(row=rr, column=col, value=f"=(1-{pcell})*{ub}+{pcell}*{bb}").number_format = "0.00000"
            ws.cell(row=rr, column=col + 1, value=py[f"q_약세_h{h}"]).number_format = "0.00000"
            ws.cell(row=rr + 1, column=1, value=f"h={h}: q_전환 = (1-p)·(1-{key}[강→강]) + p·(1-{key}[약→약])")
            ws.cell(row=rr + 1, column=col, value=f"=(1-{pcell})*(1-{uu})+{pcell}*(1-{bb})").number_format = "0.00000"
            ws.cell(row=rr + 1, column=col + 1, value=py[f"q_전환_h{h}"]).number_format = "0.00000"
        rr += 2
    ws.column_dimensions["A"].width = 62
    for c in range(2, 26):
        ws.column_dimensions[L(c)].width = 11

    # ── 예측vs실제 ─────────────────────────────────────────────────────────
    ws = wb.create_sheet("예측vs실제")
    bcols = ["버전", "h", "예측시점", "목표시점", "p_t", "강→강^h", "강→약^h", "약→강^h", "약→약^h", "q_약세", "y_약세", "L_t", "y_전환", "q_전환",
             "B1_q_약세", "B2_q_약세", "B1_q_전환", "B2_q_전환", "최근구간", "목표잠정", "이월의존구간",
             "q_약세_검산", "q_전환_검산", "Brier_약세", "Brier_약세_B1", "Brier_약세_B2", "LL_약세", "LL_약세_B1", "LL_약세_B2",
             "Brier_전환", "Brier_전환_B1", "Brier_전환_B2", "LL_전환", "LL_전환_B1", "LL_전환_B2", "정확_약세"]
    ws.append(bcols)
    bi = {c: i + 1 for i, c in enumerate(bcols)}
    def ll(qc, yc, k):
        qq = f"MIN(MAX({qc}{k},{CLIP_LO}),{CLIP_HI})"
        return f"=-({yc}{k}*LN({qq})+(1-{yc}{k})*LN(1-{qq}))"
    for k, r in enumerate(dfB.itertuples(index=False), start=2):
        d = dict(zip(dfB.columns, r))
        ws.append([d["버전"], int(d["h"]), d["예측시점"].to_pydatetime(), d["목표시점"].to_pydatetime(), float(d["p_t"]),
                   float(d["강→강^h"]), float(d["강→약^h"]), float(d["약→강^h"]), float(d["약→약^h"]), float(d["q_약세"]), int(d["y_약세"]), int(d["L_t"]),
                   int(d["y_전환"]), float(d["q_전환"]), float(d["B1_q_약세"]), float(d["B2_q_약세"]), float(d["B1_q_전환"]),
                   (None if pd.isna(d["B2_q_전환"]) else float(d["B2_q_전환"])), int(d["최근구간"]), int(d["목표잠정"]), int(d["이월의존구간"])])
        E, F, G, I, J, K, M, N, O, Pc, Q, R = (L(bi[x]) for x in ["p_t", "강→강^h", "강→약^h", "약→약^h", "q_약세", "y_약세", "y_전환", "q_전환", "B1_q_약세", "B2_q_약세", "B1_q_전환", "B2_q_전환"])
        ws.cell(row=k, column=bi["q_약세_검산"], value=f"=(1-{E}{k})*{G}{k}+{E}{k}*{I}{k}")
        ws.cell(row=k, column=bi["q_전환_검산"], value=f"=(1-{E}{k})*(1-{F}{k})+{E}{k}*(1-{I}{k})")
        ws.cell(row=k, column=bi["Brier_약세"], value=f"=({J}{k}-{K}{k})^2")
        ws.cell(row=k, column=bi["Brier_약세_B1"], value=f"=({O}{k}-{K}{k})^2")
        ws.cell(row=k, column=bi["Brier_약세_B2"], value=f"=({Pc}{k}-{K}{k})^2")
        ws.cell(row=k, column=bi["LL_약세"], value=ll(J, K, k)); ws.cell(row=k, column=bi["LL_약세_B1"], value=ll(O, K, k)); ws.cell(row=k, column=bi["LL_약세_B2"], value=ll(Pc, K, k))
        ws.cell(row=k, column=bi["Brier_전환"], value=f"=({N}{k}-{M}{k})^2")
        ws.cell(row=k, column=bi["Brier_전환_B1"], value=f"=({Q}{k}-{M}{k})^2")
        ws.cell(row=k, column=bi["Brier_전환_B2"], value=f"=({R}{k}-{M}{k})^2")
        ws.cell(row=k, column=bi["LL_전환"], value=ll(N, M, k)); ws.cell(row=k, column=bi["LL_전환_B1"], value=ll(Q, M, k)); ws.cell(row=k, column=bi["LL_전환_B2"], value=ll(R, M, k))
        ws.cell(row=k, column=bi["정확_약세"], value=f"=IF(({J}{k}>0.5)=({K}{k}=1),1,0)")
    style_header(ws, len(bcols))
    for row in ws.iter_rows(min_row=2):
        row[2].number_format = "yyyy-mm-dd"; row[3].number_format = "yyyy-mm-dd"
        for j in range(4, len(bcols)):
            row[j].number_format = "0.00000"
    ws.freeze_panes = "E2"
    last_b = len(dfB) + 1
    info["pred_rows"] = len(dfB)

    # ── 지표 (수식) ────────────────────────────────────────────────────────
    ws = wb.create_sheet("지표")
    ws["A1"] = "(B) 전환확률 지표 — 예측vs실제 시트의 AVERAGEIFS/COUNTIFS 수식. '파이썬' 열은 step5_evaluate.py 계산값(대조용)"; ws["A1"].font = BOLD
    ms = ["n", "전환발생수", "Brier_전환", "Brier_전환_B1", "Brier_전환_B2", "skill_전환_B1", "skill_전환_B2", "LL_전환", "LL_전환_B1", "LL_전환_B2",
          "Brier_약세", "Brier_약세_B1", "Brier_약세_B2", "skill_약세_B1", "skill_약세_B2", "LL_약세", "LL_약세_B1", "LL_약세_B2", "정확도_약세@0.5"]
    hdr = ["버전", "h", "구간"] + [x for m in ms for x in (m, f"{m}(파이썬)")]
    for j, h_ in enumerate(hdr, 1):
        ws.cell(row=2, column=j, value=h_)
    style_header(ws, len(hdr), row=2)
    def rng(col):
        return f"'예측vs실제'!${L(bi[col])}$2:${L(bi[col])}${last_b}"
    row = 3
    for ver in ["①당시고정", "②재적합"]:
        for h in [1, 4, 13]:
            for wname, extra in [("전체(2013-01~2026-10)", f",{rng('이월의존구간')},0"), ("최근(2026-06-05~)", f",{rng('최근구간')},1"), ("②가용(2011-08~2026-10)", "")]:
                crit = f"{rng('버전')},\"{ver}\",{rng('h')},{h}{extra}"
                py = dfM[(dfM["버전"] == ver) & (dfM["h"] == h) & (dfM["구간"] == wname)]
                py = py.iloc[0] if len(py) else None
                ws.cell(row=row, column=1, value=ver); ws.cell(row=row, column=2, value=h); ws.cell(row=row, column=3, value=wname)
                col = 4
                cells = {}
                for m in ms:
                    if m == "n":
                        f = f"=COUNTIFS({crit})"
                    elif m == "전환발생수":
                        f = f"=SUMIFS({rng('y_전환')},{crit})"
                    elif m.startswith("skill_"):
                        base = m.replace("skill_", "")           # 전환_B1 → Brier_전환 / Brier_전환_B1
                        kind, bl = base.split("_")
                        f = f"=IFERROR(1-{L(cells['Brier_'+kind])}{row}/{L(cells['Brier_'+kind+'_'+bl])}{row},\"\")"
                    elif m == "정확도_약세@0.5":
                        f = f"=IFERROR(AVERAGEIFS({rng('정확_약세')},{crit}),\"\")"
                    else:
                        f = f"=IFERROR(AVERAGEIFS({rng(m)},{crit}),\"\")"
                    c1 = ws.cell(row=row, column=col, value=f); cells[m] = col
                    c2 = ws.cell(row=row, column=col + 1, value=(None if py is None or m not in py or pd.isna(py[m]) else float(py[m])))
                    c2.font = GREEN
                    if m not in ("n", "전환발생수"):
                        c1.number_format = "0.0000"; c2.number_format = "0.0000"
                    col += 2
                row += 1
    # (A) 블록
    row += 2
    ws.cell(row=row, column=1, value="(A) 국면 진단 지표 — 국면확률 시트 수식 (p_t 대 실제국면_θT)").font = BOLD; row += 1
    ahdr = ["버전", "구간", "n", "Brier", "Brier(파이썬)", "log_loss", "log_loss(파이썬)", "정확도", "정확도(파이썬)", "TP", "FP", "FN", "TN"]
    for j, h_ in enumerate(ahdr, 1):
        ws.cell(row=row, column=j, value=h_)
    style_header(ws, len(ahdr), row=row); row += 1
    def trng(col):
        return f"'국면확률'!${L(ci[col])}$2:${L(ci[col])}${last_ts}"
    for ver, tag, pcol in [("①당시고정", "①", "P약세_①당시고정"), ("②재적합", "②", "P약세_②재적합"), ("참고:θT전체표본필터", "θT필터", "P약세_θT필터")]:
        for wname, wcol in [("전체(2013-01~2026-10)", "평가구간"), ("최근(2026-06-05~)", "최근구간"), ("②가용(2011-08~2026-10)", "②가용구간")]:
            py = dfA[(dfA["버전"] == ver) & (dfA["구간"] == wname)].iloc[0]
            ws.cell(row=row, column=1, value=ver); ws.cell(row=row, column=2, value=wname)
            ws.cell(row=row, column=3, value=f"=SUMPRODUCT(--({trng(wcol)}=1),--ISNUMBER({trng(pcol)}))")
            ws.cell(row=row, column=4, value=f"=IFERROR(AVERAGEIFS({trng(tag+'Brier')},{trng(wcol)},1),\"\")").number_format = "0.0000"
            ws.cell(row=row, column=5, value=float(py["Brier"])).number_format = "0.0000"
            ws.cell(row=row, column=6, value=f"=IFERROR(AVERAGEIFS({trng(tag+'LL')},{trng(wcol)},1),\"\")").number_format = "0.0000"
            ws.cell(row=row, column=7, value=float(py["log_loss"])).number_format = "0.0000"
            if tag != "θT필터":
                ws.cell(row=row, column=8, value=f"=IFERROR(AVERAGEIFS({trng(tag+'정확')},{trng(wcol)},1),\"\")").number_format = "0.0000"
            ws.cell(row=row, column=9, value=float(py["정확도"])).number_format = "0.0000"
            P, Nn, W = trng(pcol), trng("실제국면_θT"), trng(wcol)
            ws.cell(row=row, column=10, value=f"=SUMPRODUCT(--({W}=1),--ISNUMBER({P}),--({P}>0.5),--({Nn}=1))")
            ws.cell(row=row, column=11, value=f"=SUMPRODUCT(--({W}=1),--ISNUMBER({P}),--({P}>0.5),--({Nn}=0))")
            ws.cell(row=row, column=12, value=f"=SUMPRODUCT(--({W}=1),--ISNUMBER({P}),--({P}<=0.5),--({Nn}=1))")
            ws.cell(row=row, column=13, value=f"=SUMPRODUCT(--({W}=1),--ISNUMBER({P}),--({P}<=0.5),--({Nn}=0))")
            for j in (5, 7, 9):
                ws.cell(row=row, column=j).font = GREEN
            row += 1
    ws.column_dimensions["A"].width = 20; ws.column_dimensions["C"].width = 22
    for c in range(4, len(hdr) + 1):
        ws.column_dimensions[L(c)].width = 13

    # ── 캘리브레이션 (수식) ───────────────────────────────────────────────
    ws = wb.create_sheet("캘리브레이션")
    ws["A1"] = "예측확률 10%p 구간별 건수·평균 예측확률·실제 빈도 (전체 구간 2013-01~2026-10, 이월의존구간 제외). 종류 '전환' = q_전환 대 y_전환, '약세' = q_약세 대 y_약세"; ws["A1"].font = BOLD
    chdr = ["버전", "h", "종류", "구간하한", "구간상한", "n", "평균예측확률", "실제빈도"]
    for j, h_ in enumerate(chdr, 1):
        ws.cell(row=2, column=j, value=h_)
    style_header(ws, len(chdr), row=2)
    row = 3
    edges = [round(x / 10, 1) for x in range(11)]
    for ver in ["①당시고정", "②재적합"]:
        for h in [1, 4, 13]:
            for kind, qc, yc in [("전환", "q_전환", "y_전환"), ("약세", "q_약세", "y_약세")]:
                for lo, hi in zip(edges[:-1], edges[1:]):
                    ws.cell(row=row, column=1, value=ver); ws.cell(row=row, column=2, value=h); ws.cell(row=row, column=3, value=kind)
                    ws.cell(row=row, column=4, value=lo); ws.cell(row=row, column=5, value=hi)
                    op_hi = "<=" if hi >= 1 else "<"
                    crit = f"{rng('버전')},\"{ver}\",{rng('h')},{h},{rng('이월의존구간')},0,{rng(qc)},\">=\"&D{row},{rng(qc)},\"{op_hi}\"&E{row}"
                    ws.cell(row=row, column=6, value=f"=COUNTIFS({crit})")
                    ws.cell(row=row, column=7, value=f"=IFERROR(AVERAGEIFS({rng(qc)},{crit}),\"\")").number_format = "0.0000"
                    ws.cell(row=row, column=8, value=f"=IFERROR(AVERAGEIFS({rng(yc)},{crit}),\"\")").number_format = "0.0000"
                    row += 1
    ws.freeze_panes = "A3"

    # ── 탐지지연 / 걸어서재적합현황 (값) ─────────────────────────────────
    ws = wb.create_sheet("탐지지연")
    ws.append(list(dfLag.columns))
    for r in dfLag.itertuples(index=False):
        ws.append([None if (isinstance(x, float) and pd.isna(x)) else x for x in r])
    style_header(ws, len(dfLag.columns)); ws.freeze_panes = "A2"
    ws = wb.create_sheet("걸어서재적합현황")
    ws.append(list(dfW.columns))
    for r in dfW.itertuples(index=False):
        ws.append([None if (isinstance(x, float) and pd.isna(x)) else (int(x) if isinstance(x, (int, float)) and float(x).is_integer() else x) for x in r])
    style_header(ws, len(dfW.columns))

    for w in wb.worksheets:
        set_font(w)
    wb.save(path)
    info["path"] = str(path)
    return info


if __name__ == "__main__":
    args = sys.argv[1:]
    max_rows = None
    if "--max-rows" in args:
        i = args.index("--max-rows"); max_rows = int(args[i + 1]); del args[i:i + 2]
    out = Path(args[0]) if args else OUT / "전환확률_검증.xlsx"
    print(build(out, max_rows=max_rows))
