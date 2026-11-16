# =============================================================================
# validation/step5_verify_xlsx.py — 워크북 수식 검증 (LibreOffice 없음 → 'formulas' 패키지로 계산)
#   1) 축소본(각 표 뒤에서 N행)을 같은 builder 로 만들어 전체 수식을 계산 → 오류 셀 유무, 파이썬 재계산과 대조
#   2) 전체본은 수식 문자열을 파이썬으로 동일하게 재계산해 '지표' 시트의 파이썬 열과 대조 (formulas 계산은 선택)
# =============================================================================
from __future__ import annotations
import re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5_build_xlsx import build, OUT  # noqa: E402
import formulas  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
tmp = OUT.parent / "tmp" / f"verify_reduced_{N}.xlsx"
info = build(tmp, max_rows=N)
print("reduced workbook:", info)
t0 = time.time()
xl = formulas.ExcelModel().loads(str(tmp)).finish()
sol = xl.calculate()
print(f"formulas calculated {len(sol)} cells in {time.time()-t0:.0f}s")

def cell(sheet, ref):
    key = [k for k in sol if k.upper().endswith(f"{sheet.upper()}'!{ref.upper()}") or k.upper().endswith(f"{sheet.upper()}!{ref.upper()}")]
    if not key:
        return None
    v = sol[key[0]].value
    return v[0][0] if hasattr(v, "shape") else v

errs = []
for k, v in sol.items():
    val = v.value if hasattr(v, "value") else v
    arr = np.array(val, dtype=object).ravel() if hasattr(val, "shape") or isinstance(val, (list, tuple)) else [val]
    for a in arr:
        if isinstance(a, formulas.functions.XlError) or (isinstance(a, str) and a.startswith("#")):
            errs.append((k, str(a)))
print("error cells:", len(errs), errs[:10])

# 대조 1: 예측vs실제 수식 vs pandas
dfB = pd.read_csv(OUT / "전환확률_예측vs실제.csv").groupby(["버전", "h"], sort=False).tail(N).reset_index(drop=True)
def ll(q, y):
    qc = np.clip(q, 0.01, 0.99); return -(y * np.log(qc) + (1 - y) * np.log(1 - qc))
import openpyxl
from openpyxl.utils import get_column_letter
_hdr = [c.value for c in openpyxl.load_workbook(tmp, read_only=True)["예측vs실제"][1]]
COL = {name: get_column_letter(i + 1) for i, name in enumerate(_hdr)}   # 헤더명 → 열 문자 (열 삽입에 안전)
checks = []
for i in range(min(len(dfB), 40)):
    r = i + 2
    row = dfB.iloc[i]
    checks.append(abs(cell("예측vs실제", f"{COL['Brier_약세']}{r}") - (row["q_약세"] - row["y_약세"]) ** 2))
    checks.append(abs(cell("예측vs실제", f"{COL['LL_약세']}{r}") - ll(row["q_약세"], row["y_약세"])))
    checks.append(abs(cell("예측vs실제", f"{COL['Brier_전환']}{r}") - (row["q_전환"] - row["y_전환"]) ** 2))
    checks.append(abs(cell("예측vs실제", f"{COL['LL_전환']}{r}") - ll(row["q_전환"], row["y_전환"])))
    checks.append(abs(cell("예측vs실제", f"{COL['q_약세_검산']}{r}") - row["q_약세"]))
    checks.append(abs(cell("예측vs실제", f"{COL['q_전환_검산']}{r}") - row["q_전환"]))
print("예측vs실제 행 수식 vs pandas 최대차:", max(checks))

# 대조 2: 지표 시트 (축소본 기준 AVERAGEIFS) vs pandas
mism = []
row = 3
for ver in ["①당시고정", "②재적합"]:
    for h in [1, 4, 13]:
        for wname, rec in [("전체", lambda g: g[g["이월의존구간"] == 0]), ("최근", lambda g: g[g["최근구간"] == 1]), ("②가용", lambda g: g)]:
            g = rec(dfB[(dfB["버전"] == ver) & (dfB["h"] == h)])
            n_x = cell("지표", f"D{row}"); b_x = cell("지표", f"H{row}"); s_x = cell("지표", f"P{row}")
            if len(g):
                n_p = len(g); b_p = float(((g["q_전환"] - g["y_전환"]) ** 2).mean()); b2 = float(((g["B2_q_전환"] - g["y_전환"]) ** 2).mean())
                s_p = 1 - b_p / b2 if b2 > 0 else ""
                mism.append((row, ver, h, wname, n_x, n_p, abs(float(b_x) - b_p) if isinstance(b_x, (int, float)) else b_x, s_x, s_p))
            row += 1
bad = [m for m in mism if not (m[4] == m[5] and isinstance(m[6], float) and m[6] < 1e-9)]
print("지표 시트 대조 (n 일치·Brier_전환 차<1e-9 조건) 불일치:", len(bad), bad[:4])

# 대조 3: 전이행렬 거듭제곱·손검산
import json
summ = json.load(open(OUT / "step5_summary.json", encoding="utf-8"))
P0 = np.array(summ["theta0"]["P"])
P13 = np.linalg.matrix_power(P0, 13); P4 = np.linalg.matrix_power(P0, 4)
d1 = abs(cell("전이행렬", "V5") - P13[0, 0]) + abs(cell("전이행렬", "W6") - P13[1, 1]) + abs(cell("전이행렬", "J5") - P4[0, 0])
print("전이행렬 P^4/P^13 수식 vs numpy 차:", d1)
hc = summ["hand_check"]["①당시고정"]
print("손검산 ① h=1 q_약세 수식:", cell("전이행렬", "B30"), "파이썬:", hc["q_약세_h1"], "| h=4 수식:", cell("전이행렬", "B32"), "파이썬:", hc["q_약세_h4"], "| h=13 수식:", cell("전이행렬", "B34"), "파이썬:", hc["q_약세_h13"])
print("손검산 ② h=1 q_약세 수식:", cell("전이행렬", "D30"), "파이썬:", summ["hand_check"]["②재적합"]["q_약세_h1"])
# 대조 4: 캘리브레이션 첫 블록 n 합 = 그룹 행수
tot = sum((cell("캘리브레이션", f"F{r}") or 0) for r in range(3, 13))
print("캘리브레이션 ①h1 전환 n 합:", tot, "| 그룹 행수(이월의존 제외):", len(dfB[(dfB['버전']=='①당시고정') & (dfB['h']==1) & (dfB['이월의존구간']==0)]))
