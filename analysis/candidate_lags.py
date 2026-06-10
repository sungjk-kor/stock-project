# 사전백색화 CCF 기반 후보 lag — 워크포워드 입력용
# 생성 기준: MAX_LAG=16, STRONG_MULT=2.5, MAX_CANDIDATES_PER_VAR=2

candidate_lags = {
    "financial_conditions": [1, 5],
    "vix_log_return": [1, 2],
    "cpi_change": [4, 10],
}

strong_lags    # strong만 = {
    "financial_conditions": [1],
}

marginal_lags  # marginal만 = {
    "financial_conditions": [5],
    "vix_log_return": [1, 2],
    "cpi_change": [4, 10],
}
