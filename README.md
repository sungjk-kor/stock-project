# Stock Index Forecast Streamlit

SARIMAX를 이용해 S&P500 주간 로그수익률의 중기 추세를 예측하는 Streamlit 프로젝트입니다.

## 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 폴더 구조

```text
stock-AI-project/
├─ app.py
├─ requirements.txt
├─ README.md
├─ data/
│  ├─ raw/
│  └─ processed/
├─ utils/
├─ pages/
└─ notebooks/
```

## 데이터 준비

최종 학습 데이터는 `data/processed/final_dataset.csv`에 저장합니다.
필수 컬럼은 날짜 컬럼(`date`)과 목표 변수(`sp500_log_return`)입니다.

## 주요 화면

- 홈: 프로젝트 요약과 데이터 상태 확인
- 데이터 확인: 원천/가공 데이터 미리보기
- 거시 변수: 입력 변수 그룹별 추이 탐색
- 예측: SARIMAX 학습, 백테스트, 예측 결과 확인
