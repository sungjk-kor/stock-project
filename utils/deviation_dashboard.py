# =============================================================================
# deviation_dashboard.py  —  국면 이탈 점수 정적 HTML 대시보드 생성
#
# 목적:
#   평온강세 이탈 점수를 시각화하고, S3/GitHub Pages에 올릴 수 있는
#   단일 index.html을 생성합니다.
#   ★ 현재 상태 측정(coincident) 지표이며 예측이 아닙니다.
#
# =============================================================================

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class DeviationDashboard:
    """국면 이탈 점수 대시보드 생성 클래스"""

    def __init__(self, csv_path: str = "data/processed/final_dataset.csv"):
        """
        초기화

        Args:
            csv_path: 최종 데이터셋 경로
        """
        self.csv_path = csv_path
        self.df = None
        self.df_valid = None
        self.stats = {}
        self.config = {
            "crisis_periods": [
                ("2008-09-01", "2009-06-30"),
                ("2020-02-15", "2020-06-30"),
            ],
            "features": ["dxy_chg", "y10_chg", "financial_conditions", "credit_spread", "vix"],
            "feature_names": {
                "dxy_chg": "DXY 13주 변화율",
                "y10_chg": "10Y 금리 변화(pp)",
                "financial_conditions": "NFCI 금융여건",
                "credit_spread": "HY 신용스프레드",
                "vix": "VIX",
            },
            "regime_colors": {
                "peacetime": "#2563eb",      # 파랑 - 평온강세
                "caution": "#f59e0b",         # 주황 - 위험주의
                "rising_rates": "#ef4444",    # 빨강 - 금리상승기
                "falling_rates": "#10b981",   # 초록 - 금리하락기
            }
        }

    def load_data(self) -> bool:
        """
        데이터 로드 및 검증

        Returns:
            bool: 성공 여부
        """
        print("[1/5] 데이터 로드 중...")

        try:
            self.df = pd.read_csv(self.csv_path, parse_dates=["date"])
            self.df = self.df.sort_values("date").reset_index(drop=True)
            print(f"  ✓ 로드: {len(self.df):,}주  ({self.df['date'].min().date()} ~ {self.df['date'].max().date()})")

            # 필수 원본 컬럼 확인 (dxy_chg, y10_chg는 계산으로 생성)
            required_raw = ["dxy", "treasury_10y", "financial_conditions", "credit_spread", "vix"]
            missing = [col for col in required_raw if col not in self.df.columns]
            if missing:
                print(f"  ✗ 누락 컬럼: {missing}")
                return False

            return True
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            return False

    def calculate_features(self) -> bool:
        """
        변수 생성 및 표준화

        Returns:
            bool: 성공 여부
        """
        print("[2/5] 변수 생성 및 표준화 중...")

        try:
            # 변수 생성
            self.df["dxy_chg"] = np.log(self.df["dxy"] / self.df["dxy"].shift(13))
            self.df["y10_chg"] = self.df["treasury_10y"] - self.df["treasury_10y"].shift(13)

            # 위기 구간 마스크
            crisis_mask = pd.Series(False, index=self.df.index)
            for s, e in self.config["crisis_periods"]:
                crisis_mask |= (self.df["date"] >= s) & (self.df["date"] <= e)

            # 비위기 구간으로 표준화 학습
            features = self.config["features"]
            df_nc = self.df[~crisis_mask].dropna(subset=features)

            sc = StandardScaler()
            sc.fit(df_nc[features])

            # 전체에 적용
            self.df_valid = self.df.dropna(subset=features).copy()
            X_std = sc.transform(self.df_valid[features])

            for i, feat in enumerate(features):
                self.df_valid[f"__std_{feat}"] = X_std[:, i]

            self.df_valid["crisis"] = crisis_mask.reindex(self.df_valid.index, fill_value=False).values

            print(f"  ✓ 표준화 완료: {len(self.df_valid)}주")

            # 저장용
            self.scaler = sc
            self.crisis_mask = crisis_mask
            self.X_std = X_std

            return True
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            return False

    def fit_clusters(self) -> bool:
        """
        KMeans 군집화 및 센트로이드 계산

        Returns:
            bool: 성공 여부
        """
        print("[3/5] 군집화 및 센트로이드 계산 중...")

        try:
            features = self.config["features"]

            # 비위기 표준화값으로 KMeans fit
            df_nc = self.df[~self.crisis_mask].dropna(subset=features)
            X_nc = self.scaler.transform(df_nc[features])

            km = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
            km.fit(X_nc)

            # 클러스터 정체성 파악
            df_nc2 = df_nc.copy()
            df_nc2["cl4"] = km.predict(X_nc)

            vix_by_cl = {c: df_nc2.loc[df_nc2["cl4"] == c, "vix"].mean() for c in range(4)}
            nfci_by_cl = {c: df_nc2.loc[df_nc2["cl4"] == c, "financial_conditions"].mean() for c in range(4)}

            peacetime_label = min(vix_by_cl, key=vix_by_cl.get)
            danger_label = max(nfci_by_cl, key=nfci_by_cl.get)

            C_peace = km.cluster_centers_[peacetime_label]
            C_danger = km.cluster_centers_[danger_label]

            print(f"  ✓ 평온강세 클러스터: {peacetime_label}번")
            print(f"  ✓ 위험주의 클러스터: {danger_label}번")

            # 저장용
            self.km = km
            self.peacetime_label = peacetime_label
            self.danger_label = danger_label
            self.C_peace = C_peace
            self.C_danger = C_danger

            return True
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            return False

    def calculate_scores(self) -> bool:
        """
        이탈 점수 및 기여도 계산

        Returns:
            bool: 성공 여부
        """
        print("[4/5] 이탈 점수 및 기여도 계산 중...")

        try:
            # 이탈 방향 벡터
            d_raw = self.C_danger - self.C_peace
            d_norm = d_raw / np.linalg.norm(d_raw)

            # 이탈 점수 계산
            dev_vecs = self.X_std - self.C_peace
            scores = dev_vecs @ d_norm
            contribs = dev_vecs * d_norm

            # 데이터프레임에 추가
            self.df_valid["score"] = scores
            features = self.config["features"]
            for i, feat in enumerate(features):
                self.df_valid[f"contrib_{feat}"] = contribs[:, i]

            # GMM 소속확률 계산
            df_nc = self.df[~self.crisis_mask].dropna(subset=features)
            X_nc = self.scaler.transform(df_nc[features])

            gm = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=5)
            gm.fit(X_nc)

            gm_means = gm.means_
            peace_comp = int(np.argmin(np.linalg.norm(gm_means - self.C_peace, axis=1)))
            danger_comp = int(np.argmin(np.linalg.norm(gm_means - self.C_danger, axis=1)))

            p_all = gm.predict_proba(self.X_std)
            self.df_valid["p_peace"] = p_all[:, peace_comp]
            self.df_valid["p_danger"] = p_all[:, danger_comp]

            print(f"  ✓ 이탈 점수 계산 완료")

            # 저장용
            self.d_norm = d_norm
            self.gm = gm

            return True
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            return False

    def calculate_stats(self) -> bool:
        """
        통계치 계산 (현재값, 역대값 등)

        Returns:
            bool: 성공 여부
        """
        print("[5/5] 통계치 계산 중...")

        try:
            # 비위기 구간 통계
            nc_scores = self.df_valid.loc[~self.df_valid["crisis"], "score"]

            # 비위기 기준 극값
            max_score_nc = float(nc_scores.max())
            max_date_nc = self.df_valid.loc[nc_scores.idxmax(), "date"].date()
            min_score_nc = float(nc_scores.min())
            min_date_nc = self.df_valid.loc[nc_scores.idxmin(), "date"].date()

            # 전체 범위 극값 (위기 포함, 백분위 바에 표시)
            all_scores = self.df_valid["score"]
            max_score = float(all_scores.max())
            max_date = self.df_valid.loc[all_scores.idxmax(), "date"].date()
            min_score = float(all_scores.min())
            min_date = self.df_valid.loc[all_scores.idxmin(), "date"].date()
            mean_score = float(nc_scores.mean())
            std_score = float(nc_scores.std())
            median_score = float(nc_scores.median())

            # 현재값
            latest_row = self.df_valid.dropna(subset=self.config["features"]).iloc[-1]
            current_score = float(latest_row["score"])
            current_date = latest_row["date"].date()
            current_pct = float((nc_scores < current_score).mean() * 100)
            current_peace = float(latest_row["p_peace"])

            # 지표별 기여도
            contrib_cols = {f: float(latest_row[f"contrib_{f}"]) for f in self.config["features"]}

            # 사분위수 계산
            q25 = float(nc_scores.quantile(0.25))
            q50 = float(nc_scores.quantile(0.50))
            q75 = float(nc_scores.quantile(0.75))

            self.stats = {
                "current_score": round(current_score, 4),
                "current_date": str(current_date),
                "current_pct": round(current_pct, 1),
                "current_peace": round(current_peace, 3),
                "max_score": round(max_score, 4),
                "max_date": str(max_date),
                "min_score": round(min_score, 4),
                "min_date": str(min_date),
                "mean_score": round(mean_score, 4),
                "median_score": round(median_score, 4),
                "std_score": round(std_score, 4),
                "q25": round(q25, 4),
                "q50": round(q50, 4),
                "q75": round(q75, 4),
                "contrib": contrib_cols,
            }

            # 최대/최저 점수를 가진 행의 날짜
            max_row = self.df_valid[self.df_valid["score"] == max_score].iloc[0] if len(self.df_valid[self.df_valid["score"] == max_score]) > 0 else None
            min_row = self.df_valid[self.df_valid["score"] == min_score].iloc[0] if len(self.df_valid[self.df_valid["score"] == min_score]) > 0 else None

            # 최대/최저 이벤트명과 날짜 찾기
            events = self.load_events()
            max_event_name = ""
            max_event_date = max_row["date"].strftime("%Y-%m-%d") if max_row is not None else ""
            min_event_name = ""
            min_event_date = min_row["date"].strftime("%Y-%m-%d") if min_row is not None else ""

            # 최대/최저 날짜에 가장 가까운 이벤트 찾기
            if events:
                if max_row is not None:
                    max_dt = max_row["date"]
                    closest_event = min(events, key=lambda e: abs((pd.Timestamp(e['date']) - max_dt).days))
                    max_event_name = closest_event['label']

                if min_row is not None:
                    min_dt = min_row["date"]
                    closest_event = min(events, key=lambda e: abs((pd.Timestamp(e['date']) - min_dt).days))
                    min_event_name = closest_event['label']

            self.stats["max_event_name"] = max_event_name
            self.stats["max_event_date"] = max_event_date
            self.stats["min_event_name"] = min_event_name
            self.stats["min_event_date"] = min_event_date

            print(f"  ✓ 통계치 계산 완료")
            print(f"    - 현재 점수: {self.stats['current_score']:+.4f}")
            print(f"    - 백분위: {self.stats['current_pct']:.1f}%")
            print(f"    - P(평온강세): {self.stats['current_peace']:.3f}")
            if max_event_name:
                print(f"    - 최대 이벤트: {max_event_name} ({max_event_date})")
            if min_event_name:
                print(f"    - 최소 이벤트: {min_event_name} ({min_event_date})")

            return True
        except Exception as e:
            print(f"  ✗ 오류: {e}")
            return False

    def prepare_chart_data(self) -> dict:
        """
        Plotly 차트용 데이터 준비 (국면 라벨, 배경 shapes, 이벤트 포함)

        Returns:
            dict: 차트 데이터
        """
        # 국면 라벨 추가 (백분위 기준 — KPI 카드와 동일)
        # 비위기 구간 기준 백분위 계산
        scores_nc = self.df_valid.loc[~self.df_valid["crisis"], "score"]

        def get_regime_label(score):
            """
            백분위 기반 4단계 분류 (KPI 카드와 동일)
            - peacetime: 하위 33% (평온강세)
            - caution: 33~66% (위험주시)
            - rising_rates: 66~85% (고위험)
            - falling_rates: 상위 15% (극위기)
            """
            # 비위기 구간에서 이 점수의 백분위 계산
            percentile = (scores_nc < score).mean() * 100

            if percentile < 33:
                return "peacetime"
            elif percentile < 66:
                return "caution"
            elif percentile < 85:
                return "rising_rates"
            else:
                return "falling_rates"

        self.df_valid["regime_label"] = self.df_valid["score"].apply(get_regime_label)

        # 연속된 국면 기간을 찾기 (배경 shapes 생성용)
        regime_periods = []
        current_regime = None
        start_idx = 0

        for idx, (i, row) in enumerate(self.df_valid.iterrows()):
            regime = row["regime_label"]
            if regime != current_regime:
                if current_regime is not None:
                    regime_periods.append({
                        "regime": current_regime,
                        "start_date": str(self.df_valid.iloc[start_idx]["date"].date()),
                        "end_date": str(self.df_valid.iloc[idx - 1]["date"].date()),
                        "start_idx": start_idx,
                        "end_idx": idx - 1,
                    })
                current_regime = regime
                start_idx = idx

        if current_regime is not None:
            regime_periods.append({
                "regime": current_regime,
                "start_date": str(self.df_valid.iloc[start_idx]["date"].date()),
                "end_date": str(self.df_valid.iloc[-1]["date"].date()),
                "start_idx": start_idx,
                "end_idx": len(self.df_valid) - 1,
            })

        # S&P500 누적 수익률 계산 (시작점부터의 로그 수익률)
        sp500_cumulative_returns = None
        if "sp500" in self.df_valid.columns:
            sp500_data = self.df_valid[["sp500"]].copy()
            # 누적 로그 수익률 = log(현재/시작점) * 100
            first_price = sp500_data["sp500"].iloc[0]
            sp500_data["cumulative_returns"] = np.log(sp500_data["sp500"] / first_price) * 100
            sp500_cumulative_returns = sp500_data["cumulative_returns"].round(2).tolist()

        # 시계열 데이터
        chart_data = {
            "dates": self.df_valid["date"].astype(str).tolist(),
            "scores": self.df_valid["score"].round(4).tolist(),
            "crisis": self.df_valid["crisis"].astype(int).tolist(),
            "p_peace": self.df_valid["p_peace"].round(3).tolist(),
            "regime_label": self.df_valid["regime_label"].tolist(),
            "sp500_cumulative_returns": sp500_cumulative_returns,  # S&P500 누적 수익률 (%)
            "regime_periods": regime_periods,
        }
        return chart_data

    def load_events(self, events_path: str = "data/raw/events.csv") -> list[dict]:
        """
        이벤트 데이터 로드 (이탈점수 정보 포함)

        주간 데이터(W-FRI)에 맞춰 가장 가까운 금요일로 매칭합니다.

        Args:
            events_path: 이벤트 파일 경로

        Returns:
            list: 이벤트 리스트 (date, label, event_type, score, percentile, original_date)
        """
        try:
            events_df = pd.read_csv(events_path, parse_dates=["date"])

            # memo 또는 event_type 컬럼 사용
            label_col = "memo" if "memo" in events_df.columns else "event_type"

            events = []
            for _, row in events_df.iterrows():
                original_date = row["date"]

                # 가장 가까운 금요일(W-FRI)로 라운딩
                # 월요일(0)=4, ..., 금요일(4)=0으로 변환
                days_since_friday = (original_date.weekday() - 4) % 7
                nearest_friday = original_date - pd.Timedelta(days=days_since_friday)

                # 최종 데이터셋의 데이터 범위 확인
                if nearest_friday < self.df_valid["date"].min():
                    # 데이터 범위 이전: 스킵
                    continue

                event_date_str = str(nearest_friday.date())
                original_date_str = str(original_date.date())

                # 해당 날짜의 이탈점수 찾기
                matching_rows = self.df_valid[self.df_valid["date"].astype(str) == event_date_str]
                if not matching_rows.empty:
                    event_score = float(matching_rows.iloc[0]["score"])
                    regime = str(matching_rows.iloc[0].get("regime_label", ""))
                    # 백분위 계산 (비위기 기준)
                    nc_scores = self.df_valid.loc[~self.df_valid["crisis"], "score"]
                    percentile = float((nc_scores < event_score).mean() * 100)
                else:
                    event_score = None
                    percentile = None
                    regime = None

                events.append({
                    "date": event_date_str,  # 매칭된 금요일
                    "original_date": original_date_str,  # 원래 이벤트 날짜
                    "label": str(row[label_col]) if pd.notna(row[label_col]) else "이벤트",
                    "event_type": str(row.get("event_type", "")) if "event_type" in events_df.columns else "",
                    "score": event_score,
                    "percentile": percentile,
                    "regime": regime
                })

            print(f"  ✓ 이벤트 로드: {len(events)}개 (금요일 매칭, 점수 정보 포함)")
            return events
        except Exception as e:
            print(f"  ⚠ 이벤트 로드 실패: {e}")
            return []

    def run(self) -> bool:
        """
        전체 파이프라인 실행

        Returns:
            bool: 성공 여부
        """
        print("\n" + "=" * 65)
        print("■ 국면 이탈 점수 대시보드 — 데이터 준비")
        print("=" * 65 + "\n")

        steps = [
            self.load_data,
            self.calculate_features,
            self.fit_clusters,
            self.calculate_scores,
            self.calculate_stats,
        ]

        for step in steps:
            if not step():
                print("\n✗ 파이프라인 중단")
                return False

        print("\n" + "=" * 65)
        print("✓ 데이터 준비 완료!")
        print("=" * 65 + "\n")
        return True


    def generate_html(self, output_path: str = "analysis/dashboard/index.html") -> bool:
        """
        정적 HTML 대시보드 생성

        Args:
            output_path: 출력 경로

        Returns:
            bool: 성공 여부
        """
        print("\n" + "=" * 65)
        print("■ HTML 대시보드 생성 중...")
        print("=" * 65 + "\n")

        try:
            # 차트 데이터 준비
            chart_data = self.prepare_chart_data()
            events = self.load_events()

            # 이벤트 데이터
            events_json = json.dumps(events, ensure_ascii=False)

            # 통계 데이터
            stats_json = json.dumps(self.stats, ensure_ascii=False, default=str)

            # 현재 점수에 따른 국면 배지
            score = self.stats["current_score"]
            pct = self.stats["current_pct"]

            if pct < 33:
                regime_badge = "🟢 평온강세"
                regime_color = self.config["regime_colors"]["peacetime"]
            elif pct < 66:
                regime_badge = "🟡 위험주의"
                regime_color = self.config["regime_colors"]["caution"]
            elif pct < 85:
                regime_badge = "🔴 금리상승기"
                regime_color = self.config["regime_colors"]["rising_rates"]
            else:
                regime_badge = "🔵 금리하락기"
                regime_color = self.config["regime_colors"]["falling_rates"]

            # HTML 템플릿
            html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>국면 이탈 점수 대시보드</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 24px;
            color: #1f2937;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 20px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 24px;
            margin-bottom: 4px;
        }}

        .header p {{
            font-size: 12px;
            opacity: 0.9;
            margin-bottom: 2px;
        }}

        .content {{
            padding: 20px 20px;
        }}

        /* ── 요약 섹션 ────────────────────────────────────────────────── */
        .summary-section {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }}

        .regime-badge {{
            grid-column: 1 / -1;
            padding: 16px 24px;
            background: {regime_color}22;
            border-left: 4px solid {regime_color};
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .regime-badge-text {{
            font-size: 24px;
            font-weight: 700;
            color: {regime_color};
        }}

        .regime-badge-sub {{
            font-size: 13px;
            color: #6b7280;
        }}

        .kpi-card {{
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}

        .kpi-label {{
            font-size: 12px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 24px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 4px;
        }}

        .kpi-sub {{
            font-size: 12px;
            color: #9ca3af;
        }}

        /* ── 차트 섹션 ────────────────────────────────────────────────── */
        .chart-section {{
            margin-bottom: 16px;
        }}

        .chart-title {{
            font-size: 15px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 10px;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 8px;
        }}

        .chart-container {{
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 8px;
            margin-bottom: 12px;
        }}

        .chart-container > div {{
            width: 100%;
            height: 320px;
        }}

        /* ── 토글 버튼 ────────────────────────────────────────────────── */
        .toggle-buttons {{
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
        }}

        button {{
            padding: 8px 16px;
            border: 1px solid #d1d5db;
            background: white;
            color: #374151;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s;
        }}

        button:hover {{
            background: #f3f4f6;
        }}

        button.active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        /* ── 범례 ────────────────────────────────────────────────────── */
        .legend {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
            padding: 12px;
            background: #f9fafb;
            border-radius: 6px;
            border: 1px solid #e5e7eb;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}

        /* ── 하단 고지사항 ────────────────────────────────────────────── */
        .disclaimer {{
            background: #fef3c7;
            border: 1px solid #fcd34d;
            border-radius: 8px;
            padding: 16px;
            margin-top: 32px;
            font-size: 13px;
            color: #78350f;
            line-height: 1.6;
        }}

        .disclaimer strong {{
            display: block;
            margin-bottom: 8px;
            color: #b45309;
        }}

        /* ── 반응형 ────────────────────────────────────────────────────── */
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 24px;
            }}

            .content {{
                padding: 16px;
            }}

            .summary-section {{
                grid-template-columns: 1fr;
            }}

            .kpi-value {{
                font-size: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <div class="header">
            <h1>📊 시장 국면 이탈 점수 대시보드</h1>
            <p>평온강세로부터의 이탈 정도를 측정하는 동시적 지표(coincident indicator)</p>
            <p style="font-size: 12px; margin-top: 8px;">마지막 업데이트: {self.stats['current_date']} | 데이터 기간: {self.df_valid['date'].min().date()} ~ {self.stats['current_date']}</p>
        </div>

        <div class="content">
            <!-- 요약 섹션 -->
            <div class="summary-section">
                <div class="regime-badge">
                    <div style="font-size: 32px;">{regime_badge.split()[0]}</div>
                    <div>
                        <div class="regime-badge-text">{regime_badge}</div>
                        <div class="regime-badge-sub">현재 시장 국면</div>
                    </div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-label">현재 이탈 점수</div>
                    <div class="kpi-value">{self.stats['current_score']:+.3f}</div>
                    <div class="kpi-sub">음수 = 평온 | 양수 = 이탈</div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-label">백분위 (비위기)</div>
                    <div class="kpi-value">{self.stats['current_pct']:.1f}%</div>
                    <div class="kpi-sub">낮을수록 평온함<br><em style="font-size: 11px; color: #6b7280;">(역사 기준 상대 위치)</em></div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-label">P(평온강세)</div>
                    <div class="kpi-value">{self.stats['current_peace']:.1%}</div>
                    <div class="kpi-sub">GMM 소속확률<br><em style="font-size: 11px; color: #6b7280;">(1에 가까울수록 평온강세)</em></div>
                </div>

                <div class="kpi-card">
                    <div class="kpi-label">역대 최대</div>
                    <div class="kpi-value">{self.stats['max_score']:+.3f}</div>
                    <div class="kpi-sub">{self.stats['max_date']}<br><em style="font-size: 11px; color: #6b7280;">(금융위기 최악)</em></div>
                </div>
            </div>

            <!-- 점수 산출 방식 설명 -->
            <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 14px; margin-top: 16px;">
                <h3 style="margin: 0 0 10px 0; font-size: 13px; font-weight: 600; color: #0369a1;">🔍 점수 산출 방식</h3>
                <div style="font-size: 12px; color: #164e63; line-height: 1.6;">
                    <strong>이탈점수 = {self.stats['current_score']:+.4f}</strong><br>
                    ↓ 이 점수는 5개 지표(DXY, 10Y금리, NFCI, HY스프레드, VIX)를 표준화한 후,<br>
                    &nbsp;&nbsp;평온강세 국면의 중심에서 얼마나 벗어났는지를 측정합니다.<br>
                    <br>
                    <strong>백분위 = {self.stats['current_pct']:.1f}%</strong><br>
                    ↓ 역사 기준으로 현재가 하위 {self.stats['current_pct']:.1f}%에 위치함을 의미합니다.<br>
                    &nbsp;&nbsp;(낮을수록 평온함, 높을수록 극단적)<br>
                    <br>
                    <strong>국면 판정: 🟢 평온강세</strong><br>
                    ↓ 백분위 {self.stats['current_pct']:.1f}% &lt; 33% 기준으로 평온강세로 분류됨<br>
                    &nbsp;&nbsp;(현재는 역사상 가장 평온한 시기 중 하나)
                </div>
            </div>

            <!-- 레짐 설명 -->
            <div style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; margin-bottom: 16px;">
                <h3 style="margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: #111827;">📊 시장 국면 해석</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
                    <div style="padding: 10px; background: #dbeafe; border-left: 3px solid #2563eb; border-radius: 4px;">
                        <div style="font-weight: 600; color: #1e40af; margin-bottom: 4px;">🟢 평온강세 (하위 33%)</div>
                        <div style="font-size: 12px; color: #1e3a8a; line-height: 1.4;">VIX 낮음, 신용스프레드 좁음, 금융여건 완화. 투자 심리 긍정적.</div>
                    </div>
                    <div style="padding: 10px; background: #fed7aa; border-left: 3px solid #f59e0b; border-radius: 4px;">
                        <div style="font-weight: 600; color: #92400e; margin-bottom: 4px;">🟡 위험주시 (33~66%)</div>
                        <div style="font-size: 12px; color: #78350f; line-height: 1.4;">위험신호 증가, VIX 상승. 시장 불안 시작. 주의 필요.</div>
                    </div>
                    <div style="padding: 10px; background: #fecaca; border-left: 3px solid #ef4444; border-radius: 4px;">
                        <div style="font-weight: 600; color: #991b1b; margin-bottom: 4px;">🔴 고위험 (66~85%)</div>
                        <div style="font-size: 12px; color: #7f1d1d; line-height: 1.4;">신용스프레드 확대, VIX 높음. 시장 스트레스 심화.</div>
                    </div>
                    <div style="padding: 10px; background: #bfdbfe; border-left: 3px solid #1e3a8a; border-radius: 4px;">
                        <div style="font-weight: 600; color: #1e3a8a; margin-bottom: 4px;">🔵 극위기 (상위 15%)</div>
                        <div style="font-size: 12px; color: #1e3a8a; line-height: 1.4;">금융위기 수준. NFCI 급증, 극도의 시장 공포. 2008, 2020년 같은 시기.</div>
                    </div>
                </div>
            </div>

            <!-- 시계열 차트 -->
            <div class="chart-section">
                <h2 class="chart-title">📈 이탈 점수 시계열</h2>
                <div class="toggle-buttons">
                    <button class="active" onclick="toggleTimeRange('all')">전 기간</button>
                    <button onclick="toggleTimeRange('recent')">최근 2년</button>
                </div>
                <div class="chart-container">
                    <div id="timeseriesChart"></div>
                </div>
            </div>

            <!-- 위치 바 -->
            <div class="chart-section">
                <h2 class="chart-title">📍 역사적 위치</h2>
                <div class="chart-container">
                    <div id="positionBarChart"></div>
                </div>
            </div>

            <!-- 기여 분해 -->
            <div class="chart-section">
                <h2 class="chart-title">🔍 현재 지표별 기여도</h2>
                <div class="chart-container">
                    <div id="contributionChart"></div>
                </div>
            </div>

            <!-- 이벤트별 이탈점수 -->
            <div class="chart-section">
                <h2 class="chart-title">📍 역사적 이벤트별 이탈점수</h2>
                <div style="overflow-x: auto; background: #f9fafb; border-radius: 6px; padding: 12px; border: 1px solid #e5e7eb;">
                    <table id="eventsTable" style="width: 100%; border-collapse: collapse; font-size: 12px;">
                        <thead>
                            <tr style="background: #f3f4f6; border-bottom: 2px solid #e5e7eb;">
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #374151;">날짜</th>
                                <th style="padding: 10px; text-align: left; font-weight: 600; color: #374151;">이벤트</th>
                                <th style="padding: 10px; text-align: right; font-weight: 600; color: #374151;">이탈점수</th>
                                <th style="padding: 10px; text-align: right; font-weight: 600; color: #374151;">백분위</th>
                                <th style="padding: 10px; text-align: center; font-weight: 600; color: #374151;">국면</th>
                            </tr>
                        </thead>
                        <tbody id="eventsTableBody">
                        </tbody>
                    </table>
                </div>
                <p style="margin-top: 12px; font-size: 12px; color: #6b7280; line-height: 1.5;">
                    💡 <strong>해석:</strong> 각 이벤트 발생 시점의 이탈점수와 백분위를 표시합니다.
                    현재 백분위(28.8%)와 비교하여 과거 이벤트들이 얼마나 큰 영향을 미쳤는지 가늠할 수 있습니다.
                </p>
            </div>

            <!-- 고지사항 -->
            <div class="disclaimer">
                <strong>⚠️ 중요 고지사항</strong>
                ✓ 본 대시보드는 <strong>현재 시점의 시장 국면을 측정</strong>하는 동시적 지표입니다.<br>
                ✓ 미래 수익률을 예측하지 않습니다.<br>
                ✓ 투자 권유 또는 조언이 아닙니다.<br>
                ✓ 과거 데이터 기준 센트로이드 및 표준화 파라미터는 고정되어 있습니다.
            </div>
        </div>
    </div>

    <script>
        // 데이터 임베드
        const stats = {stats_json};
        const chartData = {json.dumps(chart_data, ensure_ascii=False, default=str)};
        const eventsData = {events_json};

        // 국면 색상 맵
        const regimeColors = {{
            peacetime: {{ color: '#2563eb', opacity: 0.15, name: '평온강세' }},
            caution: {{ color: '#f59e0b', opacity: 0.15, name: '위험주의' }},
            rising_rates: {{ color: '#ef4444', opacity: 0.15, name: '금리상승기' }},
            falling_rates: {{ color: '#10b981', opacity: 0.15, name: '금리하락기' }}
        }};

        // 시계열 차트 렌더 (이탈점수 + S&P500 누적 수익률 + 이벤트 호버)
        function renderTimeseries(range = 'all') {{
            let dates = chartData.dates;
            let scores = chartData.scores;
            let sp500CumulativeReturns = chartData.sp500_cumulative_returns;
            let pPeace = chartData.p_peace;
            let regimeLabelsList = chartData.regime_label;
            let eventsFiltered = eventsData.slice(); // 이벤트 복사

            // 범위 필터
            let startIdx = 0;
            let endIdx = dates.length - 1;
            let rangeStartDate = dates[0];
            let rangeEndDate = dates[dates.length - 1];

            if (range === 'recent') {{
                const n = dates.length;
                startIdx = Math.max(0, n - 104); // 약 2년 = 104주
                rangeStartDate = dates[startIdx];
                rangeEndDate = dates[endIdx];

                // 범위 내 이벤트만 필터링
                eventsFiltered = eventsData.filter(e => e.date >= rangeStartDate && e.date <= rangeEndDate);
            }}

            dates = dates.slice(startIdx, endIdx + 1);
            scores = scores.slice(startIdx, endIdx + 1);
            sp500CumulativeReturns = sp500CumulativeReturns.slice(startIdx, endIdx + 1);
            pPeace = pPeace.slice(startIdx, endIdx + 1);
            regimeLabelsList = regimeLabelsList.slice(startIdx, endIdx + 1);

            // 이벤트 맵 (호버시 빠른 조회용)
            const eventMap = {{}};
            eventsFiltered.forEach(event => {{
                if (!eventMap[event.date]) {{
                    eventMap[event.date] = [];
                }}
                eventMap[event.date].push(event.label);
            }});

            // 레짐 라벨 한글화
            const regimeNameMap = {{
                'peacetime': '🟦 평온강세',
                'caution': '🟨 위험주의',
                'rising_rates': '🔴 금리상승기',
                'falling_rates': '🟩 금리하락기'
            }};

            // 호버용 통합 정보 (레짐 + 이벤트)
            const hoverInfo = dates.map((d, idx) => {{
                const regime = regimeNameMap[regimeLabelsList[idx]] || regimeLabelsList[idx];
                const events = eventMap[d] ? '<br>📍 ' + eventMap[d].join('<br>📍 ') : '';
                return regime + events;
            }});

            // 메인 선 그래프: 이탈 점수
            const traceScore = {{
                x: dates,
                y: scores,
                mode: 'lines',
                name: '이탈 점수',
                line: {{ color: '#667eea', width: 2 }},
                yaxis: 'y',
                customdata: hoverInfo,
                hovertemplate: '<b>%{{x}}</b><br>이탈점수: %{{y:+.4f}}<br>%{{customdata}}<extra></extra>',
                fill: 'tozeroy',
                fillcolor: 'rgba(102, 126, 234, 0.1)'
            }};

            // 이중축: S&P500 누적 수익률
            const traceSP500 = {{
                x: dates,
                y: sp500CumulativeReturns,
                mode: 'lines',
                name: 'S&P500 누적 수익률 (%)',
                line: {{ color: '#10b981', width: 1.5, dash: 'solid' }},
                yaxis: 'y2',
                customdata: hoverInfo,
                hovertemplate: '<b>%{{x}}</b><br>누적 수익률: %{{y:+.2f}}%<br>%{{customdata}}<extra></extra>',
                opacity: 0.7
            }};

            // 배경 shapes (국면별 색칠)
            const shapes = [];
            if (chartData.regime_periods) {{
                chartData.regime_periods.forEach(period => {{
                    if (period.start_date >= dates[0] && period.end_date <= dates[dates.length - 1]) {{
                        const color = regimeColors[period.regime];
                        shapes.push({{
                            type: 'rect',
                            xref: 'x',
                            yref: 'paper',
                            x0: period.start_date,
                            x1: period.end_date,
                            y0: 0,
                            y1: 1,
                            fillcolor: color.color,
                            opacity: color.opacity,
                            line: {{ width: 0 }}
                        }});
                    }}
                }});
            }}

            // 이벤트 마커 (점 + hover로 라벨 표시)
            const annotations = [];
            const eventMarkers = []; // 이벤트 점들
            const eventDates = new Set(eventsData.map(e => e.date));

            eventsData.forEach(event => {{
                if (dates.includes(event.date)) {{
                    const idx = dates.indexOf(event.date);
                    if (idx >= 0) {{
                        eventMarkers.push({{
                            x: event.date,
                            y: scores[idx],
                            label: event.label
                        }});

                        // 수직선 (점선, 얇게)
                        shapes.push({{
                            type: 'line',
                            xref: 'x',
                            yref: 'paper',
                            x0: event.date,
                            x1: event.date,
                            y0: 0,
                            y1: 1,
                            line: {{
                                color: '#d1d5db',
                                width: 1,
                                dash: 'dash'
                            }}
                        }});
                    }}
                }}
            }});

            // 현재 시점 강조
            const currentIdx = dates.length - 1;
            const currentDate = dates[currentIdx];
            annotations.push({{
                x: currentDate,
                y: scores[currentIdx],
                text: '현재',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#667eea',
                ax: 0,
                ay: 30,
                bgcolor: '#667eea',
                bordercolor: '#667eea',
                font: {{ size: 12, color: '#fff', weight: 'bold' }},
                xanchor: 'center'
            }});

            // 역대 최대 강조
            const maxIdx = scores.indexOf(Math.max(...scores));
            if (maxIdx !== currentIdx) {{
                annotations.push({{
                    x: dates[maxIdx],
                    y: scores[maxIdx],
                    text: '역대최대',
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 1,
                    arrowcolor: '#ef4444',
                    ax: 0,
                    ay: -40,
                    bgcolor: '#ef4444',
                    bordercolor: '#ef4444',
                    font: {{ size: 11, color: '#fff' }},
                    xanchor: 'center'
                }});
            }}

            const layout = {{
                title: '',
                xaxis: {{
                    title: '날짜',
                    showgrid: true,
                    gridwidth: 1,
                    gridcolor: '#f3f4f6'
                }},
                yaxis: {{
                    title: '이탈 점수',
                    titlefont: {{ color: '#667eea' }},
                    tickfont: {{ color: '#667eea' }},
                    showgrid: true,
                    gridwidth: 1,
                    gridcolor: '#f3f4f6',
                    zeroline: true,
                    zerolinecolor: '#e5e7eb',
                    zerolinewidth: 2
                }},
                yaxis2: {{
                    title: 'S&P500 13주 수익률 (%)',
                    titlefont: {{ color: '#10b981' }},
                    tickfont: {{ color: '#10b981' }},
                    overlaying: 'y',
                    side: 'right'
                }},
                shapes: shapes,
                annotations: annotations,
                hovermode: 'x unified',
                margin: {{ l: 60, r: 80, t: 80, b: 80 }},
                height: 380,
                showlegend: true,
                plot_bgcolor: '#fafafa',
                paper_bgcolor: 'white',
                legend: {{
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.05,
                    xanchor: 'left',
                    x: 0,
                    bgcolor: 'rgba(255, 255, 255, 0.8)',
                    bordercolor: '#e5e7eb',
                    borderwidth: 1
                }}
            }};

            // 레짐 범례 (보이지 않는 trace)
            const tracePeacetime = {{
                x: [null], y: [null],
                mode: 'markers',
                marker: {{ size: 10, color: '#2563eb' }},
                name: '평온강세',
                showlegend: true,
                hoverinfo: 'skip'
            }};
            const traceCaution = {{
                x: [null], y: [null],
                mode: 'markers',
                marker: {{ size: 10, color: '#f59e0b' }},
                name: '위험주의',
                showlegend: true,
                hoverinfo: 'skip'
            }};
            const traceRising = {{
                x: [null], y: [null],
                mode: 'markers',
                marker: {{ size: 10, color: '#ef4444' }},
                name: '금리상승기',
                showlegend: true,
                hoverinfo: 'skip'
            }};
            const traceFalling = {{
                x: [null], y: [null],
                mode: 'markers',
                marker: {{ size: 10, color: '#10b981' }},
                name: '금리하락기',
                showlegend: true,
                hoverinfo: 'skip'
            }};

            // 이벤트 마커 trace (점으로 표시)
            const traceEventMarkers = {{
                x: eventMarkers.map(e => e.x),
                y: eventMarkers.map(e => e.y),
                mode: 'markers',
                name: '이벤트',
                marker: {{
                    size: 8,
                    color: '#ef4444',
                    symbol: 'diamond',
                    opacity: 0.8
                }},
                yaxis: 'y',
                customdata: eventMarkers.map(e => e.label),
                hovertemplate: '<b>📍 %{{customdata}}</b><br>점수: %{{y:+.4f}}<extra></extra>'
            }};

            Plotly.newPlot('timeseriesChart', [traceScore, traceSP500, traceEventMarkers, tracePeacetime, traceCaution, traceRising, traceFalling], layout, {{ responsive: true }});
        }}

        // 위치 바 렌더 (과거 극값 + 사분위수 + 이벤트 마커)
        function renderPositionBar() {{
            // 배경 막대들
            const backgroundTraces = [
                {{
                    x: [0, 100],
                    y: ['', ''],
                    type: 'bar',
                    orientation: 'h',
                    marker: {{ color: '#e5e7eb' }},
                    hoverinfo: 'skip',
                    showlegend: false
                }}
            ];

            // 사분위수 선
            const shapes = [
                {{
                    type: 'line',
                    xref: 'x',
                    yref: 'paper',
                    x0: stats.q25 * 100,
                    x1: stats.q25 * 100,
                    y0: 0,
                    y1: 1,
                    line: {{ color: '#9ca3af', width: 1, dash: 'dot' }}
                }},
                {{
                    type: 'line',
                    xref: 'x',
                    yref: 'paper',
                    x0: stats.q50 * 100,
                    x1: stats.q50 * 100,
                    y0: 0,
                    y1: 1,
                    line: {{ color: '#6b7280', width: 2 }}
                }},
                {{
                    type: 'line',
                    xref: 'x',
                    yref: 'paper',
                    x0: stats.q75 * 100,
                    x1: stats.q75 * 100,
                    y0: 0,
                    y1: 1,
                    line: {{ color: '#9ca3af', width: 1, dash: 'dot' }}
                }}
            ];

            // 이벤트 마커 (수직선)
            eventsData.forEach(event => {{
                if (event.percentile !== null) {{
                    shapes.push({{
                        type: 'line',
                        xref: 'x',
                        yref: 'paper',
                        x0: event.percentile,
                        x1: event.percentile,
                        y0: 0,
                        y1: 1,
                        line: {{
                            color: '#d1d5db',
                            width: 1,
                            dash: 'dash'
                        }},
                        opacity: 0.5
                    }});
                }}
            }});

            // annotations: 극값 및 사분위수 라벨 (상단 배치)
            // Python에서 계산된 이벤트 정보 사용
            const minEventLabel = stats.min_event_name
                ? stats.min_event_name + '<br><small style="color: #059669;">(' + stats.min_event_date + ')</small>'
                : '';
            const maxEventLabel = stats.max_event_name
                ? stats.max_event_name + '<br><small style="color: #991b1b;">(' + stats.max_event_date + ')</small>'
                : '';

            const annotations = [
                {{
                    x: 0,
                    y: 1.15,
                    text: '최저<br><b>' + stats.min_score.toFixed(3) + '</b>' + (minEventLabel ? '<br>' + minEventLabel : ''),
                    showarrow: false,
                    font: {{ size: 11, color: '#10b981' }},
                    xanchor: 'left'
                }},
                {{
                    x: 100,
                    y: 1.15,
                    text: '최대<br><b>' + stats.max_score.toFixed(3) + '</b>' + (maxEventLabel ? '<br>' + maxEventLabel : ''),
                    showarrow: false,
                    font: {{ size: 11, color: '#ef4444' }},
                    xanchor: 'right'
                }},
                {{
                    x: stats.current_pct,
                    y: -0.35,
                    text: '◆ 현재<br>' + stats.current_score.toFixed(3) + '<br>' + stats.current_pct.toFixed(1) + '%',
                    showarrow: true,
                    arrowhead: 2,
                    arrowwidth: 2,
                    arrowcolor: '#667eea',
                    bgcolor: '#667eea',
                    bordercolor: '#667eea',
                    font: {{ size: 12, color: '#fff', weight: 'bold' }},
                    xanchor: 'center'
                }}
            ];

            // 이벤트 점 (hover로 라벨 표시, 라벨 텍스트 제거)
            // — annotation 제거, customdata에 라벨만 저장

            // 이벤트 마커 trace
            const eventX = [];
            const eventLabels = [];
            eventsData.forEach(event => {{
                if (event.percentile !== null) {{
                    eventX.push(event.percentile);
                    eventLabels.push(event.label);
                }}
            }});

            const traceEventPoints = {{
                x: eventX,
                y: Array(eventX.length).fill(''),
                mode: 'markers',
                marker: {{
                    size: 10,
                    color: '#ef4444',
                    symbol: 'diamond',
                    opacity: 0.7
                }},
                customdata: eventLabels,
                hovertemplate: '<b>📍 %{{customdata}}</b><br>백분위: %{{x:.1f}}%<extra></extra>',
                showlegend: false
            }};

            const layout = {{
                xaxis: {{ title: '백분위 (%) — 낮을수록 평온함', range: [-5, 105] }},
                yaxis: {{ title: '', showticklabels: false }},
                shapes: shapes,
                annotations: annotations,
                margin: {{ l: 50, r: 30, t: 40, b: 80 }},
                height: 250,
                showlegend: false,
                plot_bgcolor: '#f9fafb',
                hovermode: 'closest'
            }};

            Plotly.newPlot('positionBarChart', [backgroundTraces[0], traceEventPoints], layout, {{ responsive: true }});
        }}

        // 기여 분해 렌더 (정렬 + 범례)
        function renderContribution() {{
            const contrib = stats.contrib;
            const features = ['dxy_chg', 'y10_chg', 'financial_conditions', 'credit_spread', 'vix'];
            const featureNames = {{
                dxy_chg: 'DXY 13주 변화율',
                y10_chg: '10Y 금리 변화(pp)',
                financial_conditions: 'NFCI 금융여건',
                credit_spread: 'HY 신용스프레드',
                vix: 'VIX'
            }};

            // 기여도 정렬 (절댓값 기준)
            const items = features.map(f => ({{
                feature: f,
                name: featureNames[f],
                value: contrib[f],
                abs: Math.abs(contrib[f])
            }}));
            items.sort((a, b) => b.abs - a.abs);

            const names = items.map(x => x.name);
            const values = items.map(x => x.value);
            const colors = items.map(x => x.value > 0 ? '#ef4444' : '#10b981'); // 이탈(빨강) vs 완화(초록)

            const trace = {{
                x: values,
                y: names,
                type: 'bar',
                orientation: 'h',
                marker: {{ color: colors }},
                text: values.map(v => v.toFixed(4)),
                textposition: 'auto',
                hovertemplate: '<b>%{{y}}</b><br>기여도: %{{x:+.4f}}<extra></extra>'
            }};

            // Zero line
            const shapes = [{{
                type: 'line',
                xref: 'x',
                yref: 'paper',
                x0: 0,
                x1: 0,
                y0: 0,
                y1: 1,
                line: {{ color: '#d1d5db', width: 2 }}
            }}];

            // 범례 text
            const totalAbsContrib = values.reduce((sum, v) => sum + Math.abs(v), 0);

            const layout = {{
                xaxis: {{ title: '기여도' }},
                yaxis: {{ title: '' }},
                shapes: shapes,
                margin: {{ l: 180, r: 100, t: 40, b: 50 }},
                height: 320,
                showlegend: false,
                plot_bgcolor: '#f9fafb',
                annotations: [
                    {{
                        x: Math.max(...values) * 1.1,
                        y: -0.15,
                        text: '<b>범례:</b> 🔴 이탈 | 🟢 완화 | 합계: ' + stats.current_score.toFixed(3),
                        showarrow: false,
                        xref: 'x',
                        yref: 'paper',
                        xanchor: 'left',
                        font: {{ size: 12, color: '#374151' }}
                    }}
                ]
            }};

            Plotly.newPlot('contributionChart', [trace], layout, {{ responsive: true }});
        }}

        // 토글 함수
        function toggleTimeRange(range) {{
            document.querySelectorAll('.toggle-buttons button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');
            renderTimeseries(range);
        }}

        // 이벤트 테이블 채우기
        function fillEventsTable() {{
            const tbody = document.getElementById('eventsTableBody');
            tbody.innerHTML = '';

            eventsData.forEach(event => {{
                const row = document.createElement('tr');
                row.style.borderBottom = '1px solid #e5e7eb';

                // 국면 배지 색상
                const regimeColors = {{
                    'peacetime': {{ name: '평온강세', bg: '#dbeafe', color: '#1e40af' }},
                    'caution': {{ name: '위험주의', bg: '#fed7aa', color: '#92400e' }},
                    'rising_rates': {{ name: '고위험', bg: '#fecaca', color: '#991b1b' }},
                    'falling_rates': {{ name: '극위기', bg: '#bfdbfe', color: '#1e3a8a' }}
                }};

                // 해당 날짜의 국면 찾기
                const dateIdx = chartData.dates.indexOf(event.date);
                let regimeLabel = '—';
                let regimeBg = '#f3f4f6';
                let regimeColor = '#6b7280';

                if (dateIdx >= 0) {{
                    const regime = chartData.regime_label[dateIdx];
                    const regimeInfo = regimeColors[regime] || {{ name: regime, bg: '#f3f4f6', color: '#6b7280' }};
                    regimeLabel = regimeInfo.name;
                    regimeBg = regimeInfo.bg;
                    regimeColor = regimeInfo.color;
                }}

                const scoreDisplay = event.score !== null ? event.score.toFixed(3) : '—';
                const percentileDisplay = event.percentile !== null ? event.percentile.toFixed(1) + '%' : '—';

                // 원래 날짜와 매칭된 날짜가 다른 경우 표시
                const dateDisplay = event.original_date && event.original_date !== event.date
                    ? event.original_date + '<br><small style="color: #9ca3af;">(→ ' + event.date + ')</small>'
                    : event.date;

                row.innerHTML = `
                    <td style="padding: 10px; color: #374151; font-size: 12px;">${{dateDisplay}}</td>
                    <td style="padding: 10px; color: #374151;">${{event.label}}</td>
                    <td style="padding: 10px; text-align: right; color: #374151; font-weight: 600;">${{scoreDisplay}}</td>
                    <td style="padding: 10px; text-align: right; color: #374151; font-weight: 600;">${{percentileDisplay}}</td>
                    <td style="padding: 10px; text-align: center;">
                        <span style="background: ${{regimeBg}}; color: ${{regimeColor}}; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 11px;">
                            ${{regimeLabel}}
                        </span>
                    </td>
                `;

                tbody.appendChild(row);
            }});
        }}

        // 초기 렌더
        document.addEventListener('DOMContentLoaded', function() {{
            renderTimeseries('all');
            renderPositionBar();
            renderContribution();
            fillEventsTable();
        }});
    </script>
</body>
</html>
"""

            # 디렉토리 생성
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # 파일 저장
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"✓ HTML 생성 완료: {output_path}")
            return True

        except Exception as e:
            print(f"✗ 오류: {e}")
            import traceback
            traceback.print_exc()
            return False


def generate_deviation_dashboard_html(
    csv_path: str = "data/processed/final_dataset.csv",
    output_path: str = "analysis/dashboard/index.html"
) -> tuple[bool, str]:
    """
    국면 이탈 점수 대시보드 생성 (Streamlit에서 호출용)

    Args:
        csv_path: 최종 데이터셋 경로
        output_path: 출력 HTML 경로

    Returns:
        tuple: (성공 여부, 메시지)
    """
    try:
        dashboard = DeviationDashboard(csv_path=csv_path)

        if not dashboard.run():
            return False, "데이터 준비 실패"

        if not dashboard.generate_html(output_path=output_path):
            return False, "HTML 생성 실패"

        return True, f"✓ 대시보드 생성 완료\n현재 점수: {dashboard.stats['current_score']:+.4f} | 백분위: {dashboard.stats['current_pct']:.1f}%"

    except Exception as e:
        import traceback
        return False, f"오류: {e}\n{traceback.format_exc()}"


# 테스트용 메인 함수
if __name__ == "__main__":
    dashboard = DeviationDashboard()
    if dashboard.run():
        print(f"현재 통계: {dashboard.stats}")
        chart_data = dashboard.prepare_chart_data()
        print(f"차트 데이터 행 수: {len(chart_data['dates'])}")

        # HTML 생성
        dashboard.generate_html()
