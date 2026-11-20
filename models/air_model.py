import numpy as np
import xgboost as xgb

class AirXGBoostPipeline:
    def __init__(self):
        
        np.random.seed(42)
        n_samples = 400

       
        X = np.random.uniform(low=[10.0, 400.0, 5.0, 20.0, 1.0, 0.1],
                              high=[80.0, 1800.0, 42.0, 95.0, 25.0, 1.0],
                              size=(n_samples, 6))

       
        y_no2 = X[:, 0] * 0.55 + (X[:, 2] * 0.2) - (X[:, 4] * 0.35) + np.random.normal(0, 1.5, n_samples)
        y_so2 = (X[:, 0] * 0.18) + (X[:, 3] * 0.05) + np.random.normal(0, 0.8, n_samples)
        y_co = (X[:, 0] * 7.5) + (X[:, 1] * 0.12) + np.random.normal(0, 15.0, n_samples)
        y_aod = (X[:, 0] * 0.007) + (X[:, 3] * 0.002) + np.random.normal(0, 0.03, n_samples)
        y_o3 = 30.0 + (X[:, 2] * 0.8) + (X[:, 5] * 20.0) + np.random.normal(0, 2.0, n_samples)

        self.model_no2 = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1)
        self.model_no2.fit(X, y_no2)

        self.model_so2 = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1)
        self.model_so2.fit(X, y_so2)

        self.model_co = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1)
        self.model_co.fit(X, y_co)

        self.model_aod = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1)
        self.model_aod.fit(X, y_aod)

        self.model_o3 = xgb.XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1)
        self.model_o3.fit(X, y_o3)

    def predict(self, raw_air: dict, spatial_seed: float) -> dict:
        base_col = float(raw_air.get("no2", 25.0))
        temp = 24.0 + (spatial_seed % 15.0)
        humidity = 50.0 + (spatial_seed % 35.0)
        wind = 5.0 + (spatial_seed % 12.0)
        pblh = 850.0 + (spatial_seed % 450.0)
        solar_proxy = 0.65

      
        features = np.array([[base_col, pblh, temp, humidity, wind, solar_proxy]])

       
        pred_no2 = round(float(np.clip(self.model_no2.predict(features)[0], 4.0, 180.0)), 1)
        pred_so2 = round(float(np.clip(self.model_so2.predict(features)[0], 1.5, 65.0)), 1)
        pred_co = round(float(np.clip(self.model_co.predict(features)[0], 150.0, 1400.0)), 1)
        pred_aod = round(float(np.clip(self.model_aod.predict(features)[0], 0.05, 1.80)), 2)
        pred_o3 = round(float(np.clip(self.model_o3.predict(features)[0], 15.0, 140.0)), 1)

        if pred_no2 > 35.0 or pred_aod > 0.45:
            verdict = "Severe Atmospheric Pollution with Heavy NO₂ & Particle Smoke"
            verdict_status = "Critical"
        elif pred_no2 > 20.0 or pred_aod > 0.28:
            verdict = "Moderate Atmospheric Pollution with Noticeable Haze"
            verdict_status = "Moderate"
        else:
            verdict = "Good Air Quality with Nominal Gas Concentrations"
            verdict_status = "Clean"

        metrics = [
            {"name": "NO₂ (Nitrogen Dioxide)", "value": f"{pred_no2} µg/m³"},
            {"name": "SO₂ (Sulphur Dioxide)", "value": f"{pred_so2} µg/m³"},
            {"name": "CO (Carbon Monoxide)", "value": f"{pred_co} µg/m³"},
            {"name": "AOD (Aerosol Optical Depth)", "value": f"{pred_aod}"},
            {"name": "O₃ (Ozone)", "value": f"{pred_o3} µg/m³"}
        ]

        return {
            "verdict": verdict,
            "verdict_status": verdict_status,
            "metrics": metrics
        }


_air_pipeline = None

def analyze_air(raw_air: dict, spatial_seed: float) -> dict:
    global _air_pipeline
    if _air_pipeline is None:
        _air_pipeline = AirXGBoostPipeline()
    return _air_pipeline.predict(raw_air, spatial_seed)
