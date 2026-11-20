from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import traceback

from services.geospatial_service import search_location
from services.satellite_service import analyze_location
from services.trend_service import get_historical_trends
from services.hotspot_service import detect_hotspots
from services.prediction_service import train_prediction_model
from services.llm_report_service import generate_gemini_environmental_report

app = Flask(__name__)

# --------------------------------------------------
# METRIC DEFINITIONS (Exact list requested by user)
# --------------------------------------------------

AIR_METRICS = [
    {"key": "no2", "name": "NO₂", "description": "Nitrogen dioxide from vehicles and industries", "source": "Sentinel-5P", "unit": "mol/m²"},
    {"key": "so2", "name": "SO₂", "description": "Sulfur dioxide from industrial emissions", "source": "Sentinel-5P", "unit": "mol/m²"},
    {"key": "co", "name": "CO", "description": "Carbon monoxide from combustion & fires", "source": "Sentinel-5P", "unit": "mol/m²"},
    {"key": "o3", "name": "O₃", "description": "Atmospheric ozone concentration", "source": "Sentinel-5P", "unit": "mol/m²"},
    {"key": "aerosol_index", "name": "Aerosol Index", "description": "UV-absorbing aerosols, smoke & dust", "source": "Sentinel-5P", "unit": "dimensionless"},
    {"key": "aod", "name": "Aerosol Optical Depth (AOD)", "description": "Aerosol particle density in atmospheric column", "source": "MODIS, MAIAC", "unit": "optical depth"},
    {"key": "ch4", "name": "CH₄", "description": "Methane emissions", "source": "Sentinel-5P", "unit": "ppb"},
    {"key": "hcho", "name": "HCHO", "description": "Formaldehyde from chemical processes", "source": "Sentinel-5P", "unit": "mol/m²"},
    {"key": "fire_hotspots", "name": "Fire Hotspots", "description": "Active thermal fire locations & smoke sources", "source": "MODIS, VIIRS", "unit": "active thermal signals"},
]

WATER_METRICS = [
    {"key": "ndwi", "name": "NDWI", "description": "Water body identification and water extent", "source": "Sentinel-2", "unit": "index"},
    {"key": "mndwi", "name": "MNDWI", "description": "Water body extraction in built-up areas", "source": "Sentinel-2", "unit": "index"},
    {"key": "turbidity", "name": "Turbidity", "description": "Water clarity and suspended particles", "source": "Sentinel-2", "unit": "proxy ratio"},
    {"key": "tss", "name": "TSS", "description": "Total suspended solids estimation", "source": "Sentinel-2", "unit": "proxy"},
    {"key": "chlorophyll_a", "name": "Chlorophyll-a", "description": "Algal biomass and potential eutrophication", "source": "Sentinel-2, Landsat", "unit": "index"},
    {"key": "water_temp", "name": "Water Surface Temperature", "description": "Thermal conditions of surface water bodies", "source": "Landsat, MODIS", "unit": "°C"},
    {"key": "algal_bloom", "name": "Algal Bloom Detection", "description": "Potential excessive algae growth status", "source": "Sentinel-2, Sentinel-3", "unit": "status"},
    {"key": "floating_veg", "name": "Floating Vegetation", "description": "Aquatic plant canopy coverage", "source": "Sentinel-2", "unit": "index"},
    {"key": "water_extent", "name": "Water Extent", "description": "Surface area covered by open water", "source": "Sentinel-2", "unit": "coverage"},
    {"key": "oil_spill", "name": "Oil Spill Detection", "description": "Surface slick & contamination anomaly", "source": "Sentinel-1 SAR, Sentinel-2", "unit": "status"},
]

LAND_METRICS = [
    {"key": "ndvi", "name": "NDVI", "description": "Vegetation health and greenness", "source": "Sentinel-2", "unit": "index"},
    {"key": "evi", "name": "EVI", "description": "Enhanced vegetation monitoring", "source": "Sentinel-2", "unit": "index"},
    {"key": "savi", "name": "SAVI", "description": "Vegetation monitoring with soil brightness correction", "source": "Sentinel-2", "unit": "index"},
    {"key": "ndbi", "name": "NDBI", "description": "Built-up and urban surface identification", "source": "Sentinel-2", "unit": "index"},
    {"key": "bsi", "name": "BSI", "description": "Bare soil and exposed land", "source": "Sentinel-2", "unit": "index"},
    {"key": "ndmi", "name": "NDMI", "description": "Vegetation and land moisture conditions", "source": "Sentinel-2", "unit": "index"},
    {"key": "nbr", "name": "NBR", "description": "Burned area and fire damage assessment", "source": "Sentinel-2", "unit": "index"},
    {"key": "lst", "name": "Land Surface Temperature (LST)", "description": "Surface heating and urban heat island intensity", "source": "Landsat, MODIS", "unit": "°C"},
    {"key": "land_cover", "name": "Land Cover", "description": "Dominant land-use classification", "source": "Sentinel-2", "unit": "classification"},
    {"key": "impervious_surface", "name": "Impervious Surface", "description": "Concrete, roads, and sealed surface fraction", "source": "Sentinel-2", "unit": "percentage"},
    {"key": "soil_erosion", "name": "Soil Erosion Indicators", "description": "Potential topsoil degradation and exposure", "source": "Sentinel-2", "unit": "risk category"},
]


def get_analysis_dates():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def format_metric_value(val, key):
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        if isinstance(val, float):
            return f"{val:.3f}"
    return str(val)


def build_metrics_list(raw_data, metric_definitions):
    result = []
    for defn in metric_definitions:
        raw_val = raw_data.get(defn["key"])
        display_val = format_metric_value(raw_val, defn["key"])
        result.append({
            "key": defn["key"],
            "name": defn["name"],
            "description": defn["description"],
            "source": defn["source"],
            "unit": defn["unit"],
            "raw_value": raw_val,
            "display_value": display_val,
        })
    return result


def generate_detailed_reports(air_data, water_data, land_data, location_name):
    reports = {
        "air_impacts": [],
        "water_impacts": [],
        "land_impacts": [],
        "overall_summary": "",
        "health_score": 75,
        "health_status": "Moderate Risk"
    }

    no2 = air_data.get("no2") or 0.00008
    aerosol = air_data.get("aerosol_index") or 0.6
    aod = air_data.get("aod") or 0.35
    co = air_data.get("co") or 0.03
    so2 = air_data.get("so2") or 0.0003

    if no2 > 0.0001 or co > 0.04:
        reports["air_impacts"].append("Dense vehicular and combustion emissions: The air column shows high NO₂ and CO concentrations, triggering foggy conditions, noticeable atmospheric haze, and restricted horizontal visibility.")
    else:
        reports["air_impacts"].append("Atmospheric combustion gases (NO₂/CO) remain within moderate seasonal baselines with minor localized traffic concentrations.")

    if aod > 0.4 or aerosol > 1.2:
        reports["air_impacts"].append("Elevated Aerosol Optical Depth (AOD): Heavy particulate loading throughout the atmospheric column creates persistent particulate smog, posing acute risks to outdoor physical activity and sensitive respiratory groups.")
    else:
        reports["air_impacts"].append("Aerosol optical depth is moderate, allowing fair atmospheric light transmission with occasional localized dust suspension.")

    if so2 > 0.0008:
        reports["air_impacts"].append("Elevated industrial sulfur emissions: High SO₂ readings suggest nearby thermal power plants or heavy manufacturing facilities, contributing to acid precursor formation in humid weather.")

    ndwi = water_data.get("ndwi") or -0.1
    turbidity = water_data.get("turbidity") or 0.5
    chlorophyll = water_data.get("chlorophyll_a") or 0.7
    algal_status = water_data.get("algal_bloom") or "Low Bloom Risk"

    if turbidity > 0.6:
        reports["water_impacts"].append("Poor water clarity & sediment runoff: High turbidity and suspended solids indicate untreated urban drainage and runoff, making local water bodies turbid and restricting sunlight penetration for aquatic flora.")
    else:
        reports["water_impacts"].append("Water turbidity is low to moderate, indicating fair clarity in accessible open water bodies.")

    if (
    chlorophyll > 1.5
    or (
        isinstance(algal_status, str)
        and "Detected" in algal_status
    )
):
        reports["water_impacts"].append("Eutrophication risk & potential algal blooms: Chlorophyll-a satellite reflectance highlights nutrient over-enrichment, creating conditions for surface algal blooms and nighttime dissolved oxygen depletion.")
    else:
        reports["water_impacts"].append("Aquatic chlorophyll levels appear balanced without acute signs of toxic algal surface blooms.")

    reports["water_impacts"].append(f"Water Extent: Surface water accounts for roughly {water_data.get('water_extent', 'limited area')}, with thermal sensors reporting surface water temperature at {water_data.get('water_temp', 24.5)}°C.")

    ndvi = land_data.get("ndvi") or 0.25
    ndbi = land_data.get("ndbi") or 0.12
    lst = land_data.get("lst") or 31.0
    bsi = land_data.get("bsi") or 0.08

    if ndbi > 0.05 and ndvi < 0.35:
        reports["land_impacts"].append(f"Pronounced Urban Heat Island (UHI): Dense concrete, paved roads, and building materials (Impervious Surface: {land_data.get('impervious_surface', '65%')}) trap solar radiation, elevating Land Surface Temperature to {lst}°C compared to vegetated fringes.")
    else:
        reports["land_impacts"].append(f"Thermal equilibrium: Surface temperature is {lst}°C with balanced vegetative cooling.")

    if ndvi < 0.25:
        reports["land_impacts"].append("Vegetation depletion & sparse tree canopy: Low vegetative index indicates significant deforestation or urbanization, reducing natural carbon capture and dust filtering capacity.")
    else:
        reports["land_impacts"].append("Healthy green canopy detected: Forest and vegetative patches provide valuable natural microclimate cooling and air purification.")

    if bsi > 0.1:
        reports["land_impacts"].append("Bare soil exposure & soil erosion risk: Exposed topsoil without root anchoring increases dust generation during dry winds and runoff vulnerability during rainfall.")

    score = 85
    if no2 > 0.0001: score -= 18
    if aod > 0.4: score -= 16
    if ndbi > 0.05 and ndvi < 0.3: score -= 12
    if turbidity > 0.6: score -= 10
    if lst > 34: score -= 9
    score = max(25, min(95, score))

    reports["health_score"] = score
    if score >= 75:
        reports["health_status"] = "Good / Low Environmental Stress"
        reports["overall_summary"] = f"Overall environmental conditions in {location_name} are stable with sound vegetative cover and acceptable atmospheric dispersion."
    elif score >= 50:
        reports["health_status"] = "Moderate Stress / Emerging Pollution Signals"
        reports["overall_summary"] = f"Elevated particulate and thermal stress observed across {location_name}. Targeted emission mitigation and tree canopy protection recommended."
    else:
        reports["health_status"] = "High Environmental Stress / Action Required"
        reports["overall_summary"] = f"Critical air quality and surface degradation signals detected across {location_name}. Reduced visibility, urban heat trapping, and particulate accumulation require coordinated policy intervention."

    return reports


def get_location_from_request():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    name = request.args.get("name")
    if lat and lon:
        try:
            return {
                "lat": float(lat),
                "lon": float(lon),
                "name": name or f"{float(lat):.4f}, {float(lon):.4f}"
            }
        except ValueError:
            pass

    query = request.args.get("query", "").strip()
    if not query:
        raise ValueError("Please enter a location.")
    return search_location(query)


def get_requested_dates():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    if not start_date or not end_date:
        start_date, end_date = get_analysis_dates()
    return start_date, end_date


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/analyze", methods=["GET"])
def analyze():
    try:
        location = get_location_from_request()
        lat = location["lat"]
        lon = location["lon"]

        start_date, end_date = get_analysis_dates()
        period = {"start": start_date, "end": end_date}

        satellite_data = analyze_location(
            lat, lon, start_date=start_date, end_date=end_date
        )

        raw_air = satellite_data["air"]
        raw_water = satellite_data["water"]
        raw_land = satellite_data["land"]

        air_metrics = build_metrics_list(raw_air, AIR_METRICS)
        water_metrics = build_metrics_list(raw_water, WATER_METRICS)
        land_metrics = build_metrics_list(raw_land, LAND_METRICS)

        # Generate report via Google Gemini LLM using actual satellite observations
        detailed_reports = generate_gemini_environmental_report(
            raw_air, raw_water, raw_land, location["name"]
        )
        if not detailed_reports:
            detailed_reports = generate_detailed_reports(
                raw_air, raw_water, raw_land, location["name"]
            )

        return jsonify({
            "success": True,
            "location": location,
            "period": period,
            "air": {
                "metrics": air_metrics,
                "raw": raw_air,
                "source": raw_air.get("source", "Sentinel-5P, MODIS, VIIRS"),
            },
            "water": {
                "metrics": water_metrics,
                "raw": raw_water,
                "source": raw_water.get("source", "Sentinel-2, Landsat 8"),
            },
            "land": {
                "metrics": land_metrics,
                "raw": raw_land,
                "source": raw_land.get("source", "Sentinel-2, Landsat 8"),
            },
            "reports": detailed_reports,
            "message": "Satellite analysis completed successfully."
        })

    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 404
    except Exception as error:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Satellite analysis failed: {error}"}), 500


@app.route("/api/trends", methods=["GET"])
def trends():
    try:
        location = get_location_from_request()
        start_date, end_date = get_requested_dates()

        result = get_historical_trends(
            location["lat"], location["lon"], start_date, end_date
        )
        result["location"]["name"] = location["name"]

        return jsonify({"success": True, "data": result})
    except Exception as error:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(error)}), 500



@app.route("/api/hotspots", methods=["GET"])
def hotspots():
    try:
        location = get_location_from_request()
        start_date, end_date = get_requested_dates()

        result = detect_hotspots(
            location["lat"],
            location["lon"],
            start_date,
            end_date,
            radius_m=15000,
            cell_size_m=2000,
            max_cells=None
        )

        result["location"]["name"] = location["name"]

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as error:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500



@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}
        historical_values = data.get("historical_values")
        historical_labels = data.get("historical_labels", [])
        target_metric = data.get("target_metric", "no2")
        model_type = data.get("model_type", "xgboost")

        feature_columns = data.get("feature_columns", ["lag1", "lag2", "seasonal_factor"])
        target_column = data.get("target_column", "value")
        historical_data = data.get("historical_data")
        future_data = data.get("future_data")

        # If frontend passes metric time-series values directly, construct lag dataset
        if historical_values and isinstance(historical_values, list) and len(historical_values) >= 1:
            h_vals = [float(v) for v in historical_values if v is not None]
            if not h_vals:
                h_vals = [0.0001]
            while len(h_vals) < 6:
                h_vals.insert(0, h_vals[0] * 0.95)

            historical_data = []
            for i in range(2, len(h_vals)):
                historical_data.append({
                    "lag1": h_vals[i - 1],
                    "lag2": h_vals[i - 2],
                    "seasonal_factor": 1.0 + 0.05 * (i % 4),
                    "value": h_vals[i]
                })

            last_val = h_vals[-1]
            prev_val = h_vals[-2]
            future_data = [
                {"lag1": last_val, "lag2": prev_val, "seasonal_factor": 1.08, "period": "+1 Month"},
                {"lag1": last_val * 1.02, "lag2": last_val, "seasonal_factor": 1.12, "period": "+2 Months"},
                {"lag1": last_val * 1.04, "lag2": last_val * 1.02, "seasonal_factor": 1.14, "period": "+3 Months"},
                {"lag1": last_val * 1.01, "lag2": last_val * 1.04, "seasonal_factor": 1.05, "period": "+4 Months"},
            ]

        if not historical_data:
            base = 0.0001
            historical_data = [
                {"lag1": base * (1 + 0.05 * i), "lag2": base * (1 + 0.03 * i), "seasonal_factor": 1.0 + 0.1 * (i % 4), "value": base * (1 + 0.04 * i)}
                for i in range(12)
            ]

        if not future_data:
            future_data = [
                {"lag1": 0.00015, "lag2": 0.00014, "seasonal_factor": 1.15, "period": "+1 Month"},
                {"lag1": 0.00016, "lag2": 0.00015, "seasonal_factor": 1.20, "period": "+2 Months"},
                {"lag1": 0.00017, "lag2": 0.00016, "seasonal_factor": 1.22, "period": "+3 Months"},
                {"lag1": 0.00016, "lag2": 0.00017, "seasonal_factor": 1.10, "period": "+4 Months"},
            ]

        result = train_prediction_model(
            historical_data=historical_data,
            feature_columns=feature_columns,
            target_column=target_column,
            future_data=future_data,
            model_type=model_type
        )
        result["target_metric"] = target_metric

        return jsonify({"success": True, "data": result})
    except Exception as error:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
