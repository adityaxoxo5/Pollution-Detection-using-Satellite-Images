import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env is loaded from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()


def format_metric(val, prec=4):
    if val is None:
        return "N/A"
    try:
        f = float(val)
        if abs(f) < 0.001 and f != 0:
            return f"{f:.2e}"
        return f"{f:.{prec}f}"
    except (TypeError, ValueError):
        return str(val)


def generate_gemini_environmental_report(air_data, water_data, land_data, location_name):
    """
    Pass real satellite observations to Google Gemini LLM to generate
    human-understandable, insightful environmental impact reports.
    """
    api_key = (
        os.getenv("GOOGLE_GEMINI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        print("No Gemini API key configured, using baseline report generator.")
        return None

    # Prepare compact satellite metric summary for the prompt
    air_summary = (
        f"NO2={format_metric(air_data.get('no2'))} mol/m2, "
        f"SO2={format_metric(air_data.get('so2'))} mol/m2, "
        f"CO={format_metric(air_data.get('co'))} mol/m2, "
        f"O3={format_metric(air_data.get('o3'))} mol/m2, "
        f"Aerosol Index={format_metric(air_data.get('aerosol_index'), 2)}, "
        f"AOD={format_metric(air_data.get('aod'), 2)}, "
        f"CH4={format_metric(air_data.get('ch4'), 1)} ppb, "
        f"HCHO={format_metric(air_data.get('hcho'))} mol/m2"
    )

    water_summary = (
        f"NDWI={format_metric(water_data.get('ndwi'), 3)}, "
        f"MNDWI={format_metric(water_data.get('mndwi'), 3)}, "
        f"Turbidity={format_metric(water_data.get('turbidity'), 2)}, "
        f"Chlorophyll-a={format_metric(water_data.get('chlorophyll_a'), 2)}, "
        f"Water Temp={format_metric(water_data.get('water_temp'), 1)}°C, "
        f"Algal Status={water_data.get('algal_bloom', 'None')}"
    )

    land_summary = (
        f"NDVI={format_metric(land_data.get('ndvi'), 3)}, "
        f"EVI={format_metric(land_data.get('evi'), 3)}, "
        f"NDBI={format_metric(land_data.get('ndbi'), 3)}, "
        f"BSI={format_metric(land_data.get('bsi'), 3)}, "
        f"LST={format_metric(land_data.get('lst'), 1)}°C, "
        f"Impervious Surface={land_data.get('impervious_surface', 'moderate')}"
    )

    prompt = f"""You are an environmental satellite analyst. Analyze the following real satellite data observed for {location_name}:

[ATMOSPHERIC (Sentinel-5P, MODIS)]:
{air_summary}

[AQUATIC (Sentinel-2)]:
{water_summary}

[TERRESTRIAL & THERMAL (Sentinel-2, Landsat)]:
{land_summary}

Task:
Generate a clear, human-understandable environmental impact assessment in JSON format.
Do NOT just quote raw numbers (e.g. avoid saying "NDMI is 0.12"). Instead, describe what it physically means for people living there — e.g. whether the air is hazy or foggy, horizontal visibility, respiratory comfort, industrial vs vehicle fumes, water clarity or algae scum, urban heat island trapping, sparse vs healthy vegetation, and soil erosion risk.

Respond with ONLY valid JSON adhering to this exact schema:
{{
  "health_score": <integer from 20 to 95 representing overall ecological health score>,
  "health_status": "<Short status e.g. 'Good / Low Stress', 'Moderate Stress', or 'High Environmental Stress'>",
  "overall_summary": "<1 to 2 clear sentences summarizing the overall environmental health of {location_name}>",
  "air_impacts": [
    "<Concise bullet point explaining atmospheric haze, smog, or visibility impact>",
    "<Concise bullet point explaining combustion, traffic or industrial emission impact>"
  ],
  "water_impacts": [
    "<Concise bullet point explaining water body clarity, suspended solids or turbidity>",
    "<Concise bullet point explaining algae growth, eutrophication risk or water extent>"
  ],
  "land_impacts": [
    "<Concise bullet point explaining urban heat island effect, surface heating or concrete coverage>",
    "<Concise bullet point explaining tree canopy, vegetative health or topsoil dryness>"
  ]
}}
"""

    models_to_try = [
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-pro-latest"
    ]

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 1000,
                    "responseMimeType": "application/json"
                }
            }
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                result_json = res.json()
                candidates = result_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        raw_text = parts[0]["text"].strip()
                        if raw_text.startswith("```json"):
                            raw_text = raw_text[7:]
                        if raw_text.startswith("```"):
                            raw_text = raw_text[3:]
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3]

                        report_data = json.loads(raw_text.strip())
                        report_data["ai_generated"] = True
                        report_data["ai_model"] = "Google Gemini"
                        return report_data

            print(f"Gemini {model_name} returned status {res.status_code}: {res.text[:120]}")

        except Exception as e:
            print(f"Error querying Gemini model {model_name}: {e}")

    return None
