
from datetime import datetime


def format_value(value, decimals=4):
    """Format numerical values for readable reports."""

    if value is None:
        return "Not available"

    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return "Not available"


def create_section(title, data, description):
    """Create a standard report section."""

    return {
        "title": title,
        "description": description,
        "data": data
    }


def generate_environmental_report(
    location,
    analysis,
    trends=None,
    hotspots=None,
    predictions=None
):
    """
    Generate an on-screen environmental analysis report.

    This function does not create a PDF or download a file.
    It returns structured data that the frontend can display.
    """

    trends = trends or {}
    hotspots = hotspots or {}
    predictions = predictions or {}

    # Location information
    location_name = location.get("name", "Unknown location")

    latitude = format_value(location.get("lat"), 6)
    longitude = format_value(location.get("lon"), 6)

    # Extract domain information
    air = analysis.get("air", {}).get("metrics", {})
    water = analysis.get("water", {}).get("metrics", {})
    land = analysis.get("land", {}).get("metrics", {})

    # Air section
    air_data = {
        "NO2 column density": format_value(
            air.get("no2_column_mol_m2")
        ),
        "Aerosol index": format_value(
            air.get("aerosol_index")
        ),
        "Observations": air.get("observations", "Not available"),
        "Data source": air.get("source", "Not available")
    }

    # Water section
    water_data = {
        "NDWI": format_value(water.get("ndwi")),
        "MNDWI": format_value(water.get("mndwi")),
        "Observations": water.get("observations", "Not available"),
        "Data source": water.get("source", "Not available")
    }

    # Land section
    land_data = {
        "NDVI": format_value(land.get("ndvi")),
        "NDBI": format_value(land.get("ndbi")),
        "BSI": format_value(land.get("bsi")),
        "Observations": land.get("observations", "Not available"),
        "Data source": land.get("source", "Not available")
    }

    # Historical trend information
    trend_data = {
        "Total periods": trends.get("total_periods", 0),
        "Interval": trends.get("interval", "Not available"),
        "Date range": trends.get("date_range", {})
    }

    # Hotspot information
    hotspot_data = {
        "Cells analyzed": hotspots.get(
            "total_cells_analyzed", 0
        ),
        "Potential hotspots identified": hotspots.get(
            "hotspot_count", 0
        ),
        "Method": hotspots.get("method", "Not available")
    }

    # Prediction information
    prediction_data = {
        "Model": predictions.get("model", "Not available"),
        "Target": predictions.get("target", "Not available"),
        "Evaluation": predictions.get("evaluation", {}),
        "Predictions": predictions.get("predictions", [])
    }

    # Create report sections
    sections = [
        create_section(
            "Air Quality Indicators",
            air_data,
            "Satellite-derived atmospheric indicators."
        ),

        create_section(
            "Water Surface Indicators",
            water_data,
            "Spectral indicators associated with surface water."
        ),

        create_section(
            "Land Surface Indicators",
            land_data,
            "Satellite indicators related to vegetation, "
            "built-up surfaces and bare soil."
        ),

        create_section(
            "Historical Environmental Trends",
            trend_data,
            "Monthly satellite observations across the "
            "selected analysis period."
        ),

        create_section(
            "Environmental Hotspot Analysis",
            hotspot_data,
            "Relative comparison of environmental signals "
            "across geographical grid cells."
        ),

        create_section(
            "Machine Learning Predictions",
            prediction_data,
            "Predictions generated using the supplied "
            "historical training observations."
        )
    ]

    # Scientific limitations
    limitations = [
        "Satellite observations are not equivalent to "
        "ground-level pollution measurements.",

        "NDVI, NDWI, MNDWI, NDBI and BSI are spectral indicators, "
        "not direct chemical measurements.",

        "Hotspot scores represent relative environmental signals "
        "within the selected study region.",

        "Machine learning predictions depend on the quality "
        "and quantity of training data.",

        "Ground-based measurements are recommended for "
        "independent validation."
    ]

    # Final report
    report = {
        "report_title": "TERRASCOPE Environmental Assessment Report",

        "generated_at": datetime.now().astimezone().isoformat(),

        "location": {
            "name": location_name,
            "latitude": latitude,
            "longitude": longitude
        },

        "analysis_period": analysis.get("period", {}),

        "overview": (
            "This report presents satellite-derived air, water "
            "and land indicators, historical observations, "
            "potential environmental hotspots and available "
            "machine learning predictions."
        ),

        "sections": sections,

        "scientific_limitations": limitations,

        "report_status": "Generated successfully"
    }

    return report
