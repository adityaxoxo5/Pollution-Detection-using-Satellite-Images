
import math
import requests

from services.geospatial_service import create_grid
from services.satellite_service import extract_grid_observations


NOMINATIM_UA = "TERRASCOPE-Environmental-Platform/1.0"

# Relative weights. These are initial analytical weights,
# not scientifically calibrated pollution-health thresholds.
FEATURE_WEIGHTS = {
    "NO2": 0.20,
    "SO2": 0.10,
    "CO": 0.10,
    "AEROSOL_INDEX": 0.15,
    "NDBI": 0.10,
    "BSI": 0.10,
    "LST": 0.10,
    "LOW_NDVI": 0.05,
    "TURBIDITY_PROXY": 0.05,
    "TSS_PROXY": 0.05
}

FEATURE_LABELS = {
    "NO2": "Relative NO2 signal",
    "SO2": "Relative SO2 signal",
    "CO": "Relative CO signal",
    "AEROSOL_INDEX": "Absorbing aerosol indicator",
    "NDBI": "Built-up surface indicator",
    "BSI": "Bare soil indicator",
    "LST": "Land surface temperature",
    "LOW_NDVI": "Lower vegetation index",
    "TURBIDITY_PROXY": "Turbidity spectral proxy",
    "TSS_PROXY": "Suspended sediment spectral proxy"
}


# ==================================================
# 1. REVERSE GEOCODING
# ==================================================

def reverse_geocode(lat, lon):
    """
    Obtain a locality name from OpenStreetMap.

    Returns None if the service is unavailable.
    """

    try:
        url = "https://nominatim.openstreetmap.org/reverse"

        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 14
        }

        headers = {
            "User-Agent": NOMINATIM_UA
        }

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=4
        )

        response.raise_for_status()

        data = response.json()
        address = data.get("address", {})

        return (
            address.get("suburb")
            or address.get("neighbourhood")
            or address.get("industrial")
            or address.get("quarter")
            or address.get("residential")
            or address.get("city_district")
            or address.get("town")
            or address.get("village")
            or address.get("city")
        )

    except Exception as e:
        print("Reverse geocoding failed:", e)
        return None


# ==================================================
# 2. SAFE SATELLITE VALUE EXTRACTION
# ==================================================

def get_value(cell, domain, indicator):
    """
    Read only valid numeric satellite observations.
    """

    item = (
        cell.get(domain, {})
        .get(indicator, {})
    )

    if not isinstance(item, dict):
        return None

    if item.get("status") != "available":
        return None

    value = item.get("value")

    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


# ==================================================
# 3. BUILD ENVIRONMENTAL FEATURES
# ==================================================

def build_cell_features(cell):
    """
    Collect actual satellite-derived indicators.

    Higher values are treated as relatively elevated
    indicators, except NDVI where lower values are used.
    """

    return {
        "NO2": get_value(cell, "air", "no2"),

        "SO2": get_value(cell, "air", "so2"),

        "CO": get_value(cell, "air", "co"),

        "AEROSOL_INDEX": get_value(
            cell, "air", "aerosol_index"
        ),

        "NDBI": get_value(cell, "land", "NDBI"),

        "BSI": get_value(cell, "land", "BSI"),

        "LST": get_value(cell, "land", "LST_C"),

        "LOW_NDVI": get_value(
            cell, "land", "NDVI"
        ),

        "TURBIDITY_PROXY": get_value(
            cell, "water", "TURBIDITY_PROXY"
        ),

        "TSS_PROXY": get_value(
            cell, "water", "TSS_PROXY"
        )
    }


# ==================================================
# 4. NORMALIZE INDICATORS RELATIVE TO STUDY AREA
# ==================================================

def percentile_scores(values):
    """
    Convert values to relative percentile scores from 0 to 100.

    Tied values receive the average rank.
    """

    valid = [
        (cell_id, value)
        for cell_id, value in values.items()
        if value is not None
    ]

    if len(valid) < 2:
        return {}

    sorted_values = sorted(
        value for _, value in valid
    )

    scores = {}

    for cell_id, value in valid:

        lower = sum(
            1 for item in sorted_values
            if item < value
        )

        equal = sum(
            1 for item in sorted_values
            if item == value
        )

        average_rank = lower + (equal - 1) / 2

        percentile = (
            average_rank / (len(sorted_values) - 1)
        ) * 100

        scores[cell_id] = percentile

    return scores


def calculate_feature_percentiles(cells):
    """
    Calculate relative percentiles for each indicator.

    An indicator is used only if at least two cells
    contain valid observations.
    """

    feature_values = {
        name: {}
        for name in FEATURE_WEIGHTS
    }

    for cell in cells:

        cell_id = cell["cell_id"]

        for feature in FEATURE_WEIGHTS:

            value = cell["features"].get(feature)

            if value is not None:

                # Low NDVI represents a relatively
                # lower vegetation-index signal.
                if feature == "LOW_NDVI":
                    value = -value

                feature_values[feature][cell_id] = value

    percentiles = {}

    for feature, values in feature_values.items():

        percentiles[feature] = percentile_scores(values)

    return percentiles


# ==================================================
# 5. CALCULATE TRANSPARENT RELATIVE RISK SCORES
# ==================================================

def calculate_risk_scores(cells, percentiles):
    """
    Weighted average of available indicator percentiles.

    At least two comparable indicators are required
    before a cell receives a score.
    """

    for cell in cells:

        cell_id = cell["cell_id"]

        contributions = []
        used_features = []

        for feature, weight in FEATURE_WEIGHTS.items():

            feature_scores = percentiles.get(feature, {})

            if cell_id not in feature_scores:
                continue

            score = feature_scores[cell_id]

            contributions.append(
                (score, weight, feature)
            )

            used_features.append(feature)

        if len(contributions) < 2:

            cell["risk_score"] = None
            cell["category"] = "Insufficient observations"
            cell["driver"] = "Insufficient comparable indicators"
            cell["context"] = (
                "More valid satellite observations are required"
            )
            cell["features_used"] = used_features

            continue

        weighted_sum = sum(
            score * weight
            for score, weight, _ in contributions
        )

        total_weight = sum(
            weight
            for _, weight, _ in contributions
        )

        risk_score = weighted_sum / total_weight

        # Identify the strongest weighted relative indicator.
        strongest = max(
            contributions,
            key=lambda item: item[0] * item[1]
        )

        driver = FEATURE_LABELS[strongest[2]]

        cell["risk_score"] = round(risk_score, 2)

        cell["driver"] = driver

        cell["features_used"] = used_features

        cell["category"] = (
            "High Relative Environmental Stress"
            if risk_score >= 75
            else "Elevated Relative Environmental Stress"
            if risk_score >= 50
            else "Lower Relative Environmental Stress"
        )

        cell["context"] = (
            "Relative satellite-indicator pattern; "
            "requires independent ground validation"
        )


# ==================================================
# 6. BOUNDED SPATIAL CLUSTERING
# ==================================================

def coordinates_to_meters(lat, lon, reference_lat):
    """
    Convert latitude/longitude to local metre coordinates.
    Suitable for a city-scale study area.
    """
    x = lon * 111320 * math.cos(math.radians(reference_lat))
    y = lat * 110540
    return x, y


def bounded_spatial_clustering(points, eps, max_diameter):
    """
    Group nearby candidate cells into compact spatial clusters.

    eps:
        Maximum neighbour distance used to grow a cluster.

    max_diameter:
        Maximum allowed distance between any two points
        in the same cluster. This limits chain-linking, where
        a long row of neighbouring cells becomes one huge cluster.

    Every candidate is retained, including isolated candidates.
    """
    labels = [-1] * len(points)
    cluster_id = 0

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    unassigned = set(range(len(points)))

    while unassigned:
        # Start from the highest-priority remaining point.
        seed = min(unassigned)
        unassigned.remove(seed)

        cluster_members = [seed]
        queue = [seed]
        queue_index = 0
        labels[seed] = cluster_id

        while queue_index < len(queue):
            current = queue[queue_index]
            queue_index += 1

            nearby = [
                index for index in list(unassigned)
                if distance(points[current], points[index]) <= eps
            ]

            for candidate_index in nearby:
                # Prevent long, thin chain-linked clusters.
                fits_diameter = all(
                    distance(points[candidate_index], points[member])
                    <= max_diameter
                    for member in cluster_members
                )

                if fits_diameter:
                    unassigned.remove(candidate_index)
                    cluster_members.append(candidate_index)
                    queue.append(candidate_index)
                    labels[candidate_index] = cluster_id

        cluster_id += 1

    return labels


# ==================================================
# 7. MAIN HOTSPOT DETECTION
# ==================================================

def detect_hotspots(
    lat,
    lon,
    start_date,
    end_date,
    radius_m=15000,
    cell_size_m=2000,
    max_cells=None
):
    """
    Identify relative environmental hotspot candidates
    using actual satellite observations.

    Scores are relative to the selected study area.
    No synthetic values or random risk scores are used.
    """

    # Generate geographical grid.
    grid = create_grid(
        lat,
        lon,
        radius_m=radius_m,
        cell_size_m=cell_size_m
    )

    cells = grid["cells"]

    if max_cells is not None:
        cells = cells[:max_cells]

    if not cells:
        raise ValueError("No study grid cells were generated.")

    # Extract real satellite observations.
    analysis_grid = dict(grid)
    analysis_grid["cells"] = cells

    satellite_data = extract_grid_observations(
        analysis_grid,
        str(start_date),
        str(end_date)
    )

    if satellite_data.get("status") == "unavailable":
        raise RuntimeError(
            satellite_data.get(
                "reason",
                "Satellite observations unavailable"
            )
        )

    satellite_cells = satellite_data.get("cells", [])

    if not satellite_cells:
        raise RuntimeError(
            "No satellite cell observations were returned."
        )

    # Build features for every returned cell.
    results = []

    for cell in satellite_cells:
        features = build_cell_features(cell)

        valid_count = sum(
            value is not None
            for value in features.values()
        )

        results.append({
            "cell_id": cell["cell_id"],
            "center": cell["center"],
            "features": features,
            "valid_indicators": valid_count,
            "risk_score": None,
            "category": "Insufficient observations",
            "driver": "Insufficient comparable indicators",
            "context": (
                "More valid satellite observations are required"
            )
        })

    # Normalize using actual observations from this study area.
    percentiles = calculate_feature_percentiles(results)

    # Calculate weighted relative scores.
    calculate_risk_scores(results, percentiles)

    scored_cells = [
        cell for cell in results
        if cell["risk_score"] is not None
    ]

    # Sort all cells by relative score, keeping unscored cells last.
    results.sort(
        key=lambda item: (
            item["risk_score"]
            if item["risk_score"] is not None
            else -1
        ),
        reverse=True
    )

    # No fabricated hotspot if there is insufficient evidence.
    if not scored_cells:
        return {
            "location": {
                "latitude": float(lat),
                "longitude": float(lon)
            },
            "period": {
                "start": str(start_date),
                "end": str(end_date)
            },
            "total_cells_analyzed": len(results),
            "scored_cells": 0,
            "hotspots": [],
            "all_cells": results,
            "candidate_threshold": None,
            "method": (
                "Satellite-derived relative indicator analysis. "
                "Insufficient comparable observations for scoring."
            ),
            "data_status": "insufficient_data"
        }

    # Adaptive candidate threshold:
    # retain cells at or above the 65th percentile, with a
    # minimum relative score of 50.
    scores = sorted(
        cell["risk_score"]
        for cell in scored_cells
    )

    percentile_index = max(
        0,
        math.ceil(len(scores) * 0.65) - 1
    )

    candidate_threshold = max(
        50.0,
        scores[percentile_index]
    )

    candidates = [
        cell for cell in scored_cells
        if cell["risk_score"] >= candidate_threshold
    ]

    # Prioritize higher-scoring candidates when building compact groups.
    candidates.sort(
        key=lambda cell: cell["risk_score"],
        reverse=True
    )

    hotspots = []

    if candidates:
        reference_lat = float(lat)

        points = [
            coordinates_to_meters(
                cell["center"]["lat"],
                cell["center"]["lon"],
                reference_lat
            )
            for cell in candidates
        ]

        # Neighbouring 2 km cells can connect, but a cluster is
        # not allowed to stretch into an unlimited chain.
        labels = bounded_spatial_clustering(
            points,
            eps=cell_size_m * 1.05,
            max_diameter=cell_size_m * 3.0
        )

        cluster_ids = sorted(set(labels))

        for cluster_id in cluster_ids:
            cluster_cells = [
                candidates[index]
                for index, label in enumerate(labels)
                if label == cluster_id
            ]

            if not cluster_cells:
                continue

            center_lat = sum(
                cell["center"]["lat"]
                for cell in cluster_cells
            ) / len(cluster_cells)

            center_lon = sum(
                cell["center"]["lon"]
                for cell in cluster_cells
            ) / len(cluster_cells)

            average_risk = sum(
                cell["risk_score"]
                for cell in cluster_cells
            ) / len(cluster_cells)

            drivers = [
                cell["driver"]
                for cell in cluster_cells
            ]

            driver = max(
                set(drivers),
                key=drivers.count
            )

            category = (
                "High Relative Environmental Stress"
                if average_risk >= 75
                else "Elevated Relative Environmental Stress"
                if average_risk >= 50
                else "Lower Relative Environmental Stress"
            )

            hotspots.append({
                "cell_id": cluster_cells[0]["cell_id"],
                "cell_ids": [
                    cell["cell_id"]
                    for cell in cluster_cells
                ],
                "center": {
                    "lat": center_lat,
                    "lon": center_lon
                },
                "risk_score": round(average_risk, 2),
                "average_risk": round(average_risk, 2),
                "category": category,
                "driver": driver,
                "context": (
                    "Compact cluster of elevated satellite-derived "
                    "environmental indicators. Relative score; "
                    "independent ground validation is required."
                ),
                "cells_in_cluster": len(cluster_cells),
                "name": None
            })

        # Rank all detected clusters by their average relative score.
        hotspots.sort(
            key=lambda item: item["risk_score"],
            reverse=True
        )

        # Display no more than 10 detected clusters.
        # This does not manufacture clusters if fewer are present.
        hotspots = hotspots[:10]

        # Reverse-geocode only the highest-ranked cluster to avoid
        # sending many requests to the public Nominatim service.
        if hotspots:
            top = hotspots[0]

            name = reverse_geocode(
                top["center"]["lat"],
                top["center"]["lon"]
            )

            if name:
                top["name"] = name
            else:
                top["name"] = (
                    f"Area near "
                    f"{top['center']['lat']:.4f}, "
                    f"{top['center']['lon']:.4f}"
                )

        # Other clusters use their actual centroid coordinates.
        for hotspot in hotspots[1:]:
            hotspot["name"] = (
                f"Area near "
                f"{hotspot['center']['lat']:.4f}, "
                f"{hotspot['center']['lon']:.4f}"
            )

    return {
        "location": {
            "latitude": float(lat),
            "longitude": float(lon)
        },
        "period": {
            "start": str(start_date),
            "end": str(end_date)
        },
        "total_cells_analyzed": len(results),
        "scored_cells": len(scored_cells),
        "hotspots": hotspots,
        "all_cells": results,
        "candidate_threshold": candidate_threshold,
        "candidate_cells": len(candidates),
        "method": (
            "Satellite-derived weighted relative indicator scoring "
            "with adaptive candidate selection and bounded spatial "
            "clustering. Sources: Sentinel-5P, Sentinel-2 and Landsat. "
            "Cluster diameter is limited to three grid-cell widths."
        ),
        "data_status": "available"
    }
