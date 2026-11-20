'''import requests
import math

# --------------------------------------------------
# 1. GEOGRAPHICAL COORDINATE VALIDATION
# --------------------------------------------------

def validate_coordinates(lat, lon):
    lat = float(lat)
    lon = float(lon)

    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    return lat, lon


# --------------------------------------------------
# 2. GLOBAL LOCATION SEARCH
# --------------------------------------------------

def search_location(place):
    place = place.strip()
    if not place:
        raise ValueError("Please enter a location.")

    try:
        parts = place.replace(",", " ").split()
        if len(parts) == 2:
            lat, lon = validate_coordinates(parts[0], parts[1])
            return {
                "name": f"{lat:.4f}, {lon:.4f}",
                "lat": lat,
                "lon": lon,
                "type": "coordinates"
            }
    except ValueError:
        pass

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": place,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "TERRASCOPE-Environmental-Platform/1.0"
    }

    response = requests.get(url, params=params, headers=headers, timeout=8)
    response.raise_for_status()

    results = response.json()
    if not results:
        raise ValueError("Location not found.")

    result = results[0]
    lat, lon = validate_coordinates(result["lat"], result["lon"])

    return {
        "name": result["display_name"],
        "lat": lat,
        "lon": lon,
        "type": result.get("type", "location")
    }


# --------------------------------------------------
# 3. CREATE STUDY REGION (Pure mathematical geometry)
# --------------------------------------------------

def create_study_region(lat, lon, radius_m=5000):
    lat, lon = validate_coordinates(lat, lon)
    radius_m = float(radius_m)

    if not 100 <= radius_m <= 100000:
        raise ValueError("Study radius must be between 100 and 100000 metres.")

    return {
        "type": "Circle",
        "coordinates": [lon, lat],
        "radius_m": radius_m
    }


# --------------------------------------------------
# 4. CREATE GEOGRAPHICAL GRID (Blazing fast pure math)
# --------------------------------------------------

def create_grid(lat, lon, radius_m=5000, cell_size_m=2000):
    lat, lon = validate_coordinates(lat, lon)
    if cell_size_m <= 0:
        raise ValueError("Grid cell size must be greater than zero.")

    region = create_study_region(lat, lon, radius_m)

    latitude_offset = radius_m / 111320
    cos_lat = max(abs(math.cos(math.radians(lat))), 0.01)
    longitude_offset = radius_m / (111320 * cos_lat)

    lat_step = cell_size_m / 111320
    lon_step = cell_size_m / (111320 * cos_lat)

    cells = []
    row = 0
    current_lat = lat - latitude_offset

    while current_lat < lat + latitude_offset and row < 4:
        col = 0
        current_lon = lon - longitude_offset
        while current_lon < lon + longitude_offset and col < 4:
            min_lon = current_lon
            min_lat = current_lat
            max_lon = current_lon + lon_step
            max_lat = current_lat + lat_step

            cells.append({
                "id": f"cell_{row}_{col}",
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "center": {
                    "lat": round((min_lat + max_lat) / 2, 5),
                    "lon": round((min_lon + max_lon) / 2, 5)
                }
            })
            current_lon += lon_step
            col += 1
        current_lat += lat_step
        row += 1

    return {
        "region": region,
        "cells": cells,
        "total_cells": len(cells)
    }
'''

import requests
import math


# --------------------------------------------------
# 1. GEOGRAPHICAL COORDINATE VALIDATION
# --------------------------------------------------

def validate_coordinates(lat, lon):
    lat = float(lat)
    lon = float(lon)

    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    return lat, lon


# --------------------------------------------------
# 2. GLOBAL LOCATION SEARCH
# --------------------------------------------------

def search_location(place):
    place = place.strip()

    if not place:
        raise ValueError("Please enter a location.")

    try:
        parts = place.replace(",", " ").split()

        if len(parts) == 2:
            lat, lon = validate_coordinates(parts[0], parts[1])

            return {
                "name": f"{lat:.4f}, {lon:.4f}",
                "lat": lat,
                "lon": lon,
                "type": "coordinates"
            }

    except ValueError:
        pass

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": place,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1
    }

    headers = {
        "User-Agent": "TERRASCOPE-Environmental-Platform/1.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=8
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        raise ValueError("Location not found.")

    result = results[0]

    lat, lon = validate_coordinates(
        result["lat"],
        result["lon"]
    )

    return {
        "name": result["display_name"],
        "lat": lat,
        "lon": lon,
        "type": result.get("type", "location")
    }


# --------------------------------------------------
# 3. CREATE STUDY REGION
# --------------------------------------------------

def create_study_region(lat, lon, radius_m=5000):

    lat, lon = validate_coordinates(lat, lon)

    radius_m = float(radius_m)

    if not 100 <= radius_m <= 100000:
        raise ValueError(
            "Study radius must be between 100 and 100000 metres."
        )

    return {
        "type": "Circle",
        "coordinates": [lon, lat],
        "radius_m": radius_m
    }


# --------------------------------------------------
# 4. CREATE COMPLETE GEOGRAPHICAL GRID
# --------------------------------------------------

def create_grid(
    lat,
    lon,
    radius_m=5000,
    cell_size_m=2000
):

    lat, lon = validate_coordinates(lat, lon)

    radius_m = float(radius_m)
    cell_size_m = float(cell_size_m)

    if not 100 <= radius_m <= 100000:
        raise ValueError("Invalid study radius.")

    if cell_size_m <= 0:
        raise ValueError("Grid cell size must be greater than zero.")

    # Prevent excessive grid generation
    estimated_cells = math.ceil(
        (2 * radius_m / cell_size_m) ** 2
    )

    if estimated_cells > 2500:
        raise ValueError(
            "Grid too large. Increase cell_size_m or reduce radius_m."
        )

    region = create_study_region(
        lat,
        lon,
        radius_m
    )

    # Convert metres into approximate latitude/longitude degrees
    lat_step = cell_size_m / 111320

    cos_lat = max(
        abs(math.cos(math.radians(lat))),
        0.01
    )

    lon_step = cell_size_m / (111320 * cos_lat)

    latitude_offset = radius_m / 111320
    longitude_offset = radius_m / (111320 * cos_lat)

    min_region_lat = lat - latitude_offset
    max_region_lat = lat + latitude_offset

    min_region_lon = lon - longitude_offset
    max_region_lon = lon + longitude_offset

    cells = []

    row = 0
    current_lat = min_region_lat

    while current_lat < max_region_lat:

        col = 0
        current_lon = min_region_lon

        while current_lon < max_region_lon:

            min_lon = current_lon
            min_lat = current_lat

            max_lon = min(
                current_lon + lon_step,
                max_region_lon
            )

            max_lat = min(
                current_lat + lat_step,
                max_region_lat
            )

            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Approximate distance from study centre
            north_distance = (
                (center_lat - lat) * 111320
            )

            east_distance = (
                (center_lon - lon)
                * 111320
                * cos_lat
            )

            distance = math.sqrt(
                north_distance ** 2 +
                east_distance ** 2
            )

            # Retain cells whose centres are inside the radius
            if distance <= radius_m:

                cells.append({
                    "id": f"cell_{row}_{col}",

                    "row": row,
                    "col": col,

                    "bbox": [
                        min_lon,
                        min_lat,
                        max_lon,
                        max_lat
                    ],

                    "center": {
                        "lat": round(center_lat, 6),
                        "lon": round(center_lon, 6)
                    },

                    "distance_from_center_m": round(
                        distance,
                        2
                    )
                })

            current_lon += lon_step
            col += 1

        current_lat += lat_step
        row += 1

    if not cells:
        raise ValueError(
            "No geographical grid cells generated."
        )

    return {
        "region": region,
        "cells": cells,
        "total_cells": len(cells),
        "cell_size_m": cell_size_m,
        "radius_m": radius_m
    }
