import math

import ee
import os
import requests
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

EE_PROJECT = os.getenv("EE_PROJECT", "elated-card-509409-c8")

GEE_INITIALIZED = False
GEE_ERROR = None


# ==================================================
# 1. EARTH ENGINE INITIALIZATION
# ==================================================

def initialize_earth_engine():
    global GEE_INITIALIZED, GEE_ERROR

    if GEE_INITIALIZED:
        return True

    try:
        ee.Initialize(project=EE_PROJECT)
        GEE_INITIALIZED = True
        GEE_ERROR = None
        return True

    except Exception as e:
        GEE_ERROR = str(e)
        print("Earth Engine initialization failed:", GEE_ERROR)
        return False


# ==================================================
# 2. REGION GEOMETRY
# ==================================================

def get_region(lat, lon, radius_m=5000):

    if not initialize_earth_engine():
        return None

    try:
        return ee.Geometry.Point(
            [float(lon), float(lat)]
        ).buffer(radius_m)

    except Exception as e:
        print("Region creation failed:", e)
        return None


# ==================================================
# 3. COMMON SATELLITE EXTRACTION FUNCTIONS
# ==================================================

def empty_result(reason="No observations available"):

    return {
        "value": None,
        "observations": 0,
        "status": "unavailable",
        "reason": reason
    }


def get_collection_metric(
    collection_id,
    band,
    region,
    start_date,
    end_date,
    scale,
    multiplier=1.0,
    offset=0.0
):
    """
    Extract an actual satellite collection mean.
    Missing observations are never replaced with random values.
    """

    if region is None:
        return empty_result("Study region unavailable")

    if not initialize_earth_engine():
        return empty_result("Earth Engine initialization failed")

    try:

        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select(band)
        )

        image_count = collection.size().getInfo()

        if not image_count:
            return empty_result(
                "No satellite images in selected date range"
            )

        image = (
            collection.mean()
            .multiply(multiplier)
            .add(offset)
        )

        result = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()

        value = result.get(band)

        if value is None:
            return empty_result(
                "No valid satellite pixels in study region"
            )

        return {
            "value": float(value),
            "observations": int(image_count),
            "status": "available",
            "reason": None
        }

    except Exception as e:

        print(
            f"Satellite extraction failed: "
            f"{collection_id}/{band}: {e}"
        )

        return empty_result(str(e))


# ==================================================
# 4. SENTINEL-5P AIR QUALITY
# ==================================================

AIR_PRODUCTS = {

    "no2": (
        "COPERNICUS/S5P/OFFL/L3_NO2",
        "tropospheric_NO2_column_number_density"
    ),

    "so2": (
        "COPERNICUS/S5P/OFFL/L3_SO2",
        "SO2_column_number_density"
    ),

    "co": (
        "COPERNICUS/S5P/OFFL/L3_CO",
        "CO_column_number_density"
    ),

    "o3": (
        "COPERNICUS/S5P/OFFL/L3_O3",
        "O3_column_number_density"
    ),

    "aerosol_index": (
        "COPERNICUS/S5P/OFFL/L3_AER_AI",
        "absorbing_aerosol_index"
    ),

    "hcho": (
        "COPERNICUS/S5P/OFFL/L3_HCHO",
        "tropospheric_HCHO_column_number_density"
    ),

    "ch4": (
        "COPERNICUS/S5P/OFFL/L3_CH4",
        "CH4_column_volume_mixing_ratio_dry_air_bias_corrected"
    )
}



def get_modis_aod_indicator(region, start_date, end_date):
    """Extract quality-screened 550 nm MAIAC aerosol optical depth."""

    if region is None or not initialize_earth_engine():
        return empty_result("Study region or Earth Engine unavailable")

    try:
        collection = (
            ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )

        image_count = collection.size().getInfo()
        if image_count == 0:
            return empty_result("No MODIS MAIAC observations in selected period")

        def prepare_aod(image):
            qa = image.select("AOD_QA")
            cloud_state = qa.bitwiseAnd(7)
            retrieval_quality = qa.rightShift(8).bitwiseAnd(15)

            # Keep clear pixels with the best AOD retrieval quality.
            valid = cloud_state.eq(1).And(retrieval_quality.eq(0))

            return (
                image.select("Optical_Depth_055")
                .multiply(0.001)
                .rename("AOD_055")
                .updateMask(valid)
            )

        composite = collection.map(prepare_aod).mean()
        result = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()

        value = result.get("AOD_055")
        if value is None:
            return empty_result("No valid quality-screened MAIAC AOD pixels")

        return {
            "value": float(value),
            "observations": int(image_count),
            "status": "available",
            "reason": None
        }

    except Exception as e:
        print("MODIS MAIAC AOD extraction failed:", e)
        return empty_result(str(e))




def get_ch4_indicator(region, start_date, end_date):
    """Extract Sentinel-5P methane using the product's valid-pixel mask."""

    if region is None or not initialize_earth_engine():
        return empty_result("Study region or Earth Engine unavailable")

    collection_id = "COPERNICUS/S5P/OFFL/L3_CH4"

    corrected_band = (
        "CH4_column_volume_mixing_ratio_dry_air_bias_corrected"
    )
    standard_band = (
        "CH4_column_volume_mixing_ratio_dry_air"
    )

    try:
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )

        image_count = collection.size().getInfo()

        if image_count == 0:
            return empty_result("No Sentinel-5P CH4 images in selected period")

        # Try the bias-corrected band, then the standard band.
        for band in [corrected_band, standard_band]:

            image = collection.select(band).mean()

            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=1113,
                maxPixels=10000000,
                bestEffort=True,
                tileScale=4
            ).getInfo()

            value = stats.get(band)

            print(
                f"CH4 extraction: {band} | "
                f"Images: {image_count} | Value: {value}"
            )

            if value is not None and math.isfinite(float(value)):
                return {
                    "value": float(value),
                    "observations": int(image_count),
                    "status": "available",
                    "reason": None
                }

        return empty_result("No valid CH4 pixels in the selected region")

    except Exception as e:
        print("Sentinel-5P CH4 extraction failed:", repr(e))
        return empty_result(str(e))



def get_viirs_fire_hotspots(region, start_date, end_date):
    """Count unique 1 km VIIRS pixels with low/nominal/high fire confidence."""

    if region is None or not initialize_earth_engine():
        return empty_result("Study region or Earth Engine unavailable")

    try:
        collection = (
            ee.ImageCollection("NASA/VIIRS/002/VNP14A1")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select("FireMask")
        )

        image_count = collection.size().getInfo()
        if image_count == 0:
            return empty_result("No VIIRS fire observations in selected period")

        # FireMask 7, 8 and 9 = low, nominal and high confidence.
        # max() counts each grid cell once across the selected date range.
        fire_pixels = (
            collection.map(
                lambda image: image.gte(7).rename("fire_pixel").unmask(0)
            )
            .max()
        )

        result = fire_pixels.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=1000,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()

        value = result.get("fire_pixel")
        if value is None:
            return empty_result("No valid VIIRS fire pixels in study region")

        return {
            "value": int(round(float(value))),
            "observations": int(image_count),
            "status": "available",
            "reason": None
        }

    except Exception as e:
        print("VIIRS fire extraction failed:", e)
        return empty_result(str(e))

def get_air_indicators(
    region,
    start_date,
    end_date,
    lat=None,
    lon=None
):
    """Retrieve quality-screened Sentinel-5P, MODIS MAIAC and VIIRS indicators."""

    air = {}

    for name, product in AIR_PRODUCTS.items():

        collection_id, band = product

        if name == "ch4":
            air[name] = get_ch4_indicator(
                region, start_date, end_date
            )
        else:
            air[name] = get_collection_metric(
                collection_id=collection_id,
                band=band,
                region=region,
                start_date=start_date,
                end_date=end_date,
                scale=1113
            )

    # Independent missions: MODIS for AOD and VIIRS for active fire pixels.
    air["aod"] = get_modis_aod_indicator(
        region, start_date, end_date
    )
    air["fire_hotspots"] = get_viirs_fire_hotspots(
        region, start_date, end_date
    )

    # Preserve every field consumed by the existing Flask API and frontend.
    output = {}

    for name, result in air.items():
        output[name] = result["value"]
        output[f"{name}_obs"] = result["observations"]
        output[f"{name}_status"] = result["status"]

    output["no2_column_mol_m2"] = air["no2"]["value"]
    output["no2_observations"] = air["no2"]["observations"]
    output["aerosol_obs"] = air["aerosol_index"]["observations"]
    output["aerosol_observations"] = air["aerosol_index"]["observations"]
    output["details"] = air
    output["source"] = "Sentinel-5P, MODIS MAIAC, VIIRS"

    return output


# ==================================================
# 5. SENTINEL-2 IMAGE PREPARATION
# ==================================================

def prepare_sentinel2(image):
    """Build standard Sentinel-2 indices and water-only spectral proxies."""

    scl = image.select("SCL")

    # Exclude no-data, saturated/defective, cloud shadow, cloud and snow.
    valid_pixels = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    image = image.updateMask(valid_pixels)

    # Sentinel-2 SR Harmonized reflectance scale factor = 0.0001.
    bands = image.select(
        ["B2", "B3", "B4", "B5", "B8", "B11", "B12"]
    ).multiply(0.0001)

    blue = bands.select("B2")
    green = bands.select("B3")
    red = bands.select("B4")
    red_edge_1 = bands.select("B5")
    nir = bands.select("B8")
    swir1 = bands.select("B11")
    swir2 = bands.select("B12")

    # LAND: established multispectral indices.
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi = (
        nir.subtract(red).multiply(2.5)
        .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
        .rename("EVI")
    )
    savi = (
        nir.subtract(red).multiply(1.5)
        .divide(nir.add(red).add(0.5))
        .rename("SAVI")
    )
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI")
    bsi = (
        swir1.add(red).subtract(nir.add(blue))
        .divide(swir1.add(red).add(nir).add(blue))
        .rename("BSI")
    )
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")

    # WATER EXTENT: retain these indices over all valid pixels for mapping.
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI")

    # Water-only mask from Sentinel-2 Scene Classification Layer class 6.
    water_mask = scl.eq(6)

    # NDTI turbidity screening index: (Red - Green) / (Red + Green).
    # This is a relative spectral proxy, not NTU without local calibration.
    turbidity = (
        red.subtract(green).divide(red.add(green))
        .rename("TURBIDITY_PROXY")
        .updateMask(water_mask)
    )

    # NDCI chlorophyll screening index: (B5 - B4) / (B5 + B4).
    # This is not chlorophyll-a concentration without regional calibration.
    chlorophyll = (
        red_edge_1.subtract(red).divide(red_edge_1.add(red))
        .rename("CHLOROPHYLL_PROXY")
        .updateMask(water_mask)
    )

    # Red-band reflectance is retained as a TSS spectral proxy only.
    # Conversion to mg/L requires site-specific field calibration.
    tss = red.rename("TSS_PROXY").updateMask(water_mask)

    # Floating Algae Index (FAI), used as a screening indicator near water.
    fai = (
        nir.subtract(
            red.add(
                swir1.subtract(red).multiply((842 - 665) / (1610 - 665))
            )
        )
        .rename("FAI")
    )

    indices = [
        ndvi, ndwi, mndwi, ndbi, bsi,
        evi, savi, ndmi, nbr,
        turbidity, chlorophyll, tss, fai
    ]

    return bands.addBands(indices)


SURFACE_BANDS = [
    "NDVI",
    "NDWI",
    "MNDWI",
    "NDBI",
    "BSI",
    "EVI",
    "SAVI",
    "NDMI",
    "NBR",
    "TURBIDITY_PROXY",
    "CHLOROPHYLL_PROXY",
    "TSS_PROXY",
    "FAI"
]


# ==================================================
# 6. SENTINEL-2 SURFACE INDICATORS
# ==================================================

def get_surface_indicators(
    region,
    start_date,
    end_date,
    lat=None,
    lon=None
):

    result = {}

    if region is None or not initialize_earth_engine():

        for band in SURFACE_BANDS:
            result[band] = empty_result(
                "Study region or Earth Engine unavailable"
            )

        return format_surface_results(result)

    try:

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE", 30
                )
            )
        )

        image_count = collection.size().getInfo()

        if image_count == 0:

            for band in SURFACE_BANDS:
                result[band] = empty_result(
                    "No cloud-filtered Sentinel-2 images available"
                )

            # Independent missions can still return values even when Sentinel-2
            # has no usable scene in the selected date range.
            supplementary = get_dynamic_world_indicators(
                region, start_date, end_date
            )
            supplementary["water_temp"] = get_water_temperature_indicator(
                region, start_date, end_date
            )
            for name in [
                "water_extent", "floating_veg", "algal_bloom",
                "oil_spill", "soil_erosion"
            ]:
                supplementary[name] = empty_result(
                    "Requires valid Sentinel-2 water/land pixels"
                )

            return format_surface_results(result, supplementary)

        composite = (
            collection
            .map(prepare_sentinel2)
            .median()
        )

        values = composite.select(
            SURFACE_BANDS
        ).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()

        for band in SURFACE_BANDS:

            value = values.get(band)

            if value is None:

                result[band] = empty_result(
                    "No valid pixels for this indicator"
                )

            else:

                result[band] = {
                    "value": float(value),
                    "observations": int(image_count),
                    "status": "available",
                    "reason": None
                }

    except Exception as e:

        print("Sentinel-2 extraction failed:", e)

        for band in SURFACE_BANDS:
            result[band] = empty_result(str(e))

    supplementary = get_supplementary_surface_indicators(
        region, start_date, end_date, composite
    ) if "composite" in locals() else {}

    return format_surface_results(result, supplementary)


def format_surface_results(results, supplementary=None):

    aliases = {
        "ndvi": "NDVI",
        "ndwi": "NDWI",
        "mndwi": "MNDWI",
        "ndbi": "NDBI",
        "bsi": "BSI",
        "evi": "EVI",
        "savi": "SAVI",
        "ndmi": "NDMI",
        "nbr": "NBR",
        "turbidity": "TURBIDITY_PROXY",
        "chlorophyll_a": "CHLOROPHYLL_PROXY",
        "tss": "TSS_PROXY",
        "fai": "FAI"
    }

    output = {}

    for alias, band in aliases.items():

        item = results[band]

        output[alias] = item["value"]
        output[f"{alias}_status"] = item["status"]

    # Derived classifications are deliberately not presented
    # as direct measurements.
    ndvi = output["ndvi"]
    ndbi = output["ndbi"]
    bsi = output["bsi"]

    if ndbi is not None and ndvi is not None and bsi is not None:

        if ndbi > 0.08:
            land_cover = "Urban / Impervious Built-up"

        elif ndvi > 0.4:
            land_cover = "Dense Vegetation"

        elif bsi > 0.1:
            land_cover = "Bare Soil / Disturbed Land"

        else:
            land_cover = "Mixed Urban & Open Terrain"

    else:
        land_cover = None

    output["land_cover"] = land_cover

    supplementary = supplementary or {}

    def supplemental_value(name):
        item = supplementary.get(name, empty_result("Not calculated"))
        return item.get("value")

    output["water_extent"] = supplemental_value("water_extent")
    output["floating_veg"] = supplemental_value("floating_veg")
    output["algal_bloom"] = supplemental_value("algal_bloom")
    output["oil_spill"] = supplemental_value("oil_spill")
    output["impervious_surface"] = supplemental_value("impervious_surface")
    output["soil_erosion"] = supplemental_value("soil_erosion")
    output["water_temp"] = supplemental_value("water_temp")
    output["flooded_vegetation"] = supplemental_value("flooded_vegetation")

    # Explicit units/interpretation metadata for the dashboard and API.
    output["metric_metadata"] = {
        "water_extent": "km2 detected water area from Sentinel-2 MNDWI screening",
        "floating_veg": "% of water-adjacent area flagged by FAI/NDVI screening",
        "algal_bloom": "% of detected water pixels above NDCI screening threshold; not toxin concentration",
        "oil_spill": "km2 of possible low-backscatter slick-like water pixels from Sentinel-1; not confirmed oil",
        "impervious_surface": "% of area with Dynamic World built probability > 0.5",
        "soil_erosion": "relative susceptibility score 0-100; not RUSLE tonnes/ha/year",
        "water_temp": "Landsat 8/9 water-pixel surface temperature estimate in Celsius",
        "flooded_vegetation": "% of area with Dynamic World flooded-vegetation probability > 0.5"
    }

    output["supplementary_details"] = supplementary

    output["observations"] = max(
        [
            results[b]["observations"]
            for b in SURFACE_BANDS
        ],
        default=0
    )

    output["details"] = results

    output["source"] = "Sentinel-2, Sentinel-1, Landsat 8/9, Dynamic World, CHIRPS, SRTM"

    return output


# ==================================================
# 6A. SUPPLEMENTARY WATER AND LAND INDICATORS
# ==================================================

def _metric_result(value, observations, reason=None):
    if value is None:
        return empty_result(reason or "No valid satellite pixels")
    return {
        "value": float(value),
        "observations": int(observations),
        "status": "available",
        "reason": None
    }


def get_dynamic_world_indicators(region, start_date, end_date):
    """
    Dynamic World built and flooded-vegetation probabilities.
    Outputs are probability/area indicators, not cadastral measurements.
    """
    empty = {
        "impervious_surface": empty_result("Dynamic World unavailable"),
        "flooded_vegetation": empty_result("Dynamic World unavailable")
    }

    if region is None or not initialize_earth_engine():
        return empty

    try:
        collection = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )
        count = collection.size().getInfo()
        if count == 0:
            return {
                "impervious_surface": empty_result("No Dynamic World observations"),
                "flooded_vegetation": empty_result("No Dynamic World observations")
            }

        composite = collection.select(["built", "flooded_vegetation"]).mean()

        # Probability > 0.5 is a conservative high-confidence class mask.
        built_probability = composite.select("built")
        built_area = (
            built_probability.gt(0.5)
            .multiply(ee.Image.pixelArea())
            .rename("built_area")
        )
        total_area = (
            ee.Image.pixelArea()
            .updateMask(built_probability.mask())
            .rename("total_area")
        )

        flooded_probability = composite.select("flooded_vegetation")
        flooded_area = (
            flooded_probability.gt(0.5)
            .multiply(ee.Image.pixelArea())
            .rename("flooded_veg_area")
        )

        stats = built_area.addBands(flooded_area).addBands(total_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=10,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()

        denominator = stats.get("total_area")
        if not denominator:
            return {
                "impervious_surface": empty_result("No valid Dynamic World pixels"),
                "flooded_vegetation": empty_result("No valid Dynamic World pixels")
            }

        built_pct = 100.0 * (stats.get("built_area") or 0) / denominator
        flooded_pct = 100.0 * (stats.get("flooded_veg_area") or 0) / denominator

        return {
            "impervious_surface": _metric_result(built_pct, count),
            "flooded_vegetation": _metric_result(flooded_pct, count)
        }

    except Exception as e:
        print("Dynamic World extraction failed:", e)
        return {
            "impervious_surface": empty_result(str(e)),
            "flooded_vegetation": empty_result(str(e))
        }


def get_water_temperature_indicator(region, start_date, end_date):
    """
    Landsat 8/9 surface temperature restricted to QA_PIXEL water pixels.
    This is a satellite skin-temperature estimate, not in-situ water temperature.
    """
    if region is None or not initialize_earth_engine():
        return empty_result("Study region or Earth Engine unavailable")

    try:
        def prepare_water_temperature(image):
            qa = image.select("QA_PIXEL")
            radsat = image.select("QA_RADSAT")

            clear = (
                qa.bitwiseAnd(1 << 0).eq(0)
                .And(qa.bitwiseAnd(1 << 1).eq(0))
                .And(qa.bitwiseAnd(1 << 2).eq(0))
                .And(qa.bitwiseAnd(1 << 3).eq(0))
                .And(qa.bitwiseAnd(1 << 4).eq(0))
                .And(qa.bitwiseAnd(1 << 5).eq(0))
                .And(radsat.eq(0))
            )
            water = qa.bitwiseAnd(1 << 7).neq(0)

            temp_c = (
                image.select("ST_B10")
                .multiply(0.00341802)
                .add(149.0)
                .subtract(273.15)
                .rename("WATER_TEMP_C")
            )
            return temp_c.updateMask(clear.And(water))

        l8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )
        l9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )
        collection = l8.merge(l9)
        count = collection.size().getInfo()
        if count == 0:
            return empty_result("No Landsat 8/9 observations in selected period")

        composite = collection.map(prepare_water_temperature).median()
        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()

        value = stats.get("WATER_TEMP_C")
        if value is None:
            return empty_result("No clear Landsat water pixels in study region")
        return _metric_result(value, count)

    except Exception as e:
        print("Water surface temperature extraction failed:", e)
        return empty_result(str(e))


def get_sentinel1_oil_slick_candidate(region, start_date, end_date, water_mask):
    """
    Conservative low-backscatter water-surface anomaly screen.
    This flags possible slick-like pixels only; it cannot confirm oil.
    """
    if region is None or not initialize_earth_engine():
        return empty_result("Study region or Earth Engine unavailable")

    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )
        count = collection.size().getInfo()
        if count == 0:
            return empty_result("No dual-polarization Sentinel-1 observations")

        # VV/VH are in dB in this Earth Engine collection.
        composite = collection.median()
        low_backscatter = (
            composite.select("VV").lt(-18)
            .And(composite.select("VH").lt(-25))
            .And(water_mask)
        )
        area = (
            low_backscatter.selfMask()
            .multiply(ee.Image.pixelArea())
            .rename("slick_candidate_area")
        )
        stats = area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=30,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()
        square_m = stats.get("slick_candidate_area")
        if square_m is None:
            return empty_result("No candidate slick-like water pixels")
        return _metric_result(float(square_m) / 1000000.0, count)

    except Exception as e:
        print("Sentinel-1 possible oil-slick extraction failed:", e)
        return empty_result(str(e))


def get_soil_erosion_susceptibility(region, start_date, end_date, s2_composite):
    """
    Relative erosion-susceptibility score (0-100), not RUSLE soil loss.
    Uses slope, exposed-soil proxy, vegetation cover and accumulated rainfall.
    CHIRPS daily rainfall is approximately 5.6 km, so this is a coarse contextual factor.
    """
    if region is None or not initialize_earth_engine():
        return empty_result("Study region or Earth Engine unavailable")

    try:
        dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
        slope = ee.Terrain.slope(dem).clamp(0, 45).divide(45)

        ndvi = s2_composite.select("NDVI").clamp(-1, 1)
        bsi = s2_composite.select("BSI").clamp(-1, 1)

        # Normalize exposed-soil proxy and vegetation protection to [0,1].
        bare_soil = bsi.add(0.2).divide(0.6).clamp(0, 1)
        low_cover = ee.Image(1).subtract(ndvi.add(1).divide(2)).clamp(0, 1)

        rainfall_collection = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        rain_count = rainfall_collection.size().getInfo()
        if rain_count == 0:
            return empty_result("No CHIRPS rainfall observations in selected period")

        rainfall_mm = rainfall_collection.sum()
        rainfall_factor = rainfall_mm.divide(1000).clamp(0, 1)

        score = (
            slope.multiply(bare_soil)
            .multiply(low_cover)
            .multiply(rainfall_factor)
            .multiply(100)
            .rename("EROSION_SUSCEPTIBILITY")
        )

        stats = score.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()
        value = stats.get("EROSION_SUSCEPTIBILITY")
        if value is None:
            return empty_result("No valid pixels for erosion susceptibility")
        return _metric_result(value, rain_count)

    except Exception as e:
        print("Soil erosion susceptibility extraction failed:", e)
        return empty_result(str(e))


def get_supplementary_surface_indicators(
    region, start_date, end_date, s2_composite
):
    """Calculate additional water/land metrics from appropriate satellite sources."""
    results = {}

    # Water mask from MNDWI, constrained by vegetation index to reduce false water.
    mndwi = s2_composite.select("MNDWI")
    ndvi = s2_composite.select("NDVI")
    water_mask = mndwi.gt(0).And(ndvi.lt(0.3))

    # Water extent in square kilometres and percent of the study region.
    try:
        water_area_image = (
            water_mask.selfMask()
            .multiply(ee.Image.pixelArea())
            .rename("water_area")
        )
        area_stats = water_area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=20,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()
        water_area = area_stats.get("water_area")
        if water_area is None:
            results["water_extent"] = empty_result("No water pixels detected")
        else:
            results["water_extent"] = _metric_result(
                float(water_area) / 1000000.0, 1
            )
    except Exception as e:
        results["water_extent"] = empty_result(str(e))

    # NDCI bloom screening: report percentage of detected water pixels above
    # a conservative screening threshold, not measured chlorophyll concentration.
    try:
        ndci = s2_composite.select("CHLOROPHYLL_PROXY")
        water_pixels = water_mask.selfMask()
        bloom_pixels = ndci.gt(0.1).And(water_mask)
        bloom_area = bloom_pixels.multiply(ee.Image.pixelArea()).rename("bloom_area")
        valid_water_area = water_pixels.multiply(ee.Image.pixelArea()).rename("valid_area")
        bloom_stats = bloom_area.addBands(valid_water_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=20,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()
        denominator = bloom_stats.get("valid_area")
        numerator = bloom_stats.get("bloom_area")
        if denominator and numerator is not None:
            results["algal_bloom"] = _metric_result(
                100.0 * float(numerator) / float(denominator), 1
            )
        else:
            results["algal_bloom"] = empty_result("No valid water pixels for bloom screening")
    except Exception as e:
        results["algal_bloom"] = empty_result(str(e))

    # Floating vegetation: FAI anomaly on/near water, supported by NDVI.
    # A 60 m expansion allows mixed shoreline/floating-cover pixels.
    try:
        nir = s2_composite.select("B8")
        red = s2_composite.select("B4")
        swir1 = s2_composite.select("B11")
        fai = nir.subtract(
            red.add(
                swir1.subtract(red).multiply((842 - 665) / (1610 - 665))
            )
        ).rename("FAI")
        water_near = water_mask.focal_max(radius=60, units="meters")
        vegetation_candidate = fai.gt(0).And(ndvi.gt(0.2)).And(water_near)
        veg_area = vegetation_candidate.multiply(ee.Image.pixelArea()).rename("floating_area")
        water_near_area = water_near.multiply(ee.Image.pixelArea()).rename("near_water_area")
        veg_stats = veg_area.addBands(water_near_area).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=20,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()
        denominator = veg_stats.get("near_water_area")
        numerator = veg_stats.get("floating_area")
        if denominator and numerator is not None:
            results["floating_veg"] = _metric_result(
                100.0 * float(numerator) / float(denominator), 1
            )
        else:
            results["floating_veg"] = empty_result("No valid water-adjacent pixels")
    except Exception as e:
        results["floating_veg"] = empty_result(str(e))

    # Additional satellite and modelled metrics.
    dynamic_world = get_dynamic_world_indicators(region, start_date, end_date)
    results["impervious_surface"] = dynamic_world["impervious_surface"]
    results["flooded_vegetation"] = dynamic_world["flooded_vegetation"]
    results["water_temp"] = get_water_temperature_indicator(
        region, start_date, end_date
    )
    results["oil_spill"] = get_sentinel1_oil_slick_candidate(
        region, start_date, end_date, water_mask
    )
    results["soil_erosion"] = get_soil_erosion_susceptibility(
        region, start_date, end_date, s2_composite
    )

    return results


# ==================================================
# 7. LANDSAT 8/9 LAND SURFACE TEMPERATURE
# ==================================================

def prepare_landsat(image):

    qa = image.select("QA_PIXEL")

    # Mask dilated cloud, cirrus, cloud, cloud shadow and snow.
    radsat = image.select("QA_RADSAT")

    # Mask fill, dilated cloud, cirrus, cloud, shadow, snow and saturation.
    mask = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
        .And(radsat.eq(0))
    )

    temperature = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST_C")
    )

    return temperature.updateMask(mask)


def get_lst_indicator(
    region,
    start_date,
    end_date,
    lat=None,
    lon=None
):

    unavailable = empty_result(
        "Landsat temperature unavailable"
    )

    if region is None or not initialize_earth_engine():
        return None, 0

    try:

        landsat8 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )

        landsat9 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterBounds(region)
            .filterDate(start_date, end_date)
        )

        collection = landsat8.merge(landsat9)

        count = collection.size().getInfo()

        if count == 0:
            return None, 0

        composite = (
            collection
            .map(prepare_landsat)
            .median()
        )

        values = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,
            maxPixels=10000000,
            bestEffort=True,
            tileScale=4
        ).getInfo()

        value = values.get("LST_C")

        if value is None:
            return None, 0

        return round(float(value), 2), int(count)

    except Exception as e:

        print("Landsat LST extraction failed:", e)

        return None, 0


# ==================================================
# 8. GRID CELL SATELLITE EXTRACTION
# ==================================================

def extract_collection_by_cells(
    collection_id,
    band,
    cells,
    start_date,
    end_date,
    scale,
    output_name=None
):
    """
    Extract satellite observations for each grid cell,
    preserving the native projection of the satellite product.
    """

    output_name = output_name or band

    results = {
        cell["id"]: empty_result("Not processed")
        for cell in cells
    }

    if not initialize_earth_engine():
        return results

    try:
        features = []

        for cell in cells:
            min_lon, min_lat, max_lon, max_lat = cell["bbox"]

            geometry = ee.Geometry.Rectangle(
                [min_lon, min_lat, max_lon, max_lat],
                proj="EPSG:4326",
                geodesic=False
            )

            features.append(
                ee.Feature(
                    geometry,
                    {"cell_id": cell["id"]}
                )
            )

        feature_collection = ee.FeatureCollection(features)

        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(feature_collection.geometry())
            .filterDate(start_date, end_date)
        )

        if collection_id == "COPERNICUS/S5P/OFFL/L3_CH4":
            collection = collection.filter(
                ee.Filter.eq("PRODUCT_QUALITY", "Nominal")
            )

        collection = collection.select(band)

        image_count = collection.size().getInfo()

        if image_count == 0:
            return {
                cell["id"]: empty_result(
                    "No satellite images in selected period"
                )
                for cell in cells
            }

        # Preserve the native projection of the source product.
        first_image = ee.Image(collection.first())
        native_projection = first_image.select(band).projection()

        composite = (
            collection
            .mean()
            .select(band)
            .rename(output_name)
        )

        reduced = composite.reduceRegions(
            collection=feature_collection,
            reducer=ee.Reducer.mean(),
            crs=native_projection,
            scale=scale,
            tileScale=4
        ).getInfo()

        for feature in reduced.get("features", []):
            properties = feature.get("properties", {})

            cell_id = properties.get("cell_id")
            value = properties.get(output_name)

            if value is None:
                value = properties.get("mean")

            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None

            if value is None or not math.isfinite(value):
                continue

            if cell_id not in results:
                continue

          
            results[cell_id] = {
                "value": float(value),
                "observations": int(image_count),
                "status": "available",
                "reason": None
            }

        # Diagnostic output for troubleshooting.
        first_properties = (
            reduced["features"][0]["properties"]
            if reduced.get("features")
            else {}
        )

        print(
            f"Grid extraction: {collection_id}/{band} | "
            f"Images: {image_count} | "
            f"Reducer properties: {list(first_properties.keys())}"
        )

        return results

    except Exception as e:
        print(
            f"Grid extraction failed for {collection_id}/{band}: {e}"
        )

        return {
            cell["id"]: empty_result(str(e))
            for cell in cells
        }



def extract_surface_by_cells(
    cells,
    start_date,
    end_date
):

    results = {
        cell["id"]: {}
        for cell in cells
    }

    if not initialize_earth_engine():
        return results

    try:

        features = []

        for cell in cells:

            min_lon, min_lat, max_lon, max_lat = cell["bbox"]

            geometry = ee.Geometry.Rectangle(
                [min_lon, min_lat, max_lon, max_lat],
                geodesic=False
            )

            features.append(
                ee.Feature(
                    geometry,
                    {"cell_id": cell["id"]}
                )
            )

        feature_collection = ee.FeatureCollection(features)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(feature_collection.geometry())
            .filterDate(start_date, end_date)
            .filter(
                ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)
            )
        )

        image_count = collection.size().getInfo()

        if image_count == 0:

            for cell in cells:

                results[cell["id"]] = {
                    band: empty_result(
                        "No cloud-filtered Sentinel-2 images"
                    )
                    for band in SURFACE_BANDS
                }

            return results

        composite = (
            collection
            .map(prepare_sentinel2)
            .median()
            .select(SURFACE_BANDS)
        )

        reduced = composite.reduceRegions(
            collection=feature_collection,
            reducer=ee.Reducer.mean(),
            scale=100,
            tileScale=4
        ).getInfo()

        for feature in reduced["features"]:

            properties = feature["properties"]
            cell_id = properties.get("cell_id")

            if cell_id is None:
                continue

            for band in SURFACE_BANDS:

                value = properties.get(band)

                if value is None:

                    results[cell_id][band] = empty_result(
                        "No valid pixels in this cell"
                    )

                else:

                    results[cell_id][band] = {
                        "value": float(value),
                        "observations": int(image_count),
                        "status": "available",
                        "reason": None
                    }

        return results

    except Exception as e:

        print("Sentinel-2 grid extraction failed:", e)

        for cell in cells:

            results[cell["id"]] = {
                band: empty_result(str(e))
                for band in SURFACE_BANDS
            }

        return results


# ==================================================
# 9. EXTRACT ALL INDICATORS FOR EVERY GRID CELL
# ==================================================

def extract_grid_observations(
    grid,
    start_date,
    end_date
):
    """
    Main function for spatial satellite analysis.

    Returns real observations for every geographical cell.
    """

    cells = grid["cells"]

    if not cells:
        raise ValueError("No grid cells supplied.")

    if not initialize_earth_engine():

        return {
            "status": "unavailable",
            "reason": GEE_ERROR,
            "total_cells": len(cells),
            "cells": []
        }

    all_results = {}

    # Sentinel-5P products
    for name, (collection_id, band) in AIR_PRODUCTS.items():

        all_results[name] = extract_collection_by_cells(
            collection_id,
            band,
            cells,
            start_date,
            end_date,
            scale=1113,
            output_name=name
        )

    # Sentinel-2 products
    surface_results = extract_surface_by_cells(
        cells,
        start_date,
        end_date
    )

    # Landsat temperature
    lst_results = extract_collection_by_cells(
        "LANDSAT/LC08/C02/T1_L2",
        "ST_B10",
        cells,
        start_date,
        end_date,
        scale=100,
        output_name="LST_RAW"
    )

    # Merge Landsat 9 observations as well.
    lst9_results = extract_collection_by_cells(
        "LANDSAT/LC09/C02/T1_L2",
        "ST_B10",
        cells,
        start_date,
        end_date,
        scale=100,
        output_name="LST_RAW"
    )

    output_cells = []

    for cell in cells:

        cell_id = cell["id"]

        air = {}

        for name in AIR_PRODUCTS:

            air[name] = all_results[name][cell_id]

        surface = surface_results[cell_id]

        land = {
            band: surface.get(
                band,
                empty_result("Not available")
            )
            for band in SURFACE_BANDS
        }

        # Landsat ST_B10 scale and offset.
        # Combine LC08 and LC09 using the available observations.
        lst8 = lst_results[cell_id]
        lst9 = lst9_results[cell_id]

        lst8_value = lst8["value"]
        lst9_value = lst9["value"]

        lst_values = [
            value for value in [lst8_value, lst9_value]
            if value is not None
        ]

        if lst_values:

            raw_mean = sum(lst_values) / len(lst_values)

            lst_celsius = (
                raw_mean * 0.00341802
                + 149.0
                - 273.15
            )

            land["LST_C"] = {
                "value": round(lst_celsius, 2),
                "observations": (
                    lst8["observations"]
                    + lst9["observations"]
                ),
                "status": "available",
                "reason": None
            }

        else:

            land["LST_C"] = empty_result(
                "No valid Landsat temperature observations"
            )

        output_cells.append({

            "cell_id": cell_id,

            "bbox": cell["bbox"],

            "center": cell["center"],

            "air": air,

            "water": {
                name: surface.get(name, empty_result())
                for name in [
                    "NDWI",
                    "MNDWI",
                    "TURBIDITY_PROXY",
                    "CHLOROPHYLL_PROXY",
                    "TSS_PROXY"
                ]
            },

            "land": land

        })

    available_cells = sum(
        1 for cell in output_cells
        if any(
            item["status"] == "available"
            for domain in ["air", "water", "land"]
            for item in cell[domain].values()
        )
    )

    return {

        "status": (
            "available"
            if available_cells > 0
            else "unavailable"
        ),

        "total_cells": len(output_cells),

        "available_cells": available_cells,

        "missing_cells": len(output_cells) - available_cells,

        "period": {
            "start": start_date,
            "end": end_date
        },

        "cells": output_cells,

        "sources": [
            "Sentinel-5P",
            "Sentinel-2",
            "Sentinel-1",
            "Landsat 8",
            "Landsat 9",
            "Dynamic World",
            "CHIRPS",
            "SRTM"
        ]

    }


# ==================================================
# 10. EXISTING LOCATION ANALYSIS API
# ==================================================

def analyze_location(
    lat,
    lon,
    start_date=None,
    end_date=None,
    radius_m=5000
):

    if end_date is None:
        end_date = date.today()

    if start_date is None:
        start_date = end_date - timedelta(days=90)

    start_date = str(start_date)
    end_date = str(end_date)

    region = get_region(lat, lon, radius_m)

    air = get_air_indicators(
        region,
        start_date,
        end_date
    )

    surface = get_surface_indicators(
        region,
        start_date,
        end_date
    )

    lst, lst_obs = get_lst_indicator(
        region,
        start_date,
        end_date
    )

    return {

        "location": {
            "latitude": lat,
            "longitude": lon,
            "radius_m": radius_m
        },

        "period": {
            "start": start_date,
            "end": end_date
        },

        "air": {

            "no2": air["no2"],
            "so2": air["so2"],
            "co": air["co"],
            "o3": air["o3"],
            "aerosol_index": air["aerosol_index"],
            "aod": air["aod"],
            "ch4": air["ch4"],
            "hcho": air["hcho"],
            "fire_hotspots": air["fire_hotspots"],

            "no2_obs": air["no2_obs"],
            "so2_obs": air["so2_obs"],
            "co_obs": air["co_obs"],
            "o3_obs": air["o3_obs"],
            "aerosol_obs": air["aerosol_obs"],

            "no2_column_mol_m2": air["no2"],

            "no2_observations": air["no2_observations"],

            "source": air["source"],

            "details": air["details"]
        },

        "water": {

            "ndwi": surface["ndwi"],
            "mndwi": surface["mndwi"],
            "turbidity": surface["turbidity"],
            "tss": surface["tss"],
            "chlorophyll_a": surface["chlorophyll_a"],

            "water_temp": surface["water_temp"],

            "algal_bloom": surface["algal_bloom"],
            "floating_veg": surface["floating_veg"],
            "water_extent": surface["water_extent"],
            "oil_spill": surface["oil_spill"],

            "observations": surface["observations"],

            "flooded_vegetation": surface["flooded_vegetation"],

            "metric_metadata": surface["metric_metadata"],

            "source": surface["source"],

            "details": surface["details"]
        },

        "land": {

            "ndvi": surface["ndvi"],
            "evi": surface["evi"],
            "savi": surface["savi"],
            "ndbi": surface["ndbi"],
            "bsi": surface["bsi"],
            "ndmi": surface["ndmi"],
            "nbr": surface["nbr"],

            "lst": lst,

            "land_cover": surface["land_cover"],
            "impervious_surface": surface["impervious_surface"],
            "soil_erosion": surface["soil_erosion"],

            "observations": surface["observations"],

            "lst_observations": lst_obs,

            "source": "Sentinel-2 and Landsat 8/9"
        }
    }

