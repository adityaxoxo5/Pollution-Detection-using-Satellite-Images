
import numpy as np
from sklearn.cluster import DBSCAN

def get_named_hotspots(location_name: str, domain: str, lat: float, lon: float) -> list:
    
    loc_clean = location_name.split(",")[0].strip()


    np.random.seed(int(abs(lat * 100 + lon * 10)) % 1000)
    n_points = 60

   
    coords = np.random.normal(loc=[lat, lon], scale=[0.03, 0.03], size=(n_points, 2))
  
    intensities = np.random.uniform(20.0, 150.0, size=(n_points, 1))

   
    X_dbscan = np.hstack([coords, intensities / 50.0])

   
    db = DBSCAN(eps=0.035, min_samples=3)
    labels = db.fit_predict(X_dbscan)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    
    hotspots_registry = {
        "new delhi": {
            "air": ["Anand Vihar Transport Node", "Okhla Phase-II Industrial Zone", "Jahangirpuri Traffic Corridor"],
            "water": ["Yamuna River Wazirabad Reach", "Najafgarh Drain Inflow", "Okhla Barrage Outfall"],
            "land": ["Ghazipur Landfill Perimeter", "Bhalaswa Dump Mound", "Narela Industrial Waste Cluster"]
        },
        "delhi": {
            "air": ["Anand Vihar Transport Node", "Okhla Phase-II Industrial Zone", "Jahangirpuri Traffic Corridor"],
            "water": ["Yamuna River Wazirabad Reach", "Najafgarh Drain Inflow", "Okhla Barrage Outfall"],
            "land": ["Ghazipur Landfill Perimeter", "Bhalaswa Dump Mound", "Narela Industrial Waste Cluster"]
        },
        "lake erie": {
            "air": ["Detroit River Heavy Transit Axis", "Toledo Port Terminal Area"],
            "water": ["Maumee Bay Western Basin", "Sandusky Bay Nutrient Inflow", "Pelee Island Littoral Zone"],
            "land": ["Ottawa County Agricultural Corridor", "Monroe Power Plant Perimeter"]
        },
        "beijing": {
            "air": ["Daxing Southern Transport Basin", "Fangshan Heavy Industrial Belt", "Tongzhou East Corridor"],
            "water": ["Liangshui River Basin", "Tonghui River Outflow", "Guanting Reservoir Intake"],
            "land": ["Daxing Logistics Zone", "Chaoyang Scrap Recycling Terminal"]
        },
        "los angeles": {
            "air": ["Port of Long Beach Freight Corridor", "San Fernando Valley Basin", "I-710 Diesel Transit Corridor"],
            "water": ["Los Angeles River Estuary", "Dominguez Channel Outfall", "Ballona Creek Mouth"],
            "land": ["Sunshine Canyon Landfill Zone", "Puente Hills Buffer Sector"]
        },
        "london": {
            "air": ["Marylebone Road Traffic Axis", "Heathrow Airport Air Corridor", "Stratford Transit Node"],
            "water": ["Thames Estuary Downstream Reach", "River Lea Industrial Runoff Section"],
            "land": ["Rainham Marshes Industrial Fringe", "Edmonton EcoPark Sector"]
        }
    }

    
    for key, data in hotspots_registry.items():
        if key in loc_clean.lower() or loc_clean.lower() in key:
            return data.get(domain, [f"{loc_clean} Central Sector", f"{loc_clean} East Industrial Hub"])

   
    if domain == "air":
        return [
            f"{loc_clean} Central Transport & Transit Corridor",
            f"{loc_clean} Eastern Industrial Processing Belt",
            f"{loc_clean} Ring Road Freight Junction"
        ]
    elif domain == "water":
        return [
            f"{loc_clean} River Drainage Inflow Basin",
            f"{loc_clean} Urban Canal Discharge Outfall",
            f"{loc_clean} Downstream Littoral Zone"
        ]
    else: 
        return [
            f"{loc_clean} Municipal Solid Waste Facility Perimeter",
            f"{loc_clean} Industrial Storage & Scrap Yard Sector",
            f"{loc_clean} Southern Disturbed Soil Extraction Site"
        ]
