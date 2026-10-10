"""
Tools for interacting with Farm Database, Soil Sensors, and Weather Services.

Demonstrates:
- Tool Use / Function Calling
- ReAct Loop (Action -> Observation)
- Exception Handling & Recovery (Simulated network failure with cache fallback)
"""

import os
import json
from typing import Dict, Any, Optional

# Load local farm mock database
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "farm_data.json")

def load_farm_database() -> Dict[str, Any]:
    try:
        with open(DATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"farms": {}, "cached_weather": {}}


# Global toggle to demonstrate tool failure and recovery once
_SIMULATED_WEATHER_FAIL_TRIGGERED = False


def get_farm_details(farm_id: str = "farm_01") -> Dict[str, Any]:
    """
    Retrieves static profile for GreenLeaf Farm.
    """
    db = load_farm_database()
    return db.get("farms", {}).get(farm_id, {
        "name": "GreenLeaf Farm",
        "crop": "Tomato",
        "soil_type": "Loamy Sand",
        "target_moisture_range": [40, 70]
    })


def get_soil_data(farm_id: str = "farm_01") -> Dict[str, Any]:
    """
    TOOL: Soil Sensor Telemetry
    Retrieves live soil moisture, pH, and nitrogen levels.
    """
    farm = get_farm_details(farm_id)
    sensors = farm.get("current_sensors", {
        "moisture": 28,
        "ph": 6.4,
        "nitrogen_ppm": 32
    })
    return {
        "farm_id": farm_id,
        "moisture": sensors.get("moisture", 28),
        "ph": sensors.get("ph", 6.4),
        "nitrogen_ppm": sensors.get("nitrogen_ppm", 32),
        "status": "warning" if sensors.get("moisture", 28) < 30 else "optimal"
    }


def get_weather_data(location: str = "GreenLeaf Farm", simulate_failure: bool = True) -> Dict[str, Any]:
    """
    TOOL: Live Weather Station API
    Demonstrates Exception Handling & Recovery.
    Simulates a transient 503 error on live API, catches it, and recovers via cached telemetry.
    """
    global _SIMULATED_WEATHER_FAIL_TRIGGERED
    db = load_farm_database()

    # Demonstrate simulated failure and recovery
    if simulate_failure and not _SIMULATED_WEATHER_FAIL_TRIGGERED:
        _SIMULATED_WEATHER_FAIL_TRIGGERED = True
        print("    [TOOL ERROR] Live Weather Station API timed out (503 Service Unavailable).")
        print("    [RECOVERY] Fallback mechanism engaged: Retrieving cached satellite weather telemetry...")
        cached = db.get("cached_weather", {}).get(location, {
            "temperature": 34,
            "humidity": 68,
            "rain_probability": 10,
            "heat_stress_alert": True
        })
        cached["_source"] = "cached_fallback_station"
        cached["_recovered_from_error"] = "Live API timeout"
        return cached

    # Normal fetch
    cached = db.get("cached_weather", {}).get(location, {
        "temperature": 34,
        "humidity": 68,
        "rain_probability": 10,
        "heat_stress_alert": True
    })
    cached["_source"] = "live_weather_service"
    return cached
