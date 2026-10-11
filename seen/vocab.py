"""Shared label vocabulary. Lanes must emit these strings (add new ones here via the integrator)."""

VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle")
PERSON = "person"
PLATE = "plate"
VIN = "vin"
WHEEL = "wheel"
HOOD_OPEN = "hood_open"
LIFT = "lift"

# Damage detections use the "damage:" prefix, tyres use "tyre:".
DAMAGE_TYPES = ("dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat", "other")
TYRE_CONDITIONS = ("good", "worn", "defective")
DOT = "dot"  # tyre DOT date code; Det.text holds the 4 digits (WWYY)

# Bay states (must match contracts/events.schema.json#/$defs/BayState)
BAY_STATES = (
    "car_present", "raised", "wheel_off", "hood_open",
    "person_at_car", "person_under_car", "work", "idle", "washing",
)

# Router scenes (must match contracts Route.scene)
SCENES = ("gate", "bay", "wash", "street", "closeup", "motorcycle", "unknown")

# Work-order operations (must match contracts/workorder.schema.json Operation.code)
OPERATIONS = ("car_in_bay", "lift_raise", "wheel_removal", "hood_open", "under_car_work", "person_at_car", "wash")


def damage(kind: str) -> str:
    return f"damage:{kind}"


def tyre(cond: str) -> str:
    return f"tyre:{cond}"
