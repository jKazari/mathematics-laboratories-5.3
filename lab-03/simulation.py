import numpy as np
from math import isinf
from scipy.optimize import minimize_scalar

# -----------------------------------------------------------
# Adjustable parameters
# -----------------------------------------------------------

# Refractive indices
n_air = 1.000
n_plexi = 1.491
n_water = 1.333  # initial guess; change or fit below

# Geometry (mm)
R_outer = 30.0
R_inner = 27.0
wall_thickness = 3.0
partition_thickness = 2.0
cavity_thickness = 26.0

# Measured focal lengths (mm)
f_measured = {
    "empty": -500.0,
    "one_filled": 132.0,
    "both_filled": 55.0
}

# -----------------------------------------------------------
# Paraxial tracing
# -----------------------------------------------------------

def refract(n1, n2, R, s):
    """Return image distance s' after refraction at spherical surface."""
    if isinf(R):
        # Flat surface
        if s == np.inf:
            return np.inf
        return (n2 / n1) * s
    if s == np.inf:
        return n2 * R / (n2 - n1)
    return 1 / ((n2 - n1) / (n2 * R) - n1 / (n2 * s))

def trace_system(surfaces):
    """Trace paraxial rays for object at infinity, return focal length."""
    s = np.inf  # object at infinity
    for i, surf in enumerate(surfaces):
        n1, n2, R, t = surf["n1"], surf["n2"], surf["R"], surf["t"]
        s_prime = refract(n1, n2, R, s)
        # For last surface: focal length = s' in final medium
        if i == len(surfaces) - 1:
            return s_prime
        # Otherwise translate to next vertex
        s = t - s_prime
    return None  # should not reach

# -----------------------------------------------------------
# Define optical systems
# -----------------------------------------------------------

def build_systems(n_water):
    """Return dict of three systems (empty, one_filled, both_filled)."""
    systems = {}

    # Empty container (both cavities air)
    systems["empty"] = [
        {"n1": n_air, "n2": n_plexi, "R": +R_outer, "t": wall_thickness},
        {"n1": n_plexi, "n2": n_air, "R": +R_inner, "t": cavity_thickness},
        {"n1": n_air, "n2": n_plexi, "R": np.inf, "t": partition_thickness},
        {"n1": n_plexi, "n2": n_air, "R": np.inf, "t": cavity_thickness},
        {"n1": n_air, "n2": n_plexi, "R": -R_inner, "t": wall_thickness},
        {"n1": n_plexi, "n2": n_air, "R": -R_outer, "t": 0},
    ]

    # One compartment filled (plano-convex)
    systems["one_filled"] = [
        {"n1": n_air, "n2": n_plexi, "R": +R_outer, "t": wall_thickness},
        {"n1": n_plexi, "n2": n_air, "R": +R_inner, "t": cavity_thickness},
        {"n1": n_air, "n2": n_plexi, "R": np.inf, "t": partition_thickness},
        {"n1": n_plexi, "n2": n_water, "R": np.inf, "t": cavity_thickness},
        {"n1": n_water, "n2": n_plexi, "R": -R_inner, "t": wall_thickness},
        {"n1": n_plexi, "n2": n_air, "R": -R_outer, "t": 0},
    ]

    # Both compartments filled (biconvex)
    systems["both_filled"] = [
        {"n1": n_air, "n2": n_plexi, "R": +R_outer, "t": wall_thickness},
        {"n1": n_plexi, "n2": n_water, "R": +R_inner, "t": cavity_thickness},
        {"n1": n_water, "n2": n_plexi, "R": np.inf, "t": partition_thickness},
        {"n1": n_plexi, "n2": n_water, "R": np.inf, "t": cavity_thickness},
        {"n1": n_water, "n2": n_plexi, "R": -R_inner, "t": wall_thickness},
        {"n1": n_plexi, "n2": n_air, "R": -R_outer, "t": 0},
    ]

    return systems

# -----------------------------------------------------------
# Evaluation and fitting
# -----------------------------------------------------------

def predict_focals(n_water):
    systems = build_systems(n_water)
    focals = {name: trace_system(sys) for name, sys in systems.items()}
    return focals

def error_function(n_water):
    focals = predict_focals(n_water)
    err = 0.0
    for key, f_meas in f_measured.items():
        f_model = focals[key]
        # Use relative error weighting
        err += ((abs(f_model) - abs(f_meas)) / f_meas) ** 2
    return err

# -----------------------------------------------------------
# Run & display
# -----------------------------------------------------------

if __name__ == "__main__":
    # Single evaluation
    focals = predict_focals(n_water)
    print(f"Predicted focal lengths for n_water = {n_water:.4f}:")
    for name, f in focals.items():
        print(f"  {name:12s}: {f:8.2f} mm")

    # Fit n_water to measured focals (minimize total error)
    res = minimize_scalar(error_function, bounds=(1.0, 1.6), method="bounded")
    print("\nBest-fit water index:", res.x)
    focals_fit = predict_focals(res.x)
    for name, f in focals_fit.items():
        print(f"  {name:12s}: {f:8.2f} mm (measured {f_measured[name]:.1f})")
