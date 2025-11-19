import matplotlib.pyplot as plt
import numpy as np

angles_deg = np.arange(0, 181, 30)
angles_rad = np.deg2rad(angles_deg)

# Each dataset represents one measurement series (e.g. color or condition)
dataset = {
    "Biały": {
        "left":  np.array([57, 35, 35, 75, 65, 20, 90]),  # left side of field
        "right": np.array([90, 66, 25, 22, 38, 20, 57]),  # right side of field
        "color": "black",
    },
    "Niebieski": {
        "left":  np.array([55, 44, 25, 40, 20, 30, 81]),
        "right": np.array([81, 16, 12, 12, 40, 40, 55]),
        "color": "tab:blue",
    },
    "Czerwony": {
        "left":  np.array([58, 50, 30, 28, 30, 32, 80]),
        "right": np.array([80, 32, 20, 20, 35, 25, 58]),
        "color": "tab:red",
    },
    "Zielony": {
        "left":  np.array([69, 22, 20, 20, 20, 28, 57]),
        "right": np.array([57, 22, 18, 20, 20, 22, 69]),
        "color": "tab:green",
    },
}

def plot_visual_field(datasets, r_error):

    thetas_full = np.concatenate([angles_rad, np.pi + angles_rad])

    for label, data in datasets.items():
        values_full = np.concatenate([data["right"], data["left"]])
        errors_full = np.full_like(values_full, r_error)

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_thetagrids(np.arange(0, 360, 30))
        ax.set_rgrids(np.arange(10, 100, 10))
        ax.set_rlim(0, 90)

        ax.errorbar(
            thetas_full, values_full, yerr=errors_full,
            fmt='-o', capsize=3, color=data["color"]
        )

        ax.set_title(f"Mapa pola widzenia obu oczu – {label}")
        plt.tight_layout()
        plt.show()


# Show all colors seperately with 5° angular error
plot_visual_field(dataset, r_error=5)