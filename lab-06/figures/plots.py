import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.arange(-50, 51, 10)
y1 = np.array([28.93, 31.93, 33.27, 35.90, 34.90, 39.30, 39.30, 38.00, 32.40, 29.13, 25.47])
y2 = np.array([33.17, 34.27, 34.33, 32.67, 42.33, 41.13, 37.83, 36.33, 33.20, 30.80, 31.30])

# Vertical error bars (for y-values)
y1_err = np.array([1.32, 1.42, 2.14, 1.92, 0.76, 0.82, 2.26, 0.93, 1.09, 0.58, 2.29])
y2_err = np.array([3.87, 1.82, 2.24, 0.64, 1.88, 0.79, 1.64, 2.02, 0.93, 1.31, 3.48])

# Fit quadratic trend lines
coef1 = np.polyfit(x, y1, 2)
coef2 = np.polyfit(x, y2, 2)

p1 = np.poly1d(coef1)
p2 = np.poly1d(coef2)

x_fit = np.linspace(-50, 50, 300)

# Define consistent colors
color_left = 'tab:blue'
color_right = 'tab:red'

plt.figure(figsize=(10, 7))

# Scatter + error bars
# plt.errorbar(x, y1, yerr=y1_err, xerr=3, fmt='o', 
#              color=color_left, label="Lewe oko", capsize=0)

plt.errorbar(x, y2, yerr=y2_err, xerr=3, fmt='s', 
             color=color_right, label="Prawe oko", capsize=0)

# Trend lines with matching colors but different style
# plt.plot(x_fit, p1(x_fit), color=color_left, linestyle='--')
plt.plot(x_fit, p2(x_fit), color=color_right, linestyle='--')

plt.xlabel("Kąt widzenia [°]")
plt.ylabel("Częstotliwość progowa [Hz]")

plt.xticks(np.arange(-50, 51, 10))
plt.grid(linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()
