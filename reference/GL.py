import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Constants
CONV = 0.0980665  # kg/cm^2 to MPa

# --- HIGH-PRECISION SAMPLING (kg/cm^2) ---

# Curve I: Initial Extension (Rupture at ~40% ext)
x1 = np.array([0, 5, 10, 15, 20, 30, 38, 40, 45, 60, 80, 115, 150, 180])
y1 = np.array(
    [0, 6.0, 9.8, 11.8, 12.8, 13.6, 14.1, 14.2, 13.8, 13.5, 13.6, 14.2, 14.9, 15.6]
)

# Curve II: Retraction (Unloading path)
# Note: This is lower than III due to hysteresis
x2 = np.array([115, 100, 85, 70, 55, 40, 25, 15, 5, 0])
y2 = np.array([14.2, 12.8, 11.5, 10.2, 8.8, 7.2, 5.0, 3.0, 0.8, 0])

# Curve III: Second Extension (Reloading path)
# Closely follows Curve II but slightly stiffer
x3 = np.array([0, 5, 15, 25, 40, 55, 70, 85, 100, 115])
y3 = np.array([0, 1.2, 3.8, 5.8, 8.0, 9.5, 10.8, 12.0, 13.2, 14.2])


# --- INTERPOLATION FOR SMOOTHNESS ---
def get_smooth_data(x, y):
    f = interp1d(x, y, kind="quadratic")  # Quadratic for smoother natural rubber paths
    x_new = np.linspace(x.min(), x.max(), 500)
    return x_new, f(x_new)


ext1, stress1 = get_smooth_data(x1, y1 * CONV)
ext2, stress2 = get_smooth_data(x2, y2 * CONV)
ext3, stress3 = get_smooth_data(x3, y3 * CONV)

# --- PLOTTING ---
plt.figure(figsize=(10, 7), facecolor="white")
plt.plot(ext1, stress1, color="black", label="I: First Extension", linewidth=2.5)
plt.plot(
    ext2, stress2, color="black", linestyle="--", label="II: Retraction", linewidth=1.2
)
plt.plot(
    ext3,
    stress3,
    color="black",
    linestyle=":",
    label="III: Second Extension",
    linewidth=1.5,
)

# Axis Styling
plt.title("Digitized Figure 2 (Improved Accuracy)", fontsize=13)
plt.xlabel("Extension (%)", fontsize=11)
plt.ylabel("Tensile Stress (MPa)", fontsize=11)
plt.xlim(0, 180)
plt.ylim(0, 1.6)
plt.grid(True, linestyle="-", alpha=0.2)
plt.legend(frameon=False)

plt.tight_layout()
plt.show()
