"""
Visualize and fit the Gent-Lindley (GL) data.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar
from formulas_paper import equivalent_modulus, max_pressure

colormap_plot = cm.get_cmap("viridis")
aspect_ratio_GL = 2.0 / 0.19
print(f"Gent-Lindley aspect ratio: {aspect_ratio_GL:.2f}")
# --- 1. RAW DATA ---
x_all = np.array(
    [
        0.0086,
        0.0173,
        0.0328,
        # Plot splines with slope in legend
        0.0691,
        0.1157,
        0.1295,
        0.1295,
        0.1416,
        0.1416,
        0.1520,
        0.1520,
        0.1727,
        0.1727,
        0.1900,
        0.1900,
        0.2193,
        0.2193,
        0.2539,
        0.2867,
        0.3092,
        0.3247,
        0.3247,
        0.3420,
        0.3420,
        0.3610,
        0.3765,
        0.3921,
        0.3921,
        0.4041,
        0.4041,
        0.4162,
        0.4352,
        0.4525,
        0.4663,
        0.4905,
        0.5164,
        0.5354,
        0.5648,
        0.5786,
        0.5993,
        0.6321,
        0.6580,
        0.6857,
        0.7098,
        0.7323,
        0.7565,
        0.8100,
        0.8756,
        0.9603,
        1.0725,
        1.1503,
        0.9568,
        0.8256,
        0.4715,
        0.4007,
        0.3558,
        0.3040,
        0.2642,
        0.2314,
        0.2055,
        0.1813,
        0.1468,
        0.1123,
        0.0777,
        0.0259,
        0.1140,
        0.1606,
        0.1813,
        0.2124,
        0.2504,
        0.2988,
        0.3523,
        0.4439,
        0.5078,
        0.6252,
        0.6252,
        0.8187,
        1.1554,
        1.4180,
    ]
)

# Conversion factor: 1 Kg/cm2 = 0.0980665 MPa
y_kgcm2 = np.array(
    [
        0.2718,
        1.0873,
        2.4465,
        5.3130,
        8.8221,
        9.7858,
        9.7858,
        10.4283,
        10.4283,
        10.8979,
        10.8979,
        11.3180,
        11.3180,
        11.7381,
        11.7381,
        12.1334,
        12.1334,
        12.6030,
        13.0231,
        13.2702,
        13.4679,
        13.4679,
        13.6409,
        13.6409,
        13.8386,
        13.9868,
        14.0610,
        14.0610,
        13.9374,
        13.9374,
        13.8880,
        13.7891,
        13.8138,
        13.6903,
        13.6656,
        13.6161,
        13.5173,
        13.5173,
        13.4926,
        13.3443,
        13.3690,
        13.3937,
        13.3937,
        13.3690,
        13.3196,
        13.3443,
        13.3690,
        13.4679,
        13.6161,
        13.8386,
        14.0362,
        12.5288,
        11.7133,
        9.9588,
        9.4893,
        9.0445,
        8.3773,
        7.6853,
        6.8204,
        6.0049,
        5.1153,
        3.7809,
        2.3970,
        1.2850,
        -0.1483,
        3.8303,
        5.6096,
        6.5486,
        7.3641,
        8.1796,
        8.8962,
        9.5140,
        10.1565,
        10.5272,
        11.1203,
        11.1203,
        11.9605,
        13.2702,
        14.2834,
    ]
)

CONVERSION_FACTOR = 0.0980665
y_mpa = y_kgcm2 * CONVERSION_FACTOR


# %%
# --- 2. INTERPOLATION ---


def create_branch_spline(x_segment, y_segment):
    sort_idx = np.argsort(x_segment)
    xs, ys = x_segment[sort_idx], y_segment[sort_idx]
    unique_x, indices = np.unique(xs, return_inverse=True)
    unique_y = np.bincount(indices, weights=ys) / np.bincount(indices)
    return PchipInterpolator(unique_x, unique_y, extrapolate=True)


# Re-segmented with MPa values
idx_break = (
    50  # 0-based index, so x_all[50] is the last of Branch I and first of Branch II
)
idx_break2 = 64  # end of Branch II

points_I = np.column_stack(
    (x_all[0 : idx_break + 1], y_mpa[0 : idx_break + 1])
)  # 0 to 50 inclusive
points_II = np.column_stack(
    (x_all[idx_break : idx_break2 + 1], y_mpa[idx_break : idx_break2 + 1])
)  # 50 to 64 inclusive

# Prepend the last point of Branch II to the points of Branch III
points_III = np.column_stack((x_all[idx_break2 + 1 :], y_mpa[idx_break2 + 1 :]))
if points_III.shape[0] > 0:
    last_point_II = points_II[-1]
    points_III = np.vstack((last_point_II, points_III))

f_I = create_branch_spline(points_I[:, 0], points_I[:, 1])
f_II = create_branch_spline(points_II[:, 0], points_II[:, 1])
f_III = create_branch_spline(points_III[:, 0], points_III[:, 1])
# --- 3. PLOTTING ---
# Use viridis colormap for three branches
colormap_plot = plt.cm.viridis  # RdPu
colors = [colormap_plot(i) for i in np.linspace(0.15, 0.85, 3)]

# %%

x_ranges = [
    np.linspace(min(points_I[:, 0]), max(points_I[:, 1]), 300),
    np.linspace(points_II[0, 0], points_II[-1, 0], 300),
    np.linspace(points_III[0, 0], points_III[-1, 0], 300),
]


# Estimate slope in region 0 to 0.1 for each branch
def estimate_slope(x, y, x_min=0, x_max=0.1):
    mask = (x >= x_min) & (x <= x_max)
    if np.sum(mask) < 2:
        # Fallback: use first and third point if possible
        if len(x) >= 3:
            dx = x[2] - x[0]
            dy = y[2] - y[0]
            if dx != 0:
                slope = dy / dx
                intercept = y[0] - slope * x[0]
                print(
                    f"[WARN] Fallback slope (pt1-pt3): {slope:.3f} (x0={x[0]:.3f}, x2={x[2]:.3f})"
                )
                return slope, (slope, intercept)
        print("[WARN] Not enough points to estimate slope.")
        return np.nan, (np.nan, np.nan)
    x_fit = x[mask]
    y_fit = y[mask]
    coefs = np.polyfit(x_fit, y_fit, 1)
    slope = coefs[0]
    intercept = coefs[1]
    return slope, (slope, intercept)


# %%
slope_Ib, (m_Ib, b_Ib) = estimate_slope(
    points_I[:, 0], points_I[:, 1], x_min=0.2, x_max=0.3
)
print(f"Estimated slope in 0.2-0.3 for Branch I: {slope_Ib:.3f} MPa")
print(
    f"Estimated crossing with x axis in 0.2-0.3 for Branch I: {m_Ib:.3f}, {b_Ib:.3f} MPa"
)

# %%
# Branch I
slope_I, (m_I, b_I) = estimate_slope(points_I[:, 0], points_I[:, 1])
# Branch II
slope_II, (m_II, b_II) = estimate_slope(
    points_II[:, 0], points_II[:, 1], x_min=0.03, x_max=0.15
)
# Branch III
slope_III, (m_III, b_III) = estimate_slope(
    points_III[:, 0], points_III[:, 1], x_min=0.03, x_max=0.15
)  # Focus on 0.1 to 0.2 for Branch III
print(f"Estimated slopes:")
print(f"{slope_I:.3f}")
print(f"{slope_II:.3f}")
print(f"{slope_III:.3f}")
# %%
# Plot splines
plt.figure(figsize=(8, 4))
plt.plot(x_ranges[0], f_I(x_ranges[0]), color=colors[0], label="Branch", linewidth=2)
plt.plot(
    x_ranges[1],
    f_II(x_ranges[1]),
    color=colors[1],
    label="Branch II",
    linewidth=2,
)
plt.plot(
    x_ranges[2],
    f_III(x_ranges[2]),
    color=colors[2],
    label="Branch III",
    linewidth=2,
)
# plt.plot(x_all, y_mpa, "o", color="k", label="Original Data", markersize=4, alpha=0.7)


# Plot linear approximations as dashed lines with slope in legend
linear_fits = [
    (m_I, b_I, colors[0], slope_I, "I", 0, 0.4),
    (m_II, b_II, colors[1], slope_II, "II", 0.03, 0.4),
    (m_III, b_III, colors[2], slope_III, "III", 0.03, 0.4),
    (m_Ib, b_Ib, colors[0], slope_Ib, "Ib", 0.0, 0.5),
]
for m, b, color, slope, branch, x_start, x_end in linear_fits:
    x_lin = np.linspace(x_start, x_end, 100)
    y_lin = m * x_lin + b
    plt.plot(
        x_lin,
        y_lin,
        linestyle="--",
        color=color,
        label=f"Branch {branch} Linear Slope: {slope:.2f}",
        linewidth=1.5,
        alpha=0.7,
    )
plt.ylim(-0.1, 1.5)
plt.xlim(0, 1.2)
plt.xlabel(r"$\Delta/H$", fontsize=12)
plt.ylabel(r"$F/S \,\mathrm{{(MPa)}}$", fontsize=12)

plt.grid(True, alpha=0.3)


print(f"Estimated slopes:")
print(f"{slope_I:.3f} MPa")
print(f"{slope_II:.3f} MPa")
print(f"{slope_III:.3f} MPa")

# Quick Check
print(f"Yield Point (Branch I) Stress: {f_I(0.3921):.2f} MPa")
#

# plot the equivalent modulus curve for the estimated mu
mu_for_plot = 0.36  # 0.21
kappa_for_plot = 36  # 100
p_c_for_plot = 5 / 2 * mu_for_plot
plt.title(
    rf"Gent-Lindley Data and Linear Approximations for $\mu={mu_for_plot:.2f}$, $\kappa={kappa_for_plot:.0f}$",
    fontsize=14,
)

xs = np.linspace(0, 1.0, 200)
plt.plot(
    xs,
    (
        equivalent_modulus(
            mu0=mu_for_plot,
            H=1.0,
            R=aspect_ratio_GL,
            geometry="3d",
            compressible=True,
            kappa0=kappa_for_plot,
        )
        * xs
    ),
    color="gray",
    label=rf"$E_\mathrm{{eq}}$ for $\mu={mu_for_plot:.2f}$",
    linewidth=3,
    alpha=0.4,
    linestyle="-",
)
plt.plot(
    xs,
    p_c_for_plot + (2 / 3) * mu_for_plot * xs,
    color="gray",
    label=rf"Uxial strain",
    linewidth=3,
    alpha=0.4,
    linestyle="-",
)


# plot the equivalent modulus curve for the estimated mu

# Use 'cool' colormap for the modulus curves
cool_cmap = plt.cm.cool
mu_values = [mu_for_plot]  # np.linspace(0.21, 0.49, 5)
for i, mu_for_plot in enumerate(mu_values):
    if len(mu_values) > 1:
        color = cool_cmap(i / (len(mu_values) - 1))
    else:
        color = "gray"
    p_c_for_plot = 5 / 2 * mu_for_plot
    xs = np.linspace(0, 1.0, 200)
    plt.plot(
        xs,
        (
            equivalent_modulus(
                mu0=mu_for_plot,
                H=1.0,
                R=aspect_ratio_GL,
                geometry="3d",
                compressible=True,
                kappa0=kappa_for_plot,
            )
            * xs
        ),
        label=rf"$E_\mathrm{{eq}}$ for $\mu={mu_for_plot:.2f}$",
        linewidth=3,
        alpha=0.3,
        linestyle="-",
        color=color,
    )
    plt.plot(
        xs,
        p_c_for_plot + (2 / 3) * mu_for_plot * xs,
        # label=rf"Uxial strain",
        linewidth=3,
        alpha=0.3,
        linestyle="-",
        color=color,
    )
# Find the loading (Delta/H) where max_pressure = 5/2 * mu for mu=0.21


def pressure_residual(Delta):
    obj = max_pressure(
        mu0=1,
        Delta=Delta,
        H=1,
        R=aspect_ratio_GL,
        geometry="3d",
        compressible=True,
        kappa0=kappa_for_plot,
    ) - (5 / 2)
    print(f"Delta: {Delta:.4f}, Residual: {obj:.4e}")
    return obj


sol_pressure = root_scalar(pressure_residual, bracket=[-0.01, 0.2], method="bisect")

if sol_pressure.converged:
    Delta_critical = sol_pressure.root
    print(f"Critical loading (Delta/H) for max pressure = 5/2 mu: {Delta_critical:.4f}")
    Delta_critical_inc = 10 / 3 * (0.01)
    print(
        f"Critical loading (Delta/H) for max pressure = 5/2 mu: {Delta_critical_inc:.4f}"
    )
    plt.axvline(
        Delta_critical,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Critical loading ($\Delta/H={Delta_critical:.2f}$)",
    )
else:
    print("[ERROR] Could not find critical loading for max pressure.")
plt.legend()
plt.tight_layout()
plt.show()
# %%
