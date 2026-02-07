import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar
import numpy as np
import os

# Try importing your local references, otherwise warn or mock if needed
try:
    from reference.formulas_paper import equivalent_modulus, max_pressure
except ImportError:
    import sys
    from pathlib import Path

    # Attempt to find the path relative to this file
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from reference.formulas_paper import equivalent_modulus, max_pressure
    except ImportError:
        print(
            "Warning: Could not import 'equivalent_modulus' or 'max_pressure'. Plotting will fail."
        )


class GentLindleyData:
    """
    Class to collect Gent-Lindley (GL) data and provide tools for visualization and fitting.
    """

    CONVERSION_FACTOR = 0.0980665
    ASPECT_RATIO_GL = 2.0 / 0.19
    IDX_BREAK = 50  # End of Branch I
    IDX_BREAK2 = 64  # End of Branch II

    def __init__(self, *, x_all=None, y_kgcm2=None):
        print(f"Gent-Lindley aspect ratio: {self.ASPECT_RATIO_GL:.2f}")

        # FIX: Updated to new matplotlib API (or use plt.cm.viridis)
        self.colormap_plot = plt.cm.viridis

        # Default data if not provided
        if x_all is None:
            x_all = np.array(
                [
                    0.0086,
                    0.0173,
                    0.0328,
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
        if y_kgcm2 is None:
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

        self.x_all = x_all
        # FIX: Removed duplicate assignment of self.y_mpa
        self.y_mpa = y_kgcm2 * self.CONVERSION_FACTOR

        self.points_I = np.column_stack(
            (self.x_all[0 : self.IDX_BREAK + 1], self.y_mpa[0 : self.IDX_BREAK + 1])
        )
        self.points_II = np.column_stack(
            (
                self.x_all[self.IDX_BREAK : self.IDX_BREAK2 + 1],
                self.y_mpa[self.IDX_BREAK : self.IDX_BREAK2 + 1],
            )
        )
        points_III = np.column_stack(
            (self.x_all[self.IDX_BREAK2 + 1 :], self.y_mpa[self.IDX_BREAK2 + 1 :])
        )
        if points_III.shape[0] > 0:
            last_point_II = self.points_II[-1]
            points_III = np.vstack((last_point_II, points_III))
        self.points_III = points_III

        self.f_I = self.create_branch_spline(self.points_I[:, 0], self.points_I[:, 1])
        self.f_II = self.create_branch_spline(
            self.points_II[:, 0], self.points_II[:, 1]
        )
        self.f_III = self.create_branch_spline(
            self.points_III[:, 0], self.points_III[:, 1]
        )

        self.colors = [self.colormap_plot(i) for i in np.linspace(0.15, 0.85, 3)]

        # CRITICAL FIX: Changed max(points_I[:, 1]) to max(points_I[:, 0])
        # Using index 1 (Y-axis) for X-range caused massive distortion.
        self.x_ranges = [
            np.linspace(min(self.points_I[:, 0]), max(self.points_I[:, 0]), 300),
            np.linspace(self.points_II[0, 0], self.points_II[-1, 0], 300),
            np.linspace(self.points_III[0, 0], self.points_III[-1, 0], 300),
        ]

        self.slope_Ib, (self.m_Ib, self.b_Ib) = self.estimate_slope(
            self.points_I[:, 0], self.points_I[:, 1], x_min=0.2, x_max=0.3
        )
        self.slope_I, (self.m_I, self.b_I) = self.estimate_slope(
            self.points_I[:, 0], self.points_I[:, 1]
        )
        self.slope_II, (self.m_II, self.b_II) = self.estimate_slope(
            self.points_II[:, 0], self.points_II[:, 1], x_min=0.03, x_max=0.15
        )
        self.slope_III, (self.m_III, self.b_III) = self.estimate_slope(
            self.points_III[:, 0], self.points_III[:, 1], x_min=0.03, x_max=0.15
        )

    @staticmethod
    def create_branch_spline(x_segment, y_segment):
        sort_idx = np.argsort(x_segment)
        xs, ys = x_segment[sort_idx], y_segment[sort_idx]
        unique_x, indices = np.unique(xs, return_inverse=True)
        unique_y = np.bincount(indices, weights=ys) / np.bincount(indices)
        return PchipInterpolator(unique_x, unique_y, extrapolate=True)

    @staticmethod
    def estimate_slope(x, y, x_min=0, x_max=0.1):
        mask = (x >= x_min) & (x <= x_max)
        if np.sum(mask) < 2:
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

    def print_slopes(self):
        print("Estimated slopes:")
        print(f"Branch I: {self.slope_I:.3f} MPa")
        print(f"Branch II: {self.slope_II:.3f} MPa")
        print(f"Branch III: {self.slope_III:.3f} MPa")
        print(f"Branch I (0.2-0.3): {self.slope_Ib:.3f} MPa")

    def get_branch_functions(self):
        return self.f_I, self.f_II, self.f_III

    def get_branch_slopes(self):
        return {
            "I": (self.slope_I, self.m_I, self.b_I),
            "II": (self.slope_II, self.m_II, self.b_II),
            "III": (self.slope_III, self.m_III, self.b_III),
            "Ib": (self.slope_Ib, self.m_Ib, self.b_Ib),
        }

    def get_colors(self):
        return self.colors

    def get_x_ranges(self):
        return self.x_ranges

    def get_points(self):
        return self.points_I, self.points_II, self.points_III

    def get_raw_data(self):
        return self.x_all, self.y_mpa

    def plot_branches_and_fits(self, save_path="GL_branches_and_fits.png"):
        f_I, f_II, f_III = self.get_branch_functions()
        x_ranges = self.get_x_ranges()
        colors = self.get_colors()

        m_I, b_I = self.m_I, self.b_I
        m_II, b_II = self.m_II, self.b_II
        m_III, b_III = self.m_III, self.b_III
        m_Ib, b_Ib = self.m_Ib, self.b_Ib
        slope_I, slope_II, slope_III, slope_Ib = (
            self.slope_I,
            self.slope_II,
            self.slope_III,
            self.slope_Ib,
        )

        plt.figure(figsize=(8, 4))
        plt.plot(
            x_ranges[0],
            f_I(x_ranges[0]),
            color=colors[0],
            label="Branch I",
            linewidth=2,
        )
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
        # FIX: Increased limit to 1.5 to include all data (max data is ~1.42)
        plt.xlim(0, 1.5)
        plt.xlabel(r"$\Delta/H$", fontsize=12)
        plt.ylabel(r"$F/S \,\mathrm{{(MPa)}}$", fontsize=12)
        plt.grid(True, alpha=0.3)

        mu_for_plot = 0.36
        kappa_for_plot = 36
        p_c_for_plot = 5 / 2 * mu_for_plot

        plt.title(
            rf"Gent-Lindley Data and Linear Approximations for $\mu={mu_for_plot:.2f}$, $\kappa={kappa_for_plot:.0f}$",
            fontsize=14,
        )

        xs = np.linspace(0, 1.5, 200)

        # Plot theoretical Equivalent Modulus
        # NOTE: This depends on external import 'equivalent_modulus'
        try:
            y_eq = (
                equivalent_modulus(
                    mu0=mu_for_plot,
                    H=1.0,
                    R=self.ASPECT_RATIO_GL,
                    geometry="3d",
                    compressible=True,
                    kappa0=kappa_for_plot,
                )
                * xs
            )

            plt.plot(
                xs,
                y_eq,
                color="gray",
                label=rf"$E_\mathrm{{eq}}$ for $\mu={mu_for_plot:.2f}$",
                linewidth=3,
                alpha=0.4,
                linestyle="-",
            )
        except NameError:
            pass

        # Plot Uniaxial Strain
        # FIX: Corrected label typo "Uxial" -> "Uniaxial"
        plt.plot(
            xs,
            p_c_for_plot + (2 / 3) * mu_for_plot * xs,
            color="gray",
            label="Uniaxial strain",
            linewidth=3,
            alpha=0.4,
            linestyle="--",
        )

        # FIX: Removed redundant loop that re-plotted the exact same lines.
        # If you need to plot multiple Mu values, add the loop back here iterating over a list.

        # Calculate Critical Pressure
        def pressure_residual(Delta):
            # NOTE: Depends on external import 'max_pressure'
            try:
                val = max_pressure(
                    mu0=1,
                    Delta=Delta,
                    H=1,
                    R=self.ASPECT_RATIO_GL,
                    geometry="3d",
                    compressible=True,
                    kappa0=kappa_for_plot,
                )
                return val - (5 / 2)
            except NameError:
                return 1  # Fallback to avoid crash if import missing

        # Only attempt root finding if imports succeeded
        if "max_pressure" in globals():
            sol_pressure = root_scalar(
                pressure_residual, bracket=[-0.01, 0.2], method="bisect"
            )
            if sol_pressure.converged:
                Delta_critical = sol_pressure.root
                plt.axvline(
                    Delta_critical,
                    color="black",
                    linestyle=":",
                    linewidth=2,
                    label=f"Critical loading ($\Delta/H={Delta_critical:.2f}$)",
                )

        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    gl_data = GentLindleyData()
    gl_data.print_slopes()
    gl_data.plot_branches_and_fits()
