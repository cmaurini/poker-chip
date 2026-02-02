"""
Experimental data from Gent and Lindley (1959)
"""

import numpy as np

GL_fig2_x = np.array(
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

GL_fig2_y_Kgcm2 = np.array(
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

Kgcm2_to_MPa = 0.0980665
GL_fig2_y_MPa = GL_fig2_y_Kgcm2 * Kgcm2_to_MPa

# correction term due to the machine compliance
D_GL = 20.0  # mm
thickness_GL = 1.9  # mm
H_GL = thickness_GL / 2  # mm
R_GL = D_GL / 2
surface = np.pi * (D_GL / 2) ** 2
stiffness_KT_oscar = 1930.0  # (N / m)
ET = stiffness_KT_oscar / surface * (H_GL * 2)
GL_fig2_x_corrected = GL_fig2_x - GL_fig2_y_MPa / ET
mu_GL = 0.59  # MPa, fitted value from Gent and Lindley data E=18kg/cm^2, mu=6kg/cm^2=0.59MPa
kappa_GL = 1000.0  # MPa, assumed value

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    from formulas_paper import Eeq_3d

    # plot original and corrected data

    plt.figure(figsize=(8, 6))
    plt.plot(GL_fig2_x, GL_fig2_y_MPa, "-", label="Original data")
    # plt.plot(
    #    GL_fig2_x_corrected,
    #    GL_fig2_y_MPa / mu_GL,
    #    "--",
    #    label="Corrected data",
    #    color="gray",
    # )
    # add analytical estimate for comparison of the curve with Eeq_3d

    eta_vals = np.linspace(0.01, 1.0, 100)
    # Eeq_3d_vals = Eeq_3d(mu0=mu_GL / 2, kappa0=kappa_GL, #H=H_GL, R=R_GL) * eta_vals
    # plt.plot(
    #    eta_vals,
    #    Eeq_3d_vals,
    #    "k--",
    #    label="Eeq_3d analytical",
    # )
    # add incompressible estimate
    from formulas_paper import (
        Eeq_3d_inc,
        Eeq_3d_inc_exam,
        p_max_3d_inc,
    )

    mu_GL = 0.2  # MPa, fitted value from Gent and Lindley data E=18kg/cm^2, mu=6kg/cm^2=0.59MPa
    Eeq_3d_inc_vals = Eeq_3d_inc(mu0=mu_GL / 2, H=H_GL, R=R_GL) * eta_vals
    p_max_3d_inc_vals = p_max_3d_inc(mu0=mu_GL, Delta=eta_vals * H_GL, H=H_GL, R=R_GL)
    plt.plot(
        eta_vals,
        Eeq_3d_inc_vals,
        "k:",
        label="Eeq_3d incompressible",
    )
    plt.plot(
        eta_vals,
        Eeq_3d_inc_exam(mu0=mu_GL, H=H_GL, R=R_GL) * eta_vals,
        "g:",
        label="Eeq_3d incompressible exam",
    )
    plt.plot(
        eta_vals,
        p_max_3d_inc_vals,
        "y-",
        label="p_max_3d incompressible",
    )

    plt.plot(
        eta_vals,
        5 / 2 + eta_vals,
        ":",
        color="gray",
    )

    plt.hlines(y=2.5, xmin=0, xmax=1.0, color="gray", linestyle="--")
    plt.vlines(x=0.1, ymin=0, ymax=5, color="gray", linestyle="--")
    plt.legend()
    plt.xlabel("Extension (%)")
    plt.ylabel(r"Tensile Stress (MPa) / $\mu$")
    plt.title("Gent and Lindley (1959)")
    plt.ylim(0, 5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outdir = "figures"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, "gent_lindley_plot.pdf"), dpi=300)
    plt.close()
