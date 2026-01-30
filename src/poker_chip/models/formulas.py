import scipy.special
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

plt.style.use("science")


GL_fig2_x = np.array(
    [
        0.008635578583765112,
        0.017271157167530225,
        0.03281519861830742,
        0.0690846286701209,
        0.1157167530224525,
        0.12953367875647667,
        0.12953367875647667,
        0.14162348877374784,
        0.14162348877374784,
        0.15198618307426598,
        0.15198618307426598,
        0.17271157167530224,
        0.17271157167530224,
        0.18998272884283246,
        0.18998272884283246,
        0.21934369602763384,
        0.21934369602763384,
        0.2538860103626943,
        0.2867012089810017,
        0.309153713298791,
        0.3246977547495682,
        0.3246977547495682,
        0.34196891191709844,
        0.34196891191709844,
        0.3609671848013817,
        0.3765112262521589,
        0.3920552677029361,
        0.3920552677029361,
        0.40414507772020725,
        0.40414507772020725,
        0.4162348877374784,
        0.43523316062176165,
        0.45250431778929184,
        0.46632124352331605,
        0.4905008635578584,
        0.5164075993091537,
        0.5354058721934369,
        0.5647668393782384,
        0.5785837651122625,
        0.5993091537132987,
        0.6321243523316062,
        0.6580310880829016,
        0.6856649395509499,
        0.7098445595854922,
        0.7322970639032815,
        0.7564766839378237,
        0.8100172711571675,
        0.8756476683937824,
        0.9602763385146804,
        1.072538860103627,
        1.1502590673575128,
        0.9568221070811744,
        0.8255613126079447,
        0.4715025906735751,
        0.4006908462867012,
        0.3557858376511226,
        0.30397236614853196,
        0.26424870466321243,
        0.231433506044905,
        0.20552677029360966,
        0.18134715025906734,
        0.1468048359240069,
        0.11226252158894645,
        0.07772020725388601,
        0.025906735751295335,
        0.11398963730569947,
        0.16062176165803108,
        0.18134715025906734,
        0.21243523316062174,
        0.2504317789291882,
        0.2987910189982729,
        0.35233160621761656,
        0.4438687392055268,
        0.5077720207253886,
        0.625215889464594,
        0.625215889464594,
        0.8186528497409326,
        1.155440414507772,
        1.4179620034542313,
    ]
)

GL_fig2_y_Kgcm2 = np.array(
    [
        0.27182866556836904,
        1.0873146622734762,
        2.4464579901153214,
        5.313014827018122,
        8.822075782537068,
        9.785831960461286,
        9.785831960461286,
        10.42833607907743,
        10.42833607907743,
        10.897858319604612,
        10.897858319604612,
        11.317957166392093,
        11.317957166392093,
        11.738056013179571,
        11.738056013179571,
        12.1334431630972,
        12.1334431630972,
        12.602965403624383,
        13.023064250411862,
        13.270181219110379,
        13.467874794069193,
        13.467874794069193,
        13.640856672158156,
        13.640856672158156,
        13.838550247116968,
        13.98682042833608,
        14.060955518945635,
        14.060955518945635,
        13.937397034596376,
        13.937397034596376,
        13.887973640856673,
        13.789126853377265,
        13.813838550247118,
        13.69028006589786,
        13.665568369028007,
        13.616144975288304,
        13.517298187808896,
        13.517298187808896,
        13.492586490939045,
        13.344316309719934,
        13.369028006589787,
        13.393739703459637,
        13.393739703459637,
        13.369028006589787,
        13.319604612850084,
        13.344316309719934,
        13.369028006589787,
        13.467874794069193,
        13.616144975288304,
        13.838550247116968,
        14.036243822075782,
        12.528830313014828,
        11.71334431630972,
        9.958813838550247,
        9.489291598023065,
        9.044481054365733,
        8.377265238879737,
        7.685337726523888,
        6.820428336079078,
        6.0049423393739705,
        5.115321252059308,
        3.7808896210873146,
        2.397034596375618,
        1.28500823723229,
        -0.1482701812191104,
        3.830313014827018,
        5.6095551894563425,
        6.548599670510709,
        7.364085667215816,
        8.179571663920923,
        8.896210873146623,
        9.514003294892916,
        10.156507413509061,
        10.527182866556837,
        11.120263591433279,
        11.120263591433279,
        11.960461285008238,
        13.270181219110379,
        14.2833607907743,
    ]
)

Kgcm2_to_MPa = 0.0980665
GL_fig2_y_MPa = GL_fig2_y_Kgcm2 * Kgcm2_to_MPa

# correction term due to the machine compliance
D_GL = 20.0
H_GL = 1.9
surface = np.pi * (D_GL / 2) ** 2
stiffness_KT_oscar = 1930.0  # (N / m)
ET = stiffness_KT_oscar / surface * H_GL
GL_fig2_x_corrected = GL_fig2_x - GL_fig2_y_MPa / ET


def t_lim_3d(sigma_bar_k, mu, L, H, eta_bar):
    t_lim = sigma_bar_k / (
        mu
        * (L / 2)
        / ((H / 2) ** 2)
        * (
            1.0 / eta_bar
            - (
                scipy.special.jv(
                    0, complex(0, 2 / L) * np.sqrt(3) * np.sqrt(eta_bar) * 0
                )
            )
            / (
                eta_bar
                * scipy.special.jv(
                    0, complex(0, 1) * np.sqrt(3) * np.sqrt(eta_bar)
                )
            )
        )
    )


def t_lim_2d():
    eta_bar = (lmbda / (3 * mu)) ** (1 / 2) * H / L
    t_lim = L * 2 * sigmap_0 / (4 * lmbda) / (1 - 1 / np.cosh(1 / eta_bar))


def pressure_3d(lmbda, mu, T, R, Delta, r=0):
    term1 = lmbda * Delta / T
    term2 = 1 - scipy.special.jv(
        0, 1j * np.sqrt(3 * mu / lmbda) * r / T
    ) / scipy.special.jv(0, 1j * np.sqrt(3 * mu / lmbda) * R / T)
    return np.real(term1 * term2)


def sigma_zz(r, z, D, H, mu, delta_u=1.0):
    R = D / 2
    H2 = H / 2
    Delta = delta_u / 2.0
    return (
        (3.0 / 4.0)
        * mu
        * Delta
        / (4 * H2**3)
        * (R**22 - r**2 + 2 * z**2 + 14.0 / 3.0 * H**2)
    )


def force(D, H, mu, delta_u=1.0):
    R = D / 2
    H2 = H / 2
    Delta = delta_u / 2.0
    return (
        (3 * np.pi * mu * Delta * R**2)
        / (8 * H2**3)
        * (1 + 16.0 / 3.0 * H2**2 / R**2)
    )


def equivalent_modulus(D, H, mu):
    k_equiv = force(D=D, H=H, mu=mu, delta_u=H) / (np.pi * D**2 / 4)
    return k_equiv


def pressure_2d(lmbda, mu, T, R, Delta, r=0):
    term1 = lmbda * Delta / T
    term2 = 1 - np.cosh(np.sqrt(3 * mu / lmbda) * r / T) / np.cosh(
        np.sqrt(3 * mu / lmbda) * R / T
    )
    return term1 * term2


def pressure_2d_incompressible(mu, T, R, Delta, x_1=0, x_2=0):
    return 3 * mu * Delta / (2 * T**2) * (R**2 - x_1**2 + T**2 + x_2**2)


def sigma_2d_incompressible(mu, T, Delta, x_1=0, x_2=0):
    factor = 3 * mu * Delta / (2 * T**2)
    sigma_22 = factor * (1 - x_1**2 + 3 * T**2 - x_2**2)
    sigma_11 = factor * (1 - 3 * x_1**2 + T**2 - x_2**2)
    sigma_12 = 2 * factor * (x_1 * x_2)
    return sigma_11, sigma_22, sigma_12


def sigma_3d_incompressible(mu, T, R, Delta, r=0, z=0):
    factor = 3 * mu * Delta / (4 * T**2)
    sigma_rr = factor * (R**2 - r**2 + 4 * z**2 - 3 * T**2 / 3)
    sigma_zz = factor * (R**2 - r**2 - 2 - 2 * z**2 + 14 / 3 * T**2)
    sigma_rz = 2 * factor * (r * z)
    sigma_theta_theta = sigma_rr
    return sigma_rr, sigma_zz, sigma_rz


def pressure_3d_incompressible(mu, T, R, Delta, r=0, z=0):
    return (
        3.0 / 4.0 * mu * Delta / T**3 * (R**2 - r**2 + 2 * T**2 / 3 + 2 * z**2)
    )


def force_3d(lmbda, mu, T, R, Delta):
    term1 = Delta * np.pi * R**2 * lmbda
    term2 = 1 - scipy.special.hyp0f1(
        2, (3 * R**2 * mu) / (4 * T**2 * lmbda)
    ) / scipy.special.hyp0f1(1, (3 * R**2 * mu) / (4 * T**2 * lmbda))
    return (term1 * term2) / T


def force_2d_incompressible(lmbda, mu, T, R, Delta):
    D = 2 * R
    return 2 * mu * Delta * D / (T**2) * (2 / 3 + 2 * T**2)


def force_3d_incompressible(mu, T, R, Delta):
    D = 2 * R
    return (
        3
        / 8
        * np.pi
        * mu
        * Delta
        * R**4
        / T**3
        * (1.0 + 0 * 16 / 3 * T**2 / D**2)
    )


def force_2d(lmbda, mu, T, R, Delta):
    term1 = (Delta * R * lmbda) / T
    term2 = (
        Delta
        * lmbda
        * np.sqrt(lmbda / mu)
        * np.tanh((np.sqrt(3) * R * np.sqrt(mu / lmbda)) / T)
    ) / np.sqrt(3)
    return 2 * (term1 - term2)


def equivalent_modulus_2d(lmbda, mu, T, R):
    """
    Equivalent modulus for an almost incompressible disk in 2d plane-strain model

    Parameters
    ----------
    lmbda : float
        Lame's first parameter
    mu : float
        Lame's second parameter
    T : float
        half-thickness of the material (total thickness is 2T)
    R : float
        Radius of the disk, total width is 2R
    """
    k_equiv = force_2d(lmbda, mu, T, R, Delta=T) / (2 * R)
    return k_equiv


def equivalent_modulus_3d(lmbda, mu, T, R):
    """
    Equivalent modulus for a 3D almost incompressible disk

    Parameters
    ----------
    lmbda : float
        Lame's first parameter
    mu : float
        Lame's second parameter
    T : float
        half-thickness of the material (total thickness is 2T)
    R : float
        Radius of the disk,  diameter is D=2R
    """
    k_equiv = force_3d(lmbda, mu, T, R, Delta=T) / (np.pi * R**2)
    return k_equiv


def equivalent_modulus_3d_incompressible(mu, T, R):
    """
    Equivalent modulus for a 3D almost incompressible disk

    Parameters
    ----------
    lmbda : float
        Lame's first parameter
    mu : float
        Lame's second parameter
    T : float
        half-thickness of the material (total thickness is 2T)
    R : float
        Radius of the disk,  diameter is D=2R
    """
    k_equiv = force_3d_incompressible(mu, T, R, Delta=T) / (np.pi * R**2)
    return k_equiv


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    import os

    # Values for Gent and Lindley papers figure 2 in Mpas
    D = 20  # mm Diameter
    H = 1.9  # mm Total thickness
    Y = 1.76  # 12.8  # 18.3  # 18 Kg/cm^3 = 1.76 # MPa
    crack_stress = 1.21  #  12.3  Kg/cm^3 # 12.3 MPa = 1.21 MPa
    n = 3.0  # Three dimension
    K_t = 2200  # GPa # Wood and Martin 1964
    lmbda_t = K_t
    mu = 0.59  # Y / 2 / (1 + 0.5)
    eta_bar = K_t / mu * (2 * H / D) ** 2
    print(f"- Data: mu: {mu:2.3f}, Y: {Y:2.3f}, eta_bar: {eta_bar:2.3f}")

    # plot the pressure

    rs = np.linspace(0, D / 2, 100)
    zs = np.linspace(-H / 2, H / 2, 100)

    if not os.path.exists("gentlindley"):
        os.makedirs("gentlindley")

    plt.plot(
        rs,
        pressure_2d(100 * mu, mu, H / 2, D / 2, H / 2, r=rs),
        label=r"$2D: \lambda=100\mu$",
    )
    plt.plot(
        rs,
        pressure_3d(100 * mu, mu, H / 2, D / 2, H, r=rs),
        label=r"$3D: \lambda=100\mu$",
    )
    plt.plot(
        rs,
        pressure_2d_incompressible(mu, H / 2, D / 2, H / 2, x_1=rs),
        label=r"$2D:\text{ incomp}$",
    )
    plt.plot(
        rs,
        pressure_3d_incompressible(mu, H / 2, D / 2, H, r=rs),
        label=r"$3D: \text{ incomp}$",
    )
    plt.legend()
    plt.xlabel(r"$r \,(\mathrm{mm})$")
    plt.ylabel(r"$p\; (\mathrm{MPa})$")
    print(
        f"- F = {force(D, H, mu, delta_u=H):2.4f}, K_equiv={equivalent_modulus(D, H, mu):2.4f}"
    )
    plt.tight_layout()
    plt.savefig("gentlindley/p.png")
    plt.close()

    plt.figure()
    # Force-displacement curve
    fig, ax1 = plt.subplots()
    stiffness_GL = GL_fig2_y_MPa[4] / GL_fig2_x[4]

    # correction term due to the machine compliance
    # surface = np.pi * (D / 2) ** 2
    # stiffness_KT_oscar = 1930.0
    # ET = stiffness_KT_oscar / surface * H
    # GL_fig2_x_corrected = GL_fig2_x - GL_fig2_y_MPa / ET

    # Gent and Lindley slopes for the first and second part of the curve
    x = [0, 0.12673267326732673, 0.21188118811881188, 0.38613861386138615]
    y = [
        -0.02849002849002849,
        9.971509971509972,
        12.393162393162394,
        14.273504273504274,
    ]

    E2 = (y[-1] - y[0]) / (x[-1] - x[0]) * Kgcm2_to_MPa
    E1 = (y[1] - y[0]) / (x[1] - x[0]) * Kgcm2_to_MPa
    print(f"- Gent-Lindley slopes in MPa are {E1:2.3f} and {E2:2.3f} MPa")
    # oscar
    x_o = [0.0, 0.024752475247524754, 0.024752475247524754]
    y_o = [0.0, 0.19514563106796118, 0.5854368932038835]
    EoT = y_o[1] / x_o[1]
    Eo = y_o[2] / x_o[2]
    print(f"- Equivalent modulus in Oscar paper: {EoT:2.3f} - {Eo:2.3f} MPa")
    # ax1.plot(x_o, y_o, "k-")

    #    ax1.plot([0, 0.2], [0, 0.2 * stiffness_GL], "r--")
    ax1.plot(
        GL_fig2_x_corrected,
        GL_fig2_y_MPa,
        "gray",
        label="Gent and Lindley fig 2 data",
        lw=2,
    )
    # plt.plot([0,1],[0,force(delta_u=H)], 'k--')
    ax1.set_xlabel("Extension")
    ax1.set_ylabel(r"$F/S\, (\mathrm{MPa})$")
    ax2 = ax1.twinx()
    mn, mx = ax1.get_ylim()
    ax2.set_ylim(mn / Kgcm2_to_MPa, mx / Kgcm2_to_MPa)
    ax2.set_ylabel(r"$F/S\, (\mathrm{Kg}/\mathrm{cm}^2)$")
    # plt.plot([0,1],[0,force(delta_u=H)], 'k--')
    #

    xs = np.array([0, 0.2])
    ax1.plot(xs, xs * EoT, "y:", label="Kumar's corrected result")
    # ax1.plot(xs, xs * Eo, "r .", label="Kumar's stiff result")
    Ea_3D_wc = equivalent_modulus_3d(lmbda_t, mu, H / 2, D / 2)
    Ea_3D_inc = equivalent_modulus_3d_incompressible(mu, H / 2, D / 2)
    ax1.plot(
        xs,
        xs * Ea_3D_wc,
        "g:",
        label="Weakly compressible model",
    )
    ax1.plot(
        xs,
        xs * Ea_3D_inc,
        "r:",
        label="Incompressible model",
    )
    ax1.set_ylim(0, 1.7)
    # add an horizontal line at 5/2*mu
    ax1.axhline(y=5 / 2 * mu, color="lightgray", linestyle=":")
    # ax1.axhline(y=5 / 6 * Y, color="gray", linestyle="--")
    ax1.legend()
    fig.savefig("gentlindley/GL-fig2.pdf")
    print(f"""
          Equivalent modulus for Gent and Lindley figure 2:
          - Gent and Lindley first part: {E1:2.1f} MPa
          - Gent and Lindley second part: {E2:2.1f} MPa
          - Weakly compressible model: {Ea_3D_wc:2.1f} MPa
          - Incompressible model: {Ea_3D_inc:2.1f} MPa
          """)

    x_fig1 = np.array(
        [
            -7.942857142857141,
            -7.799999999999998,
            -6.942857142857141,
            -6.857142857142857,
            -5.942857142857141,
            -5.828571428571428,
            -4.97142857142857,
            -4.885714285714284,
            -3.9714285714285715,
            -3.8285714285714283,
            -2.9714285714285715,
            -2.8285714285714283,
            -2,
            -1.8571428571428565,
            -0.9714285714285716,
            -0.7428571428571433,
        ]
    )
    y_fig2_MPa = Kgcm2_to_MPa * np.array(
        [
            0,
            18.608058608058613,
            0,
            15.384615384615383,
            0,
            14.505494505494504,
            0,
            9.963369963369962,
            0,
            7.9120879120879115,
            0,
            7.1794871794871815,
            -0.14652014652014111,
            4.542124542124541,
            -0.14652014652014111,
            3.8095238095238106,
        ]
    )

    plt.figure()
    Hs_GLfig1 = (
        np.array([0.056, 0.086, 0.145, 0.183, 0.320, 0.365, 0.565, 0.980]) * 10
    )
    slopes = 0 * Hs_GLfig1
    for i in range(0, len(Hs_GLfig1)):
        dx = (x_fig1[2 * i + 1] - x_fig1[2 * i]) * Hs_GLfig1[i]
        dy = y_fig2_MPa[2 * i + 1] - y_fig2_MPa[2 * i]
        slopes[i] = dy / dx
    plt.plot(Hs_GLfig1, slopes * Hs_GLfig1, "o")
    plt.plot(
        Hs_GLfig1,
        equivalent_modulus_3d_incompressible(mu, Hs_GLfig1 / 2, D / 2),
        "*",
    )
    plt.ylabel("Slope (MPa)")
    plt.xlabel("H (cm)")
    plt.ylim(0, 35)
    plt.tight_layout()
    plt.savefig("gentlindley/slopes.pdf")
    print(np.pi * (D / 2) ** 2 * 5 * mu / 4)
