# Poker Chip Phase-Field Fracture Simulation Library

Phase-field fracture simulations of poker chips and other geometries using FEniCSx (dolfinx) and alternating minimization algorithms.

`

### Installation

**Option 1: Using conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate poker-chip
```




## Solved Problem

### Energy Formulation

The total energy functional for the phase-field fracture problem is:

$$\mathcal{E}(u, \alpha) = \int_{\Omega} \psi_e(\varepsilon(u), \alpha) \, d\Omega + \int_{\Omega} \psi_d(\alpha, \nabla\alpha) \, d\Omega$$

where the elastic energy includes a **decomposition into deviatoric and volumetric parts with independent damage modulation**:

$$\psi_e(\varepsilon, \alpha) = a_\mu(\alpha) \psi_e^\text{dev}(\varepsilon_\text{dev}) + a_\kappa(\alpha) \psi_e^\text{vol}(\text{tr}(\varepsilon), \alpha)$$

**Deviatoric part** (volume-preserving deformation):
$$\psi_e^\text{dev}(\varepsilon_\text{dev}) = \mu(\alpha) \varepsilon_\text{dev} : \varepsilon_\text{dev}$$

**Volumetric part** (compressive/tensile):
$$\psi_e^\text{vol}(\text{tr}(\varepsilon), \alpha) = \frac{\kappa(\alpha)}{2}(\text{tr}(\varepsilon) - \varepsilon_\text{nl})^2 + p_\text{cav} \sigma(\alpha) \varepsilon_\text{nl}$$

**Damage modulation functions** (different for shear and bulk moduli):
$$\mu(\alpha) = \frac{\mu_0 (1 - w(\alpha))}{1 + (\gamma_\mu - 1) w(\alpha)} + k_\text{res}$$

$$\kappa(\alpha) = \frac{\kappa_0 (1 - w(\alpha))}{1 + (\gamma_\kappa - 1) w(\alpha)} + k_\text{res}$$

**Damage dissipation with gradient regularization**:
$$\psi_d(\alpha, \nabla\alpha) = w_1 w(\alpha) + w_1 \ell^2 |\nabla\alpha|^2$$

where:

- **$u$** is the displacement field
- **$\alpha$** is the phase-field damage variable (0 = sound, 1 = fully damaged)
- **$\varepsilon(u) = \frac{1}{2}(\nabla u + \nabla u^T)$** is the strain tensor
- **$\varepsilon_\text{dev}$** is the deviatoric strain (distortion)
- **$\text{tr}(\varepsilon)$** is the volumetric strain (dilation)
- **$\varepsilon_\text{nl}$** is the nonlinear volumetric strain parameter
- **$w(\alpha) = \alpha$** is the dissipated energy function
- **$\mu(\alpha)$** and **$\kappa(\alpha)$** are damage-modulated shear and bulk moduli
- **$\mu_0, \kappa_0$** are the sound (undamaged) moduli
- **$\gamma_\mu, \gamma_\kappa$** are material parameters controlling damage rate
- **$k_\text{res}$** is the residual stiffness
- **$\ell$** is the internal length scale (gradient regularization)
- **$w_1$** is the damage dissipation coefficient
- **$p_\text{cav}$** is the cavity pressure
- **$\sigma(\alpha)$** is a pressure-modulation function

### Variational Formulation

The coupled problem is solved using alternating minimization over the displacement $u$ and damage $\alpha$:

**Step 1: Minimize over $u$ (elastic step)**

Find $u \in \mathcal{U}$ such that:

$$\int_{\Omega} a(\alpha) \sigma(\varepsilon(u)) : \varepsilon(v) \, d\Omega = \int_{\Omega} f \cdot v \, d\Omega + \int_{\partial\Omega_N} t \cdot v \, d\Gamma \quad \forall v \in \mathcal{V}$$

where $\sigma = \frac{\partial \psi_e}{\partial \varepsilon}$ is the stress tensor, $f$ is the body force, and $t$ is the applied traction.

**Step 2: Minimize over $\alpha$ (damage step)**

Find $\alpha \in \mathcal{A}$ with $0 \leq \alpha \leq 1$ such that:

$$\int_{\Omega} \left[ a'(\alpha) \psi_e(\varepsilon(u)) + w'(\alpha) \right] \beta \, d\Omega = 0 \quad \forall \beta \in \mathcal{B}$$

where $a'(\alpha) = -2(1-\alpha)$ and $w'(\alpha) = 1$.

The inequality constraint $0 \leq \alpha \leq 1$ is enforced using a constrained optimization approach.

### Material Parameters

The model depends on the following independent and derived material parameters:

**Independent parameters:**
- **$\mu_0$** (mu): Shear modulus of sound material (typically 0.59 for GL data, but acts as a stress rescaling)
- **$\kappa_0$** (kappa): Bulk modulus of sound material
- **$\ell$**: Internal length scale (controls crack regularization width)
- **$w_1$**: Damage dissipation coefficient (normalized by dimensional analysis, often set to 1.0)
- **$\gamma_\mu$**: Damage rate parameter for shear modulus (controls how fast shear stiffness degrades)
- **$\gamma_\kappa$**: Damage rate parameter for bulk modulus
- **$p_\text{cav}$**: Cavity/hydrostatic pressure coefficient (default: $5\mu_0/2 = 1.475$)
- **$e_c$**: Critical strain parameter for cavity nucleation
- **$k_\text{res}$**: Residual stiffness ratio (default: $10^{-6}$)

**Derived parameters (computed from independent ones):**
- **$G_c = w_1 \pi \ell$**: Griffith energy release rate (critical strain energy release)
- **$\tau_c = \sqrt{2\mu_0 w_1 / \gamma_\mu}$**: Critical shear stress for damage initiation
- **$p_c$**: Critical hydrostatic pressure ($p_c = \frac{p_\text{cav}}{\sqrt{1-e_c}}$)
- **$\gamma_\kappa$**: Computed to achieve target critical stress in volumetric mode

**Material functions:**
- **$w(\alpha) = \alpha$**: Dissipated energy function (linear damage evolution)
- **$w_f(\alpha) = 1 - (1-\alpha)^2$**: Damage function used in modulation
- **$\sigma_p(\alpha) = (1-\alpha)^2 + k_\text{res}$**: Pressure modulation function
- **$\varepsilon_\text{nl}(\varepsilon, \alpha)$**: Nonlinear volumetric strain (cavity activation) 