# MicroC parameter files and their conversion to WarpBioCell units

Source: `diffusion-parameters.txt` and `input-parameters.txt` of the MicroC model
(Jayathilake et al. 2024, PLoS Comput Biol 20(3): e1011944; repository CBigOxf/jayathilake2022),
provided by the user on 2026-09-22. MicroC is a 2-D NetLogo lattice model with a Boolean
gene network per cell; only the oxygen-related entries are used here. Interpretation of the
units is ours and marked where it is an assumption.

## diffusion-parameters.txt

| Name | Init. | BC | Dif. coef. [m²/s] | Consumption [mol/cell/s] | Production [mol/cell/s] |
|---|---|---|---|---|---|
| FGF | 0.0e-6 | 0.0e-6 | 2.2e-10 | 2.0e-18 | 0.0e-21 |
| EGFRD | 0.0e-3 | 0.0e-3 | 2.2e-10 | 4.0e-17 | 0.0 |
| FGFRD | 0.0e-3 | 0.0e-3 | 2.2e-10 | 4.0e-17 | 0.0 |
| TGFA | 0.000 | 0.000 | 5.18e-11 | 2.0e-17 | 2.0e-20 |
| Oxygen | 0.07 | 0.07 | 1.0e-9 | 3.0e-17 | 0.00 |
| Glucose | 5.00 | 5.00 | 6.70e-11 | 3.0e-15 | 0.00 |
| GI | 0.000 | 0.000 | 5.18e-11 | 2.0e-17 | 0.0e-20 |
| HGF | 2.0e-6 | 2.0e-6 | 8.50e-11 | 2.0e-18 | 0.0e-21 |
| cMETD | 0.0e-3 | 0.0e-3 | 2.2e-10 | 4.0e-17 | 0.0 |
| H | 4e-5 | 4e-5 | 1.0e-9 | 3.0e-15 | 2.0e-20 |
| pH | 7.4 | 7.4 | 1.0e-9 | 0.0 | 0.0 |
| Lactate | 5.00 | 1.00 | 6.70e-11 | 0.0e-15 | 3.0e-15 |
| MCT1D | 0.0e-6 | 0.0e-6 | 2.2e-10 | 4.0e-17 | 0.0 |
| GLUT1D | 0.0e-6 | 0.0e-6 | 2.2e-10 | 4.0e-17 | 0.0 |

## input-parameters.txt (Boolean-network input nodes)

| Input | Initial | Threshold |
|---|---|---|
| DNA_damage | 0 | 0.5 |
| EGFR_stimulus | 0 | 1.0e-6 |
| TGFBR_stimulus | 0 | 0.5 |
| FGFR_stimulus | 0 | 1.0e-6 |
| Oxygen_supply | 0.10 | 0.022 |
| EGFRI | 0 | 5e-3 |
| FGFRI | 0 | 5e-3 |
| Glucose_supply | 5.0 | 4.0 |
| Growth_Inhibitor | 0.0 | 5e-5 |
| cMET_stimulus | 0 | 1.0e-6 |
| cMETI | 0 | 5e-3 |
| pH_min | 8.0 | 6.0 |
| MCT1_stimulus | 5.0 | 1.5 |
| MCT1I | 0 | 17.0e-6 |
| MCT4I | 0 | 1.0 |
| GLUT1I | 0 | 4.0e-6 |

An input node is active when the local field value crosses its threshold; e.g.
`Oxygen_supply` is ON above 2.2 % O₂, `Glucose_supply` ON above 4.0 mM.

## Oxygen entries converted (assumptions stated)

| Quantity | MicroC | Assumed unit | WarpBioCell value | Label |
|---|---|---|---|---|
| Boundary / initial O₂ | 0.07 | volume fraction (7 % O₂; the paper's runs use 3–9 %) | 7 × 7.13 = **50 mmHg** | literature-derived (MicroC) |
| Hypoxia threshold (`Oxygen_supply`) | 0.022 | fraction (2.2 % O₂) | **15.7 mmHg** | literature-derived (MicroC) |
| Diffusion coefficient | 1.0e-9 | m²/s | 1000 µm²/s = **3.6e6 µm²/h** | literature-derived (MicroC) |
| Consumption | 3.0e-17 | mol/cell/s | ÷ α (1.3e-21 mol µm⁻³ mmHg⁻¹) = **8.3e7 mmHg µm³/h** | estimated (needs MicroC's own solubility convention to be exact) |
| Half-saturation K_O₂ | 0.45 % (paper, baseline of the sensitivity analysis) | % O₂ | **3.2 mmHg** | literature-derived (MicroC) |
| Necrosis | O₂ *and* glucose below their critical values | — | oxygen-only stand-in: `death_threshold_mmHg` = 3.6 (0.5 % O₂) | illustrative |

Conversion factor 7.13 mmHg per % O₂ (37 °C, 1 atm, humidified); α = 1.3 µM/mmHg.

Not available in these files: the cell-cycle time (`T_Division`), the phenotype and diffusion
update intervals (`T_Phenotype`, `T_Diffusion`), the quiescence wait time (`T_Q`), the
Boolean network itself (S1 File, GINsim `.zginml`). The network is what turns these inputs
into Proliferation / Apoptosis / Growth_Arrest / Necrosis; without it, WarpBioCell's built-in
rules stand in for it (`configs/microc_oxygen.yaml`).

## Glucose and lactate entries used in `configs/tumor_spheroid_metabolic.yaml` (Milestone 9)

The file's "Consumption 3.0e-15" for glucose is far above what the paper's stoichiometry
implies (it would deplete glucose within ~30 µm of the surface), so the rates come from
Eqs 2–8 instead, applied to this model's oxygen uptake q_O₂ = 1.4e8 mmHg µm³/h
(≈ 5e-17 mol/cell/s). Conversion: 1 mM = 1e-18 mol/µm³, so mol/cell/s × 3600 / 1e-18 gives
mM µm³/h per cell.

| Quantity | MicroC | WarpBioCell value | Label |
|---|---|---|---|
| Glucose diffusion | 6.7e-11 m²/s | 2.4e5 µm²/h | literature-derived (MicroC) |
| Glucose boundary | 5.00 mM | 5.0 mM | literature-derived (MicroC) |
| Glucose activation threshold (`Glucose_supply`) | 4.0 mM | `glucose_threshold_mM: 4.0` | literature-derived (MicroC) |
| K_G | 0.04 mM (baseline of S11 Fig) | 0.04 mM | estimated |
| OXPHOS glucose uptake | q_O₂/6 (Eq. 3) | 3.0e4 mM µm³/h | estimated (stoichiometry) |
| Glycolytic glucose uptake | q_O₂/6 · A₀/2, A₀ = 30 (Eq. 3) | 4.5e5 mM µm³/h | estimated (stoichiometry) |
| Lactate production | 2 × glycolytic glucose uptake (Eq. 7) | 9.0e5 mM µm³/h × G/(K_G+G) | estimated (stoichiometry) |
| Lactate diffusion / boundary | 6.7e-11 m²/s / 1.00 mM | 2.4e5 µm²/h / 1.0 mM | literature-derived (MicroC) |
| Oxygen uptake of glycolytic cells | K · glycoATP, K = 0.5 | `uptake_state_factors: {hypoxic: 0.5}` | literature-derived (MicroC) |
| Necrosis | O₂ and glucose below critical values | `necrosis_requires_glucose: true`, glucose death threshold 0.5 mM | rule literature-derived, threshold illustrative |

Still missing for a like-for-like reproduction: lactate uptake by oxygenated cells (MCT1,
Eq. 8), H⁺/pH, a 2-D slab geometry, the cell-cycle timing, and the network.
