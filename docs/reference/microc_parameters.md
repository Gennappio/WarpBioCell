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

What a like-for-like reproduction still needs beyond oxygen: glucose (D = 67 µm²/s, uptake
3e-15 mol/cell/s, 5 mM boundary, 4 mM threshold), lactate (produced at 3e-15 mol/cell/s by
glycolytic cells, 1 mM boundary), H⁺/pH, a 2-D slab geometry, and the network.
