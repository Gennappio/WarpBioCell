# Spheroid with glucose and lactate (Milestone 9, first part)

Run: `configs/tumor_spheroid_metabolic.yaml` (seed 0, CPU, 2026-09-22): the baseline spheroid
plus two species solved on the same grid with MicroC's stoichiometry (Jayathilake et al.
2024, Eqs 2–8) applied to this model's oxygen uptake:

| state | oxygen uptake | glucose uptake | lactate production |
|---|---:|---:|---:|
| proliferative / quiescent (OXPHOS) | q_O₂ = 1.4e8 mmHg µm³/h | q_O₂/6 → 3.0e4 mM µm³/h | 0 |
| hypoxic (glycolytic) | 0.5 q_O₂ (MicroC K = 0.5) | q_O₂/6 · A₀/2 = 15× → 4.5e5 mM µm³/h | 2 per glucose → 9.0e5 mM µm³/h, × G/(K_G + G) |
| dead | 0 | 0 | 0 |

Glucose: D = 2.4e5 µm²/h (MicroC), K_G = 0.04 mM, 5 mM at the box faces; lactate: same D,
1 mM at the faces, no uptake (no MCT1 in this rule set). Lifecycle: division also ramps with
glucose between 0.5 and 4 mM (MicroC's `Glucose_supply` threshold), and necrosis follows
MicroC's rule — the anoxic death rate applies only when oxygen **and** glucose are below
their death thresholds (2 mmHg, 0.5 mM). HYPOXIC cells are the glycolytic ones: the
rule-based stand-in for MicroC's metabolic network (Milestone 11). 7 days, 103 s wall time
(three species: ~4× the oxygen-only cost).

## Time course, against the oxygen-only baseline

| day | cells | proliferative | hypoxic | dead (baseline dead) | R99 [µm] | O₂ min [mmHg] | glucose min / mean [mM] | lactate max [mM] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 712 | 1 077 | 0 | 0 (0) | 123 | 13.9 | 4.82 / 4.86 | 1.0 |
| 3 | 3 709 | 1 795 | 603 | 0 (0) | 157 | 4.5 | 4.67 / 4.78 | 1.1 |
| 4 | 5 122 | 2 225 | 1 687 | 0 (0) | 174 | 3.3 | 3.69 / 4.43 | 3.0 |
| 5 | 6 831 | 2 633 | 2 902 | 0 (189) | 191 | 2.8 | 2.25 / 3.61 | 6.0 |
| 6 | 8 538 | 3 112 | 4 062 | 0 (672) | 205 | 2.3 | 1.08 / 2.85 | 8.4 |
| 7 | 10 368 | 3 751 | 4 959 | 2 (1 405) | 218 | 2.0 | 0.06 / 2.01 | 10.5 |

Radial profile at day 7 (20 µm shells):

| shell outer [µm] | O₂ [mmHg] | glucose [mM] | lactate [mM] | proliferative | quiescent | hypoxic |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 2.0 | 0.07 | 10.5 | 0.00 | 0.00 | 0.82 |
| 60 | 2.4 | 0.23 | 10.2 | 0.00 | 0.00 | 1.00 |
| 100 | 3.6 | 0.66 | 9.3 | 0.00 | 0.00 | 1.00 |
| 140 | 5.7 | 1.41 | 7.8 | 0.00 | 0.00 | 1.00 |
| 180 | 9.2 | 2.37 | 5.9 | 0.40 | 0.55 | 0.04 |
| 220 | 14.6 | 3.09 | 4.5 | 0.98 | 0.02 | 0.00 |

## What changed with metabolism

1. **The necrotic core almost disappears.** Two dead cells at day 7 against 1 405 in the
   oxygen-only baseline, with the same growth (10 368 vs 11 686 cells, R99 218 vs 225 µm).
   The cause is not the glucose rule but the halved oxygen consumption of glycolytic cells:
   the hypoxic core now settles at 2.0–2.4 mmHg, just at the 2 mmHg death threshold, instead
   of dropping below it. MicroC's K = 0.5 factor therefore makes hypoxia self-limiting in
   this model. Where oxygen does fall below 2 mmHg, glucose is also gone (0.07 mM at the
   centre), so the AND rule would allow necrosis; it is the oxygen floor that prevents it.
2. **Glucose is depleted from the inside, lactate accumulates.** Glucose goes from 5 mM at
   the faces to 0.07 mM at the centre once the glycolytic core is large (day 5–7), lactate
   rises to 10.5 mM in the core with a monotone gradient outward. MicroC's Fig. 6 shows the
   same pattern in 2-D at smaller amplitude (glucose 3.5–5 mM, lactate up to ~5 mM, with a
   lactate sink through MCT1 that this rule set lacks).
3. **Glucose does not limit growth here.** The proliferating rim (outer 40 µm) sees
   3.1–3.3 mM, below MicroC's 4 mM activation threshold, so the ramp reduces its division
   rate by ~20 %; the population is 11 % smaller than the baseline at day 7 — a modest,
   oxygen-independent effect.

## Caveats

* The metabolic phenotype is a rule (state → rates); MicroC's network switches phenotypes
  through ATP production and includes lactate uptake by oxygenated cells (reverse Warburg),
  which changes the lactate and oxygen budgets. Milestone 11.
* Stoichiometric rates are *estimated*: they inherit the uncertainty of q_O₂ and of A₀ = 30.
* Three species cost ~4× the oxygen-only run on the CPU; the solver residual is now relative
  to the local field magnitude (`|r| / max(6C, C_boundary)`), which was necessary for a
  produced species whose interior values exceed its boundary value ten-fold.
