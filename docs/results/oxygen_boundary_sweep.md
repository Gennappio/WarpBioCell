# Sensitivity to the boundary oxygen: the first scientific experiment

Sweep `configs/sweeps/oxygen_boundary.yaml` on the baseline `configs/tumor_spheroid.yaml`:
`oxygen.boundary_mmHg` ∈ {20, 38, 60, 100, 150} × seeds {0, 1, 2}, 12 simulated days each, all
other parameters at their (illustrative / estimated) defaults, box fixed at 800 µm. 15 runs,
19 min in total on the CPU (45–87 s each). Directory: `runs/2026-09-22T13-23-59_oxygen_boundary`
(not committed); analysis by `examples/analyze_sweep.py --zero-order-oxygen`. Figure:
`docs/results/figures/oxygen_boundary_sweep.png`.

## Results (mean ± sd over 3 seeds)

| O_b [mmHg] | hypoxia onset R99 [µm] | zero-order estimate [µm] | necrosis onset R99 [µm] | living cells, day 12 | R99, day 12 [µm] | viable rim, day 12 [µm] | necrotic fraction, day 12 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | ≤ 100.5 ± 0.4 (initial cluster already hypoxic) | 79 | 133.3 ± 0.8 | 8 185 ± 14 | 253.3 ± 0.6 | 93 ± 1 | 0.36 |
| 38 | 142.6 ± 0.8 | 131 | 173.3 ± 0.3 | 22 949 ± 82 | 310.6 ± 0.9 | 111 ± 1 | 0.29 |
| 60 | 185.4 ± 1.1 | 183 | 213.4 ± 0.8 | 29 879 ± 59 | 316.8 ± 0.8 | 157 ± 1 | 0.15 |
| 100 | 251.6 ± 0.9 | 276 | 279.8 ± 1.5 | 35 019 ± 58 | 317.9 ± 0.7 | 238 ± 1 | 0.02 |
| 150 | none within 12 days (R99 = 320) | none for R < 400 | none | 35 942 ± 143 | 318.5 ± 0.9 | 319 ± 1 | 0.00 |

Onset radii have a resolution of one metrics interval (1 h ≈ 0.7 µm of radial growth).

## What the experiment says

1. **The boundary oxygen sets the internal structure, not the growth rate.** For O_b ≥ 38 mmHg
   the spheroid radius at day 12 is 311–319 µm whatever the oxygen, because radial growth is
   set by the proliferating rim (30–40 µm thick, always oxygenated) and proceeds at 18 µm/day
   in every case. Oxygen decides how much of the interior is hypoxic and necrotic: necrotic
   fraction 0.29 → 0.15 → 0.02 → 0 and viable rim 111 → 157 → 238 → 319 µm from 38 to
   150 mmHg. Only at 20 mmHg does growth itself slow (R99 = 253 µm): the rim is hypoxic too,
   no shell is majority-proliferative, and growth continues only through the reduced hypoxic
   division rate.
2. **Hypoxia onset follows the diffusion scaling.** The onset radius grows from 143 to 252 µm
   between 38 and 100 mmHg, and the zero-order estimate (uniform consumption, Dirichlet
   boundary at L = 400 µm, ρ = 2.3×10⁻⁴ cells/µm³) tracks it within 10 %: 131, 183, 276 µm.
   The estimate is below the simulation at low O_b (Michaelis–Menten saturation reduces the
   consumption near the threshold) and above it at 100 mmHg (the box is a cube: most of its
   faces are farther than 400 µm from the centre, so the far-field drop is larger than the
   spherical estimate assumes). Both deviations have the expected sign.
3. **Necrosis appears about 30 µm of radius after hypoxia**, consistently (133/173/213/280 vs
   100/143/185/252 µm): the radial distance between the 8 mmHg and 2 mmHg thresholds.
4. **The vessel distance matters as much as the oxygen value.** At 150 mmHg with the box at
   400 µm nothing becomes hypoxic within 12 days, while the same boundary value at 600 µm
   (`docs/results/spheroid_baseline.md`, 1200 µm box) gave onset at R99 = 290 µm. Any
   comparison with a culture experiment must therefore state the effective distance between
   the spheroid and the well-mixed medium, not just its oxygen content.
5. **Seed-to-seed spread is negligible for these observables**: ±1 µm on onset radii, ±0.3 %
   on the living count at 1 000+ initial cells. Single-seed runs are adequate for population
   and radius metrics at this scale; replicates matter only for rare events (first dead cell,
   small clusters).

## Against the literature (order of magnitude only)

Cultured spheroids in air-saturated medium develop necrotic cores at diameters of roughly
400–600 µm with viable rims of 100–220 µm. Here, a boundary of 100–150 mmHg at 400–600 µm from
the centre gives onset diameters of 500–580 µm and viable rims of 240–320 µm before necrosis;
with tissue-like 38 mmHg the viable rim settles at ~110 µm. These are the right ranges with
parameters that were never fitted, which is what the model needs to show at this stage; a
quantitative comparison needs the dataset still to be chosen (TODO.md).

## Caveats

* All parameters illustrative or estimated; the uptake rate q_max and the cell density set the
  scale of every radius in this study (both enter as the product ρ q).
* The Dirichlet cube is a crude vessel model; the zero-order estimate uses a sphere of radius
  L = 400 µm and is therefore only indicative at large spheroid sizes.
* At 20 mmHg the onset radius is censored by the initial cluster size.
* CPU only, one machine, Warp 1.17.0.
