# Baseline tumor spheroid: first quantitative analysis

Run: `configs/tumor_spheroid.yaml` as committed (seed 0, CPU, 2026-09-22), 1000 initial cells,
7 simulated days, oxygen boundary 38 mmHg on the faces of an 800 µm box. Wall time 27 s on the
development Mac (CPU, single thread). Everything below emerges from the rules in AGENTS.md; no
outcome is prescribed. All parameters are illustrative or estimated (see the YAML), so the
numbers are a sanity check of the model's behaviour, not a prediction.

Figures: `docs/results/figures/baseline_*.png` (population and radii over time, radial profile
and mid-plane slice at day 7). A second run at culture-like oxygen (150 mmHg, 1200 µm box) is
reported at the end.

## Time course

| day | living | total | R99 [µm] | non-prolif. core | hypoxic core | necrotic core | viable rim | prolif. rim | O₂ min at cells | hypoxic | dead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 1000 | 99 | 0 | 0 | 0 | 99 | 99 | 20.4 | 0% | 0% |
| 1 | 1712 | 1712 | 123 | 80 | 0 | 0 | 123 | 43 | 13.9 | 0% | 0% |
| 2 | 2557 | 2557 | 140 | 100 | 0 | 0 | 140 | 40 | 8.7 | 0% | 0% |
| 3 | 3701 | 3701 | 157 | 120 | 80 | 0 | 157 | 37 | 4.4 | 16% | 0% |
| 4 | 5108 | 5108 | 173 | 140 | 120 | 0 | 173 | 33 | 2.0 | 35% | 0% |
| 5 | 6679 | 6868 | 190 | 160 | 140 | 60 | 130 | 30 | 1.9 | 46% | 3% |
| 6 | 8454 | 9126 | 208 | 180 | 160 | 80 | 128 | 28 | 1.9 | 51% | 7% |
| 7 | 10281 | 11686 | 225 | 180 | 180 | 100 | 125 | 45 | 1.9 | 52% | 12% |

Core radii are the outer edge of the outermost 20 µm shell in which at least half of the cells
are non-proliferative / hypoxic-or-dead / dead (`metrics/population.py::shell_radii`);
rims are R99 minus the corresponding core radius. Resolution: one shell, 20 µm.

Events:

* first hypoxic cell at t = 52 h, when R99 = 144 µm (2733 cells);
* first dead cell at t = 97 h, R99 = 174 µm (5173 cells); a necrotic core (≥ 50 % of a shell)
  from t = 99 h;
* the minimum oxygen at the cells stops falling at 1.9 mmHg from day 4 on: cells below the
  2 mmHg death threshold die at 0.5 /h and stop consuming, so the anoxic core regulates
  itself to just below the threshold. This is an emergent feedback, not a rule.

## Growth

* Effective growth rate of the living population: 0.019 /h over days 0–2 (doubling 37 h)
  against a free rate of 0.0289 /h (24 h): contact inhibition alone slows growth by a third
  before any hypoxia exists.
* Days 4–7: 0.0096 /h (doubling 72 h), with the proliferating rim confined to the outer
  30–45 µm.
* R99 grows linearly at 0.71 µm/h = 17 µm/day from day 4 on — the classic linear phase of
  spheroid growth once proliferation is restricted to a rim of constant thickness.

## Spheroid structure at day 7

From the radial profile (`baseline_radial_profile_day7.png`):

| shell [µm] | composition | O₂ [mmHg] |
|---|---|---:|
| 0–100 | dead (necrotic core) | 1.9 |
| 100–180 | hypoxic, alive, not dividing at full rate | 2–8 |
| 180–200 | mixed hypoxic / quiescent / proliferative | 8–10 |
| 200–225 | proliferative rim | 10–17 |

Viable rim ≈ 125–130 µm and stable from day 5 (it grows exactly as fast as the necrotic
core), proliferating rim ≈ 30–45 µm. Both are in the range reported for multicellular
spheroids (viable rims of roughly 100–200 µm), although with illustrative parameters the
agreement is only order-of-magnitude.

## Check against the zero-order diffusion estimate

With uniform consumption `ρ q` inside a sphere of radius R and a Dirichlet boundary at
distance L = 400 µm from the centre, the oxygen at the centre is approximately

```text
O(0) ≈ O_b − ρ q R³ / (3 D) · (1/R − 1/L) − ρ q R² / (6 D)
```

Using the measured density at day 3, ρ = 2.3×10⁻⁴ cells/µm³ (0.94 cells per (16 µm)³), and
the model parameters (q = 1.4×10⁸ mmHg µm³/h, D = 7.2×10⁶ µm²/h, O_b = 38 mmHg):

| R [µm] | O(0) estimate [mmHg] |
|---:|---:|
| 100 | 19.4 |
| 120 | 12.3 |
| 140 | 4.4 |
| 160 | −3.9 (i.e. anoxic) |

The estimate puts the 8 mmHg hypoxia threshold at R ≈ 131 µm; the simulation reaches it at
R99 = 144 µm. The difference has the expected sign: Michaelis–Menten uptake saturates at low
oxygen (less consumption than zero-order near the core) and R99 overestimates the radius of
the consuming mass. The field solver and the coupling therefore behave as the continuum
picture says they should.

## Comparison with MicroC (qualitative)

Jayathilake et al. 2024 (PLoS Comput Biol, 2-D lattice, 800 µm domain, 6 % O₂ ≈ 43 mmHg at the
boundary) report the same sequence: growth, oxygen depletion at the centre as the tumour
enlarges, a hypoxic population emerging after the tumour reaches a critical size, and
necrosis when oxygen and glucose are both low. Their time scale is much longer (hypoxia after
~25 days) because their model is two-dimensional (less shielding), slower growing, and
includes glucose and lactate metabolism. A like-for-like reproduction (2-D slab, their rates,
oxygen only) is the natural next validation step and is not attempted here.

## Culture-like oxygen: 150 mmHg boundary, 1200 µm box

Same configuration with `oxygen.boundary_mmHg=150` and `oxygen.grid.box_um=1200`
(air-saturated medium far from the spheroid), run for 12 days on the 61³ grid.

| day | living | total | R99 [µm] | non-prolif. core | hypoxic core | necrotic core | viable rim | prolif. rim | O₂ min at cells | hypoxic | dead |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 1000 | 99 | 0 | 0 | 0 | 99 | 99 | 129.2 | 0% | 0% |
| 2 | 2557 | 2557 | 140 | 100 | 0 | 0 | 140 | 40 | 111.5 | 0% | 0% |
| 4 | 5130 | 5130 | 173 | 140 | 0 | 0 | 173 | 33 | 90.6 | 0% | 0% |
| 6 | 9265 | 9265 | 210 | 180 | 0 | 0 | 210 | 30 | 65.4 | 0% | 0% |
| 8 | 15714 | 15714 | 247 | 200 | 0 | 0 | 247 | 47 | 36.7 | 0% | 0% |
| 10 | 24319 | 24319 | 284 | 240 | 0 | 0 | 284 | 44 | 11.3 | 0% | 0% |
| 11 | 29691 | 29691 | 301 | 260 | 80 | 0 | 301 | 41 | 3.6 | 3% | 0% |
| 12 | 35698 | 35843 | 319 | 280 | 140 | 40 | 279 | 39 | 1.9 | 8% | 0% |

First hypoxic cell at t = 249 h (day 10.4) with R99 = 290 µm and 26 300 cells; first dead
cell at t = 276 h, R99 = 310 µm; necrotic core (≥ 50 % of a shell) from t = 278 h. Until then
the spheroid is entirely viable and grows with the same linear radius rate as the baseline
(18 µm/day from day 4 on) and the same proliferating rim (30–47 µm), since neither depends on
oxygen while every cell is above the hypoxia threshold. Wall time 189 s (CPU).

Raising the boundary oxygen four-fold moves the onset of hypoxia from R ≈ 145 µm to
R ≈ 290 µm (diameter ≈ 580 µm), i.e. into the 400–600 µm diameter range
where necrotic cores are observed in cultured spheroids. This is the single most sensitive
parameter identified so far and the first candidate for the oxygen-boundary sensitivity study
planned in docs/roadmap.md.

## Caveats

* Single seed; stochastic spread of the population counts is a few per cent at these sizes
  (`tests/test_lifecycle.py` quantifies it for the branching process) but has not been
  measured for the coupled model.
* The Dirichlet box acts as a vessel at fixed distance: the far-field drop between the box and
  the spheroid surface is a large part of the total (see the estimate above). Results depend
  on `box_um` for that reason, which is physical (vessel distance) but must be stated.
* Daughters inherit the parent radius, so the cell density inside the spheroid is set by the
  mechanics, not by a growth model; ρ = 0.94 cells per (16 µm)³ at day 3.
* CPU only; identical physics on CUDA is expected but has not been run.
