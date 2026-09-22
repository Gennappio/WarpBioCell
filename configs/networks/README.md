# Boolean networks

* `microc_jaya.bnd` / `microc_jaya.cfg` — the MicroC gene regulatory network (Jayathilake et
  al. 2024, PLoS Comput Biol 20(3): e1011944; MAPK core from Grieco et al. 2013 plus HIF,
  metabolic and transporter nodes), in MaBoSS format as distributed with the OpenCellComms
  MicroC adapter (`opencellcomms_adapters/MicroC/data/jaya.bnd`, "Converted from GraphML").
  106 nodes, 25 input nodes (rates 0), unit rates for the rest; the `.cfg` sets the initial
  state distribution (genes 50/50, fate and input nodes OFF). Copied here unchanged on
  2026-09-22 so WarpBioCell's tests and configurations are self-contained.

Formats read by `warpbiocell.network`: MaBoSS `.bnd` (+ optional `.cfg`) and BoolNet `.bnet`.
