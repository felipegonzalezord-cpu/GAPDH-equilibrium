# GAPDH Equilibrium Explorer

**Where the GAPDH reaction stops, and why.**

An interactive Streamlit app that calculates how far the glyceraldehyde-3-phosphate dehydrogenase
reaction goes before it reaches equilibrium:

```
D-GAP + NAD⁺ + Pi  ⇌  1,3-BPG + NADH + H⁺
```

It shows how much GAP is oxidised, how much NAD⁺ is reduced and the ΔA₃₄₀ you should read, starting
from the lab's current assay (0.5 mM GAP, 1 mM NAD⁺, 10 mM sodium phosphate, 1 mM EDTA, pH 8.5, 25 °C).
Every value can be changed.

Live app: **https://gapdh-equilibrium.streamlit.app**

## What it does

Ten tabs:

1. **Equilibrium** — initial, change and equilibrium concentrations; the Γ = K′ check; teaching shortcuts
   (quadratic and square-root rule); warnings for photometric linearity, NADH in the start mixture and small signals.
2. **Energy profile** — ΔG′ and the forward/backward flux ratio along the reaction, with snapshots a–d
   (including an NADH spike at the plateau).
3. **pH & buffer** — K′ and ΔG′° vs pH, % GAP oxidised vs pH, phosphate speciation, buffer capacity and pH drift.
4. **Outcome maps** — ΔA₃₄₀ vs GAP, % oxidised vs Pi, a pH × Pi heatmap and the "two-fold change" table.
5. **Ionic strength** — contributions to I, activity coefficients, Kγ vs I and the plateau vs KCl (approximate).
6. **Time course** — simulated A₃₄₀(t) for phosphate at the chosen pH, phosphate at pH 7.0 and arsenate.
7. **Reverse & spikes** — a reverse-direction calculator and "perturb the plateau" predictions with a simulated trace.
8. **From data to K′** — K′ from a measured plateau with propagated uncertainty, and CSV upload (`time_min, A340`).
9. **Experiment planner** — predicted readouts for experiment sets A–E, downloadable as CSV.
10. **Theory & references** — every equation, the assumptions, Gibbs energy values and references.

Downloads: inputs + results as JSON (sidebar), every table as CSV, every chart as PNG (camera icon in the chart toolbar).

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Put it on Streamlit Community Cloud

1. Create a GitHub repository (for example `gapdh-equilibrium`) and upload everything in this folder,
   keeping the folder structure (`gapdh/`, `tests/`, `.streamlit/`).
2. Go to https://share.streamlit.io, sign in with GitHub and choose **Create app → Deploy a public app from GitHub**.
3. Pick the repository, branch `main` and main file `app.py`.
4. In **App URL**, type `gapdh-equilibrium` (→ https://gapdh-equilibrium.streamlit.app). If it is taken, choose
   another name and update the URL in the companion PDF and in `gapdh/constants.py` (`APP_URL`).
5. Under **Advanced settings**, choose Python 3.11 and click **Deploy**. Later pushes to GitHub redeploy automatically.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

`tests/test_validation.py` checks every validation number of the build spec (section 4) plus the expected tables
of section 5. `tests/test_app.py` renders all ten tabs at the defaults, for every preset and at extreme inputs.

## Repository layout

```
app.py                     Streamlit UI only; no science in here
gapdh/constants.py         every constant, with its source
gapdh/thermo.py            K′(pH, T), ΔG′°, equilibrium solver, Γ, ΔG′ profile, inverse
gapdh/buffer.py            buffer capacity and pH drift
gapdh/activity.py          ionic strength, Debye–Hückel γ, Kγ correction
gapdh/kinetics.py          time-course simulator
gapdh/presets.py           named presets
gapdh/model.py             one prediction pipeline that combines the modules above
tests/                     validation and app smoke tests
.streamlit/config.toml     accent colour for light and dark themes
```

## Assumptions

1. One H⁺ is released per turnover.
2. Concentrations are used as activities unless the ionic-strength correction is on.
3. The enzyme binds a negligible share of the reactants.
4. No Mg²⁺ complexation.
5. No non-enzymatic decay of GAP or 1,3-BPG in the equilibrium solver. GAP decays with t½ ≈ 75 min at 37 °C
   (Phillips & Thornalley 1993), so slow drifts after the plateau are not modelled.

Also: the textbook K′ (0.078 M⁻¹ at pH 7) does not state its ionic strength — the app assumes 0.25 M and says so;
buffer capacity is evaluated at the starting pH; the EDTA charge is fixed at −3.5 (its pH 8.5 value).

## Notes on the build spec

All section-4 numbers are reproduced. Three points differ in detail from how the spec states them:

* **TAPS buffer capacity.** The spec gives β ≈ 28.4 mM/pH for "+ 50 mM TAPS". That is TAPS alone; the total with
  10 mM phosphate is 29.5 mM/pH. The drift (≈ 0.003 pH units) is the same either way.
* **Kγ.** The spec's Kγ values (0.141 at I = 0.04 M, 0.028 at 0.25 M) treat Pi as fully HPO₄²⁻. With the
  speciation weighting the spec also asks for (98 % HPO₄²⁻ at pH 8.5), Kγ is 0.139 and 0.027. The KCl result
  (K′ × 2.9, plateau 0.146 mM) is unchanged.
* **Reference ionic strength.** The "I 0.04 → 0.14 M" result treats the current mixture as the reference. With the
  default I_ref = 0.25 M, switching the ionic-strength correction on lowers the default plateau to ΔA₃₄₀ ≈ 0.28,
  because the mixture (I ≈ 0.04 M) is far from 0.25 M. The Ionic strength tab offers both reference choices.

The phosphate pKa₂ is 7.2 for buffer capacity (spec §2) and 6.8 for speciation and charges (spec §3.5).

## References

* Greinert T, Vogel K, Mühlenweg J-K, Sadowski G, Maskow T, Held C (2020). *Fluid Phase Equilibria* 517:112597.
  https://doi.org/10.1016/j.fluid.2020.112597
* Greinert T, Vogel K, Maskow T, Held C (2021). *Sci Rep* 11:6125. https://doi.org/10.1038/s41598-021-85594-8
* Cornell NW, Leadbetter M, Veech RL (1979). *J Biol Chem* 254:6522–6527. https://doi.org/10.1016/S0021-9258(18)50399-9
* Veech RL, Lawson JW, Cornell NW, Krebs HA (1979). *J Biol Chem* 254:6538–6547.
* Trentham DR, McMurray CH, Pogson CI (1969). *Biochem J* 114:19–24. https://pubmed.ncbi.nlm.nih.gov/4309306
* Moniot S et al. (2008). *J Biol Chem* 283:21693–21702. https://doi.org/10.1074/jbc.M802286200
* Textbook ΔG′° = +6.3 kJ/mol: https://boku.ac.at/fileadmin/data/H03000/H77000/LVA/772111_6/05_Glykolyse2.pdf
* Phillips SA, Thornalley PJ (1993). Methylglyoxal formation from triose phosphates. *Eur J Biochem*.

Layout conventions follow the Kinetics Analyzer app (https://github.com/IgnacioNAV/kinetics-analyzer).
