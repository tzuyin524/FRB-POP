FRB DM-distribution toy-model bundle
====================================

A pedagogical walk-through of six effects that shape the detected
dispersion-measure distribution of fast radio bursts, using three reference
single-dish telescopes (ASKAP-like 12 m, Parkes-like 60 m, FAST-like 300 m):

    1. Euclidean standard candle      (hard horizon)
    2. Log-normal luminosity function (soft tail)
    3. Cosmology                      (LCDM volume kernel + (1+z) K-tax)
    4. Star-formation history         (Madau-Dickinson 2014)
    5. Intra-channel DM smearing      (S_min ~ sqrt(1+(DM/1000)^2))
    6. Spectral index alpha           ((1+z)^(1-alpha) K-correction)

Calibration: ASKAP just detects a median FRB at z=1 (DM=1000) under the
bare model; the population is then made 4x fainter for the figures.


Files
-----

    frb_dm_distribution.pdf   The report (open in Preview).
    frb_dm_distribution.html  Same content, opens in any browser. Equations
                              are rendered live via MathJax (needs network).
                              Edit/restyle freely; CSS is in <style> at top.

    frb_doc_figures.py        Regenerates all six figures used in the doc.
                              Reads nothing, writes figures/fig*.png.
                              Self-contained; no command-line args.

    figures/fig1_euclidean.png  Section 1
    figures/fig2_lf.png         Section 2
    figures/fig3_cosmology.png  Section 3
    figures/fig4_sfr.png        Section 4
    figures/fig5_smear.png      Section 5
    figures/fig6_alpha.png      Section 6

    scripts/                  The six build-up scripts that produced the
                              analysis incrementally. Each adds one effect
                              on top of the previous and prints a quartile
                              table to stdout. Read them in order if you
                              want to see how the model was assembled:

        scripts/frb_dndm.py             v1: cosmology + log-normal LF
                                            (parameter-scan playground)
        scripts/frb_dndm_telescopes.py  v2: ASKAP/PKS/FAST with absolute
                                            FoV-weighted rates
        scripts/frb_dndm_shapes.py      v3: shapes only (PDF normalised)
        scripts/frb_dndm_sfr.py         v4: + Madau-Dickinson SFR weighting
        scripts/frb_dndm_smear.py       v5: + intra-channel DM smearing
        scripts/frb_dndm_alpha.py       v6: + spectral index K-correction


Requirements
------------

Python 3.10+ with numpy, scipy, matplotlib. No other dependencies.


Re-rendering the PDF
--------------------

    python3 frb_doc_figures.py
    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
        --headless --disable-gpu --no-pdf-header-footer \
        --print-to-pdf=frb_dm_distribution.pdf \
        --virtual-time-budget=15000 \
        file://$PWD/frb_dm_distribution.html


Cosmology assumed
-----------------

Flat LambdaCDM with H0 = 67.4 km/s/Mpc, Omega_m = 0.315.
DM-redshift relation simplified to DM = 1000 z.


Caveats / extensions not in the model
-------------------------------------

- No intrinsic Macquart-relation scatter (host- and IGM-DM contributions).
- No scatter-broadening from inhomogeneous plasma (DM^2 to DM^4 depending
  on screen geometry).
- FRBs assumed to track SFR with no delay; no compact-binary delay-time
  distribution.
- Single-pixel beams; no PAF / multibeam receiver multiplicities.
- Single spectral index, not a distribution over alpha.

Each of these would smear the curves further but should not change the
qualitative conclusions.
