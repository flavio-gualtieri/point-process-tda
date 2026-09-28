"""End-to-end scores: does a fitted model reproduce the pattern it was fitted to?

    kernel.py     kernel score on local configurations (primary)
    dss.py        Dawid-Sebastiani on geometric statistics (secondary)
    energy.py     whole-pattern W1 energy score (power-check baseline)
    simulate.py   observed patterns, true models, fitted models -> simulations
    endtoend.py   one cloud scored under every pipeline's fit; regret, gain, skill
"""
