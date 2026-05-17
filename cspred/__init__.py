"""UCBShift — chemical shift predictor for protein NMR.

Typical notebook usage::

    from cspred import calc_sing_pdb

    # Both ML and transfer-prediction modules (default)
    df = calc_sing_pdb("protein.pdb", pH=7.0)

    # ML only (no BLAST / mTM-align required)
    df = calc_sing_pdb("protein.pdb", pH=7.0, TP=False)

    # Transfer prediction only
    df = calc_sing_pdb("protein.pdb", pH=7.0, ML=False)
"""

from CSpred import calc_sing_pdb, build_input, data_preprocessing  # noqa: F401
from ucbshifty import main as predict_shifty  # noqa: F401

__all__ = ["calc_sing_pdb", "build_input", "data_preprocessing", "predict_shifty"]
