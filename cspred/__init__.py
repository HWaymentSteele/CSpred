"""UCBShift — chemical shift predictor for protein NMR.

Typical notebook usage::

    from cspred import predict_shifts

    # Both ML and transfer-prediction modules (default)
    df = predict_shifts("protein.pdb", pH=7.0)

    # ML only (no BLAST / mTM-align required)
    df = predict_shifts("protein.pdb", pH=7.0, TP=False)

    # Transfer prediction only
    df = predict_shifts("protein.pdb", pH=7.0, ML=False)
"""

from CSpred import predict_shifts, build_input, data_preprocessing  # noqa: F401
from ucbshifty import main as predict_shifty  # noqa: F401

__all__ = ["predict_shifts", "build_input", "data_preprocessing", "predict_shifty"]
