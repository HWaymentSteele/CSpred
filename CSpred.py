#!/usr/bin/env python3
# UCBShift chemical shift predictor — main entry point.
# Combines UCBShift-X (ML) and UCBShift-Y (transfer) predictions.

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib

from spartap_features import PDB_SPARTAp_DataReader
from data_prep_functions import (
    ha23ambigfix, Add_res_spec_feats, feat_pwr,
    hbondd_cols, cos_cols,
    dssp_pp_cols, dssp_energy_cols,
    rcoil_cols, ring_cols,
    sparta_rename_map,
)
import ucbshifty
import toolbox

if sys.version_info < (3, 8):
    raise RuntimeError("Python >= 3.8 required")

pd.options.mode.chained_assignment = None

SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
ML_MODEL_PATH = SCRIPT_PATH / "models"


def build_input(
    pdb_file_name: str,
    pH: float = 5.0,
    rcfeats: bool = True,
    hse: bool = True,
    hbrad: Optional[list] = None,
) -> pd.DataFrame:
    """Extract SPARTA+ features from *pdb_file_name* and return a DataFrame."""
    if hbrad is None:
        hbrad = [5.0, 5.0, 5.0]
    reader = PDB_SPARTAp_DataReader()
    pdb_data = reader.df_from_file_3res(
        pdb_file_name, rcshifts=rcfeats, hse=hse,
        first_chain_only=False, sequence_columns=0, hbrad=hbrad,
    )
    pdb_data["pH"] = pH
    return pdb_data


def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps: HA2/HA3 fix, hydrophobicity, power features, column drops."""
    data = data.copy()
    data = data[sorted(data.columns)]
    data = ha23ambigfix(data, mode=0)
    Add_res_spec_feats(data, include_onehot=False)
    data = feat_pwr(data, hbondd_cols + cos_cols, [2])
    data = feat_pwr(data, hbondd_cols, [-1, -2, -3])
    dropped = set(dssp_pp_cols + dssp_energy_cols + rcoil_cols + [
        'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1',
        'FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'RES',
        'CHAIN', 'RESNAME_ip1', 'RESNAME_im1', 'BMRB_RES_NUM',
        'CG', 'RCI_S2', 'MATCHED_BMRB', 'identifier',
    ])
    data = data.drop(dropped & set(data.columns), axis=1)
    return data


def prepare_data_for_atom(data: pd.DataFrame, atom: str) -> pd.DataFrame:
    """Keep only the ring-current column relevant to *atom*; zero-fill NaNs."""
    dat = data.copy()
    ring_col = atom + '_RC'
    cols_to_remove = [c for c in ring_cols if c != ring_col]
    dat = dat.drop([c for c in cols_to_remove if c in dat.columns], axis=1)
    dat[ring_col] = dat[ring_col].fillna(0)
    return dat


def predict_shifts(
    pdb_file_name: str,
    pH: float = 5.0,
    TP: bool = True,
    TP_pred: Optional[pd.DataFrame] = None,
    ML: bool = True,
    test: bool = False,
) -> pd.DataFrame:
    """Predict chemical shifts for a single PDB file.

    TP  — use UCBShift-Y (transfer prediction) module
    ML  — use UCBShift-X (machine learning) module
    test — exclude mode for UCBShift-Y (for benchmarking)
    """
    if not ML_MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Models directory not found: {ML_MODEL_PATH}")
    if pH < 2 or pH > 12:
        print("Warning! Extreme pH — predictions may be unreliable.")

    preds = pd.DataFrame()

    if TP:
        if TP_pred is None:
            print("Calculating UCBShift-Y predictions ...")
            hashed = str(hash(pdb_file_name) % ((sys.maxsize + 1) * 2)) + '/'
            TP_pred = ucbshifty.main(pdb_file_name, 1, exclude=test, custom_working_dir=hashed)
        if not ML:
            preds = TP_pred[["RESNUM", "RESNAME"]].copy()
            for atom in toolbox.ATOMS:
                rc = TP_pred[f"{atom}_RC"] if f"{atom}_RC" in TP_pred.columns else 0
                preds[f"{atom}_Y"] = TP_pred[atom] + rc

    if ML:
        print("Generating features ...")
        feats = build_input(pdb_file_name, pH)
        feats.rename(columns=sparta_rename_map, inplace=True)
        resnames = feats["RESNAME"]
        resnums = feats["RES_NUM"]
        rcoils = feats[rcoil_cols]
        feats = data_preprocessing(feats)

        result: dict = {"RESNUM": resnums, "RESNAME": resnames}
        for atom in toolbox.ATOMS:
            print(f"Calculating UCBShift-X predictions for {atom} ...")
            atom_feats = prepare_data_for_atom(feats, atom)
            r0 = joblib.load(ML_MODEL_PATH / f"{atom}_R0.sav")
            r0_pred = r0.predict(atom_feats.values)

            feats_r1 = atom_feats.copy()
            feats_r1["R0_PRED"] = r0_pred
            r1 = joblib.load(ML_MODEL_PATH / f"{atom}_R1.sav")
            r1_pred = r1.predict(feats_r1.values)
            result[f"{atom}_X"] = r1_pred + rcoils[f"RCOIL_{atom}"]

            if TP:
                print(f"Calculating UCBShift predictions for {atom} ...")
                feats_r2 = atom_feats.copy()
                feats_r2["RESNAME"] = resnames
                feats_r2["RESNUM"] = resnums
                tp_atom = TP_pred[[
                    "RESNAME", "RESNUM", atom,
                    f"{atom}_BEST_REF_SCORE", f"{atom}_BEST_REF_COV", f"{atom}_BEST_REF_MATCH",
                ]]
                feats_r2 = pd.merge(feats_r2, tp_atom, on="RESNUM", suffixes=("", "_TP"), how="left")
                result[f"{atom}_Y"] = feats_r2[atom].values
                result[f"{atom}_BEST_REF_SCORE"] = feats_r2[f"{atom}_BEST_REF_SCORE"].values
                result[f"{atom}_BEST_REF_COV"] = feats_r2[f"{atom}_BEST_REF_COV"].values
                result[f"{atom}_BEST_REF_MATCH"] = feats_r2[f"{atom}_BEST_REF_MATCH"].values
                valid = (feats_r2.RESNAME == feats_r2.RESNAME_TP) & feats_r2[atom].notnull()
                feats_r2[atom] -= rcoils[f"RCOIL_{atom}"].values
                feats_r2["R0_PRED"] = r0_pred
                valid_feats_r2 = feats_r2.drop(["RESNAME", "RESNUM", "RESNAME_TP"], axis=1)[valid]
                r2_pred = r1_pred.copy()
                if len(valid_feats_r2):
                    r2 = joblib.load(ML_MODEL_PATH / f"{atom}_R2.sav")
                    r2_pred[valid] = r2.predict(valid_feats_r2.values)
                result[f"{atom}_UCBShift"] = r2_pred + rcoils[f"RCOIL_{atom}"]

        preds = pd.DataFrame(result)
    return preds


# ── CLI ───────────────────────────────────────────────────────────────────────

def cli() -> None:
    """Entry point for the ``cspred`` command-line tool."""
    global ML_MODEL_PATH
    parser = argparse.ArgumentParser(
        description=(
            "UCBShift: NMR chemical shift predictor for protein backbone atoms "
            "(H, Hα, C′, Cα, Cβ, N) in aqueous solution. Combines an ML module "
            "(UCBShift-X) with a transfer-prediction module (UCBShift-Y)."
        )
    )
    parser.add_argument("input", help="Query PDB file (or batch list if --batch is set)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Input is a text file with one PDB path per line (optional pH on same line)")
    parser.add_argument("--output", "-o", default="shifts.csv",
                        help="Output CSV (or output directory in batch mode)")
    parser.add_argument("--worker", "-w", type=int, default=4,
                        help="Number of CPU cores for batch mode (currently unused)")
    parser.add_argument("--shifty_only", "-y", "-Y", action="store_true",
                        help="Use only UCBShift-Y (transfer prediction)")
    parser.add_argument("--shiftx_only", "-x", "-X", action="store_true",
                        help="Use only UCBShift-X (machine learning)")
    parser.add_argument("--pH", "-pH", "-ph", type=float, default=5.0,
                        help="pH value (default: 5)")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Use test-mode BLAST database (for benchmarking)")
    parser.add_argument("--models", help="Alternate path to models directory")
    args = parser.parse_args()

    if args.models:
        models_path = Path(args.models)
        if not models_path.is_dir():
            raise FileNotFoundError(f"Models directory not found: {args.models}")
        ML_MODEL_PATH = models_path

    if not args.batch:
        preds = predict_shifts(
            args.input, args.pH,
            TP=not args.shiftx_only, ML=not args.shifty_only, test=args.test,
        )
        preds.to_csv(args.output, index=False)
    else:
        inputs = []
        with open(args.input) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 1:
                    parts.append(args.pH)
                else:
                    parts[-1] = float(parts[-1])
                inputs.append(parts)

        save_prefix = "" if args.output == "shifts.csv" else args.output.rstrip("/") + "/"
        for idx, (pdb_file, ph) in enumerate(inputs):
            preds = predict_shifts(
                pdb_file, ph,
                TP=not args.shiftx_only, ML=not args.shifty_only, test=args.test,
            )
            out_name = save_prefix + os.path.basename(pdb_file).replace(".pdb", ".csv")
            preds.to_csv(out_name, index=False)
            print(f"Finished prediction for {pdb_file} ({idx + 1}/{len(inputs)})")

    print("Complete!")


if __name__ == "__main__":
    cli()
