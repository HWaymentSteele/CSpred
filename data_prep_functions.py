#!/usr/bin/env python3

import numpy as np
import pandas as pd

try:
    from Bio.Data import IUPACData
except ImportError:
    from Bio.SeqUtils import IUPACData  # biopython < 1.80

from typing import Optional

atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']

paper_order = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE',
               'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER',
               'THR', 'VAL', 'TRP', 'TYR']

# Wimley WC & White SH (1996). Nature Struct. Biol. 3:842-848.
hydrophobic_dict = {
    'LYS': 1.81, 'GLN': 0.19, 'THR': 0.11, 'ASP': 0.5, 'GLU': 0.12,
    'ARG': 1.0, 'LEU': -0.69, 'TRP': -0.24, 'VAL': -0.53, 'ILE': -0.81,
    'PRO': -0.31, 'MET': -0.44, 'ASN': 0.43, 'SER': 0.33, 'ALA': 0.33,
    'GLY': 1.14, 'TYR': 0.23, 'HIS': -0.06, 'PHE': -0.58, 'CYS': 0.22,
}

col_phipsi = [i + j for i in ['PHI_', 'PSI_'] for j in ['COS_i-1', 'SIN_i-1']]
col_phipsi += [i + j for i in ['PHI_', 'PSI_'] for j in ['COS_i', 'SIN_i']]
col_phipsi += [i + j for i in ['PHI_', 'PSI_'] for j in ['COS_i+1', 'SIN_i+1']]
col_chi = [i + j + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
col_hbprev = ['O_' + i + '_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbond = [i + j + '_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_hbnext = ['HN_' + i + '_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS']]
col_s2 = ['S2' + i for i in ['_i-1', '_i', '_i+1']]
struc_cols = col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2

blosum_names = [f'BLOSUM62_NUM_{paper_order[i]}' for i in range(20)]
col_blosum = [blosum_names[i] + j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
seq_cols = col_blosum
bin_seq_cols = [
    'BINSEQREP_' + sorted(IUPACData.protein_letters_3to1.keys())[i].upper() + j
    for j in ['_i-1', '_i', '_i+1'] for i in range(20)
]
rcoil_cols = ['RCOIL_' + atom for atom in atom_names]
ring_cols = [atom + '_RC' for atom in atom_names]
all_cols = struc_cols + seq_cols + rcoil_cols + ring_cols
all_cols_bin = struc_cols + bin_seq_cols + rcoil_cols + ring_cols

hsea_names = ['HSE_CA' + i for i in ['_U', '_D', '_Angle']]
hseb_names = ['HSE_CB' + i for i in ['_U', '_D']]
hse_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in hsea_names + hseb_names]
dssp_ss_names = ['A_HELIX_SS', 'B_BRIDGE_SS', 'STRAND_SS', '3_10_HELIX_SS', 'PI_HELIX_SS', 'TURN_SS', 'BEND_SS', 'NONE_SS']
dssp_asa_names = ['REL_ASA', 'ABS_ASA']
dssp_pp_names = ['DSSP_PHI', 'DSSP_PSI']
dssp_hb_names = ['NH-O1_ENERGY', 'NH-O2_ENERGY', 'O-NH1_ENERGY', 'O-NH2_ENERGY']
dssp_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_ss_names + dssp_asa_names + dssp_pp_names + dssp_hb_names]
dssp_energy_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_hb_names]
dssp_expp_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_ss_names + dssp_asa_names + dssp_hb_names]
dssp_ssi_cols = [name + '_i' for name in dssp_ss_names]
dssp_norm_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_asa_names + dssp_hb_names]
dssp_pp_cols = [name + i for i in ['_i-1', '_i', '_i+1'] for name in dssp_pp_names]

ext_seq_cols = ['RESNAME_i' + i + str(j) for i in ['+', '-'] for j in range(1, 21)]

sparta_ring_cols = [atom + '_RING' for atom in atom_names]
sparta_rcoil_cols = ['RC_' + atom for atom in atom_names]
sparta_rename_cols = sparta_ring_cols + sparta_rcoil_cols
sparta_rename_map = dict(zip(sparta_rename_cols, ring_cols + rcoil_cols))
sx2_rcoil_cols = ['RC_' + atom for atom in atom_names]
sx2_rename_map = dict(zip(sparta_ring_cols + sx2_rcoil_cols, ring_cols + rcoil_cols))

orig_cols = col_blosum[:20]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i-1', 'COS_i-1']]
orig_cols += [i + j + '_i-1' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += col_blosum[20:40]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i', 'COS_i']]
orig_cols += [i + j + '_i' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += col_blosum[40:]
orig_cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i+1', 'COS_i+1']]
orig_cols += [i + j + '_i+1' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
orig_cols += ['O_' + i + '_i-1' for i in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += [i + j + '_i' for i in ['HN_', 'Ha_', 'O_'] for j in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += ['HN_' + i + '_i+1' for i in ['_EXISTS', 'd_HA', '_COS_A', '_COS_H']]
orig_cols += ['S2' + i for i in ['_i-1', '_i', '_i+1']]

exist_cols = [i + 'EXISTS' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_']]
exist_cols += ['O__EXISTS_i-1', 'HN__EXISTS_i+1']
exist_cols += [i + '_EXISTS' + '_i' for i in ['Ha_', 'HN_', 'O_']]
non_exist_cols = [c for c in orig_cols if c not in exist_cols]

x10cols = [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i-1', 'COS_i-1']]
x10cols += [i + j + '_i-1' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
x10cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i', 'COS_i']]
x10cols += [i + j + '_i' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]
x10cols += [i + j for i in ['PHI_', 'PSI_'] for j in ['SIN_i+1', 'COS_i+1']]
x10cols += [i + j + '_i+1' for i in ['CHI1_', 'CHI2_'] for j in ['SIN', 'COS', 'EXISTS']]

phipsicos_cols = [i + 'COS_i-1' for i in ['PHI_', 'PSI_']]
phipsicos_cols += [i + 'COS_i' for i in ['PHI_', 'PSI_']]
phipsicos_cols += [i + 'COS_i+1' for i in ['PHI_', 'PSI_']]
chicos_cols = [i + 'COS' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_']]
hbondcos_cols = ['O_' + i + '_i-1' for i in ['_COS_H', '_COS_A']]
hbondcos_cols += [i + j + '_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['_COS_H', '_COS_A']]
hbondcos_cols += ['HN_' + i + '_i+1' for i in ['_COS_H', '_COS_A']]
cos_cols = phipsicos_cols + chicos_cols + hbondcos_cols
phipsisin_cols = [i + 'SIN_i-1' for i in ['PHI_', 'PSI_']]
phipsisin_cols += [i + 'SIN_i' for i in ['PHI_', 'PSI_']]
phipsisin_cols += [i + 'SIN_i-1' for i in ['PHI_', 'PSI_']]
chisin_cols = [i + 'SIN' + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_']]
sin_cols = phipsisin_cols + chisin_cols
hbondd_cols = ['O_d_HA_i-1']
hbondd_cols += [i + 'd_HA_i' for i in ['Ha_', 'HN_', 'O_']]
hbondd_cols += ['HN_d_HA_i+1']

angle_cols = cos_cols + sin_cols
noisy_cols = angle_cols + hbondd_cols + col_s2

square_cols = cos_cols + hbondd_cols + col_s2
square_cols_names = [x + '^2.0' for x in square_cols]
cols_pown1 = hbondd_cols
cols_pown2 = hbondd_cols
cols_pown3 = hbondd_cols
col_names_pown1 = [x + '^-1.0' for x in cols_pown1]
col_names_pown2 = [x + '^-2.0' for x in cols_pown2]
col_names_pown3 = [x + '^-3.0' for x in cols_pown3]

im1_cols = ['PSI_' + i for i in ['COS_i-1', 'SIN_i-1']]
ip1_cols = ['PHI_' + i for i in ['COS_i+1', 'SIN_i+1']]
im1_cols += [i + j + '_i-1' for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
ip1_cols += [i + j + '_i+1' for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
im1_cols += col_hbprev
ip1_cols += col_hbnext
im1_cols += ['S2_i-1']
ip1_cols += ['S2_i+1']
im1_cols_bin = list(im1_cols)
ip1_cols_bin = list(ip1_cols)
im1_cols += [
    'BLOSUM62_NUM_' + list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i-1'
    for i in range(20)
]
ip1_cols += [
    'BLOSUM62_NUM_' + list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i+1'
    for i in range(20)
]
im1_cols_bin += [
    'BINSEQREP_' + list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i-1'
    for i in range(20)
]
ip1_cols_bin += [
    'BINSEQREP_' + list(IUPACData.protein_letters_3to1.keys())[i].upper() + '_i+1'
    for i in range(20)
]

protein_letters = [code.upper() for code in IUPACData.protein_letters_3to1.keys()]
sp_feat_cols = [
    'BLOSUM62_NUM_ALA_i-1', 'BLOSUM62_NUM_CYS_i-1', 'BLOSUM62_NUM_ASP_i-1',
    'BLOSUM62_NUM_GLU_i-1', 'BLOSUM62_NUM_PHE_i-1', 'BLOSUM62_NUM_GLY_i-1',
    'BLOSUM62_NUM_HIS_i-1', 'BLOSUM62_NUM_ILE_i-1', 'BLOSUM62_NUM_LYS_i-1',
    'BLOSUM62_NUM_LEU_i-1', 'BLOSUM62_NUM_MET_i-1', 'BLOSUM62_NUM_ASN_i-1',
    'BLOSUM62_NUM_PRO_i-1', 'BLOSUM62_NUM_GLN_i-1', 'BLOSUM62_NUM_ARG_i-1',
    'BLOSUM62_NUM_SER_i-1', 'BLOSUM62_NUM_THR_i-1', 'BLOSUM62_NUM_VAL_i-1',
    'BLOSUM62_NUM_TRP_i-1', 'BLOSUM62_NUM_TYR_i-1',
    'PHI_SIN_i-1', 'PHI_COS_i-1', 'PSI_SIN_i-1', 'PSI_COS_i-1',
    'CHI1_SIN_i-1', 'CHI1_COS_i-1', 'CHI1_EXISTS_i-1',
    'CHI2_SIN_i-1', 'CHI2_COS_i-1', 'CHI2_EXISTS_i-1',
    'BLOSUM62_NUM_ALA_i', 'BLOSUM62_NUM_CYS_i', 'BLOSUM62_NUM_ASP_i',
    'BLOSUM62_NUM_GLU_i', 'BLOSUM62_NUM_PHE_i', 'BLOSUM62_NUM_GLY_i',
    'BLOSUM62_NUM_HIS_i', 'BLOSUM62_NUM_ILE_i', 'BLOSUM62_NUM_LYS_i',
    'BLOSUM62_NUM_LEU_i', 'BLOSUM62_NUM_MET_i', 'BLOSUM62_NUM_ASN_i',
    'BLOSUM62_NUM_PRO_i', 'BLOSUM62_NUM_GLN_i', 'BLOSUM62_NUM_ARG_i',
    'BLOSUM62_NUM_SER_i', 'BLOSUM62_NUM_THR_i', 'BLOSUM62_NUM_VAL_i',
    'BLOSUM62_NUM_TRP_i', 'BLOSUM62_NUM_TYR_i',
    'PHI_SIN_i', 'PHI_COS_i', 'PSI_SIN_i', 'PSI_COS_i',
    'CHI1_SIN_i', 'CHI1_COS_i', 'CHI1_EXISTS_i',
    'CHI2_SIN_i', 'CHI2_COS_i', 'CHI2_EXISTS_i',
    'BLOSUM62_NUM_ALA_i+1', 'BLOSUM62_NUM_CYS_i+1', 'BLOSUM62_NUM_ASP_i+1',
    'BLOSUM62_NUM_GLU_i+1', 'BLOSUM62_NUM_PHE_i+1', 'BLOSUM62_NUM_GLY_i+1',
    'BLOSUM62_NUM_HIS_i+1', 'BLOSUM62_NUM_ILE_i+1', 'BLOSUM62_NUM_LYS_i+1',
    'BLOSUM62_NUM_LEU_i+1', 'BLOSUM62_NUM_MET_i+1', 'BLOSUM62_NUM_ASN_i+1',
    'BLOSUM62_NUM_PRO_i+1', 'BLOSUM62_NUM_GLN_i+1', 'BLOSUM62_NUM_ARG_i+1',
    'BLOSUM62_NUM_SER_i+1', 'BLOSUM62_NUM_THR_i+1', 'BLOSUM62_NUM_VAL_i+1',
    'BLOSUM62_NUM_TRP_i+1', 'BLOSUM62_NUM_TYR_i+1',
    'PHI_SIN_i+1', 'PHI_COS_i+1', 'PSI_SIN_i+1', 'PSI_COS_i+1',
    'CHI1_SIN_i+1', 'CHI1_COS_i+1', 'CHI1_EXISTS_i+1',
    'CHI2_SIN_i+1', 'CHI2_COS_i+1', 'CHI2_EXISTS_i+1',
    'O__EXISTS_i-1', 'O_d_HA_i-1', 'O__COS_A_i-1', 'O__COS_H_i-1',
    'HN__EXISTS_i', 'HN_d_HA_i', 'HN__COS_A_i', 'HN__COS_H_i',
    'Ha__EXISTS_i', 'Ha_d_HA_i', 'Ha__COS_A_i', 'Ha__COS_H_i',
    'O__EXISTS_i', 'O_d_HA_i', 'O__COS_A_i', 'O__COS_H_i',
    'HN__EXISTS_i+1', 'HN_d_HA_i+1', 'HN__COS_A_i+1', 'HN__COS_H_i+1',
    'S2_i-1', 'S2_i', 'S2_i+1',
]
spartap_cols = sp_feat_cols + atom_names + [a + "_RC" for a in atom_names] + ["FILE_ID", "RESNAME", "RES_NUM"]
col_square = [f"{a}_{b}_{c}" for a in ['PHI', 'PSI'] for b in ['COS', 'SIN'] for c in ['i-1', 'i', 'i+1']]
dropped_cols = [f"DSSP_{a}_{b}" for a in ["PHI", "PSI"] for b in ['i-1', 'i', 'i+1']] + [
    "BMRB_RES_NUM", "MATCHED_BMRB", "CG", "HA2_RING", "HA3_RING", "RCI_S2", "identifier"
]
col_lift = [col for col in sp_feat_cols if "BLOSUM" not in col and "_i-1" not in col and "_i+1" not in col]
non_numerical_cols = [
    '3_10_HELIX_SS_i', "A_HELIX_SS_i", "BEND_SS_i", "B_BRIDGE_SS_i",
    "CHI1_EXISTS_i", "CHI2_EXISTS_i", "HN__EXISTS_i", "Ha__EXISTS_i",
    "NONE_SS_i", "O__EXISTS_i", "PI_HELIX_SS_i", "STRAND_SS_i", "TURN_SS_i",
] + protein_letters

cols_notinsp = dssp_cols + hse_cols + ext_seq_cols


# ── Core feature functions ────────────────────────────────────────────────────

def diff_targets(data: pd.DataFrame, rings: bool = False, coils: bool = True, drop_cols: bool = True) -> pd.DataFrame:
    df = data.copy()
    if rings:
        df[atom_names] = df[atom_names].values - df[ring_cols].fillna(0).values
        if drop_cols:
            df.drop(ring_cols, axis=1, inplace=True)
    if coils:
        df[atom_names] = df[atom_names].values - df[rcoil_cols].values
        if drop_cols:
            df.drop(rcoil_cols, axis=1, inplace=True)
    return df


def feat_pwr(data: pd.DataFrame, columns: list, pwrs: list) -> pd.DataFrame:
    dat = data.copy()
    for col in columns:
        for power in pwrs:
            new_col_name = f"{col}^{power}"
            if power < 0:
                dat[new_col_name] = 0.0
                dat.loc[dat[col] > 0, new_col_name] = dat.loc[dat[col] > 0, col] ** power
            else:
                dat[new_col_name] = dat[col] ** power
    return dat


def featsq(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Square the given columns — alias for feat_pwr with power=2."""
    return feat_pwr(data, columns, [2])


def Add_res_spec_feats(dataset: pd.DataFrame, include_onehot: bool = True) -> None:
    if include_onehot:
        for code in protein_letters:
            dataset[code] = [int(res == code) for res in dataset['RESNAME']]
    dataset["HYDROPHOBICITY"] = [hydrophobic_dict[res] for res in dataset['RESNAME']]


def ha23ambigfix(data: pd.DataFrame, mode: int = 0) -> pd.DataFrame:
    dat = data.copy()
    idx_ha = dat[dat['HA2_RING'].notnull()].index.tolist()
    if mode == 0:
        for i in idx_ha:
            dat.loc[i, 'HA_RC'] = (dat.loc[i, 'HA2_RING'] + dat.loc[i, 'HA3_RING']) / 2
    elif mode == 1:
        for i in idx_ha:
            dat.loc[i, 'HA_RC'] = dat.loc[i, 'HA2_RING']
    elif mode == 2:
        for i in idx_ha:
            dat.loc[i, 'HA_RC'] = dat.loc[i, 'HA3_RING']
    dat = dat.drop(['HA2_RING', 'HA3_RING'], axis=1)
    return dat


def dihedral_purifier(data: pd.DataFrame, tol: float = 0.001, drop_cols: bool = True, set_nans: bool = True) -> pd.DataFrame:
    dat = data.copy()
    phipsidev = np.abs(
        dat['PHI_COS_i'] - np.cos(np.pi / 180 * dat['DSSP_PHI_i'].astype(np.float64).values)
    )
    good_idxs = (
        ((dat['PHI_COS_i'] == 0) & (dat['PHI_SIN_i'] == 0))
        | (phipsidev == 1)
        | (phipsidev <= tol)
    )
    if set_nans:
        dat.loc[~good_idxs, atom_names] = np.nan
    else:
        dat = dat.loc[good_idxs]
    if drop_cols:
        dat = dat.drop(
            ['DSSP_' + i + j for i in ['PHI', 'PSI'] for j in ['_i-1', '_i', '_i+1']],
            axis=1,
        )
    return dat


def dssp_purifier(data: pd.DataFrame, set_nans: bool = True) -> pd.DataFrame:
    dat = data.copy()
    bad_idxs = dat[
        (dat[dssp_ssi_cols[0]] == 0) & (dat[dssp_ssi_cols[1]] == 0) &
        (dat[dssp_ssi_cols[2]] == 0) & (dat[dssp_ssi_cols[3]] == 0) &
        (dat[dssp_ssi_cols[4]] == 0) & (dat[dssp_ssi_cols[5]] == 0) &
        (dat[dssp_ssi_cols[6]] == 0) & (dat[dssp_ssi_cols[7]] == 0)
    ].index
    if set_nans:
        dat.loc[bad_idxs, atom_names] = np.nan
    else:
        dat = dat.drop(bad_idxs)
    return dat


def medianize(data: pd.DataFrame, cols: list, medians: Optional[list] = None):
    dat = data.copy()
    meds = medians if medians is not None else [dat.loc[dat[col] != 0, col].median() for col in cols]
    med_dict = dict(zip(cols, meds))
    for col in cols:
        dat.loc[dat[col] == 0, col] = med_dict[col]
    return meds, dat


def rc_fix(data: pd.DataFrame, use_null: bool = False) -> None:
    if use_null:
        checklist = {"N": ["PRO", np.nan], "H": ["PRO", np.nan], "CB": ["GLY", np.nan]}
    else:
        checklist = {"N": ["PRO", 119.5], "H": ["PRO", 8.23], "CB": ["GLY", 36.8]}
    for atom in atom_names:
        zero_indices = data[data["RCOIL_" + atom] == 0].index
        for idx in zero_indices:
            assert data.loc[idx, "RESNAME"] == checklist[atom][0]
            data.loc[idx, "RCOIL_" + atom] = checklist[atom][1]


def check_nan_shifts(data: pd.DataFrame, thr: int, ends: bool = False) -> pd.DataFrame:
    dat = data.copy()
    for idx in range(len(dat)):
        snums = dat.loc[idx, atom_names].count()
        file = dat.loc[idx, 'PDB_FILE_NAME']
        chain = dat.loc[idx, 'CHAIN']
        if snums < thr:
            dat.loc[idx, atom_names] = np.nan
            if idx > 0:
                file_m1 = dat.loc[idx - 1, 'PDB_FILE_NAME']
                chain_m1 = dat.loc[idx - 1, 'CHAIN']
                if (file == file_m1) and (chain == chain_m1):
                    dat.loc[idx - 1, atom_names] = np.nan
            if idx < len(dat) - 1:
                file_p1 = dat.loc[idx + 1, 'PDB_FILE_NAME']
                chain_p1 = dat.loc[idx + 1, 'CHAIN']
                if (file == file_m1) and (chain == chain_m1):
                    dat.loc[idx + 1, atom_names] = np.nan
        if ends:
            if idx == 0 or idx == len(dat) - 1:
                dat.loc[idx, atom_names] = np.nan
            elif (file != file_m1) or (chain == chain_m1):
                dat.loc[idx, atom_names] = np.nan
                dat.loc[idx - 1, atom_names] = np.nan
            elif (file != file_p1) or (chain == chain_p1):
                dat.loc[idx, atom_names] = np.nan
                dat.loc[idx + 1, atom_names] = np.nan
    return dat


def filter_outlier(data: pd.DataFrame, atom: str, outlier_path: str = "./") -> None:
    data["identifier"] = data["FILE_ID"].astype(str) + data["RES_NUM"].map(str)
    outlier = pd.read_csv(f"{outlier_path}filtered_{atom}.csv")
    for i in range(len(outlier)):
        identifier = outlier.iloc[i]["PDB"] + str(outlier.iloc[i]["RESNUM"])
        resname = outlier.iloc[i]["RESNAME"]
        filtered = data[data["identifier"] == identifier]
        assert len(filtered) <= 1
        if len(filtered) == 1:
            idx = filtered.iloc[0].name
            assert data.loc[idx, "RESNAME"] == resname
            data.loc[idx, atom] = np.nan
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(outlier)}", end="\r")
    data.drop("identifier", axis=1, inplace=True)


def raw_dprep(
    data: pd.DataFrame,
    ha23mode: int = 0,
    power_dict: Optional[dict] = None,
    diff_rings: bool = False,
) -> pd.DataFrame:
    if power_dict is None:
        power_dict = {2.0: square_cols, -1.0: hbondd_cols, -2.0: hbondd_cols, -3.0: hbondd_cols}
    dat = data.rename(columns=sx2_rename_map)
    dat.index = pd.RangeIndex(len(dat))
    dat = ha23ambigfix(dat, mode=ha23mode)
    try:
        dat = dihedral_purifier(dat, drop_cols=True)
        dat = dssp_purifier(dat)
    except KeyError:
        pass
    for pwr, cols in power_dict.items():
        dat = feat_pwr(dat, cols, [pwr])
    if diff_rings:
        for col in ring_cols:
            null_idxs = dat[dat[col].isnull()].index.tolist()
            dat.loc[null_idxs, col] = dat[col].median()
        dat = diff_targets(dat, rings=True, coils=True)
    else:
        dat = diff_targets(dat, rings=False, coils=True)
    dat.index = pd.RangeIndex(len(dat))
    return dat


def sx2_dprep(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    ha23mode: int = 0,
    med_cols: Optional[list] = None,
    add_squares: Optional[list] = None,
    med_rings: bool = False,
    diff_rings: bool = False,
    medians: bool = True,
    normalize: bool = False,
):
    if med_cols is None:
        med_cols = angle_cols + hbondd_cols
    if add_squares is None:
        add_squares = square_cols
    train_dat = train_data.rename(columns=sx2_rename_map)
    test_dat = test_data.rename(columns=sx2_rename_map)
    train_dat.index = pd.RangeIndex(len(train_dat))
    test_dat.index = pd.RangeIndex(len(test_dat))
    train_dat = dihedral_purifier(train_dat)
    test_dat = dihedral_purifier(test_dat)
    train_dat = dssp_purifier(train_dat)
    test_dat = dssp_purifier(test_dat)
    if medians:
        train_medians, train_dat = medianize(train_dat, med_cols)
        _, test_dat = medianize(test_dat, med_cols, train_medians)
    train_dat = ha23ambigfix(train_dat, mode=ha23mode)
    test_dat = ha23ambigfix(test_dat, mode=ha23mode)
    if med_rings:
        for col in ring_cols:
            null_train = train_dat[train_dat[col].isnull()].index.tolist()
            null_test = test_dat[test_dat[col].isnull()].index.tolist()
            train_dat.loc[null_train, col] = train_dat[col].median()
            test_dat.loc[null_test, col] = train_dat[col].median()
    if add_squares is not None:
        train_dat = featsq(train_dat, add_squares)
        test_dat = featsq(test_dat, add_squares)
    train_dat = diff_targets(train_dat, rings=diff_rings, coils=True)
    test_dat = diff_targets(test_dat, rings=diff_rings, coils=True)
    train_dat.index = pd.RangeIndex(len(train_dat))
    test_dat.index = pd.RangeIndex(len(test_dat))
    return train_dat, test_dat


def sp_dprep(
    data: pd.DataFrame,
    ha23mode: int = 0,
    spfeats_only: bool = False,
    med_rings: bool = True,
    add_squares: Optional[list] = None,
    diff_rings: bool = False,
) -> pd.DataFrame:
    if add_squares is None:
        add_squares = square_cols
    dat = data.rename(columns=sx2_rename_map)
    dat.index = pd.RangeIndex(len(dat))
    dat = dihedral_purifier(dat, drop_cols=not spfeats_only)
    dat = dssp_purifier(dat)
    if spfeats_only:
        dat = dat.drop(dssp_cols + hse_cols + ext_seq_cols, axis=1)
    else:
        dat = featsq(dat, square_cols)
    dat = ha23ambigfix(dat, mode=ha23mode)
    if med_rings:
        for col in ring_cols:
            null_idxs = dat[dat[col].isnull()].index.tolist()
            dat.loc[null_idxs, col] = dat[col].median()
    dat = diff_targets(dat, rings=diff_rings, coils=True)
    dat.index = pd.RangeIndex(len(dat))
    return dat


def hbond_purifier(data: pd.DataFrame, ang_tol: Optional[float] = None, drop_phi: bool = False) -> pd.DataFrame:
    dat = data.copy()
    if ang_tol is not None:
        cos_tol = np.cos((180 - ang_tol) * np.pi / 180)
        hyd_angs = [f"{i}__COS_H_{j}" for i in ['HN', 'Ha', 'O'] for j in ['i-1', 'i', 'i+1']]
        for col in hyd_angs:
            if col in dat.columns:
                dat = dat[(dat[col] < cos_tol) | (dat[col] == 0)]
    if drop_phi:
        phi_angs = [f"{i}__COS_A_{j}" for i in ['HN', 'Ha', 'O'] for j in ['i-1', 'i', 'i+1']]
        for col in phi_angs:
            if col in dat.columns:
                dat = dat.drop(col, axis=1)
    return dat
