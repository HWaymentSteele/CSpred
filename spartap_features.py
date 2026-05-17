#!/usr/bin/env python3

import math
import warnings
import pandas as pd
import numpy as np
from Bio import PDB
from Bio.PDB.PDBParser import PDBParser
from Bio.Align import substitution_matrices
from Bio import BiopythonWarning

try:
    from Bio.Data import IUPACData
except ImportError:
    from Bio.SeqUtils import IUPACData  # biopython < 1.80

warnings.simplefilter('ignore', BiopythonWarning)

# ── Module-level constants ────────────────────────────────────────────────────

atom_names = ['HA', 'H', 'CA', 'CB', 'C', 'N']

_PAPER_ORDER = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
]

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Wishart et al., J-Bio NMR 5 (1995) 67-81.
_rc_ala: dict[str, list] = {
    'N':  [123.8, 118.8, 120.4, 120.2, 120.3, 108.8, 118.2, 119.9, 120.4, 121.8,
           119.6, 118.7, np.nan, 119.8, 120.5, 115.7, 113.6, 119.2, 121.3, 120.3],
    'H':  [8.24, 8.32, 8.34, 8.42, 8.30, 8.33, 8.42, 8.00, 8.29, 8.16,
           8.28, 8.40, np.nan, 8.32, 8.23, 8.31, 8.15, 8.03, 8.25, 8.12],
    'HA': [4.32, 4.55, 4.71, 4.64, 4.35, 4.62, 3.96, 4.73, 4.17, 4.32,
           4.34, 4.48, 4.74, 4.42, 4.34, 4.3, 4.47, 4.35, 4.12, 4.66, 4.55],
    'C':  [177.8, 174.6, 176.3, 176.6, 175.8, 174.9, 174.1, 176.4, 176.6, 177.6,
           176.3, 175.2, 177.3, 176.0, 176.3, 174.6, 174.7, 176.3, 176.1, 175.9],
    'CA': [52.5, 58.2, 54.2, 56.6, 57.7, 45.1, 55.0, 61.1, 56.2, 55.1,
           55.4, 53.1, 63.3, 55.7, 56.0, 58.3, 61.8, 62.2, 57.5, 57.9],
    'CB': [19.1, 28, 41.1, 29.9, 39.6, np.nan, 29, 38.8, 33.1, 42.4,
           32.9, 38.9, 32.1, 29.4, 30.9, 63.8, 69.8, 32.9, 29.6, 38.8],
}
_rc_pro: dict[str, list] = {
    'N':  [125, 119.9, 121.4, 121.7, 120.9, 109.1, 118.2, 121.7, 121.6, 122.6,
           120.7, 119.0, np.nan, 120.6, 121.3, 116.6, 116.0, 120.5, 122.2, 120.8],
    'H':  [8.19, 8.30, 8.31, 8.34, 8.13, 8.21, 8.37, 8.06, 8.18, 8.14,
           8.25, 8.37, np.nan, 8.29, 8.2, 8.26, 8.15, 8.02, 8.09, 8.1],
    'HA': [4.62, 4.81, 4.90, 4.64, 4.9, 4.13, 5.0, 4.47, 4.60, 4.63, 4.82,
           5.0, 4.73, 4.65, 4.65, 4.78, 4.61, 4.44, 4.99, 4.84],
    'C':  [175.9, 173, 175, 174.9, 174.4, 174.5, 172.6, 175.0, 174.8, 175.7,
           174.6, 173.6, 171.4, 174.4, 174.5, 173.1, 173.2, 174.9, 174.8, 174.8],
    'CA': [50.5, 56.4, 52.2, 54.2, 55.6, 44.5, 53.3, 58.7, 54.2, 53.1,
           53.3, 51.3, 61.5, 53.7, 54.0, 56.4, 59.8, 59.8, 55.7, 55.8],
    'CB': [18.1, 27.1, 40.9, 29.2, 39.1, np.nan, 29.0, 38.7, 32.6, 41.7,
           32.4, 38.7, 30.9, 28.8, 30.2, 63.3, 69.8, 32.6, 28.9, 38.3],
}
randcoil_ala = {atom: dict(zip(_PAPER_ORDER, _rc_ala[atom])) for atom in atom_names}
randcoil_pro = {atom: dict(zip(_PAPER_ORDER, _rc_pro[atom])) for atom in atom_names}
oxidized_cys_correction = {"H": 0.11, "HA": 0.16, "C": 0, "CA": -2.8, "CB": 13.1, "N": -0.2}

secondary_struc_dict = dict(zip(
    ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-'],
    [list(np.identity(8)[i]) for i in range(8)],
))
max_asa_dict = dict(zip(
    _PAPER_ORDER,
    [121.0, 148.0, 187.0, 214.0, 228.0, 97.0, 216.0, 195.0, 230.0, 191.0,
     203.0, 187.0, 154.0, 214.0, 265.0, 143.0, 163.0, 165.0, 264.0, 255.0],
))

# 3-letter → 1-letter mapping (uppercase keys)
_THREE_TO_ONE = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
_ONE_TO_THREE = {v: k.upper() for k, v in IUPACData.protein_letters_3to1.items()}


def _three_to_one(resname: str) -> str:
    """Convert 3-letter residue code to 1-letter; returns 'X' if unknown."""
    return _THREE_TO_ONE.get(resname.upper(), 'X')


# ── Base reader ───────────────────────────────────────────────────────────────

class BaseDataReader:
    """Base class for reading structure files into DataFrames."""

    def __init__(self, columns) -> None:
        self.columns_ = columns

    def df_from_file(self, fpath: str) -> pd.DataFrame:
        raise NotImplementedError


# ── PDB / SPARTA+ feature reader ─────────────────────────────────────────────

class PDB_SPARTAp_DataReader(BaseDataReader):
    """Extract SPARTA+ and extended features from a PDB file."""

    COLS_ = {'FILE_ID': str, 'PDB_FILE_NAME': str, 'RES_NAME': str, 'PDB_RES_NUM': int, 'S2': float}

    def __init__(self) -> None:
        super().__init__(columns=PDB_SPARTAp_DataReader.COLS_.keys())

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _fix_atom_type(start_atom_type: str) -> str:
        atom_type = list(start_atom_type)
        if atom_type[0].isnumeric():
            atom_type.append(atom_type.pop(0))
        if atom_type[0] == 'D':
            atom_type[0] = 'H'
        return ''.join(atom_type)

    @staticmethod
    def _fix_res_name(start_res_name: str) -> str:
        return start_res_name if len(start_res_name) <= 3 else start_res_name[1:]

    # ── Feature calculation methods ───────────────────────────────────────────

    def get_bfactor(self, res_obj, atoms: str = 'all') -> float:
        btot = 0.0
        if atoms == 'all':
            at_list = list(res_obj.get_atoms())
            for atom in at_list:
                btot += atom.get_bfactor()
            return btot / len(at_list)
        # atoms == 'set6'
        for at in ['H', '1H', 'H1', 'D', '1D', 'D1']:
            try:
                btot += res_obj[at].get_bfactor()
                break
            except KeyError:
                pass
        for at in ['HA', '1HA']:
            try:
                btot += res_obj[at].get_bfactor()
                break
            except KeyError:
                pass
        for at in ['N', 'C', 'CA', 'CB']:
            try:
                btot += res_obj[at].get_bfactor()
            except KeyError:
                pass
        return btot / 6

    def blosum_nums(self, res_obj) -> list:
        """Return the 20 BLOSUM62 substitution scores for this residue in sorted AA order."""
        res_name = self._fix_res_name(res_obj.resname)
        code1 = _THREE_TO_ONE.get(res_name)
        if code1 is None or code1 not in BLOSUM62.alphabet:
            return [0] * 20
        # Return scores in alphabetical order of 3-letter codes (matching column naming)
        aa_sorted = sorted(_THREE_TO_ONE.keys())  # alphabetical 3-letter codes
        nums = []
        for aa in aa_sorted:
            code2 = _THREE_TO_ONE[aa]
            try:
                nums.append(int(BLOSUM62[code1, code2]))
            except (KeyError, IndexError):
                nums.append(0)
        return nums

    def binary_seqvecs(self, res_obj) -> list:
        res_name = self._fix_res_name(res_obj.resname)
        out = [0] * 20
        for i, aa in enumerate(_PAPER_ORDER):
            if res_name == aa:
                out[i] = 1
        return out

    @staticmethod
    def calc_phi_psi(chain_obj) -> list:
        polys = PDB.PPBuilder(radius=2.1).build_peptides(chain_obj)
        phi_psi_list = []
        full_res_list = []
        for poly in polys:
            phi_psi_list.append(poly.get_phi_psi_list())
            for res in poly:
                full_res_list.append(res)

        out_list = []
        for peps in phi_psi_list:
            for ang_pair in peps:
                phi_dat = [math.cos(ang_pair[0]), math.sin(ang_pair[0])] if ang_pair[0] is not None else [0, 0]
                psi_dat = [math.cos(ang_pair[1]), math.sin(ang_pair[1])] if ang_pair[1] is not None else [0, 0]
                out_list.append(phi_dat + psi_dat)
        return [out_list[i] for i in range(len(out_list)) if full_res_list[i].get_id()[2] == ' ']

    def calc_torsion_angles(self, res_obj, chi: list = None) -> list:
        if chi is None:
            chi = [1, 2]
        chi_defs = dict(
            chi1=dict(
                ARG=['N', 'CA', 'CB', 'CG'], ASN=['N', 'CA', 'CB', 'CG'],
                ASP=['N', 'CA', 'CB', 'CG'], CYS=['N', 'CA', 'CB', 'SG'],
                GLN=['N', 'CA', 'CB', 'CG'], GLU=['N', 'CA', 'CB', 'CG'],
                HIS=['N', 'CA', 'CB', 'CG'], ILE=['N', 'CA', 'CB', 'CG1'],
                LEU=['N', 'CA', 'CB', 'CG'], LYS=['N', 'CA', 'CB', 'CG'],
                MET=['N', 'CA', 'CB', 'CG'], PHE=['N', 'CA', 'CB', 'CG'],
                PRO=['N', 'CA', 'CB', 'CG'], SER=['N', 'CA', 'CB', 'OG'],
                THR=['N', 'CA', 'CB', 'OG1'], TRP=['N', 'CA', 'CB', 'CG'],
                TYR=['N', 'CA', 'CB', 'CG'], VAL=['N', 'CA', 'CB', 'CG1'],
            ),
            chi2=dict(
                ARG=['CA', 'CB', 'CG', 'CD'], ASN=['CA', 'CB', 'CG', 'OD1'],
                ASP=['CA', 'CB', 'CG', 'OD1'], GLN=['CA', 'CB', 'CG', 'CD'],
                GLU=['CA', 'CB', 'CG', 'CD'], HIS=['CA', 'CB', 'CG', 'ND1'],
                ILE=['CA', 'CB', 'CG1', 'CD1'], LEU=['CA', 'CB', 'CG', 'CD1'],
                LYS=['CA', 'CB', 'CG', 'CD'], MET=['CA', 'CB', 'CG', 'SD'],
                PHE=['CA', 'CB', 'CG', 'CD1'], PRO=['CA', 'CB', 'CG', 'CD'],
                TRP=['CA', 'CB', 'CG', 'CD1'], TYR=['CA', 'CB', 'CG', 'CD1'],
            ),
        )
        res_name = self._fix_res_name(res_obj.resname)
        chi_list = []
        for chi_idx in chi:
            key = f"chi{chi_idx}"
            if key not in chi_defs:
                chi_list += [0, 0, 0]
                continue
            if res_obj.id[0] != " ":
                chi_list += [0] * 3 * len(chi)
                break
            try:
                atom_list = chi_defs[key][res_name]
                vec_atoms = [res_obj[a] for a in atom_list]
                vectors = [a.get_vector() for a in vec_atoms]
                angle = PDB.calc_dihedral(*vectors)
                chi_list += [math.cos(angle), math.sin(angle), 1]
            except KeyError:
                chi_list += [0, 0, 0]
        return chi_list

    @staticmethod
    def s2_param(nn_tree, res_obj, chain, rad: float = 10.0, b_param: float = -0.1) -> float:
        rOsum = 0.0
        rHsum = 0.0
        try:
            prev_res = chain[res_obj.get_id()[1] - 1]
        except KeyError:
            prev_res = None
        prev_Oatom = prev_res['O'] if prev_res is not None else None
        try:
            prev_Oatom = prev_res['O']
        except (KeyError, TypeError):
            prev_Oatom = None

        if prev_Oatom is not None:
            for atom in nn_tree.search(prev_Oatom.get_coord(), rad, level='A'):
                if 'H' not in atom.get_name() and (atom - prev_Oatom) > 0.1:
                    at_res = atom.get_parent()
                    if at_res != res_obj and at_res != prev_res:
                        rOsum += math.exp(-(atom - prev_Oatom - 1.2))

        for at in ['H', '1H']:
            try:
                Hatom = res_obj[at]
                for atom in nn_tree.search(Hatom.get_coord(), rad, level='A'):
                    if 'H' not in atom.get_name() and (atom - Hatom) > 0.1:
                        at_res = atom.get_parent()
                        if at_res != res_obj and at_res != prev_res:
                            rHsum += math.exp(-(atom - Hatom - 1.2))
                break
            except KeyError:
                pass

        return math.tanh(0.8 * (rOsum + rHsum)) + b_param

    @staticmethod
    def find_nearest_atom(nn_tree, catom, rad: float, inrad: float = 0.5, atom_type='Any', excl: list = None):
        if excl is None:
            excl = []
        atoms = nn_tree.search(catom.get_coord(), rad)
        dists = [atom - catom for atom in atoms]
        if len(dists) <= 1:
            return None

        d_min = rad
        loc = None

        if atom_type == 'Any':
            for idx, dist in enumerate(dists):
                if inrad < dist < d_min and dist > 0.1 and atoms[idx] not in excl:
                    d_min = dist
                    loc = idx
        elif isinstance(atom_type, list):
            for idx, dist in enumerate(dists):
                if inrad < dist < d_min and dist > 0.1:
                    if atoms[idx].element in atom_type and atoms[idx] not in excl:
                        d_min = dist
                        loc = idx
        else:
            for idx, dist in enumerate(dists):
                if inrad < dist < d_min and dist > 0.1:
                    if atoms[idx].element == atom_type and atoms[idx] not in excl:
                        d_min = dist
                        loc = idx

        return atoms[loc] if loc is not None else None

    @staticmethod
    def find_nxtnearest_Atom(nn_tree, catom, rad: float, atom_type='Any', excl: list = None):
        if excl is None:
            excl = []
        atoms = nn_tree.search(catom.get_coord(), rad)
        dists = [atom - catom for atom in atoms]
        if len(dists) < 3:
            return None

        d_min = d_2min = rad
        min_loc = min_loc2 = None

        if atom_type == 'Any':
            for idx, dist in enumerate(dists):
                if dist < d_min and dist > 0.1:
                    d_2min, d_min = d_min, dist
                    min_loc2, min_loc = min_loc, idx
                elif d_min < dist < d_2min:
                    d_2min = dist
                    min_loc2 = idx
        else:
            for idx, dist in enumerate(dists):
                atom_name = atoms[idx].name
                if dist < d_min and dist > 0.1 and atom_type in atom_name:
                    d_2min, d_min = d_min, dist
                    min_loc2, min_loc = min_loc, idx
                elif d_min < dist < d_2min and atom_type in atom_name:
                    d_2min = dist
                    min_loc2 = idx

        return atoms[min_loc2] if min_loc2 is not None else None

    def calc_ring_currents(self, res_object_list: list) -> dict:
        res_numbers = [r.get_id()[1] for r in res_object_list]
        if len(res_numbers) > len(set(res_numbers)):
            raise ValueError('Not all residue numbers are unique in calc_ring_currents')

        intensity_factors = {'PHE': 1.05, 'TYR': 0.92, 'TRP1': 1.04, 'TRP2': 0.90, 'HIS': 0.43}
        target_factors = {'HA': 5.13, 'HA2': 5.13, 'HA3': 5.13, 'H': 7.06, 'CA': 1.5,
                          'CB': 1.00, 'N': 1.00, 'C': 1.00, '1HA': 5.13, '2HA': 5.13, '1H': 7.06}
        atom_names_target = list(target_factors.keys())

        rings = []
        for res in res_object_list:
            try:
                name = res.get_resname()
                if name == 'PHE':
                    rings.append(['PHE', (res['CG'].get_coord(), res['CD2'].get_coord(),
                                          res['CE2'].get_coord(), res['CZ'].get_coord(),
                                          res['CE1'].get_coord(), res['CD1'].get_coord())])
                elif name == 'TYR':
                    rings.append(['TYR', (res['CG'].get_coord(), res['CD2'].get_coord(),
                                          res['CE2'].get_coord(), res['CZ'].get_coord(),
                                          res['CE1'].get_coord(), res['CD1'].get_coord())])
                elif name == 'TRP':
                    rings.append(['TRP1', (res['CD2'].get_coord(), res['CE3'].get_coord(),
                                           res['CZ3'].get_coord(), res['CH2'].get_coord(),
                                           res['CZ2'].get_coord(), res['CE2'].get_coord())])
                    rings.append(['TRP2', (res['CG'].get_coord(), res['CD2'].get_coord(),
                                           res['CE2'].get_coord(), res['NE1'].get_coord(),
                                           res['CD1'].get_coord())])
                elif name == 'HIS':
                    rings.append(['HIS', (res['CG'].get_coord(), res['ND1'].get_coord(),
                                          res['CE1'].get_coord(), res['NE2'].get_coord(),
                                          res['CD2'].get_coord())])
            except Exception:
                print('error on ring')

        res_ring_shift_dict: dict = {}
        for res in res_object_list:
            resnum = res.get_id()[1]
            target_atoms = [a for a in res.get_atoms() if a.get_id() in atom_names_target]
            for atom in target_atoms:
                shift = 0.0
                for ring in rings:
                    ring_coords = ring[1]
                    normal = np.cross(ring_coords[1] - ring_coords[0], ring_coords[-1] - ring_coords[0])
                    normal = normal / np.linalg.norm(normal)
                    o = atom.get_coord() + np.dot(normal, ring_coords[0] - atom.get_coord()) * normal
                    G = 0.0
                    for i in range(len(ring_coords)):
                        j = 0 if i == len(ring_coords) - 1 else i + 1
                        r_i = ring_coords[i] - o
                        r_j = ring_coords[j] - o
                        d_r_i = ring_coords[i] - atom.get_coord()
                        d_r_j = ring_coords[j] - atom.get_coord()
                        area_ij = np.linalg.norm(np.cross(r_i, r_j)) / 2
                        sign = np.sign(np.dot(np.cross(r_i, r_j), normal))
                        area_ij *= sign
                        d_ij = 1 / np.linalg.norm(d_r_i) ** 3 + 1 / np.linalg.norm(d_r_j) ** 3
                        G += d_ij * area_ij
                    shift += G * intensity_factors[ring[0]] * target_factors[atom.get_id()]

                if resnum not in res_ring_shift_dict:
                    res_ring_shift_dict[resnum] = {}
                res_ring_shift_dict[resnum][atom.get_id() + '_RING'] = shift

            included = [k.split('_')[0] for k in res_ring_shift_dict.get(resnum, {})]
            for a in [x for x in atom_names_target if x not in included]:
                res_ring_shift_dict.setdefault(resnum, {})[a + '_RING'] = np.nan
            d = res_ring_shift_dict[resnum]
            if pd.isna(d.get('H_RING', np.nan)):
                d['H_RING'] = d.get('1H_RING', np.nan)
            if pd.isna(d.get('HA_RING', np.nan)):
                d['HA_RING'] = d.get('1HA_RING', np.nan)

        return res_ring_shift_dict

    def NH_O_bond(self, nn_tree, res_obj, im1_atoms, ip1_atoms, rad, atom0, atom1, at_type, efilt=False):
        res_i_atoms = list(res_obj.get_atoms())
        angle2 = angle1 = 0.0
        excl_at = []
        while angle2 < math.pi / 2 or angle1 < math.pi / 2:
            full_excl = (res_i_atoms + ip1_atoms + excl_at
                         if atom1.element == 'O'
                         else res_i_atoms + im1_atoms + excl_at)
            atom2 = self.find_nearest_atom(nn_tree, atom1, rad, atom_type=at_type, excl=full_excl)
            if atom2 is None:
                return [0] * 5
            excl_at.append(atom2)
            angle2 = PDB.calc_angle(atom0.get_vector(), atom1.get_vector(), atom2.get_vector())
            bond_dist = atom1 - atom2
            parent_tree = PDB.NeighborSearch(list(atom2.get_parent().get_atoms()))
            try:
                atom3 = self.find_nearest_atom(parent_tree, atom2, rad, atom_type=['N', 'C', 'O'])
                angle1 = PDB.calc_angle(atom1.get_vector(), atom2.get_vector(), atom3.get_vector())
                energy = 0.084 * 332 * (
                    1 / (atom0 - atom2) + 1 / (atom1 - atom3) - 1 / bond_dist - 1 / (atom0 - atom3)
                )
            except AttributeError:
                angle1 = math.pi * 109.5 / 180
                energy = 0.0
                if angle2 >= math.pi / 2:
                    return [bond_dist, math.cos(angle1), math.cos(angle2), 1, energy]
            if efilt and energy > -0.5:
                angle1 = 0.0
        return [bond_dist, math.cos(angle1), math.cos(angle2), 1, energy]

    def hbond_network(self, nn_tree, res_obj, rad=None, ha_bond='restrictive', efilt=False, efilter_O2=True):
        if rad is None:
            rad = [5.0, 5.0, 5.0]
        HAbond_Params = [0] * 5
        HNbond_Params = [0] * 5
        Obond_Params = [0] * 5

        HAatom = next((res_obj[at] for at in ['HA', '1HA'] if at in res_obj), None)
        HNatom = next((res_obj[at] for at in ['H', '1H', 'H1', 'D', '1D', 'D1'] if at in res_obj), None)
        Oatom = res_obj['O'] if 'O' in res_obj else None

        res_i_atoms = list(res_obj.get_atoms())
        all_res = list(res_obj.get_parent().get_parent().get_residues())
        idx_i = all_res.index(res_obj)

        def _get_neighbor_atoms(idx):
            try:
                nbr = all_res[idx]
                if res_obj.get_full_id()[2] == nbr.get_full_id()[2]:
                    return nbr, list(nbr.get_atoms())
            except IndexError:
                pass
            return None, []

        res_ip1, ip1_atoms = _get_neighbor_atoms(idx_i + 1)
        res_im1, im1_atoms = _get_neighbor_atoms(idx_i - 1)

        if HAatom is not None:
            ACatom = res_obj['CA']
            if ha_bond == 'permissive':
                HAOatom = self.find_nearest_atom(nn_tree, HAatom, rad[0], atom_type=['N', 'O'], excl=res_i_atoms)
                if HAOatom is not None:
                    HAangle2 = PDB.calc_angle(ACatom.get_vector(), HAatom.get_vector(), HAOatom.get_vector())
                    HAbond_dist = HAatom - HAOatom
                    try:
                        HAOCatom = HAOatom.get_parent()['C']
                        HAangle1 = PDB.calc_angle(HAatom.get_vector(), HAOatom.get_vector(), HAOCatom.get_vector())
                        HAenergy = 0.084 * 332 * (
                            1 / (ACatom - HAOatom) + 1 / (HAatom - HAOCatom) - 1 / HAbond_dist - 1 / (ACatom - HAOCatom)
                        )
                        flag = 1
                    except KeyError:
                        HAangle1 = math.pi * 109.5 / 180
                        HAenergy = 0.0
                        flag = 1
                    HAbond_Params = [HAbond_dist, math.cos(HAangle1), math.cos(HAangle2), flag, HAenergy]
            elif ha_bond == 'restrictive':
                excl_at_Ha: list = []
                flag = HAangle2 = HAangle1 = 0
                while HAangle2 < math.pi / 2 or HAangle1 < math.pi / 2:
                    flag = 0
                    full_excl = res_i_atoms + ip1_atoms + im1_atoms + excl_at_Ha
                    HAOatom = self.find_nearest_atom(nn_tree, HAatom, rad[0], atom_type='O', excl=full_excl)
                    if HAOatom is None:
                        HAbond_Params = [0] * 5
                        break
                    excl_at_Ha.append(HAOatom)
                    HAangle2 = PDB.calc_angle(ACatom.get_vector(), HAatom.get_vector(), HAOatom.get_vector())
                    HAbond_dist = HAatom - HAOatom
                    try:
                        HAOCatom = HAOatom.get_parent()['C']
                        HAangle1 = PDB.calc_angle(HAatom.get_vector(), HAOatom.get_vector(), HAOCatom.get_vector())
                        HAenergy = 0.084 * 332 * (
                            1 / (ACatom - HAOatom) + 1 / (HAatom - HAOCatom) - 1 / HAbond_dist - 1 / (ACatom - HAOCatom)
                        )
                        O_idx = all_res.index(HAOatom.get_parent())
                        _, O_ip1_atoms = _get_neighbor_atoms(O_idx + 1)
                        O_ip1_atoms += [HAatom]
                        sec = self.NH_O_bond(nn_tree, HAOatom.get_parent(), [], O_ip1_atoms, rad[2],
                                             HAOCatom, HAOatom, ['H', 'D'], efilt=efilter_O2)
                        flag = sec[-2]
                    except KeyError:
                        HAangle1 = math.pi * 109.5 / 180
                        HAenergy = 0.0
                        flag = 1
                    HAbond_Params = [HAbond_dist, math.cos(HAangle1), math.cos(HAangle2), flag, HAenergy]

        if HNatom is not None:
            Natom = res_obj['N']
            HNbond_Params = self.NH_O_bond(nn_tree, res_obj, im1_atoms, ip1_atoms, rad[1],
                                            Natom, HNatom, ['N', 'O'], efilt=efilt)
        if Oatom is not None:
            Catom = res_obj['C']
            Obond_Params = self.NH_O_bond(nn_tree, res_obj, im1_atoms, ip1_atoms, rad[2],
                                           Catom, Oatom, ['H', 'D'], efilt=efilt)
        return HAbond_Params + HNbond_Params + Obond_Params

    def check_disulfide(self, nn_tree, res_obj) -> bool:
        S_atom = res_obj.get('SG', None)
        if S_atom is None:
            S_atom = next((a for a in res_obj.child_list if 'S' in a.name), None)
        if S_atom is None:
            return False
        nn_S = self.find_nearest_atom(nn_tree, S_atom, 2.5, atom_type='S')
        if nn_S is None:
            return False
        return abs(nn_S.parent.id[1] - res_obj.id[1]) >= 4

    # ── Single-residue feature extraction ────────────────────────────────────

    def df_from_file_1res(
        self,
        fpath: str,
        hbrad=None,
        ha_bond: str = 'restrictive',
        efilt: bool = False,
        efilter_O2: bool = True,
        s2rad: float = 10.0,
        hse: bool = True,
        first_chain_only: bool = False,
        bfact_mode: str = 'all',
        sequence_columns: int = 0,
    ) -> pd.DataFrame:
        if hbrad is None:
            hbrad = [5.0, 5.0, 5.0]

        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure('structure', fpath)
        if "H" not in {atom.get_name() for atom in structure.get_atoms()}:
            print("Warning! No hydrogen atoms found — predictions may be degraded.")

        file_id = fpath.split('/')[-1].split('.')[0].split('_')[0]
        file_name = fpath.split('/')[-1]
        AAlet3 = sorted(_THREE_TO_ONE.keys())

        col_names = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN']
        col_names += [i + j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        col_names += [i + j for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
        col_names += [i + j for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        col_names += ['S2']
        col_names += [f'BLOSUM62_NUM_{AAlet3[i]}' for i in range(20)]
        if hse:
            col_names += ['HSE_CA' + i for i in ['_U', '_D', '_Angle']]
            col_names += ['HSE_CB' + i for i in ['_U', '_D']]
        col_names += ['A_HELIX_SS', 'B_BRIDGE_SS', 'STRAND_SS', '3_10_HELIX_SS', 'PI_HELIX_SS',
                      'TURN_SS', 'BEND_SS', 'NONE_SS', 'REL_ASA', 'ABS_ASA', 'DSSP_PHI', 'DSSP_PSI',
                      'NH-O1_ENERGY', 'NH-O2_ENERGY', 'O-NH1_ENERGY', 'O-NH2_ENERGY']
        ring_column_names = ['C_RING', 'CA_RING', 'CB_RING', 'N_RING', 'H_RING', 'HA_RING', 'HA2_RING', 'HA3_RING']
        col_names += ring_column_names + ['AVG_B']
        if sequence_columns > 0:
            seq_match_cols = (
                [f'RESNAME_i-{i}' for i in range(sequence_columns, 0, -1)]
                + [f'RESNAME_i+{i}' for i in range(1, sequence_columns + 1)]
            )
            col_names += seq_match_cols
        col_names += ["CYS_OX"]

        data = []
        for model in structure:
            nn_tree = PDB.NeighborSearch(list(model.get_atoms()))
            try:
                dssp = PDB.DSSP(model, fpath)
            except Exception:
                dssp = []
            if hse:
                hseca_calc = PDB.HSExposureCA(model)
                hsecb_calc = PDB.HSExposureCB(model)

            for chain in model:
                if sequence_columns > 0:
                    res_dict = {res.get_id(): idx for idx, res in enumerate(chain.get_unpacked_list())}
                    list_of_resnames = (
                        ['NONE'] * sequence_columns
                        + [res.resname for res in chain.get_unpacked_list()]
                        + ['NONE'] * sequence_columns
                    )
                dihedrals = self.calc_phi_psi(chain)
                polys = PDB.PPBuilder(radius=2.1).build_peptides(chain)
                poly_residues = [res for poly in polys for res in poly if res.id[2] == ' ']
                seq = ''.join(_three_to_one(r.get_resname()) for r in poly_residues)
                chain_resnum_list = [r.get_id()[1] for r in poly_residues]
                rings = self.calc_ring_currents(poly_residues)

                for l, res in enumerate(poly_residues):
                    resname = self._fix_res_name(res.resname)
                    if resname not in AAlet3:
                        raise ValueError(f'Unexpected residue in peptide loop: {fpath}')
                    if _three_to_one(resname) != seq[l]:
                        raise ValueError(f'Peptide loop inconsistency: {fpath}')

                    res_id = res.get_id()
                    resnum = res_id[1]
                    row_data = [file_id, file_name, resname, resnum, chain.get_id()]
                    row_data += dihedrals[l]
                    row_data += self.calc_torsion_angles(res)
                    row_data += self.hbond_network(nn_tree, res, hbrad, ha_bond=ha_bond,
                                                    efilt=efilt, efilter_O2=efilter_O2)
                    row_data += [self.s2_param(nn_tree, res, chain, s2rad)]
                    row_data += self.blosum_nums(res)
                    if hse:
                        try:
                            hseca = list(hseca_calc[(chain.get_id(), res_id)])
                            hsecb = list(hsecb_calc[(chain.get_id(), res_id)])[:-1]
                        except KeyError:
                            hseca, hsecb = [0] * 3, [0] * 2
                        row_data += hseca + hsecb
                    try:
                        dssp_dat = dssp[(chain.id, res.id)]
                        row_data += secondary_struc_dict[dssp_dat[2]]
                        row_data += [dssp_dat[3], dssp_dat[3] * max_asa_dict[resname]]
                        row_data += [dssp_dat[4], dssp_dat[5]]
                        row_data += [dssp_dat[7], dssp_dat[9], dssp_dat[11], dssp_dat[13]]
                    except KeyError:
                        print(f'KeyError at {(chain.id, res.id)} — skipping residue')
                        row_data += [0] * 16
                        continue
                    except TypeError:
                        print(f'DSSP failed at {(chain.id, res.id)} — skipping residue')
                        row_data += [0] * 16
                        continue
                    row_data += [rings[resnum][i] for i in ring_column_names]
                    row_data += [self.get_bfactor(res, atoms=bfact_mode)]
                    if sequence_columns > 0:
                        central = res_dict[res.get_id()] + sequence_columns
                        row_data += (
                            list_of_resnames[central - sequence_columns: central]
                            + list_of_resnames[central + 1: central + 1 + sequence_columns]
                        )
                    row_data += [self.check_disulfide(nn_tree, res) if resname == "CYS" else False]
                    data.append(row_data)

                if first_chain_only:
                    break

        return pd.DataFrame(data, columns=col_names)

    # ── Three-residue (tripeptide) feature extraction ─────────────────────────

    def df_from_file_3res(
        self,
        fpath: str,
        hbrad=None,
        ha_bond: str = 'restrictive',
        efilt: bool = False,
        efilter_O2: bool = True,
        s2rad: float = 10.0,
        rcshifts: bool = True,
        hse: bool = True,
        first_chain_only: bool = False,
        bfact_mode: str = 'all',
        sequence_columns: int = 0,
    ) -> pd.DataFrame:
        if hbrad is None:
            hbrad = [5.0, 5.0, 5.0]

        phipsi_names = [i + j for i in ['PHI_', 'PSI_'] for j in ['COS', 'SIN']]
        chi_names = [i + j for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
        hbprev_names = ['O_' + j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        hb_names = [i + j for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        hbnext_names = ['HN_' + j for j in ['d_HA', 'COS_H', 'COS_A', 'EXISTS', 'ENERGY']]
        hse_names = ['HSE_CA' + i for i in ['_U', '_D', '_Angle']] + ['HSE_CB' + i for i in ['_U', '_D']]
        ss_names = ['A_HELIX_SS', 'B_BRIDGE_SS', 'STRAND_SS', '3_10_HELIX_SS', 'PI_HELIX_SS',
                    'TURN_SS', 'BEND_SS', 'NONE_SS']
        asa_names = ['REL_ASA', 'ABS_ASA']
        dssp_phipsi_names = ['DSSP_PHI', 'DSSP_PSI']
        dssp_hbond_names = ['NH-O1_ENERGY', 'NH-O2_ENERGY', 'O-NH1_ENERGY', 'O-NH2_ENERGY']
        dssp_names = asa_names + ss_names + dssp_hbond_names + dssp_phipsi_names

        AAlet3 = sorted(_THREE_TO_ONE.keys())
        blosum_names_local = [f'BLOSUM62_NUM_{AAlet3[i]}' for i in range(20)]

        col_id = ['FILE_ID', 'PDB_FILE_NAME', 'RESNAME', 'RES_NUM', 'CHAIN']
        col_extra_resnames = ['RESNAME_im1', 'RESNAME_ip1']
        col_phipsi = [i + j + k for k in ['i-1', 'i', 'i+1'] for i in ['PHI_', 'PSI_'] for j in ['COS_', 'SIN_']]
        col_chi = [i + j + k for k in ['_i-1', '_i', '_i+1'] for i in ['CHI1_', 'CHI2_'] for j in ['COS', 'SIN', 'EXISTS']]
        col_hbprev = ['O_' + i + '_i-1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
        col_hbond = [i + j + '_i' for i in ['Ha_', 'HN_', 'O_'] for j in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
        col_hbnext = ['HN_' + i + '_i+1' for i in ['d_HA', '_COS_H', '_COS_A', '_EXISTS', '_ENERGY']]
        col_s2 = ['S2' + i for i in ['_i-1', '_i', '_i+1']]
        col_blosum = [blosum_names_local[i] + j for j in ['_i-1', '_i', '_i+1'] for i in range(20)]
        col_names = col_id + col_extra_resnames + col_phipsi + col_chi + col_hbprev + col_hbond + col_hbnext + col_s2 + col_blosum
        if rcshifts:
            col_rccs = ['RC_' + i for i in atom_names]
            col_names += col_rccs
        if hse:
            col_hse = [hse_names[i] + j for j in ['_i-1', '_i', '_i+1'] for i in range(5)]
            col_names += col_hse
        col_dssp = [i + j for j in ['_i-1', '_i', '_i+1'] for i in dssp_names]
        col_names += col_dssp
        ring_column_names = ['C_RING', 'CA_RING', 'CB_RING', 'N_RING', 'H_RING', 'HA_RING', 'HA2_RING', 'HA3_RING']
        col_names += ring_column_names
        bfact_names = ['AVG_B' + i for i in ['_i-1', '_i', '_i+1']]
        col_names += bfact_names
        if sequence_columns > 0:
            seq_match_cols = (
                [f'RESNAME_i-{i}' for i in range(sequence_columns, 0, -1)]
                + [f'RESNAME_i+{i}' for i in range(1, sequence_columns + 1)]
            )
            col_names += seq_match_cols

        df_1res_all_chains = self.df_from_file_1res(
            fpath, hbrad, ha_bond=ha_bond, efilt=efilt, efilter_O2=efilter_O2,
            s2rad=s2rad, hse=hse, first_chain_only=first_chain_only,
            bfact_mode=bfact_mode, sequence_columns=sequence_columns,
        )

        chain_dfs: list[pd.DataFrame] = []
        for chain in sorted(set(df_1res_all_chains['CHAIN'].tolist())):
            df = pd.DataFrame(columns=col_names)
            df_1res = df_1res_all_chains[df_1res_all_chains['CHAIN'] == chain].copy()
            df_1res.index = range(len(df_1res))

            res_list = df_1res['RES_NUM'].tolist()
            if len(set(res_list)) != len(res_list):
                raise ValueError(f'Duplicate residue numbers in: {fpath}')
            if (np.sort(res_list) != np.array(res_list)).any():
                raise ValueError(f'Residue numbers not ordered in: {fpath}')

            rows = []
            for i in range(len(df_1res)):
                res_i_num = df_1res.loc[i, 'RES_NUM']
                row: dict = {}
                for col in col_id:
                    row[col] = df_1res.loc[i, col]

                if (res_i_num - 1) not in res_list:
                    blosum_prev = [0] * 20
                    psi_prev = phi_prev = [0, 0]
                    chi_prev = [0] * 6
                    hb_prev = [0] * 5
                    s2_prev = 0
                    hse_prev = [0] * 5
                    resname_prev = ''
                    dssp_prev = [0] * 16
                    bfact_prev = [0]
                else:
                    blosum_prev = list(df_1res.loc[i - 1, blosum_names_local].values)
                    psi_prev = [df_1res.loc[i - 1, 'PSI_COS'], df_1res.loc[i - 1, 'PSI_SIN']]
                    phi_prev = [df_1res.loc[i - 1, 'PHI_COS'], df_1res.loc[i - 1, 'PHI_SIN']]
                    chi_prev = list(df_1res.loc[i - 1, chi_names].values)
                    hb_prev = list(df_1res.loc[i - 1, hbprev_names].values)
                    s2_prev = df_1res.loc[i - 1, 'S2']
                    if hse:
                        hse_prev = list(df_1res.loc[i - 1, hse_names].values)
                    resname_prev = df_1res.loc[i - 1, 'RESNAME']
                    dssp_prev = list(df_1res.loc[i - 1, dssp_names])
                    bfact_prev = [df_1res.loc[i - 1, 'AVG_B']]

                if (res_i_num + 1) not in res_list:
                    blosum_next = [0] * 20
                    psi_next = phi_next = [0, 0]
                    chi_next = [0] * 6
                    hb_next = [0] * 5
                    s2_next = 0
                    hse_next = [0] * 5
                    res_next = 'ALA' if rcshifts else None
                    resname_next = ''
                    dssp_next = [0] * 16
                    bfact_next = [0]
                else:
                    blosum_next = list(df_1res.loc[i + 1, blosum_names_local].values)
                    psi_next = [df_1res.loc[i + 1, 'PSI_COS'], df_1res.loc[i + 1, 'PSI_SIN']]
                    phi_next = [df_1res.loc[i + 1, 'PHI_COS'], df_1res.loc[i + 1, 'PHI_SIN']]
                    chi_next = list(df_1res.loc[i + 1, chi_names].values)
                    hb_next = list(df_1res.loc[i + 1, hbnext_names].values)
                    s2_next = df_1res.loc[i + 1, 'S2']
                    if hse:
                        hse_next = list(df_1res.loc[i + 1, hse_names].values)
                    res_next = df_1res.loc[i + 1, 'RESNAME'] if rcshifts else None
                    resname_next = df_1res.loc[i + 1, 'RESNAME']
                    dssp_next = list(df_1res.loc[i + 1, dssp_names])
                    bfact_next = [df_1res.loc[i + 1, 'AVG_B']]

                row.update(dict(zip(col_extra_resnames, [resname_prev, resname_next])))
                row.update(dict(zip(col_phipsi, phi_prev + psi_prev + list(df_1res.loc[i, phipsi_names].values) + phi_next + psi_next)))
                row.update(dict(zip(col_chi, chi_prev + list(df_1res.loc[i, chi_names].values) + chi_next)))
                row.update(dict(zip(col_hbprev, hb_prev)))
                row.update(dict(zip(col_hbond, list(df_1res.loc[i, hb_names].values))))
                row.update(dict(zip(col_hbnext, hb_next)))
                row.update(dict(zip(col_s2, [s2_prev, df_1res.loc[i, 'S2'], s2_next])))
                row.update(dict(zip(col_blosum, blosum_prev + list(df_1res.loc[i, blosum_names_local].values) + blosum_next)))
                if rcshifts:
                    resname = df_1res.loc[i, 'RESNAME']
                    rc_table = randcoil_pro if res_next == 'PRO' else randcoil_ala
                    rccs = [rc_table[j][resname] for j in atom_names]
                    if df_1res.loc[i, 'CYS_OX']:
                        if res_next == 'PRO':
                            print("Warning! Oxidized CYS preceding PRO!")
                        rccs = [v + oxidized_cys_correction[j] for v, j in zip(rccs, atom_names)]
                    row.update(dict(zip(col_rccs, rccs)))
                if hse:
                    row.update(dict(zip(col_hse, hse_prev + list(df_1res.loc[i, hse_names].values) + hse_next)))
                row.update(dict(zip(ring_column_names, list(df_1res.loc[i, ring_column_names].values))))
                row.update(dict(zip(col_dssp, dssp_prev + list(df_1res.loc[i, dssp_names]) + dssp_next)))
                row.update(dict(zip(bfact_names, bfact_prev + [df_1res.loc[i, 'AVG_B']] + bfact_next)))
                if sequence_columns > 0:
                    row.update(dict(zip(seq_match_cols, list(df_1res.loc[i, seq_match_cols]))))
                rows.append(row)

            chain_dfs.append(pd.DataFrame(rows, columns=col_names))

        return pd.concat(chain_dfs, ignore_index=True) if chain_dfs else pd.DataFrame(columns=col_names)
