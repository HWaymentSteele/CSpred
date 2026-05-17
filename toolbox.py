#!/usr/bin/env python3

try:
    from Bio.Data import IUPACData
except ImportError:
    from Bio.SeqUtils import IUPACData  # biopython < 1.80
from Bio import PDB
from urllib.request import urlopen
import pickle
import os
import re
import numpy as np
from numpy import argmin
import subprocess
import pandas as pd
from io import StringIO
import ssl
from pathlib import Path
from typing import Optional, Union

protein_dict = {pair[0].upper(): pair[1] for pair in IUPACData.protein_letters_3to1.items()}
protein_dict_reverse = {pair[1]: pair[0].upper() for pair in IUPACData.protein_letters_3to1.items()}

ATOMS: list[str] = ["H", "HA", "C", "CA", "CB", "N"]
rmse = lambda x: np.sqrt(np.square(x).mean())

try:
    check_result = subprocess.check_output(["which", "reduce"], stderr=subprocess.DEVNULL)
except Exception:
    check_result = b""
REDUCE_STATUS = len(check_result) != 0


class PDBSaver:
    """Write a PDB file from a Biopython chain object, handling disordered atoms."""

    def __init__(self) -> None:
        self.structure = None

    def set_structure(self, structure) -> None:
        self.structure = structure

    def save(self, address: Union[str, Path]) -> None:
        if self.structure is None:
            raise TypeError("Structure not set!")
        contents = []
        atom_counter = 0
        for residue in self.structure.get_residues():
            for atom in residue.get_atoms():
                if atom.is_disordered():
                    atom = atom.child_dict[sorted(atom.child_dict)[0]]
                atom_counter += 1
                contents.append(
                    f"ATOM {atom_counter:6d}  {atom.name:<4s}{residue.resname:3s} "
                    f"{self.structure.id}{residue.get_id()[1]:4d}{residue.get_id()[2]:1s}   "
                    f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
                    f"  1.00{atom.bfactor:6.2f}         {atom.element:>3s}\n"
                )
        with open(address, "w") as f:
            f.writelines(contents)


def download_pdb(
    pdb_id: str,
    chain_id: Optional[str] = None,
    destination: Optional[str] = None,
    add_hydrogen: bool = False,
    residue_whitelist: Optional[list] = None,
) -> bool:
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        data = urlopen(f"https://files.rcsb.org/download/{pdb_id}.pdb")
    except Exception:
        print(f"Cannot download PDB {pdb_id}")
        return False
    lines = [line.decode("utf-8") for line in data]
    if destination is None:
        destination = str(Path.cwd()) + "/"
    filepath = destination + (f"{pdb_id}_.pdb" if chain_id == "_" else f"{pdb_id}.pdb")
    with open(filepath, "w") as f:
        f.writelines(lines)
    if chain_id not in {None, "_"} or residue_whitelist is not None:
        parser = PDB.PDBParser(QUIET=True)
        struc = parser.get_structure(pdb_id + (chain_id or ""), filepath)
        if chain_id not in {None, "_"}:
            if len(struc) > 1:
                print(f"Multiple structures found for {pdb_id}, only the first structure is taken.")
                struc = struc[0]
            chains = [item for item in struc.get_chains() if item.id == chain_id]
            if len(chains) != 1:
                print(f"Cannot find chain {chain_id} for PDB {pdb_id}")
                return False
            struc = chains[0]
        else:
            struc = struc[0].child_list[0]
        if residue_whitelist:
            deletion = [
                residue.id for residue in struc.child_list
                if residue.resname not in residue_whitelist
            ]
            for delete_id in deletion:
                struc.detach_child(delete_id)
        io = PDBSaver()
        io.set_structure(struc)
        os.remove(filepath)
        filepath = destination + f"{pdb_id}{chain_id}.pdb"
        io.save(filepath)
    if add_hydrogen:
        assert REDUCE_STATUS, (
            "REDUCE is not correctly configured. Please make sure REDUCE is in your path.\n"
            "For more information, please visit http://kinemage.biochem.duke.edu/software/reduce.php"
        )
        os.system(f"reduce {filepath} > {filepath}.H -Quiet")
        os.rename(f"{filepath}.H", filepath)
    print(f"PDB {pdb_id} downloaded to {filepath}")
    return True


def fetch_seq(pdb_id: str, chain_id: Optional[str] = None) -> Union[str, dict]:
    if chain_id == "_":
        chain_id = "A"
    try:
        data = urlopen(
            f"https://www.rcsb.org/pdb/download/viewFastaFiles.do"
            f"?structureIdList={pdb_id}&compressionType=uncompressed"
        )
    except Exception:
        print(f"Cannot find sequence for PDB {pdb_id}")
        return ""
    seq = ""
    record_seq = False
    all_chain = chain_id is None
    all_chain_record: dict = {}
    for line in data:
        line = line.decode("utf-8")
        if not line.strip():
            continue
        if "<!DOCTYPE html" in line:
            main_page = urlopen(f"https://www.rcsb.org/structure/removed/{pdb_id}")
            content = main_page.read().decode("utf-8")
            supersede = content.split("It has been replaced (superseded) by&nbsp<a href=\"/structure/")[1][:4]
            print(f"PDB {pdb_id} has been superseded by {supersede}")
            return fetch_seq(supersede, chain_id)
        if all_chain:
            if line[0] == ">":
                if seq:
                    all_chain_record[chain_id] = seq
                seq = ""
                chain_id = line.split("|")[0].split(":")[1]
            else:
                seq += line.strip()
        else:
            if f">{pdb_id.upper()}:{chain_id.upper()}" in line:
                record_seq = True
            else:
                if line[0] == ">":
                    record_seq = False
                    if seq:
                        return seq
                elif record_seq:
                    seq += line.strip()
    if all_chain:
        all_chain_record[chain_id] = seq
        return all_chain_record
    return seq


def decode_seq(seq: str, supplementary_dict: Optional[dict] = None) -> Union[str, list]:
    lookup_dict = dict(protein_dict_reverse)
    if supplementary_dict is not None:
        lookup_dict.update(supplementary_dict)
    seq = seq.upper()
    if len(seq) == 1:
        return lookup_dict[seq]
    return [lookup_dict.get(r, "UNK") for r in seq]


def form_seq(arr: list, supplementary_dict: Optional[dict] = None) -> str:
    lookup_dict = dict(protein_dict)
    if supplementary_dict is not None:
        lookup_dict.update(supplementary_dict)
    return "".join(lookup_dict[r.upper()] for r in arr)


def load_pkl(path: Union[str, Path]) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj: object, path: Union[str, Path]) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {Path.cwd()}/{path}")


def get_pH(shift_file_path: str, default: float = 5.0) -> float:
    with open(shift_file_path) as f:
        data = f.read()
    regex = re.compile(r"pH.*\d+.*")
    pH_line = regex.search(data)
    if pH_line is None:
        return default
    pH_line = pH_line.group()
    digit_re = re.compile(r"\d+\.\d+")
    candidate_numbers = digit_re.findall(pH_line)
    int_re = re.compile(r" \d{1,2} ")
    candidate_numbers.extend(int_re.findall(pH_line))
    if not candidate_numbers:
        return default
    distances = [abs(pH_line.index(num) - pH_line.index("pH")) for num in candidate_numbers]
    return float(eval(candidate_numbers[argmin(distances)]))


def get_res(file: str) -> Optional[float]:
    with open(file) as f:
        data = f.read()
    regex = re.compile(r"RESOLUTION.*\d+\.\d+.*ANGSTROMS.")
    resolution_line = regex.search(data)
    if resolution_line is None:
        return None
    digit_re = re.compile(r"\d+\.\d+")
    resolution = digit_re.search(resolution_line.group())
    if resolution is None:
        return None
    return float(eval(resolution.group()))


def get_free_gpu() -> Optional[int]:
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
    )
    gpu_df = pd.read_csv(
        StringIO(gpu_stats.decode("utf-8").replace("MiB", "")),
        names=["memory.used", "memory.free"],
        skiprows=1,
    )
    gpu_df["usage"] = gpu_df["memory.free"] / (gpu_df["memory.used"] + gpu_df["memory.free"])
    idx = gpu_df["usage"].idxmax()
    if gpu_df.loc[idx, "usage"] < 0.1:
        return None
    return idx
