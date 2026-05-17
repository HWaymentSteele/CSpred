#!/usr/bin/env python3
# Transfer-prediction (UCBShift-Y) module.
# Predicts chemical shifts by transferring shifts from similar proteins in
# refDB via sequence alignment (BLAST) and structure alignment (mTM-align).

import os
import shutil
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import PDB, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Align
from Bio.Align import substitution_matrices

import toolbox
from save_pdb import PDBSaver

# ── Module-level constants ────────────────────────────────────────────────────

DEBUG = False
GLOBAL_TEST_CUTOFF = 0.99
SCRIPT_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
BLAST_DEFAULT_EXE = str(SCRIPT_PATH / "bins" / "ncbi-blast-2.9.0+" / "bin" / "blastp")
MTM_DEFAULT_EXE = str(SCRIPT_PATH / "bins" / "mTM-align" / "mTM-align")
os.environ["BLASTDB"] = str(SCRIPT_PATH / "refDB")

BLOSUM62 = substitution_matrices.load("BLOSUM62")

# Decompress refDB PDB files on first run
_REFDB_PDBS = SCRIPT_PATH / "refDB" / "pdbs"
if not _REFDB_PDBS.exists():
    _REFDB_PDBS.mkdir(parents=True)
    print("Decompressing mTM-align database...")
    os.system(f"tar -xzf {SCRIPT_PATH}/refDB/pdbs.tgz -C {SCRIPT_PATH}/refDB/")

# ── Random coil values (Wishart et al., J-Bio NMR, 5 (1995) 67-81) ──────────

_PAPER_ORDER = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
]

_rc_ala: dict[str, list] = {
    'N':  [123.8, 118.7, 120.4, 120.2, 120.3, 108.8, 118.2, 119.9, 120.4, 121.8,
           119.6, 118.7, np.nan, 119.8, 120.5, 115.7, 113.6, 119.2, 121.3, 120.3],
    'H':  [8.24, (8.32+8.43)/2, 8.34, 8.42, 8.30, 8.33, 8.42, 8.00, 8.29, 8.16,
           8.28, 8.40, np.nan, 8.32, 8.23, 8.31, 8.15, 8.03, 8.25, 8.12],
    'HA': [4.32, 4.55, 4.71, 4.64, 4.35, 4.62, 3.96, 4.73, 4.17, 4.32,
           4.34, 4.48, 4.74, 4.42, 4.34, 4.3, 4.47, 4.35, 4.12, 4.66, 4.55],
    'C':  [177.8, 174.6, 176.3, 176.6, 175.8, 174.9, 174.1, 176.4, 176.6, 177.6,
           176.3, 175.2, 177.3, 176.0, 176.3, 174.6, 174.7, 176.3, 176.1, 175.9],
    'CA': [52.5, (58.2+55.4)/2, 54.2, 56.6, 57.7, 45.1, 55.0, 61.1, 56.2, 55.1,
           55.4, 53.1, 63.3, 55.7, 56.0, 58.3, 61.8, 62.2, 57.5, 57.9],
    'CB': [19.1, (28+41.1)/2, 41.1, 29.9, 39.6, np.nan, 29, 38.8, 33.1, 42.4,
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

randcoil_ala = {atom: dict(zip(_PAPER_ORDER, _rc_ala[atom])) for atom in toolbox.ATOMS}
randcoil_pro = {atom: dict(zip(_PAPER_ORDER, _rc_pro[atom])) for atom in toolbox.ATOMS}

EXTERNAL_MAPPINGS = {
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS",
    "CAS": "CYS", "CSD": "CYS", "MSE": "MET", "CSO": "CYS",
}
SS_CAPS = {"H": 3.7, "HA": 3, "C": 10, "CA": 11.25, "CB": 20, "N": 22}


# ── Utility functions ─────────────────────────────────────────────────────────

def get_blosum_value(resname1: str, resname2: str) -> int:
    """Return the BLOSUM62 substitution score between two residues (3-letter codes)."""
    code1, code2 = toolbox.form_seq([resname1, resname2])
    return int(BLOSUM62[code1, code2])


def _extract_aligned_seqs(alignment) -> tuple[str, str]:
    """Parse gapped aligned sequences from a Bio.Align PairwiseAlignment object.

    Returns (target_gapped, query_gapped) where target = seq1 and query = seq2
    as passed to aligner.align(seq1, seq2).
    """
    fmt = format(alignment)
    target_parts: list[str] = []
    query_parts: list[str] = []
    for line in fmt.split("\n"):
        tokens = line.split()
        if len(tokens) >= 3:
            if tokens[0] == "target":
                target_parts.append(tokens[2])
            elif tokens[0] == "query":
                query_parts.append(tokens[2])
    return "".join(target_parts), "".join(query_parts)


def Needleman_Wunsch_alignment(seq1: str, seq2: str) -> tuple[str, str]:
    """Global sequence alignment using BLOSUM62 via Needleman-Wunsch.

    Returns (aligned_seq1, aligned_seq2) with gap characters ('-').
    """
    missing = None
    if "-" in seq1:
        missing = [s == "-" for s in seq1]
        seq1 = seq1.replace("-", "")

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.substitution_matrix = BLOSUM62

    alignment = aligner.align(seq1, seq2)[0]
    aligned1, aligned2 = _extract_aligned_seqs(alignment)

    if missing is None:
        return aligned1, aligned2

    # Re-insert the positions that were originally '-' in seq1
    final1_temp, final2_temp = [], []
    j = 0
    for s in missing:
        if s:
            final1_temp.append("-")
            final2_temp.append("-")
        else:
            while j < len(aligned1) and aligned1[j] == "-":
                final1_temp.append(aligned1[j])
                final2_temp.append(aligned2[j])
                j += 1
            if j < len(aligned1):
                final1_temp.append(aligned1[j])
                final2_temp.append(aligned2[j])
                j += 1
    if j < len(aligned1):
        final1_temp.extend(aligned1[j:])
        final2_temp.extend(aligned2[j:])

    # Remove double-gap positions introduced by the re-insertion
    final1, final2 = [], []
    for a, b in zip(final1_temp, final2_temp):
        if not (a == "-" and b == "-"):
            final1.append(a)
            final2.append(b)
    return "".join(final1), "".join(final2)


# ── PDB I/O ───────────────────────────────────────────────────────────────────

def read_sing_chain_PDB(
    path: str,
    fix_unknown_res: bool = True,
    remove_alternate_res: bool = True,
):
    """Parse a single-chain PDB file and return its chain object."""
    parser = PDB.PDBParser(QUIET=True)
    struc = parser.get_structure("query", path)
    if len(struc) > 1:
        print("Multiple models exist in this PDB file — only the first model is used.")
    struc = struc[0]
    assert len(struc) == 1, "Multiple chains exist in this PDB file!"
    chain = struc.child_list[0]

    if fix_unknown_res:
        deletion = []
        existing_resnum: list[int] = []
        for residue in chain.child_list:
            if residue.resname in EXTERNAL_MAPPINGS:
                print(
                    f"Warning: residue {residue.resname}[{residue.id[1]}] "
                    f"is recognised as {EXTERNAL_MAPPINGS[residue.resname]}"
                )
                residue.resname = EXTERNAL_MAPPINGS[residue.resname]
            elif residue.resname not in toolbox.protein_dict:
                print(f"Warning: unknown residue encountered: {residue.resname}[{residue.id[1]}]")
                deletion.append(residue.id)
            if remove_alternate_res:
                if residue.id[1] in existing_resnum:
                    print(
                        f"Warning: residue {residue.resname}[{residue.id[1]}"
                        f"{residue.id[2]}] ignored (alternate residue)"
                    )
                    deletion.append(residue.id)
                else:
                    existing_resnum.append(residue.id[1])
        for item in deletion:
            chain.detach_child(item)
        saver = PDBSaver()
        saver.set_structure(chain)
        basename = os.path.basename(path)
        saver.save(basename.replace(".pdb", "_fix.pdb"))
    return chain


def chain_to_seq(chain, fasta_output: Optional[str] = None, res_num: bool = True):
    """Extract the amino-acid sequence (and optionally residue numbers) from a chain."""
    residues, resnum = [], []
    for residue in chain.child_list:
        if residue.resname in toolbox.protein_dict:
            residues.append(residue.resname)
            resnum.append(residue.id[1])
    seq = Seq(toolbox.form_seq(residues))
    if fasta_output:
        record = SeqRecord(seq, id="query", description="")
        with open(fasta_output, "w") as f:
            f.write(record.format("fasta"))
    return (seq, resnum) if res_num else seq


# ── BLAST alignment ───────────────────────────────────────────────────────────

class blast_result:
    def __init__(self) -> None:
        self.target_name = ""
        self.score = 0.0
        self.Evalue = 0.0
        self.Lmatch = 0
        self.Tmatch = 0
        self.source_seq = ""
        self.target_seq = ""
        self.coverage = 0.0
        self._last_source_num = 0
        self._last_target_num = 0

    def parse(self, line: str) -> None:
        parts = line.split()
        self.target_name = parts[0]
        self.score = float(parts[-2])
        self.Evalue = float(parts[-1])

    def parse_match(self, line: str) -> None:
        identity_entry = line.split(",")[0]
        number_entry = next(e for e in identity_entry.split() if "/" in e)
        self.Lmatch, self.Tmatch = (int(n) for n in number_entry.split("/"))

    def parse_seq(self, line: str, obj: str) -> None:
        start_num = int(line.split()[1])
        end_num = int(line.split()[3])
        if obj == "source":
            if self.source_seq == "" or start_num == self._last_source_num + 1:
                self.source_seq += line.split()[2]
                self._last_source_num = end_num
        elif obj == "target":
            if self.target_seq == "" or start_num == self._last_target_num + 1:
                self.target_seq += line.split()[2]
                self._last_target_num = end_num

    def calc_coverage(self, len_total: int) -> None:
        assert self.source_seq != "" and self.target_seq != ""
        self.coverage = self.Lmatch / max(self.Tmatch, len_total)


def blast(
    seq,
    db_name: str = "refDB.blastdb",
    cleaning: bool = True,
    return_aligned_seq: bool = False,
    working_dir: Optional[str] = None,
) -> dict:
    if working_dir is None:
        working_dir = "blast/"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)
    if isinstance(seq, str):
        fasta_name = working_dir + os.path.split(seq)[-1]
        shutil.copy(seq, fasta_name)
    else:
        fasta_name = working_dir + "query.fasta"
        record = SeqRecord(Seq(str(seq)), id="query", description="")
        with open(fasta_name, "w") as f:
            f.write(record.format("fasta"))
    cmd = (
        f"{BLAST_DEFAULT_EXE} -db {db_name} -query {fasta_name} "
        f"-out {working_dir}blast.out > /dev/null 2>&1"
    )
    os.system(cmd)
    results: dict[str, blast_result] = {}
    mode = "ignore"
    for line in open(working_dir + "blast.out"):
        if "Sequences producing significant alignments:" in line:
            mode = "add_match"
            continue
        elif line[0] == ">":
            mode = line.split()[0].replace(">", "")
        if mode == "add_match" and line.strip():
            result = blast_result()
            result.parse(line)
            results[result.target_name] = result
        else:
            if mode != "ignore":
                if "Identities =" in line:
                    if results[mode].source_seq == "":
                        results[mode].parse_match(line)
                    else:
                        mode = "ignore"
                elif "Query" in line and return_aligned_seq:
                    results[mode].parse_seq(line, "source")
                elif "Sbjct" in line and return_aligned_seq:
                    results[mode].parse_seq(line, "target")
    for identifier in results:
        results[identifier].calc_coverage(len(seq))
    if cleaning:
        shutil.rmtree(working_dir)
    return results


# ── mTM-align structural alignment ───────────────────────────────────────────

class mTM_align_result:
    def __init__(self, pdbid: str) -> None:
        self.target_name = pdbid
        self.rmsd = 0.0
        self.TMscore = 0.0
        self.source_seq = ""
        self.target_seq = ""
        self.coverage = 0.0

    def parse_alignment(self, source_seq: str, target_seq: str) -> None:
        assert len(source_seq) == len(target_seq)
        for s, t in zip(source_seq, target_seq):
            if s == "-" and t == "-":
                continue
            self.source_seq += s
            self.target_seq += t
        matches = sum(1 for s, t in zip(self.source_seq, self.target_seq) if s == t)
        self.coverage = matches / len(self.source_seq)


def mTM_align(
    source_file: str,
    alignment_candidates: list,
    db_path: Optional[str] = None,
    cleaning: bool = True,
    working_dir: Optional[str] = None,
) -> dict:
    if db_path is None:
        db_path = str(SCRIPT_PATH / "refDB" / "pdbs") + "/"
    if working_dir is None:
        working_dir = "mTM_align/"
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    os.mkdir(working_dir)
    shutil.copy(source_file, working_dir + "query.pdb")
    for candidate in alignment_candidates:
        shutil.copy(db_path + candidate.split(".")[1] + ".pdb", f"{working_dir}{candidate}.pdb")
    with open(working_dir + "inputs", "w") as f:
        f.write("query.pdb\n")
        for candidate in alignment_candidates:
            f.write(f"{candidate}.pdb\n")
    os.chdir(working_dir)
    os.system(f"{MTM_DEFAULT_EXE} -i inputs > /dev/null 2>&1")

    results = {c: mTM_align_result(c) for c in alignment_candidates}

    with open("pairwise_rmsd.txt") as f:
        title = [t.replace(".pdb", "") for t in f.readline().split()]
        query_line = next(l for l in f if "query.pdb" in l)
    for rmsd, candidate_pdb in zip(query_line.split()[1:], title):
        if candidate_pdb != "query":
            results[candidate_pdb].rmsd = float(rmsd)

    with open("pairwise_TMscore.txt") as f:
        title = [t.replace(".pdb", "") for t in f.readline().split()]
        query_line = next(l for l in f if "query.pdb" in l)
    for score, candidate_pdb in zip(query_line.split()[1:], title):
        if candidate_pdb != "query":
            results[candidate_pdb].TMscore = float(score)

    all_alignments = SeqIO.parse("result.fasta", "fasta")
    alignment_seqs: dict = {}
    query_seq = None
    for aln in all_alignments:
        if aln.id == "query.pdb":
            query_seq = aln.seq
        else:
            alignment_seqs[aln.id.replace(".pdb", "")] = aln.seq
    for seq_id, target_seq in alignment_seqs.items():
        results[seq_id].parse_alignment(str(query_seq), str(target_seq))

    os.chdir("../")
    if cleaning:
        shutil.rmtree(working_dir)
    return results


# ── Shift assignment ──────────────────────────────────────────────────────────

def assign_aligned_shifts(
    source_seq: str,
    target_seq: str,
    target_id: str,
    refDB: pd.DataFrame,
    strict: int,
) -> list:
    """Transfer chemical shifts from a reference protein to the query sequence.

    strict: 0 = exact match only, 1 = BLOSUM62 > 0, 2 = permissive.
    Returns a list of dicts (one per query residue) with secondary shift values.
    """
    if (refDB.RES_NUM == np.arange(len(refDB)) + 1).all():
        refDB_seq = toolbox.form_seq(refDB.RESNAME)
    else:
        refDB_seq = ""
        start = 1
        for i in range(len(refDB)):
            refDB_seq += "-" * (refDB.loc[i, "RES_NUM"] - start)
            start = refDB.loc[i, "RES_NUM"] + 1
            refDB_seq += toolbox.form_seq([refDB.loc[i, "RESNAME"]])

    if len(source_seq) != len(target_seq):
        source_seq, target_seq = Needleman_Wunsch_alignment(source_seq, target_seq)
    shift_seq, pdb_seq = Needleman_Wunsch_alignment(refDB_seq, target_seq.replace("-", ""))

    # Map each position in shift_seq to a refDB row
    refDB_seq_shifts: list[dict] = []
    n = 0
    for ch in shift_seq:
        if ch == "-":
            refDB_seq_shifts.append({})
        else:
            residue = toolbox.decode_seq(ch)
            assert residue == refDB.iloc[n]["RESNAME"]
            record = {atom: refDB.iloc[n][atom] for atom in toolbox.ATOMS}
            record["TARGET_RESNAME"] = residue
            record["TARGET_RESNAME_i+1"] = refDB.iloc[n]["RESNAME_i+1"]
            refDB_seq_shifts.append(record)
            n += 1

    # Drop positions where pdb_seq has a gap
    refDB_pdb_shifts: list[dict] = [
        refDB_seq_shifts[i] for i in range(len(shift_seq)) if pdb_seq[i] != "-"
    ]

    # Map to query positions (skipping gaps in target_seq)
    query_ref_pdb_shifts: list[dict] = []
    n = 0
    for ch in target_seq:
        if ch == "-":
            query_ref_pdb_shifts.append({})
        else:
            query_ref_pdb_shifts.append(refDB_pdb_shifts[n])
            n += 1

    # Assign source residue names and apply strict rules
    results: list[dict] = []
    n = 0
    for i, ch in enumerate(source_seq):
        if ch != "-":
            shifts = dict(query_ref_pdb_shifts[i])
            try:
                shifts["SOURCE_RESNAME"] = toolbox.decode_seq(ch)
            except KeyError:
                shifts["SOURCE_RESNAME"] = "UNK"
            results.append(shifts)
            n += 1

    for i, res in enumerate(results):
        if "TARGET_RESNAME" in res:
            accept = (
                strict == 2
                or (strict == 1 and get_blosum_value(res["SOURCE_RESNAME"], res["TARGET_RESNAME"]) > 0)
                or (strict == 0 and res["SOURCE_RESNAME"] == res["TARGET_RESNAME"])
            )
            if accept:
                rc = randcoil_pro if res["TARGET_RESNAME_i+1"] == "PRO" else randcoil_ala
                for atom in toolbox.ATOMS:
                    res[atom] -= rc[atom][res["TARGET_RESNAME"]]
            else:
                results[i] = {}
        else:
            results[i] = {}
    return results


# ── Main prediction entry point ───────────────────────────────────────────────

def main(
    path: str,
    strict: int,
    secondary: bool = False,
    test: bool = False,
    exclude: bool = False,
    shifty: bool = False,
    blast_score_threshold: float = 0,
    e_value_threshold: float = 1e-10,
    long_Tmatch_threshold: int = 40,
    short_Tmatch_threshold: int = 20,
    long_match_percent_threshold: float = 0.15,
    short_match_percent_threshold: float = 0.4,
    TMscore_threshold: float = 0.8,
    rmsd_threshold: float = 1.75,
    coverage_threshold: float = 0.3,
    refDB_shifts_path: Optional[str] = None,
    custom_working_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate chemical shifts for a single PDB file using UCBShift-Y (SHIFTY++).

    Returns a DataFrame with predicted chemical shifts and alignment quality metrics.
    """
    if refDB_shifts_path is None:
        refDB_shifts_path = str(SCRIPT_PATH / "refDB" / "shifts_df") + "/"

    fixname = os.path.basename(path).replace(".pdb", "_fix.pdb")
    seq, resnum = chain_to_seq(read_sing_chain_PDB(path))
    db_name = "train.blastdb" if test else "refDB.blastdb"
    blast_results = blast(seq, db_name=db_name, return_aligned_seq=True,
                          cleaning=not DEBUG, working_dir=custom_working_dir)

    def _empty_result() -> pd.DataFrame:
        residues = toolbox.decode_seq(str(seq))
        df = pd.DataFrame({"RESNAME": residues, "RESNUM": resnum})
        for atom in toolbox.ATOMS:
            df[atom] = np.nan
            df[f"{atom}_BEST_REF_SCORE"] = 0
            df[f"{atom}_BEST_REF_COV"] = 0
            df[f"{atom}_BEST_REF_MATCH"] = 0
        return df

    candidates = []
    for result in blast_results.values():
        if result.score >= blast_score_threshold and result.Evalue <= e_value_threshold:
            long_ok = (result.Tmatch >= long_Tmatch_threshold
                       and result.Lmatch / result.Tmatch >= long_match_percent_threshold)
            short_ok = (result.Tmatch >= short_Tmatch_threshold
                        and result.Lmatch / result.Tmatch >= short_match_percent_threshold)
            if long_ok or short_ok:
                if not exclude or not result.coverage > GLOBAL_TEST_CUTOFF:
                    candidates.append(result)

    if not candidates:
        print("No sequence in database generates possible alignments")
        if os.path.exists(fixname):
            os.remove(fixname)
        return _empty_result()

    final = []
    identities = []

    if shifty:
        best_match = int(np.argmax([item.score for item in candidates]))
        final.append(candidates[best_match])
        identities.append(candidates[best_match].coverage)
        if os.path.exists(fixname):
            os.remove(fixname)
    else:
        candidate_names = [item.target_name for item in candidates]
        source_pdb = fixname if os.path.exists(fixname) else path
        mtm_results = mTM_align(source_pdb, candidate_names,
                                 cleaning=not DEBUG, working_dir=custom_working_dir)
        if os.path.exists(fixname):
            os.remove(fixname)

        blast_scores = []
        for result in mtm_results.values():
            if (result.TMscore > TMscore_threshold
                    and result.rmsd < rmsd_threshold
                    and result.coverage > coverage_threshold):
                identity = blast_results[result.target_name].coverage
                final.append(result)
                identities.append(identity)
                blast_scores.append(blast_results[result.target_name].score)
        if blast_scores:
            normalized_blast_scores = np.array(blast_scores) / np.max(blast_scores)

    if not final:
        print("No significant structure alignment is possible!")
        return _empty_result()

    print(f"Calculating using {len(final)} references with maximal identity {np.max(identities):.2f}")

    refDB = {item.target_name: pd.read_csv(f"{refDB_shifts_path}{item.target_name}.csv")
             for item in final}
    seq_str = str(seq)
    candidate_shifts = [
        assign_aligned_shifts(seq_str, candidate.target_seq, candidate.target_name,
                               refDB[candidate.target_name], strict)
        for candidate in final
    ]

    if shifty:
        scores = [1.0]
    else:
        scores = [candidate.TMscore * normalized_blast_scores[idx]
                  for idx, candidate in enumerate(final)]

    # Reconcile biopython sequence with mTM-align sequence
    if not shifty:
        mtm_recognized_seq = final[0].source_seq.replace("-", "")
        if mtm_recognized_seq != seq_str:
            bp_seq, mtm_seq = Needleman_Wunsch_alignment(seq_str, mtm_recognized_seq)
            bp_seq_list = list(bp_seq)
            for i, m in enumerate(mtm_seq):
                if m == "-" and bp_seq_list[i] != "-":
                    bp_seq_list[i] = "x"
            bp_seq_cleaned = "".join(bp_seq_list).replace("-", "")
            old_resnum = resnum
            seq_str = ""
            resnum = []
            for i, ch in enumerate(bp_seq_cleaned):
                if ch != "x":
                    seq_str += ch
                    resnum.append(old_resnum[i])

    seq_shifts = []
    for i, (char, rnum) in enumerate(zip(seq_str, resnum)):
        resname = toolbox.decode_seq(char)
        next_pro = (
            i + 1 < len(resnum)
            and resnum[i + 1] == rnum + 1
            and seq_str[i + 1] == "P"
        )
        residue_shifts: dict = {"RESNAME": resname, "RESNUM": rnum}
        for atom in toolbox.ATOMS:
            shifts, res_scores, reference_scores = [], [], []
            for cshift, score in zip(candidate_shifts, scores):
                entry = cshift[i]
                if atom in entry and not np.isnan(entry[atom]):
                    blosum_bonus = get_blosum_value(entry["SOURCE_RESNAME"], entry["TARGET_RESNAME"])
                    target_score = np.exp(score * 5) * np.exp(blosum_bonus)
                    shifts.append(entry[atom])
                    res_scores.append(target_score)
                    reference_scores.append(target_score)
                else:
                    reference_scores.append(0)

            if shifts:
                rc_diff = np.sum(np.array(shifts) * np.array(res_scores)) / np.sum(res_scores)
                rc_diff = np.clip(rc_diff, -SS_CAPS[atom], SS_CAPS[atom])
                rc_table = randcoil_pro if next_pro else randcoil_ala
                if secondary:
                    residue_shifts[atom] = rc_diff
                    residue_shifts[f"{atom}_RC"] = rc_table[atom][resname]
                else:
                    residue_shifts[atom] = rc_diff + rc_table[atom][resname]
            else:
                residue_shifts[atom] = np.nan

            max_ref = int(np.argmax(reference_scores))
            if reference_scores[max_ref] > 0:
                best = final[max_ref]
                residue_shifts[f"{atom}_BEST_REF_SCORE"] = best.TMscore * np.exp(-best.rmsd)
                residue_shifts[f"{atom}_BEST_REF_COV"] = min(best.coverage, identities[max_ref])
                residue_shifts[f"{atom}_BEST_REF_MATCH"] = int(
                    candidate_shifts[max_ref][i].get("SOURCE_RESNAME")
                    == candidate_shifts[max_ref][i].get("TARGET_RESNAME")
                )
            else:
                residue_shifts[f"{atom}_BEST_REF_SCORE"] = 0
                residue_shifts[f"{atom}_BEST_REF_COV"] = 0
                residue_shifts[f"{atom}_BEST_REF_MATCH"] = 0
        seq_shifts.append(residue_shifts)

    return pd.DataFrame(seq_shifts)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "UCBShift-Y: predict NMR chemical shifts by transferring shifts from "
            "similar proteins in refDB via sequence (BLAST) and structure (mTM-align) alignment."
        )
    )
    parser.add_argument("input", help="Query PDB file")
    parser.add_argument("--output", "-o", default="shifts.csv", help="Output CSV file")
    parser.add_argument(
        "--strict", "-s", type=int, default=1,
        help="Strictness: 0=exact match, 1=BLOSUM62>0, 2=permissive",
    )
    parser.add_argument("--secondary", "-2", action="store_true",
                        help="Output secondary shifts (observed minus random coil)")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Use the test BLAST database")
    parser.add_argument("--exclude", "-e", action="store_true",
                        help="Exclude mode: skip near-identical reference sequences")
    parser.add_argument("--shifty", "-y", action="store_true",
                        help="SHIFTY mode: use only the top BLAST hit")
    args = parser.parse_args()
    result = main(args.input, strict=args.strict, secondary=args.secondary,
                  test=args.test, exclude=args.exclude, shifty=args.shifty)
    result.to_csv(args.output, index=False)
