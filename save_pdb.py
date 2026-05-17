#!/usr/bin/env python3

from pathlib import Path
from typing import Union


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
                    f"ATOM {atom_counter:6d}{atom.name:>5s} {residue.resname:3s} "
                    f"{self.structure.id} {residue.get_id()[1]:3d}     "
                    f"{atom.coord[0]:7.3f} {atom.coord[1]:7.3f} {atom.coord[2]:7.3f}"
                    f"  1.00 {atom.bfactor:5.2f}         {atom.element:>3s}\n"
                )
        with open(address, "w") as f:
            f.writelines(contents)
