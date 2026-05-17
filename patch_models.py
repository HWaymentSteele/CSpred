#!/usr/bin/env python3
"""Patch sklearn 1.1.x tree models to be loadable with sklearn 1.3+.

Strategy: temporarily replace sklearn.tree._tree in sys.modules with a
wrapper so pickle resolves 'Tree' to PatchedTree. The proxy accepts the old
node dtype (missing missing_go_to_left), patches it, infers n_features /
n_classes / n_outputs from the node and value arrays, and emits a proper
sklearn 1.3+ __reduce__ tuple on re-save.

Must be run with scikit-learn >= 1.3 installed.
"""

import sys
import numpy as np
import joblib
import argparse
from pathlib import Path

NEW_NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold',
              'impurity', 'n_node_samples', 'weighted_n_node_samples',
              'missing_go_to_left'],
    'formats': ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8', 'u1'],
    'offsets': [0, 8, 16, 24, 32, 40, 48, 56],
    'itemsize': 64,
})

# Pre-import all Cython modules that do a C-level Tree size check at init time.
import sklearn.ensemble
import sklearn.ensemble._gb
import sklearn.tree
import sklearn.tree._tree as _real_tree_module
from sklearn.tree._tree import Tree as _RealTree


class PatchedTree:
    """Proxy that accepts old/broken node arrays and re-saves in sklearn 1.3+ format."""

    def __new__(cls, *args):
        return object.__new__(cls)

    def __setstate__(self, state):
        nodes = state.get('nodes')
        if nodes is not None and 'missing_go_to_left' not in nodes.dtype.names:
            new_nodes = np.zeros(len(nodes), dtype=NEW_NODE_DTYPE)
            for field in nodes.dtype.names:
                new_nodes[field] = nodes[field]
            new_nodes['missing_go_to_left'] = 1
            state['nodes'] = new_nodes
        self._state = state

    def __reduce__(self):
        nodes = self._state['nodes']
        values = self._state['values']
        # Infer constructor args from arrays (regression: n_outputs=1, n_classes=[1])
        n_outputs = values.shape[1]
        n_classes = np.ones(n_outputs, dtype=np.intp)
        valid_feat = nodes['feature'][nodes['feature'] >= 0]
        n_features = int(valid_feat.max()) + 1 if len(valid_feat) > 0 else 1
        # Emit (Tree, (n_features, n_classes, n_outputs), state) — sklearn 1.3+ format
        return (_RealTree, (n_features, n_classes, n_outputs), self._state)


class _PatchedTreeModule:
    """sys.modules wrapper so pickle resolves 'Tree' to PatchedTree."""
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        if name == 'Tree':
            return PatchedTree
        return getattr(self._real, name)


def patch_models(models_dir: Path, compress: int = 3) -> None:
    sav_files = sorted(models_dir.glob("*.sav"))
    if not sav_files:
        raise FileNotFoundError(f"No .sav files found in {models_dir}")

    import sklearn
    print(f"sklearn {sklearn.__version__} — patching {len(sav_files)} files in {models_dir}\n")

    for sav in sav_files:
        sys.modules['sklearn.tree._tree'] = _PatchedTreeModule(_real_tree_module)
        try:
            model = joblib.load(sav)
        finally:
            sys.modules['sklearn.tree._tree'] = _real_tree_module

        joblib.dump(model, sav, compress=compress)
        print(f"  {sav.name} — done ({sav.stat().st_size/1e6:.1f} MB)")

    print("\nVerifying all files load cleanly ...")
    for sav in sav_files:
        joblib.load(sav)
        print(f"  {sav.name} OK")

    print("\nAll models patched. scikit-learn >= 1.3 version cap can now be removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch UCBShift models for sklearn 1.3+ compatibility.")
    parser.add_argument("--models", default="ucbshift_weights", help="Path to models directory")
    parser.add_argument("--compress", type=int, default=3, metavar="1-9")
    args = parser.parse_args()
    patch_models(Path(args.models), compress=args.compress)
