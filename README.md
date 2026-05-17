# UCBShift

UCBShift is a program for predicting chemical shifts for backbone atoms and β-carbon of a protein in solution. 

## Quick start

This fork modernizes UCBShift for Python 3.8+ and current dependencies, and is intended for **academic use only**, consistent with the original UC Regents license.

### Install

```bash
pip install git+https://github.com/HWaymentSteele/CSpred.git
```

### Model weights

The trained model weights (18 `.sav` files) are hosted on Hugging Face at [hkws/ucbshift_weights](https://huggingface.co/hkws/ucbshift_weights/tree/main). Download them into a folder named `ucbshift_weights/` in your working directory:

```python
from huggingface_hub import snapshot_download
import os

repo_id = "hkws/ucbshift_weights"
local_dir = "./ucbshift_weights"

if not os.path.exists(f"{local_dir}/CA_R0.sav"):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
```

### Python / Jupyter / Colab usage

```python
from cspred import predict_shifts

# ML only — no external tools required, works in Colab
df = predict_shifts("protein.pdb", pH=7.0, TP=False)

# ML + transfer prediction (best accuracy, requires BLAST and mTM-align)
df = predict_shifts("protein.pdb", pH=7.0)

# Transfer prediction only
df = predict_shifts("protein.pdb", pH=7.0, ML=False)
```

### Full Colab setup

```python
# Install package
!pip install git+https://github.com/HWaymentSteele/CSpred.git

# Download model weights from Hugging Face
from huggingface_hub import snapshot_download
import os

repo_id = "hkws/ucbshift_weights"
local_dir = "./ucbshift_weights"

if not os.path.exists(f"{local_dir}/CA_R0.sav"):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)

# Run prediction
from cspred import predict_shifts
df = predict_shifts("protein.pdb", pH=7.0, TP=False)
df.head()
```

> **Note:** The transfer prediction module (UCBShift-Y) requires BLAST and mTM-align, which are not available in Colab. Use `TP=False` for ML-only predictions in Colab environments.

## Publication
Li, J., Bennett, K. C., Liu, Y., Martin, M. V., & Head-Gordon, T. (2020). Accurate prediction of chemical shifts for aqueous protein structure on “Real World” data. _Chemical Science_, 11(12), 3180-3191. DOI: [10.1039/C9SC06561J](https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc06561j)

## Using UCBShift through NMRBox
The original UCBShift code is available on NMRBox, which provides out-of-box using experience for UCBShift in their virtual machines. You can sign up for NMRBox here: https://nmrbox.nmrhub.org/

The rest of this README has been removed to indicate that this version of the code has been modified and stress-tested in the context of the above api usage. Users should see documentation at the original fork for original command-line usage.

## License
Copyright ©20xx  The Regents of the University of California (Regents). All Rights Reserved. Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and not-for-profit purposes, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright notice, this paragraph and the following paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 408, Berkeley, CA 94704-1362, otl@berkeley.edu.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
