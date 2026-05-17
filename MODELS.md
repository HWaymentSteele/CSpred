# Model Weights

The 18 `.sav` files in `ucbshift_weights/` are the trained UCBShift-X (R0, R1, R2) model weights for each of the six predicted atoms (H, Hα, C′, Cα, Cβ, N). They are excluded from git via `.gitignore` and must be downloaded separately.

## Downloading

Download `ucbshift_weights.tgz` from the [Dryad dataset](https://datadryad.org/stash/share/6vbrswTtNRcHk2vV3e6P1QGH1yYMhvdHDlauysTCObE) and extract into the repo root:

```bash
tar -xzf ucbshift_weights.tgz   # produces ucbshift_weights/*.sav
```

## Compressing the model files

The raw model files total ~7 GB. Running `compress_models.py` re-saves them with joblib's built-in compression, typically reducing total size by 60–75% with no loss of accuracy.

```bash
python compress_models.py
```

Options:

```
--models PATH   Path to models directory (default: ./models)
--compress 1-9  Compression level (default: 3, good balance of size vs load time)
```

After compressing, re-package for distribution:

```bash
tar -czf ucbshift_weights.tgz ucbshift_weights/
```

### Choosing a compression level

| Level | Size reduction | Load time impact |
|-------|---------------|-----------------|
| 1     | ~50%          | minimal         |
| 3     | ~65%          | small           |
| 6     | ~70%          | moderate        |
| 9     | ~75%          | significant     |

Level 3 is recommended for a hosted download. Level 1 is better if models are on a fast local disk and load time matters.

## Google Colab

```python
!pip install gdown
!gdown "YOUR_GDRIVE_FILE_ID"   # upload compressed ucbshift_weights.tgz to Drive and share publicly
!tar -xzf ucbshift_weights.tgz

from cspred import predict_shifts
df = predict_shifts("protein.pdb", pH=7.0, TP=False)
```
