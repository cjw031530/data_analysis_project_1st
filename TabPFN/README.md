# TabPFN classification

Run from the repository root after placing the source CSV there and installing `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe -m TabPFN.train
```

The official `tabpfn` package supplies the pretrained TabPFN v2 checkpoint used here. Its `fit` call prepares a labeled context; it does not retrain the checkpoint. The first run downloads the checkpoint to the user's cache outside this repository. The v2 weights are publicly downloadable and covered by the [Prior Labs License](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE), which requires attribution. CPU is selected explicitly because this environment's PyTorch installation is CPU only. Parameters are in `conf/train.yaml`.

This run uses the available NVIDIA GPU through an optional CUDA PyTorch wheel installed beside the model code, so the CPU wheel used by TabICL remains in place:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --target TabPFN\.torch_cuda "torch==2.14.1+cu126" --index-url https://download.pytorch.org/whl/cu126
$env:PYTHONPATH = (Resolve-Path 'TabPFN\.torch_cuda').Path
.\.venv\Scripts\python.exe -m TabPFN.train features.selector_jobs=2 model.device=cuda
.\.venv\Scripts\python.exe -m TabPFN.evaluate
```

When a run is trained with `model.device=cuda`, set `PYTHONPATH` to the same CUDA wheel directory for evaluation. The standard `requirements.txt` install supports the default CPU run.

The script selects 40 features using XGBoost gain on a reproducible sample from 202306–202401, fits a 1000-row stratified context, chooses a classification threshold on 202402–202403, and evaluates 202404–202405 using a context through 202403. It writes `decision.json` and `metrics.json` under `TabPFN/outputs/`. The test period is not used in feature selection or threshold choice. The undersampled file's metrics are unweighted and do not represent the original population rate.

To repeat the 202404–202405 holdout evaluation from the source CSV, or predict future rows from a CSV with the same feature columns (TARGET optional):

```powershell
.\.venv\Scripts\python.exe -m TabPFN.evaluate
.\.venv\Scripts\python.exe -m TabPFN.evaluate paths.input=future.csv context_end=202405 input_start=null input_end=null
```

Evaluation reconstructs the sampled context from the original CSV and checks its SHA-256 hash against the saved decision. `predictions.csv` is written inside the model output directory by default. Published decisions, metrics, and predictions are included in Git; checkpoints and temporary outputs are excluded.
