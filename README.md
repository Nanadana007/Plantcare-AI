# PlantCare AI

PlantCare AI is a Flask + MobileNetV2 application for plant disease detection.

## Quick Start (Any PC/Laptop)

This project includes a universal launcher (`run.py`) that works on Windows, macOS, and Linux.

### 0) Clone the repository

```bash
git clone https://github.com/Nanadana007/Plantcare-AI.git
cd Plantcare-AI
pip install -r requirements.txt
python run.py
```

### 1) Install Python

Install Python `3.10` or `3.11` and confirm:

```bash
python --version
```

If `python` is not found on Windows, use `py`.

### 2) Go to project folder

```bash
cd "/path/to/PLANTAI"
```

### 3) Run the app

Windows (CMD):

```bat
py -3.10 run.py
```

Windows (PowerShell):

```powershell
py -3.10 .\run.py
```

macOS/Linux:

```bash
python3 run.py
```

Note: if port `5000` is busy, launcher automatically picks next free port (`5001`, `5002`, ...).

Alternative launchers:

```bat
run.bat
```

```bash
bash run.sh
```

### 4) Open in browser

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## What `run.py` does automatically

- Creates `.venv` if missing
- Installs dependencies from `requirements.txt`
- Creates runtime folders (`models`, `uploads`, `results`, `static/images`)
- Verifies model files exist
- Starts Flask server on port `5000`

## Model Files Required

At least one model file must exist in `models/`:
- `best_model.keras` (preferred)
- `best_model.h5`
- `Pretrained_model.h5`

Class names file (recommended):
- `models/class_names.json`

If files are missing, `run.py` will automatically download default public files from Hugging Face on first run.

You can also override with your own URLs:

```bash
# macOS/Linux (bash/zsh)
export PLANTCARE_MODEL_URL="https://.../best_model.keras"
export PLANTCARE_CLASS_NAMES_URL="https://.../class_names.json"
python3 run.py
```

```bat
:: Windows CMD
set PLANTCARE_MODEL_URL=https://.../best_model.keras
set PLANTCARE_CLASS_NAMES_URL=https://.../class_names.json
py -3.10 run.py
```

```powershell
# Windows PowerShell
$env:PLANTCARE_MODEL_URL="https://.../best_model.keras"
$env:PLANTCARE_CLASS_NAMES_URL="https://.../class_names.json"
py -3.10 .\run.py
```

Optional local overrides (relative or absolute paths):
- `PLANTCARE_MODEL_PATH`
- `PLANTCARE_CLASS_NAMES_PATH`

## Docker Run

```bash
docker build -t plantcare-ai .
docker run --rm -p 5000:5000 plantcare-ai
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Training (Optional)

If you want to train your own model:

```bash
python -m src.train --data_dir /absolute/path/to/New_Plant_Diseases_Dataset
```

Training outputs:
- `models/best_model.keras`
- `models/class_names.json`
- plots/reports in `results/`

## Troubleshooting

- `Model not found`:
  Add model file in `models/` (`best_model.keras` recommended), then rerun.
- `Unsupported Input`:
  Model is trained for specific crop leaf classes (PlantVillage scope). Upload a clear close-up single-leaf image from supported crops.
- TensorFlow install issue:
  Use Python `3.10` or `3.11`, then rerun `run.py`.
- Port already in use:
  Launcher auto-finds a free port. You can also force one:
  - macOS/Linux: `python3 run.py --port 5510`
  - Windows CMD: `py -3.10 run.py --port 5510`

## Flask Routes

- `GET /` or `GET /home` -> home page
- `GET /about` -> about page
- `GET /upload` -> upload page
- `POST /predict` -> prediction API
- `GET /result` -> result page
- `GET /health` -> health check
