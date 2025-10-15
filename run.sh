# Cross-platform venv
if [ -d ".venv/Scripts" ]; then
  # Windows venv
  PY=".venv/Scripts/python.exe"
  PIP=".venv/Scripts/pip.exe"
  # Active le venv Windows si besoin (optionnel)
  # source .venv/Scripts/activate
else
  PY="python3"
  PIP="pip3"
  source .venv/bin/activate
fi

# puis utilise $PY et $PIP partout au lieu de "python"/"pip"
$PY -m pip install --upgrade pip
$PIP install -r requirements.txt

export SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY=copy
export SPEECHBRAIN_LOCAL_FILE_STRATEGY=copy
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

$PY evaluation.py
