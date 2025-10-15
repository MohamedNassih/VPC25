# VPC25 — Voice Privacy Challenge

Ce dépôt contient un pipeline complet pour l’anonymisation de voix et l’évaluation locale en vue de participer à **VPC25**. Il s’appuie sur des embeddings **ECAPA (SpeechBrain)**, une sélection/synthèse contrôlée, et un post-traitement acoustique léger pour préserver l’intelligibilité tout en améliorant l’anonymat.

> ✅ Cette version inclut des instructions **Windows-compatibles** (symlinks HF, FFmpeg, exécution via `python -m`) et un mode **CPU** (GPU fortement conseillé pour la vitesse).

---

## 1) Prérequis

* **Python 3.10–3.12** (testé avec 3.12)
* **ffmpeg/ffprobe** installés et accessibles dans le PATH

  * Windows (recommandé) : copiez aussi `ffmpeg.exe` et `ffprobe.exe` dans `.venv/Script[s]` (voir plus bas)
* Optionnel mais recommandé : **GPU NVIDIA + CUDA** compatible avec votre PyTorch

---

## 2) Installation rapide

```bash
# Créer et activer un venv
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

# Dépendances
pip install -r requirements.txt
```

### 2.1 Spécifique Windows (FFmpeg + stratégies locales SpeechBrain)

Dans PowerShell **après activation du venv** :

```powershell
# Rendez FFmpeg omniprésent
Copy-Item "C:\ffmpeg\bin\ffmpeg.exe"  ".\.venv\Scripts\ffmpeg.exe"  -Force
Copy-Item "C:\ffmpeg\bin\ffprobe.exe" ".\.venv\Scripts\ffprobe.exe" -Force

# Variables d'environnement pour cette session
$env:PATH = ".\.venv\Scripts;C:\ffmpeg\bin;$env:PATH"
$env:FFMPEG_BINARY=".\.venv\Scripts\ffmpeg.exe"
$env:FFPROBE_BINARY=".\.venv\Scripts\ffprobe.exe"
$env:IMAGEIO_FFMPEG_EXE=".\.venv\Scripts\ffmpeg.exe"

# Empêcher les symlinks lors des fetchs SpeechBrain/HF sous Windows
$env:SPEECHBRAIN_LOCAL_DOWNLOAD_STRATEGY="copy"
$env:SPEECHBRAIN_LOCAL_FILE_STRATEGY="copy"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"

# (Facultatif) Ponts utiles
pip install --upgrade imageio imageio-ffmpeg
```

---

## 3) Données attendues

Placez les jeux d’essai dans `evaluation_data/` :

```
evaluation_data/
  Enrollment/
    spk0001/*.wav
    spk0002/*.wav
    ...
  Trial/
    spk0001/*.wav
    spk0002/*.wav
    ...
```

* WAV mono, 16 kHz/16 bits recommandés (le pipeline resample si besoin).

---

## 4) Construction du pool d’x-vectors (ECAPA)

Créez un dictionnaire d’enceintes anonymes (pool) depuis `Enrollment` :

```powershell
# Windows / CPU
python -m scripts.build_xvector_pool `
  --enroll-root evaluation_data/Enrollment `
  --out parameters/pool/xvector_pool.npz `
  --per-speaker-avg `
  --max-files-per-speaker 10 `
  --device cpu
```

Options utiles :

* `--device cuda` pour activer le GPU.
* `--local-ckpt <dossier>` si vous avez téléchargé le modèle ECAPA en local via `huggingface_hub.snapshot_download`.

### 4.1 (Windows, cas rare) Erreur de symlink HF

Si vous rencontrez `WinError 1314` malgré les variables ci-dessus :

1. Téléchargez localement le modèle ECAPA.

   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("speechbrain/spkrec-ecapa-voxceleb", local_dir="pretrained_models/spkrec-ecapa-voxceleb")
   ```
2. (Si nécessaire) Éditez `pretrained_models/spkrec-ecapa-voxceleb/hyperparams.yaml` et mettez `collect_in: null` dans la section `pretrainer:`.
3. Copiez `label_encoder.txt` vers `label_encoder.ckpt` dans le même dossier.
4. Relancez les scripts avec `--local-ckpt pretrained_models/spkrec-ecapa-voxceleb`.

---

## 5) Sanity-check local (petit sous-ensemble)

Pour vérifier rapidement la chaîne :

```powershell
python -m scripts.sanity_eval_local `
  --root evaluation_data `
  --split Trial `
  --speakers 2 `
  --files-per-speaker 2 `
  --out evaluation_data/anonymized_local `
  --csv logs/sanity_eval_local.csv `
  --compute-ecapa-sim
```

Vous obtiendrez quelques fichiers anonymisés + un CSV avec une similarité ECAPA moyenne (plus bas = meilleur anonymat).

---

## 6) Évaluation complète

```powershell
# Windows
python .\evaluation.py

# Linux/macOS
python evaluation.py
```

* Produit des WAV anonymisés et **results.csv** à la racine.
* Sur CPU, l’exécution est plus lente — privilégiez `--device cuda` dans les scripts si vous avez un GPU.

### 6.1 Paramétrage

Le fichier **`parameters/config.yaml`** centralise les hyperparamètres (niveau d’anonymisation, post-traitement, etc.).

* **`parameters/pseudo_map.json`** : mappage pseudonyme (si vous fixez des seeds/
  correspondances stables entre runs).

---

## 7) Soumission (exemple)

```powershell
Compress-Archive -Path "evaluation_data\Anonymized\*" -DestinationPath "submission_anonymized.zip" -Force
Copy-Item results.csv submission_results.csv -Force
```

Vérifiez le format attendu par la compétition et renommez si nécessaire.

---

## 8) Détails des modules

* `utils/dsp.py` : primitives DSP (préaccentuation, filtres all-pass/warp, normalisation de loudness, etc.)
* `utils/features.py` : extraction d’embeddings ECAPA (SpeechBrain) et traits acoustiques auxiliaires
* `utils/selection.py` : sélection des cibles anonymes (pool d’x-vectors, stratégie de proximité/anti-proximité)
* `utils/synth.py` : re-synthèse/warping de caractéristiques, génération anonymisée
* `utils/postproc.py` : post-traitement (normalisation LUFS, fade, anti-click, dither optionnel)
* `scripts/build_xvector_pool.py` : création du pool d’x-vectors
* `scripts/sanity_eval_local.py` : smoke-test local sur un petit sous-ensemble
* `model.py` : pipeline haut-niveau orchestrant extraction → anonymisation → post-proc

> **Exécution recommandée des scripts** : `python -m scripts.<nom>` pour éviter les soucis d’import relatifs.

---

## 9) Bonnes pratiques & réglages

* **Équilibre anonymat/qualité** : démarrez avec les valeurs par défaut de `config.yaml`, puis ajustez le degré de transformation (warp, ratio de sélection anti-proche) et la normalisation LUFS pour minimiser l’impact WER.
* **Pool plus robuste** : augmentez `--max-files-per-speaker` et/ou diversifiez les speakers Enrollment.
* **Déterminisme** : fixez `seed` dans `config.yaml` pour reproduire un run.

---

## 10) Dépannage

* **`ffmpeg was not found`** :

  * Confirmez `ffmpeg -version` dans le même shell.
  * Sous Windows, copiez `ffmpeg.exe`/`ffprobe.exe` dans `.venv/Script[s]` et exportez les variables d’environnement ci-dessus.

* **`WinError 1314` (symlink)** :

  * Assurez `SPEECHBRAIN_LOCAL_*=copy` et `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
  * Si persistant : mettez `collect_in: null` dans le `hyperparams.yaml` du modèle téléchargé localement et créez les fichiers cibles (ex. `label_encoder.ckpt`).

* **`AttributeError: torchaudio.list_audio_backends`** :

  * Utilisez une paire **torch/torchaudio** cohérente (ex. 2.9.0+cpu / 2.9.0+cpu). Réinstallez torchaudio si nécessaire.

* **Lenteur CPU** :

  * Passez au GPU (`--device cuda` dans les scripts) ou réduisez temporairement le nombre de fichiers/speakers pour tester.

