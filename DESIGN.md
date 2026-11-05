# Design Doc: Cricket Umpire Signal Classifier

## Overview

An image classifier that identifies which signal a cricket umpire is giving — **out**, **six**, or **wide** — from a single photo. Built as a transfer-learning fine-tune of `resnet18` on a small, manually collected image set.

## Goals

- Given one image of an umpire, predict one of three signal classes.
- Keep the pipeline simple enough to run end-to-end on a laptop CPU (or a single GPU if available) with three short scripts: build folders, train, test.

## Non-goals

- No video input, no real-time/streaming inference.
- No detection/localization of the umpire in a larger scene — inputs are assumed to already be cropped/framed on the umpire.
- No web UI, API, or packaged app around the model. This project is the training/eval pipeline only.
- No automated data collection — image gathering into `data/` is a manual (or external, separately-run) step.

## Architecture

### Pipeline

```
setup_folders.py  →  data/train|val/{out,six,wide}/*.jpg  (images added manually)
                              │
                              ▼
                          train.py  →  best_model.pt
                              │
                              ▼
                         test_model.py  →  per-image predictions + accuracy
```

### Model

- **Backbone:** `torchvision.models.resnet18`, initialized with `IMAGENET1K_V1` pretrained weights.
- **Head:** final fully-connected layer replaced with `nn.Linear(in_features, 3)` for the 3 signal classes.
- **Why resnet18 + transfer learning:** the dataset is tiny (tens of images per class), so training a CNN from scratch isn't viable. A pretrained backbone gives useful low/mid-level visual features (edges, poses, limb shapes) for free; only the classifier head and fine-tuned weights need to adapt to umpire signals specifically.

### Training (`train.py`)

- Data loaded via `torchvision.datasets.ImageFolder`, which infers class labels from subfolder names under `data/train` and `data/val`.
- Train-time augmentation: random resized crop (224px, scale 0.7–1.0), random rotation (±15°), color jitter (brightness/contrast/saturation), random horizontal flip — needed to squeeze more effective variety out of a small dataset and reduce overfitting.
- Val-time transform: plain resize(256) + center-crop(224), no augmentation.
- Both splits normalized with standard ImageNet mean/std, matching what the pretrained backbone expects.
- Optimizer: SGD (lr=0.001, momentum=0.9) with a step LR scheduler (decay ×0.1 every 7 epochs), 15 epochs total, batch size 8.
- Model selection: after each epoch's validation pass, the checkpoint is saved only if validation accuracy improves — so `best_model.pt` holds the best-val-accuracy epoch, not simply the last epoch. The state dict is staged in a temp directory during training and copied to `best_model.pt` at the root only once training completes.
- Only the `state_dict` is persisted (not the full model object), so any script loading `best_model.pt` must reconstruct the same architecture first — `test_model.py` does this explicitly.

### Evaluation (`test_model.py`)

- Rebuilds a fresh `resnet18` (no pretrained weights this time — `weights=None`) with a 3-class head, then loads `best_model.pt` into it.
- Runs inference over `data/test` (also an `ImageFolder`), printing a ✓/✗ line per image (actual vs. predicted class) and an overall accuracy percentage.
- No confusion matrix, precision/recall, or per-class breakdown currently — accuracy is the only aggregate metric.

## Data

- Three classes: `out`, `six`, `wide`.
- Split across `data/train`, `data/val`, `data/test`, each with one subfolder per class (`ImageFolder` convention).
- Current volume is very small (tens of images per split-class), so metrics from `test_model.py` are directional signals, not statistically reliable measurements — this is the main practical constraint on the whole project.
- `data/test` uses capitalized folder names (`Out`, `Six`, `Wide`) while `train`/`val` use lowercase. `ImageFolder` sorts classes alphabetically per split, and both casings happen to sort into the same relative order, so label indices still align correctly today — but this is fragile if new classes are ever added, and should be made consistent.

## Known limitations

- **`requirements.txt` is a raw environment dump**, not a curated dependency list — it's missing `torch`/`torchvision` (both required to run every script here) and includes unrelated packages pulled in from other work on the same machine/venv (`openai`, `pygame`, `wxPython`, `pyicloud`, `icrawler`, etc.).
- **No automated tests.** `pytest` appears in `requirements.txt` but there's no test suite.
- **No single-image inference script.** `test_model.py` only evaluates against a labeled `data/test` folder; there's no entry point that takes one arbitrary image and returns a predicted class.
- **Small, manually curated dataset** — no documented source, licensing, or collection methodology for the images currently in `data/`.
