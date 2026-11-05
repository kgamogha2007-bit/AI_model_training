# Cricket Umpire Signal Classifier

## Setup
1. Create a virtual environment: python -m venv venv
2. Activate it: venv\Scripts\activate (Windows)
3. Install dependencies: pip install -r requirements.txt

## Files
- setup_folders.py — creates the data folder structure
- train.py — trains the model on data/train and data/val
- test_model.py — tests the trained model on data/test
- best_model.pt — the trained model weights
- data/ — training/validation/test images, organized by class