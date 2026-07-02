# Computer Vision

This repository collects computer vision experiments and notebooks/scripts in one place.

## Structure

- `CNN_Flower/` - TensorFlow image classification experiment for the TF Flowers dataset.
- `Mnist/` - MNIST preprocessing and dense model experiment.
- `Datasets/` - local data cache and extracted datasets. This folder is ignored in Git and should stay out of the public repo.
- `venv/` - local Python virtual environment. This folder is ignored in Git.

## Working Rules

- Keep each new experiment in its own folder.
- Add a short `README.md` inside a new experiment folder when the project grows.
- Commit code, config, and small sample assets only.
- Keep raw datasets, model checkpoints, and training logs out of the repository.

## Current Projects

- `CNN_Flower/cnn.py` - CNN built with TensorFlow/Keras for flower classification.
- `Mnist/ann.py` - MNIST preprocessing pipeline with a dense neural network.

## Suggested Next Step

If you want to publish this publicly, initialize Git at the repository root, create a GitHub repo named `computer-vision` or similar, and push only the tracked source files.
