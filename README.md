# aariz-cephalometric-landmark-detection
External Validation of Deep Learning for Cephalometric Landmark Detection

Training and evaluation code accompanying the manuscript "External Validation of Deep Learning for Cephalometric Landmark Detection" ( submitted for peer review).

Overview

This repository contains the full data pipeline, model architectures, and evaluation code used to:


Train a heatmap-regression U-Net and a multi-resolution (HRNet-style) comparison architecture on the Aariz cephalometric benchmark (1,000 lateral cephalograms, 7 imaging devices, 29 landmarks).
Evaluate internal performance on Aariz's official test split, benchmarked against previously published methods.
Externally validate both models — without any retraining or fine-tuning — on the independent ISBI 2015 Grand Challenge dataset (400 lateral cephalograms, different institution, country, and imaging device).


The central finding: a model with internal performance competitive with published benchmarks shows a substantial, clinically consequential generalization gap when evaluated on a genuinely independent external cohort — concentrated in the landmarks defining the SNA, SNB, and ANB angles central to orthognathic surgical diagnosis.

Repository Contents

FileDescriptionaariz_dataset.pyPyTorch Dataset for the Aariz benchmark. Handles landmark parsing, pixel-to-mm conversion (device-aware), heatmap generation, and supports both individual-annotator (Junior/Senior) and the dataset's documented combined ground-truth convention.isbi2015_dataset.pyPyTorch Dataset for the ISBI 2015 external validation set, with the 19-landmark harmonization mapping onto Aariz's 29-landmark scheme.unet_model.pyU-Net baseline architecture (7.7M parameters) and a legacy variant for loading earlier checkpoints trained at a different output resolution.hrnet_model.pyHRNet-style multi-resolution comparison architecture (4.6M parameters). A simplified variant of the published HRNet design — see manuscript Methods for details.train_eval.pyTraining loop (with checkpointing), evaluation harness (Mean Radial Error, Successful Detection Rate, bootstrap confidence intervals), device-stratified analysis, and failure-case review utilities.

Data

Neither dataset is redistributed in this repository. Both are publicly available from their original sources:


Aariz: Figshare, DOI 10.6084/m9.figshare.27986417.v1 (CC-BY license)
ISBI 2015: Kaggle mirror


Usage

pythonfrom aariz_dataset import AarizDataset
from unet_model import UNet
from train_eval import run_training, evaluate, summarize_errors

# Train
model = run_training(
    root="path/to/Aariz",
    output_dir="path/to/checkpoints",
    n_epochs=100,
    batch_size=16,
    model_class=UNet
)

# Evaluate on the official test split, using the dataset's documented
# combined (Junior+Senior) ground truth convention
test_ds = AarizDataset(root="path/to/Aariz", split="test",
                        annotator=AarizDataset.COMBINED_LABEL, heatmap_size=256)
errors = evaluate(model, test_ds, device="cuda")
summary = summarize_errors(errors)

See the manuscript's Methods section for full training hyperparameters and evaluation protocol.
