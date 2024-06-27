# Training Guide

## Introduction

This guide provides instructions on how to start training a BERT model for sequence classification using the provided script. It also describes the locations of training and testing files.

## Prerequisites

- Ensure you have installed the required libraries: `pandas`, `numpy`, `transformers`, `datasets`, `sklearn`, and `wandb`.
- Place your labeled CSV file (`labeledData.csv`) in the correct directory.

```python
.
├── data
│   ├── df1.csv
│   └── df2.csv
├── data_test.ipynb
├── labeledData.csv
├── result
│   ├── sentiment_result_1.csv
│   ├── sentiment_result_2.csv
│   ├── tokenized_1.csv
│   └── tokenized_2.csv
├── saved_models
│   └── epochs_8
├── sentiment_model.py
└── tmp_trainer
```

## Implementation

Directly execute the code, the test result and best model will be save accordingly.

```python
python sentiment_model.py
```

Approximate training time is 3min on 4x4090 GPUs.
The best model's weight, test data and training data can be found in [Google Cloud](https://drive.google.com/drive/folders/1XAWkesZweyD60VHO32IUOncZWZ4yvyWi).
