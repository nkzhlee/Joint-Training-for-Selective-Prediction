# Training and Evaluation Script

This project contains a script for training, validating, and testing a model for sequence classification and deferral policies using PyTorch and Hugging Face Transformers.

## Features

- Supports training and evaluation of sequence classification models.
- Implements a deferral policy classifier.
- Utilizes Hugging Face Transformers and PyTorch for model training.
- Logs results using [Weights & Biases (WandB)](https://wandb.ai/).
- Calculates key performance metrics, including accuracy, F1 score, and Cohen's Kappa.

## Requirements

### Libraries and Frameworks

Ensure the following dependencies are installed:

- Python 3.7+
- PyTorch
- Transformers
- tqdm
- NumPy
- scikit-learn
- Weights & Biases (wandb)

Install the dependencies using pip:
```bash
pip install torch transformers tqdm numpy scikit-learn wandb
```

### Project Structure

- **`Constants.py`**: Contains file paths and hyperparameters.
- **`DataModules.py`**: Defines the `SequenceDataset` class for loading and preprocessing data.
- **`SFRNModel.py`**: Contains the `SFRNModel` and `DeferralClassifier` implementations.

## Usage

### Arguments

The script takes the following command-line arguments:

| Argument     | Description                              | Default       |
|--------------|------------------------------------------|---------------|
| `--ckp_name` | Name for saving checkpoints              | `debug_cpt`   |
| `--device`   | Device for training (`cuda` or `cpu`)    | `cuda:0`      |

### Example Command

```bash
python ./Joint_Training/main.py --ckp_name my_model_checkpoint --device cuda:0
```

### Training and Evaluation

The training process:

1. **Data Loading**:
    - Loads training and test datasets using `SequenceDataset`.
    - Splits the training set into training and validation subsets.

2. **Model Initialization**:
    - Loads the tokenizer and initializes the `SFRNModel` and `DeferralClassifier`.

3. **Training Loop**:
    - Alternates training phases between classification and deferral policy based on epochs.
    - Tracks and logs metrics such as accuracy, F1 score, and loss.

4. **Validation and Testing**:
    - Evaluates model performance on validation and test datasets.
    - Logs results using WandB.

### Hyperparameters

Hyperparameters are loaded from `Constants.py`. Modify the values as needed for your experiment:

- `random_seed`: Seed for reproducibility.
- `model_name`: Hugging Face model identifier (e.g., `bert-base-uncased`).
- `data_split`: Fraction of data to use for validation.
- `lr`: Learning rate for the classifier.
- `p_lr`: Learning rate for the deferral policy.
- `epochs`: Total number of epochs.
- `alpha`, `beta`, `gamma`: Weights for loss functions.
- `GRADIENT_ACCUMULATION_STEPS`: Steps for gradient accumulation.

## Metrics

The script calculates and logs the following metrics:

- **Accuracy**: Overall classification accuracy.
- **F1 Score**: Macro-averaged F1 score for multi-class classification.
- **Cohen’s Kappa**: Agreement measure considering chance.
- **Deferral Policy Metrics**: Accuracy and F1 score for deferral decisions.

## Checkpoints

Checkpoints are saved after each epoch if validation accuracy and F1 score improve. Checkpoint files are saved in the `checkpoint/` directory.

## Logging

The script uses [WandB](https://wandb.ai/) for logging training and evaluation metrics. Set up your WandB project and entity in the `train()` function:

```python
wandb.init(project="Your_project", entity="Your_username", config=config_dictionary)
```

## Contact

For questions or issues, feel free to open an issue or contact the author Zhaohui Li (nkzhaohuilee@gmail.com).

