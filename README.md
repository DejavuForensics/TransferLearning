# Neural Network Transfer Learning with CIFAR-10

This repository contains an implementation of transfer learning using pre-trained neural networks with the CIFAR-10 dataset. The project demonstrates how to extract features from images using both pre-trained networks and custom CNNs, and then use these features to train an SVM (Support Vector Machine) with cross-validation.

## Project Structure

- `FeatureExtractor/`: Contains scripts for feature extraction using neural networks
- `SVM/`: Implementation of SVM with cross-validation
- `AntivirusDataset/`: Example dataset for demonstration

## Features

- Feature extraction using pre-trained networks from CIFAR-10
- Training custom CNNs for feature extraction
- Classification using SVM with cross-validation
- Results visualization and performance metrics
- Hybrid approach combining shallow and deep network features

## Requirements

Project requirements are listed in the `requirements.txt` file. To install all dependencies, follow the installation instructions specific to your operating system:

- [Windows Installation](docs/installation_windows.md)
- [Linux Installation](docs/installation_linux.md)

## Usage

### 1. Feature Extraction

To extract features using the ResNet model:
```bash
# On Windows
python FeatureExtractor/extract_features.py --model resnet --data_dir AntivirusDataset

# On Linux
python3 FeatureExtractor/extract_features.py --model resnet --data_dir AntivirusDataset
```

Available options:
- `--model`: Choose between 'resnet' or 'lenet' to use pre-trained models, or any other option to train from scratch
- `--data_dir`: Directory containing images for feature extraction

The script will generate a LIBSVM format file: `FeatureExtractor/TransferLearningAntivirus.libsvm`

### 2. SVM Classification

To run SVM classification:
```bash
# On Windows
python SVM/svm_classifier.py -dataset FeatureExtractor/TransferLearningAntivirus.libsvm

# On Linux
python3 SVM/svm_classifier.py -dataset FeatureExtractor/TransferLearningAntivirus.libsvm
```

Where:
- `-dataset`: Path to the libsvm file (optional, default: 'FeatureExtractor/TransferLearningAntivirus.libsvm')

The script will:
- Load the extracted features
- Train the SVM with cross-validation
- Display performance metrics
- Generate a detailed HTML report: `SVM/results/svm_results_report.html`

The HTML report includes:
- Confusion matrices for each fold
- Performance metrics (accuracy, precision, recall, F1-score, AUC)
- Overall statistics
- Visualizations

### 3. Hybrid Approach: Combining Shallow and Deep Networks

This project also supports a hybrid approach that combines features from both shallow and deep networks. This approach can potentially capture both low-level and high-level features for improved classification performance.

To use the hybrid approach:

1. First, convert your dataset to LIBSVM format:
```bash
python SVM/converter_libsvm.py Antivirus_Dataset_PE32_Ransomware_mELM_format.csv Antivirus_Dataset_PE32_Ransomware_SVM_format.libsvm
```

2. Then, merge the features from deep networks with the shallow network features:
```bash
python FeatureExtractor/join_repository.py FeatureExtractor/TransferLearningAntivirus.libsvm Antivirus_Dataset_PE32_Ransomware_SVM_format.libsvm
```

3. Finally, train the SVM classifier on the combined features:
```bash
python SVM/svm_classifier.py -dataset Joined_TransferLearning.libsvm
```

This hybrid approach:
- Combines features from both shallow and deep networks
- Potentially captures a wider range of feature representations
- May improve classification performance by leveraging complementary feature sets
- Uses cross-validation to ensure robust performance evaluation

The results will be saved in the same format as the standard approach, with an HTML report available at `SVM/results/svm_results_report.html`.

### 4. Parameterizable SVM with Grid/Random Search

This project also provides a more flexible SVM implementation that allows for extensive parameter tuning through grid search, halving search or random search. This can help find the optimal hyperparameters for your specific dataset.

To use the parameterizable SVM:

```bash
python SVM/svm_parameters.py --search random --n-iter 10 --cv 3 --C-min -1 --C-max 1 --C-steps 3 --g-min -1 --g-max 1 --g-steps 3 --kernels rbf linear --dataset FeatureExtractor/TransferLearningAntivirus.libsvm
```

Available parameters:
- `--search`: Search strategy ('grid', 'halving' or 'random')
- `--n-iter`: Number of iterations for random search
- `--cv`: Number of cross-validation folds
- `--C-min`: Minimum value for C parameter (log scale)
- `--C-max`: Maximum value for C parameter (log scale)
- `--C-steps`: Number of steps for C parameter
- `--g-min`: Minimum value for gamma parameter (log scale)
- `--g-max`: Maximum value for gamma parameter (log scale)
- `--g-steps`: Number of steps for gamma parameter
- `--kernels`: Space-separated list of kernels to try (e.g., 'rbf linear')
- `--dataset`: Path to the libsvm file

Example configurations:

1. Grid search with specific parameter ranges:
```bash
python SVM/svm_parameters.py --search grid --cv 5 --C-min -2 --C-max 2 --C-steps 5 --g-min -2 --g-max 2 --g-steps 5 --kernels rbf linear poly --dataset FeatureExtractor/TransferLearningAntivirus.libsvm
```

2. Random search with more iterations:
```bash
python SVM/svm_parameters.py --search random --n-iter 50 --cv 3 --C-min -3 --C-max 3 --g-min -3 --g-max 3 --kernels rbf --dataset FeatureExtractor/TransferLearningAntivirus.libsvm
```

The script will:
- Perform the specified search strategy
- Test all combinations of parameters
- Display the best parameters found
- Show performance metrics for each configuration
- Generate a detailed report of the search process

This approach is particularly useful when:
- You need to optimize SVM performance
- You want to compare different kernel functions
- You need to find the optimal balance between bias and variance
- You're working with a new dataset and need to find suitable parameters

## Contribution

Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
