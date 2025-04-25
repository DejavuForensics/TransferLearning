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
python SVM/svm_classifier.py

# On Linux
python3 SVM/svm_classifier.py
```

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

## Contribution

Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
