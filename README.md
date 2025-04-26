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

Onde:
- `-dataset`: Caminho para o arquivo libsvm (opcional, padrão: 'FeatureExtractor/TransferLearningAntivirus.libsvm')

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
python SVM/svm_classifier.py -dataset FeatureExtractor/Joined_TransferLearning.libsvm
```

This hybrid approach:
- Combines features from both shallow and deep networks
- Potentially captures a wider range of feature representations
- May improve classification performance by leveraging complementary feature sets
- Uses cross-validation to ensure robust performance evaluation

The results will be saved in the same format as the standard approach, with an HTML report available at `SVM/results/svm_results_report.html`.

## Contribution

Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
