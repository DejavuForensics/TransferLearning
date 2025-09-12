# Neural Network Transfer Learning with CIFAR-10

This repository contains an implementation of transfer learning using pre-trained neural networks with the CIFAR-10 dataset. The project demonstrates how to extract features from images using both pre-trained networks and custom CNNs, and then use these features to train an SVM (Support Vector Machine) with cross-validation.

## Project Structure

- `FeatureExtractor/`: Contains scripts for feature extraction using neural networks

## Features

- Feature extraction using pre-trained networks from CIFAR-10
- Training custom CNNs for feature extraction
- Classification using lightweight learning machines with cross-validation
- Results visualization and performance metrics

## Requirements

Project requirements are listed in the `requirements.txt` file. To install all dependencies, follow the installation instructions specific to your operating system:

- [Windows Installation](docs/installation_windows.md)
- [Linux Installation](docs/installation_linux.md)

## Usage

### 1. Feature Extraction

To extract features using the ResNet model:
```bash
python FeatureExtractor/extract_features.py -model resnet -data_benign ../IoT_ARM-main/IoT_ARM-main/benign/cuckoobox -data_malware ../IoT_ARM-main/IoT_ARM-main/malware/cuckoobox
```

Available options:
- `-model`: Choose between 'resnet' or 'lenet' to use pre-trained models, or any other option to train from scratch

The script will generate a LIBSVM format file: `FeatureExtractor/TransferLearningAntivirus.libsvm`

### 2. SVM Classification

To employ SVM classification:
- https://github.com/DejavuForensics/SVM

### 2. ELM Classification

To employ ELM classification:
- https://github.com/DejavuForensics/mELM


## Contribution

Contributions are welcome! Please open an issue to discuss proposed changes or submit a pull request.

This project is licensed under the MIT License - see the LICENSE file for details.
