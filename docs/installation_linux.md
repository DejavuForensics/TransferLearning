# Linux Installation

This guide will help you set up the environment to run the project on Linux systems.

## Prerequisites

1. Python 3.8 or higher
2. pip3 (Python package manager)
3. Git (optional, for cloning the repository)
4. build-essential (for compiling some dependencies)

## Step 1: Download the Repository

1. Clone the repository using Git:
```bash
git clone https://github.com/romariogl/NeuralNetwork_TransferLearning.git
cd NeuralNetwork_TransferLearning
```

Or download the ZIP file from the repository and extract it to a folder of your choice.

## Step 2: Install System Dependencies

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3-pip python3-venv build-essential
```

### Fedora
```bash
sudo dnf install python3-pip python3-virtualenv gcc
```

### CentOS/RHEL
```bash
sudo yum install python3-pip python3-virtualenv gcc
```

## Step 3: Create Virtual Environment

1. Navigate to the project directory
2. Run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 4: Update pip and Install Dependencies

1. With the virtual environment activated, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 5: Run the Code

1. To extract features using the ResNet model:
```bash
python3 FeatureExtractor/extract_features.py --model resnet --data_dir AntivirusDataset
```

2. To run SVM classification:
```bash
python3 SVM/svm_classifier.py
```
