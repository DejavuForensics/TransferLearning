# Windows Installation

This guide will help you set up the environment to run the project on Windows.

## Prerequisites

1. Python 3.8 or higher
2. pip (Python package manager)
3. Git (optional, for cloning the repository)

## Step 1: Download the Repository

1. Clone the repository using Git:
```bash
git clone https://github.com/romariogl/NeuralNetwork_TransferLearning.git
cd NeuralNetwork_TransferLearning
```

Or download the ZIP file from the repository and extract it to a folder of your choice.

## Step 2: Install Python

1. Visit [python.org](https://www.python.org/downloads/)
2. Download the latest version of Python for Windows
3. During installation, check the "Add Python to PATH" option
4. Complete the installation

## Step 3: Create Virtual Environment

1. Open Command Prompt (CMD) as administrator
2. Navigate to the project directory
3. Run the following commands:

```bash
python -m venv venv
venv\Scripts\activate
```

## Step 4: Install Dependencies

1. With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

## Step 5: Run the Code

1. To extract features using the ResNet model:
```bash
python FeatureExtractor/extract_features.py --model resnet --data_dir AntivirusDataset
```

2. To run SVM classification:
```bash
python SVM/svm_classifier.py
```
