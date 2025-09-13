# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import json
import csv
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Custom Networks
from networks.lenet import LeNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet
from networks.densenet import DenseNet
from networks.wide_resnet import WideResNet
from networks.capsnet import CapsNet


class FileToImageConverter:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size
        self.target_bytes = target_size[0] * target_size[1] * 3

    def _process_file(self, file_path):
        """Process .exe or .json files and extract byte-level features."""
        if file_path.lower().endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            byte_content = json.dumps(data, ensure_ascii=False).encode('utf-8')
        else:
            with open(file_path, 'rb') as f:
                byte_content = f.read()

        byte_array = np.frombuffer(byte_content, dtype=np.uint8)

        # Compute entropy of bytes
        unique, counts = np.unique(byte_array, return_counts=True)
        probabilities = counts / len(byte_array)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Avoid log(0)

        # Normalize entropy to [0, 255]
        entropy_normalized = int((entropy / 8) * 255)

        # Compute mean and standard deviation
        mean = float(np.mean(byte_array))
        std = float(np.std(byte_array))

        # Normalize mean and std to [0, 255]
        mean_normalized = int(mean)
        std_normalized = int((std / 128) * 255)

        return byte_array, entropy_normalized, mean_normalized, std_normalized

    def convert_to_image(self, file_path):
        """Convert a file into a 32x32x3 RGB image using byte features."""
        byte_array, entropy, mean, std = self._process_file(file_path)

        # Pad or truncate to target size
        if len(byte_array) < self.target_bytes:
            padding = np.zeros(self.target_bytes - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate([byte_array, padding])
        elif len(byte_array) > self.target_bytes:
            byte_array = byte_array[:self.target_bytes]

        # Generate smart-resized image with embedded statistical features
        image = self._smart_resize(byte_array, entropy, mean, std)

        # Ensure correct shape
        if image.shape != (self.target_size[0], self.target_size[1], 3):
            print(f"Warning: Image has shape {image.shape}, reshaping to {(self.target_size[0], self.target_size[1], 3)}")
            image = image.reshape(self.target_size[0], self.target_size[1], 3)

        return image

    def _smart_resize(self, byte_array, entropy, mean, std):
        """Smart resizing using linear interpolation and embedding statistical features into RGB channels."""
        if len(byte_array) == self.target_bytes:
            return byte_array.reshape(self.target_size[0], self.target_size[1], 3)

        # Linear interpolation to resize byte sequence
        x_original = np.linspace(0, 1, len(byte_array))
        x_target = np.linspace(0, 1, self.target_bytes)
        interp_func = interp1d(x_original, byte_array, kind='linear', fill_value='extrapolate')
        resized_bytes = np.clip(interp_func(x_target), 0, 255).astype(np.uint8)

        # Semente determinística e válida p/ RandomState
        seed = int(np.sum(byte_array) % (2**32 - 1))
        rng = np.random.RandomState(seed=seed)
        variation = rng.randint(0, 100, size=self.target_size[:2])

        image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)

        # Channel R: Original bytes + variation
        image[:, :, 0] = np.clip(resized_bytes.reshape(self.target_size[0], self.target_size[1]) + variation, 0, 255)

        # Channel G: Entropy + mean-weighted variation
        entropy_variation = np.clip(entropy + (variation * mean / 255.0), 0, 255)
        image[:, :, 1] = entropy_variation

        # Channel B: Combined mean and std + std-weighted variation
        mean_std_variation = np.clip((mean + std) // 2 + (variation * std / 128.0), 0, 255)
        image[:, :, 2] = mean_std_variation

        return image.astype(np.uint8)


class CustomDatasetLoader:
    def __init__(self, benign_dir, malware_dir):
        self.converter = FileToImageConverter()
        self.benign_dir = benign_dir
        self.malware_dir = malware_dir
        self.class_names = ['benign', 'malware']

    def load_data(self):
        """Load binary classification dataset from separate benign and malware directories."""
        x_data = []
        y_data = []
        file_names = []  # <- Nome do aplicativo (arquivo)

        print(f"\nLoading dataset from:")
        print(f"  Benign: {self.benign_dir}")
        print(f"  Malware: {self.malware_dir}")

        label_mapping = {'benign': -1, 'malware': +1}

        for class_name, data_dir in [('benign', self.benign_dir), ('malware', self.malware_dir)]:
            if not os.path.exists(data_dir):
                print(f"Warning: Directory '{data_dir}' does not exist. Skipping {class_name}.")
                continue

            files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            print(f"Found {len(files)} files in {class_name} directory.")

            for filename in files:
                file_path = os.path.join(data_dir, filename)
                try:
                    img_array = self.converter.convert_to_image(file_path)
                    if img_array.shape != (32, 32, 3):
                        img_array = img_array.reshape(32, 32, 3)
                    x_data.append(img_array)
                    y_data.append(label_mapping[class_name])
                    file_names.append(filename)  # guarda o nome do arquivo
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

        if len(x_data) == 0:
            raise ValueError("No valid files found in either benign or malware directories.")

        return np.array(x_data), np.array(y_data), file_names

    def save_as_libsvm(self, x_data, y_data, output_file):
        """Save dataset in LIBSVM format (sparse representation, non-zero features only)."""
        x_data = x_data.reshape(len(x_data), -1)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(len(x_data)):
                line = str(int(y_data[i]))
                for j in range(x_data.shape[1]):
                    value = x_data[i, j]
                    if value != 0:
                        line += f" {j+1}:{int(value)}"  # bytes 0..255 -> int
                f.write(line + '\n')

        print(f"LIBSVM file saved to: {output_file}")
        print(f"Total samples: {len(x_data)}")

    def save_as_csv(self, x_data, y_data, file_names, output_file, delimiter=';'):
        """
        Save dataset as CSV with format:
        1st column: application (file) name
        2nd column: output neuron(s) (here: the label 0/1)
        From 3rd column on: input neurons (flattened features)
        """
        # Achata as imagens (N, 32, 32, 3) -> (N, 3072)
        x_flat = x_data.reshape(len(x_data), -1)

        if len(file_names) != len(x_flat) or len(y_data) != len(x_flat):
            raise ValueError("Lengths of file_names, y_data, and x_data do not match.")

        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Cabeçalho: app;output;in_1;in_2;...
        header = ['app', 'output'] + [f'in_{i+1}' for i in range(x_flat.shape[1])]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delimiter)
            writer.writerow(header)
            for fname, y, features in zip(file_names, y_data, x_flat):
                # Converte y para 0 (benign) ou 1 (malware)
                label = 0 if int(y) == -1 else 1
                row = [fname, label] + [int(v) for v in features]
                writer.writerow(row)

        print(f"CSV file saved to: {output_file}")
        print(f"Total samples: {len(x_flat)}")



class Classifier:
    def __init__(self, model_name='lenet'):
        """Initialize classifier with pre-trained network."""
        self.model_defs = {
            'lenet': LeNet,
            'pure_cnn': PureCnn,
            'net_in_net': NetworkInNetwork,
            'resnet': ResNet,
            'densenet': DenseNet,
            'wide_resnet': WideResNet,
            'capsnet': CapsNet
        }

        if model_name not in self.model_defs:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.model_defs.keys())}")

        self.model = self.model_defs[model_name](load_weights=True)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Feature extractor: penultimate layer output
        self.feature_model = tf.keras.Model(
            inputs=self.model._model.input,
            outputs=self.model._model.layers[-2].output
        )

    def extract_features(self, image):
        """Extract features from the penultimate layer of the model."""
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0
        features = self.feature_model.predict(image, verbose=0)
        return features

    def classify_image(self, image):
        """Classify an image and return prediction with confidence and features."""
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        image = image.astype('float32') / 255.0

        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]
        features = self.extract_features(image)

        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {name: float(prob) for name, prob in zip(self.class_names, predictions[0])},
            'features': features[0]
        }

    def evaluate_accuracy(self, test_data=None):
        """Evaluate model accuracy on CIFAR-10 or custom test set."""
        if test_data is None:
            (_, _), (x_test, y_test) = cifar10.load_data()
            y_test = tf.keras.utils.to_categorical(y_test, len(self.class_names))
        else:
            x_test, y_test = test_data
            y_test = tf.keras.utils.to_categorical((y_test + 1) // 2, num_classes=2)  # Map [-1,1] to [0,1]

        return self.model.evaluate(x_test, y_test, verbose=0)[1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify malware/benign files using pre-trained CNNs and extract features for lightweight ML.'
    )

    # Model selection
    parser.add_argument(
        '-model', default='lenet',
        choices=['lenet', 'resnet', 'densenet', 'wide_resnet', 'capsnet'],
        help='Pre-trained model to use for feature extraction and classification.'
    )

    # Dataset paths
    parser.add_argument('-data_benign', required=True, type=str,
                        help='Path to directory containing benign files (.exe or .json)')
    parser.add_argument('-data_malware', required=True, type=str,
                        help='Path to directory containing malware files (.exe or .json)')

    # Classification mode
    parser.add_argument('-image_idx', type=int,
                        help='Index of a specific image to classify (optional). If not provided, all images are processed.')

    # Training mode
    parser.add_argument('-train', action='store_true',
                        help='Train the model on the custom dataset instead of using pre-trained weights.')
    parser.add_argument('-epochs', type=int, default=10,
                        help='Number of training epochs (used only with -train).')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Batch size for training (used only with -train).')
    parser.add_argument('-validation_split', type=float, default=0.2,
                        help='Fraction of training data used for validation (used only with -train).')

    # Attack mode (compat)
    parser.add_argument('-maxiter', default=75, type=int,
                        help='Maximum iterations for differential evolution attack.')
    parser.add_argument('-popsize', default=400, type=int,
                        help='Population size for adversarial generation per iteration.')
    parser.add_argument('-samples', default=500, type=int,
                        help='Number of image samples to attack.')
    parser.add_argument('-targeted', action='store_true',
                        help='Enable targeted attacks.')
    parser.add_argument('-save', default='FeatureExtractor/networks/results/results.pkl',
                        help='Path to save attack results (pickle format).')

    # Saídas
    parser.add_argument('-libsvm_file', default='FeatureExtractor/TransferLearningAntivirus.libsvm',
                        help='Path to save dataset in LIBSVM format.')
    parser.add_argument('-csv_file', default='FeatureExtractor/TransferLearningAntivirus.csv',
                        help='Path to save dataset in CSV format (app;output;in_1;in_2;...).')

    parser.add_argument('-verbose', action='store_true',
                        help='Print detailed information during execution.')

    args = parser.parse_args()

    # Validate input directories
    if not os.path.exists(args.data_benign):
        print(f"Error: Benign directory '{args.data_benign}' does not exist.")
        raise SystemExit(1)
    if not os.path.exists(args.data_malware):
        print(f"Error: Malware directory '{args.data_malware}' does not exist.")
        raise SystemExit(1)

    # Load dataset
    loader = CustomDatasetLoader(args.data_benign, args.data_malware)
    x_data, y_data, file_names = loader.load_data()

    # Create necessary directories
    os.makedirs('FeatureExtractor/networks/pretrained_weights', exist_ok=True)
    os.makedirs('FeatureExtractor/networks/results', exist_ok=True)

    classifier = Classifier(args.model)

    # Train model if requested
    if args.train:
        print("\nTraining model on custom dataset...")
        y_data_one_hot = tf.keras.utils.to_categorical((y_data + 1) // 2, num_classes=2)  # Map [-1,1] → [0,1]

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5",
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        if isinstance(classifier.model, LeNet):
            classifier.model.epochs = args.epochs
            classifier.model.batch_size = args.batch_size
            classifier.model.train()
            classifier.model._model.load_weights(f"FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")
            print(f"Best weights loaded from: FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")
        else:
            history = classifier.model.fit(
                x_data, y_data_one_hot,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_split=args.validation_split,
                callbacks=callbacks,
                verbose=1
            )
            classifier.model.load_weights(f"FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")
            print(f"\nTraining completed!")
            print(f"Final Accuracy: {history.history['accuracy'][-1]:.2%}")
            print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
            print(f"Best weights loaded from: FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")

    # Classify specific image or entire dataset (se desejar usar)
    indices = [args.image_idx] if args.image_idx is not None else range(len(x_data))
    # (Nada a fazer aqui por padrão — mantido por compatibilidade)

    # Save dataset in LIBSVM format
    print(f"\nSaving dataset in LIBSVM format to: {args.libsvm_file}")
    loader.save_as_libsvm(x_data, y_data, args.libsvm_file)

    # Save dataset in CSV format (app;output;in_1;...)
    print(f"\nSaving dataset in CSV format to: {args.csv_file}")
    loader.save_as_csv(x_data, y_data, file_names, args.csv_file, delimiter=';')

    print(f"\n✅ Process completed successfully. {len(x_data)} samples processed.")
