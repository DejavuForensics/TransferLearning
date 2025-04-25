import argparse
import numpy as np
import pickle
import os
import json
from PIL import Image
from scipy.interpolate import interp1d
from sklearn.datasets import dump_svmlight_file
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
        """Processa arquivos .exe ou .json"""
        if file_path.lower().endswith('.exe'):
            with open(file_path, 'rb') as f:
                byte_content = f.read()
        elif file_path.lower().endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            byte_content = json.dumps(data).encode('utf-8')
        
        # Converte para array numpy e calcula estatísticas
        byte_array = np.frombuffer(byte_content, dtype=np.uint8)
        
        # Calcula entropia dos bytes
        unique, counts = np.unique(byte_array, return_counts=True)
        probabilities = counts / len(byte_array)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normaliza a entropia para o intervalo [0, 255]
        entropy_normalized = int((entropy / 8) * 255)
        
        # Calcula média e desvio padrão dos bytes
        mean = np.mean(byte_array)
        std = np.std(byte_array)
        
        # Normaliza média e desvio padrão para o intervalo [0, 255]
        mean_normalized = int(mean)
        std_normalized = int((std / 128) * 255)
        
        return byte_array, entropy_normalized, mean_normalized, std_normalized
    
    def convert_to_image(self, file_path):
        """Converte um arquivo em imagem 32x32x3 com características adicionais"""
        byte_array, entropy, mean, std = self._process_file(file_path)
        
        # Se o arquivo for muito pequeno, preenche com zeros
        if len(byte_array) < self.target_bytes:
            padding = np.zeros(self.target_bytes - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate([byte_array, padding])
        elif len(byte_array) > self.target_bytes:
            byte_array = byte_array[:self.target_bytes]
        
        # Cria a imagem com características adicionais
        image = self._smart_resize(byte_array, entropy, mean, std)
        
        # Verifica se a imagem tem a dimensão correta
        if image.shape != (self.target_size[0], self.target_size[1], 3):
            print(f"Aviso: A imagem tem dimensão {image.shape}, redimensionando para {(self.target_size[0], self.target_size[1], 3)}")
            image = image.reshape(self.target_size[0], self.target_size[1], 3)
        
        
        return image
    
    def _smart_resize(self, byte_array, entropy, mean, std):
        """Redimensionamento inteligente com interpolação linear e adição de características"""
        if len(byte_array) == self.target_bytes:
            return byte_array.reshape(self.target_size[0], self.target_size[1], 3)
            
        # Usa interpolação linear
        x_original = np.linspace(0, 1, len(byte_array))
        x_target = np.linspace(0, 1, self.target_bytes)
        interp_func = interp1d(x_original, byte_array, kind='linear', fill_value='extrapolate')
        resized_bytes = np.clip(interp_func(x_target), 0, 255).astype(np.uint8)
        
        # Cria uma imagem RGB onde:
        # R: Bytes originais com variação baseada no hash do arquivo
        # G: Entropia com variação baseada na média
        # B: Média e desvio padrão com variação baseada no desvio padrão
        
        # Gera uma matriz de variação única para cada arquivo
        hash_value = hash(byte_array.tobytes())  # Usa o hash do conteúdo do arquivo
        variation = np.random.RandomState(seed=hash_value).randint(0, 100, size=(self.target_size[0], self.target_size[1]))
        
        # Cria a imagem RGB
        image = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        
        # Preenche o canal R com os bytes redimensionados + variação
        image[:,:,0] = np.clip(resized_bytes.reshape(self.target_size[0], self.target_size[1]) + variation, 0, 255)
        
        # Preenche o canal G com entropia + variação baseada na média
        entropy_variation = np.clip(entropy + (variation * mean / 255), 0, 255)
        image[:,:,1] = entropy_variation
        
        # Preenche o canal B com média e desvio padrão + variação baseada no desvio padrão
        mean_std_variation = np.clip((mean + std) // 2 + (variation * std / 128), 0, 255)
        image[:,:,2] = mean_std_variation
        
        return image.astype(np.uint8)
   

class CustomDatasetLoader:
    def __init__(self, data_folder, label_file=None):
        self.converter = FileToImageConverter()
        self.data_folder = data_folder
        self.label_file = label_file
        self.class_names = ['benign', 'malware']  # Classes binárias
        
    def load_data(self):
        """Carrega dados de arquivos .exe e .json com classificação binária"""
        x_data = []
        y_data = []
        
        print(f"\nIniciando carregamento de dados do diretório: {self.data_folder}")
        
        # Mapeamento de rótulos
        label_mapping = {'benign': -1, 'malware': +1}
        
        if not os.path.exists(self.data_folder):
            print(f"Pasta {self.data_folder} não encontrada")
            return
            
        # Para cada classe (benign/malware)
        for class_name in ['benign', 'malware']:
            class_folder = os.path.join(self.data_folder, class_name)
            print(f"Verificando classe {class_name} em {class_folder}")
            
            if not os.path.exists(class_folder):
                print(f"Pasta {class_folder} não encontrada")
                continue
                
            # Processa cada arquivo na pasta
            files = os.listdir(class_folder)
            print(f"Encontrados {len(files)} arquivos em {class_folder}")
            
            for filename in files:
                if filename.lower().endswith(f'.json'):
                    file_path = os.path.join(class_folder, filename)
                    try:
                        img_array = self.converter.convert_to_image(file_path)
                        # Garante que a imagem tenha o formato correto (32, 32, 3)
                        if img_array.shape != (32, 32, 3):
                            print(f"Ajustando formato da imagem de {img_array.shape} para (32, 32, 3)")
                            img_array = img_array.reshape(32, 32, 3)
                        x_data.append(img_array)
                        y_data.append(label_mapping[class_name])
                    except Exception as e:
                        print(f"Erro ao processar {file_path}: {str(e)}")
        
        return np.array(x_data), np.array(y_data)
    
    def save_as_libsvm(self, x_data, y_data, output_file):
        """Salva os dados no formato libsvm"""
        dump_svmlight_file(x_data.reshape(len(x_data), -1), y_data, output_file)

class Classifier:
    def __init__(self, model_name='lenet'):
        """Inicializa o classificador com um modelo pré-treinado.
        
        Args:
            model_name (str): Nome do modelo a ser usado ('lenet', 'resnet', etc.)
        """
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
            raise ValueError(f"Modelo {model_name} não encontrado. Modelos disponíveis: {list(self.model_defs.keys())}")
            
        self.model = self.model_defs[model_name](load_weights=True)
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    def classify_image(self, image):
        """Classifica uma imagem usando o modelo pré-treinado.
        
        Args:
            image (numpy.ndarray): Imagem para classificação (deve ter shape (1, 32, 32, 3))
            
        Returns:
            dict: Dicionário contendo:
                - class: Nome da classe prevista
                - confidence: Confiança da previsão
                - all_predictions: Probabilidades para todas as classes
        """
        # Garante que a imagem esteja no formato correto
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normaliza a imagem para o intervalo [0, 1]
        image = image.astype('float32') / 255.0
        
        # Faz a previsão
        predictions = self.model.predict(image)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': float(confidence),
            'all_predictions': {name: float(prob) for name, prob in zip(self.class_names, predictions[0])}
        }
        
    def evaluate_accuracy(self, test_data=None):
        """Avalia a acurácia do modelo no conjunto de teste.
        
        Args:
            test_data (tuple, optional): Tupla (x_test, y_test). Se None, usa CIFAR-10.
            
        Returns:
            float: Acurácia do modelo
        """
        if test_data is None:
            (_, _), (x_test, y_test) = cifar10.load_data()
        else:
            x_test, y_test = test_data
            
        y_test = tf.keras.utils.to_categorical(y_test, len(self.class_names))
        return self.model.accuracy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use pre-trained models for classification')
    parser.add_argument('--model', default='lenet', choices=['lenet', 'resnet', 'densenet', 'wide_resnet', 'capsnet'],
                       help='Model to use for classification')
    
    # Argumentos para modo de classificação
    parser.add_argument('--data_dir', required=True, help='Directory containing .exe and .json files')
    parser.add_argument('--image_idx', type=int, help='Index of image to classify from custom dataset')
    parser.add_argument('--train', action='store_true', help='Train the model with custom data instead of using pre-trained weights')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data to use for validation')
    
    # Argumentos para modo de ataque (mantidos do código original)
    parser.add_argument('--maxiter', default=75, type=int,
                       help='The maximum number of iterations in the differential evolution algorithm.')
    parser.add_argument('--popsize', default=400, type=int,
                       help='The number of adversarial images generated each iteration.')
    parser.add_argument('--samples', default=500, type=int,
                       help='The number of image samples to attack.')
    parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
    parser.add_argument('--save', default='FeatureExtractor/networks/results/results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--libsvm_file', default='FeatureExtractor/TransferLearningAntivirus.libsvm', help='Save location for LIBSVM format file')
    parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

    args = parser.parse_args()

    # Verifica se o diretório existe
    if not os.path.exists(args.data_dir):
        print(f"Erro: O diretório {args.data_dir} não existe!")
        exit(1)

    # Carrega o dataset customizado com o conversor escolhido
    loader = CustomDatasetLoader(args.data_dir)
    (x_data, y_data) = loader.load_data()
    
    # Verifica se o dataset não está vazio
    if len(x_data) == 0:
        print(f"Erro: Nenhuma imagem encontrada no diretório {args.data_dir}!")
        print("Verifique se:")
        print("1. O diretório contém arquivos .exe ou .json")
        print("2. Os arquivos estão na estrutura correta:")
        print("   - benign/")
        print("   - malware/")
        exit(1)

    class_names = ['benign', 'malware']  # Classes do CustomDatasetLoader

    # Cria o diretório se não existir
    os.makedirs('FeatureExtractor/networks/pretrained_weights', exist_ok=True)
    os.makedirs('FeatureExtractor/networks/results', exist_ok=True)

    classifier = Classifier(args.model)
    
    # Se o modo de treinamento estiver ativado
    if args.train:
        print("\nIniciando treinamento do modelo com dados customizados...")        
        # Converte os rótulos para o formato one-hot
        y_data_one_hot = tf.keras.utils.to_categorical((y_data + 1) // 2, num_classes=2)
        
        # Configura callbacks para monitoramento
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5",
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Treina o modelo usando o método train() da classe LeNet
        if isinstance(classifier.model, LeNet):
            # Configura parâmetros de treinamento
            classifier.model.epochs = args.epochs
            classifier.model.batch_size = args.batch_size
            
            # Treina o modelo
            classifier.model.train()
            
            # Carrega os melhores pesos
            classifier.model._model.load_weights(f"FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")
            print(f"\nMelhores pesos carregados de: FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")
        else:
            # Para outros modelos, usa o método fit padrão
            history = classifier.model.fit(
                x_data, y_data_one_hot,
                epochs=args.epochs,
                batch_size=args.batch_size,
                validation_split=args.validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            print("\nTreinamento concluído!")
            print(f"Acurácia final: {history.history['accuracy'][-1]:.2%}")
            print(f"Acurácia de validação: {history.history['val_accuracy'][-1]:.2%}")
            
            # Carrega os melhores pesos
            classifier.model.load_weights(f"FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")
            print(f"\nMelhores pesos carregados de: FeatureExtractor/networks/pretrained_weights/{args.model}_custom.h5")
        
    # Se um índice específico foi fornecido, processa apenas essa imagem
    if args.image_idx is not None:
        if args.image_idx >= len(x_data):
            print(f"Erro: Índice {args.image_idx} fora do intervalo! O dataset tem {len(x_data)} imagens.")
            exit(1)
        indices = [args.image_idx]
    else:
        # Processa todas as imagens
        indices = range(len(x_data))
    
    # Salva no formato LIBSVM
    print(f'Saving LIBSVM format to {args.libsvm_file}')
    with open(args.libsvm_file, 'w') as f:
        for idx in indices:
            image = x_data[idx:idx+1]  # Adiciona dimensão de batch
            
            # Classifica a imagem
            result = classifier.classify_image(image)
            
            # Mapeia as previsões para as classes do dataset customizado
            predicted_class = 'malware' if result['confidence'] > 0.5 else 'benign'
            confidence = result['confidence'] if predicted_class == 'malware' else 1 - result['confidence']                
            
            # Prepara os atributos (probabilidades das classes do CIFAR-10)
            attributes = []
            for i, (class_name, prob) in enumerate(result['all_predictions'].items(), 1):
                attributes.append(f"{i}:{prob:.4f}")
            
            # Escreve no formato LIBSVM
            line = f"{y_data[idx]} {' '.join(attributes)}\n"
            f.write(line)
            
            # Salva o resultado no formato pickle para cada imagem
            formatted_result = {
                'model': args.model,
                'image_idx': idx,
                'true_class': class_names[0 if y_data[idx] == -1 else 1],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': result['all_predictions'],
                'success': predicted_class == class_names[0 if y_data[idx] == -1 else 1]
            }
            
            # Salva cada resultado em um arquivo separado
            result_file = f"{args.save}_{idx}.pkl"
            with open(result_file, 'wb') as file:
                pickle.dump(formatted_result, file)

