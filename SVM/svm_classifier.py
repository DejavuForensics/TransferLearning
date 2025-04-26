import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
import os
import pickle
import argparse
from generate_report import calculate_metrics, generate_html_report

def load_libsvm_data(file_path):
    """Carrega dados do arquivo LIBSVM"""
    print(f"\nCarregando dados do arquivo: {file_path}")
    X, y = load_svmlight_file(file_path)
    X = X.toarray()  # Converte para array numpy denso
    return X, y

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.scaler = StandardScaler()
        self.models = []
        self.results = {}
        
    def train(self, X, y, n_splits=5):
        """
        Train SVM with 5-fold cross validation
        
        Args:
            X: Feature matrix
            y: Target labels
            n_splits: Number of folds (default: 5)
        """
        # Initialize k-fold cross validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store results for each fold
        self.results = {}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\nTraining fold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train SVM
            svm = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True)
            svm.fit(X_train_scaled, y_train)
            self.models.append(svm)
            
            # Make predictions
            y_pred = svm.predict(X_test_scaled)
            y_prob = svm.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            self.results[f'fold_{fold}'] = calculate_metrics(y_test, y_pred, y_prob)
            
            print(f"Fold {fold + 1} Results:")
            print(f"Accuracy: {self.results[f'fold_{fold}']['accuracy']:.4f}")
            print(f"Precision: {self.results[f'fold_{fold}']['precision']:.4f}")
            print(f"Recall: {self.results[f'fold_{fold}']['recall']:.4f}")
            print(f"F1-Score: {self.results[f'fold_{fold}']['f1']:.4f}")
            print(f"AUC: {self.results[f'fold_{fold}']['auc']:.4f}")
        
        # Generate HTML report
        generate_html_report(self.results)
        
        # Save results
        self.save_results()
    
    def predict(self, X):
        """
        Make predictions using all models and return average probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Average predictions and probabilities across all folds
        """
        X_scaled = self.scaler.transform(X)
        predictions = []
        probabilities = []
        
        for model in self.models:
            pred = model.predict(X_scaled)
            prob = model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred)
            probabilities.append(prob)
        
        # Average predictions and probabilities
        avg_predictions = np.mean(predictions, axis=0)
        avg_probabilities = np.mean(probabilities, axis=0)
        
        return avg_predictions, avg_probabilities
    
    def save_results(self):
        """Save results to pickle file"""
        os.makedirs('SVM/results', exist_ok=True)
        with open('SVM/results/svm_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
    
    def load_results(self):
        """Load results from pickle file"""
        with open('SVM/results/svm_results.pkl', 'rb') as f:
            self.results = pickle.load(f)

def main():
    # Configuração do parser de argumentos
    parser = argparse.ArgumentParser(description='Classificador SVM para detecção de malware')
    parser.add_argument('-dataset', type=str, required=True,
                      help='Caminho para o arquivo LIBSVM contendo os dados')
    args = parser.parse_args()
    
    # Carrega os dados do arquivo LIBSVM
    X, y = load_libsvm_data(args.dataset)
    
    # Inicializa e treina o SVM com 5-fold cross validation
    svm = SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
    svm.train(X, y)
    
    # Exemplo de previsão para um novo arquivo
    print("\nExemplo de previsão:")
    sample_idx = 0
    sample_pred, sample_prob = svm.predict(X[sample_idx].reshape(1, -1))
    print(f"Arquivo de teste {sample_idx}:")
    print(f"Classe verdadeira: {'benign' if y[sample_idx] == -1 else 'malware'}")
    print(f"Classe prevista: {'benign' if sample_pred[0] == -1 else 'malware'}")
    print(f"Probabilidade de malware: {sample_prob[0]:.4f}")

if __name__ == '__main__':
    main() 