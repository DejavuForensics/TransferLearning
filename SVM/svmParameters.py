"""
Código desenvolvido por:

Prof. Dr. Sidney Lima
Universidade Federal de Pernambuco
Departamento de Eletrônica e Sistemas
"""

import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

#========================================================================
class SVMParametersSklearn:
    def main(self, dataset_path):
        print(f"Carregando dados do arquivo: {dataset_path}")
        # lê libsvm diretamente com scikit-learn
        X, y = load_svmlight_file(dataset_path)
        X = X.toarray()  # matriz densa

        # poda por correlação
        threshold = 0.1
        y, X = self.pruning_dataset(y, X, threshold)

        # split hold-out 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=1
        )

        # grade de hiperparâmetros
        param_grid = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'C':       [10 ** i for i in range(-3, 4)],
            'gamma':   [10 ** i for i in range(-3, 4)]
        }
        grid = GridSearchCV(
            SVC(), param_grid,
            cv=10,
            return_train_score=True,
            n_jobs=-1,
            verbose=1
        )

        # executa busca em paralelo
        grid.fit(X_train, y_train)

        # prepara pastas
        results_dir = "results"
        images_dir  = os.path.join(results_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # grava parâmetros e estatísticas de CV
        with open(os.path.join(results_dir, "svm_parameters_results.txt"), "w", encoding="utf-8") as pf:
            pf.write("kernel\tC\tgamma\tmean_train\tstd_train\tmean_test\tstd_test\n")
            cv = grid.cv_results_
            for mean_tr, std_tr, mean_te, std_te, params in zip(
                cv['mean_train_score'], cv['std_train_score'],
                cv['mean_test_score'],  cv['std_test_score'],
                cv['params']
            ):
                pf.write(f"{params['kernel']}\t"
                         f"{params['C']:.3f}\t"
                         f"{params['gamma']:.3f}\t"
                         f"{mean_tr*100:.2f}\t"
                         f"{std_tr*100:.2f}\t"
                         f"{mean_te*100:.2f}\t"
                         f"{std_te*100:.2f}\n")

        # melhor estimador e avaliação final
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize='true') * 100

        # grava matriz de confusão final
        with open(os.path.join(results_dir, "svm_confusion_matrix_test.txt"), "w", encoding="utf-8") as cf:
            cf.write(f"Best params: {grid.best_params_}\n")
            cf.write("Confusion Matrix (Teste, %):\n")
            np.savetxt(cf, cm, fmt="%.2f", delimiter="\t")

        # gera relatório HTML e gráficos
        self.generate_html_report(grid, cm, images_dir)
        print(f"Relatório gerado em {results_dir}/report.html")

    def pruning_dataset(self, y, X, threshold):
        # remove colunas de variância zero
        stds = X.std(axis=0)
        nonzero = stds > 0
        X_nz = X[:, nonzero]

        # calcula correlações com controle de NaN/div0
        with np.errstate(divide='ignore', invalid='ignore'):
            cors = np.array([
                np.corrcoef(X_nz[:, i], y)[0, 1]
                if np.count_nonzero(~np.isnan(X_nz[:, i])) > 1 else 0
                for i in range(X_nz.shape[1])
            ])
        cors = np.nan_to_num(cors)

        # seleciona features
        keep_mask = np.abs(cors) >= threshold
        orig_indices = np.where(nonzero)[0][keep_mask]

        # salva CSV de seleção
        df = pd.DataFrame({
            'Original_Index': orig_indices + 1,
            'Selected_Index': np.arange(1, keep_mask.sum() + 1),
            'Correlation': cors[keep_mask]
        })
        df.to_csv('selected_features.csv', index=False)

        return y, X[:, orig_indices]

    def generate_html_report(self, grid, cm_test, images_dir):
        # plot distribuição de acurácias de teste em CV
        scores = grid.cv_results_['mean_test_score'] * 100
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=20, alpha=0.7)
        plt.xlabel('Acurácia CV (%)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Acurácias em CV')
        plt.savefig(os.path.join(images_dir, 'acc_dist.png'))
        plt.close()

        # plot matriz de confusão final
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm_test, interpolation='nearest')
        ax.set_title('Matriz de Confusão (Teste, %)')
        for (i, j), v in np.ndenumerate(cm_test):
            ax.text(j, i, f"{v:.1f}", ha='center', va='center')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'cm_test.png'))
        plt.close()

        # escreve HTML
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Relatório SVM</title></head>
<body>
  <h1>Relatório SVM</h1>
  <p>Data: {datetime.now():%Y-%m-%d %H:%M:%S}</p>
  <h2>Melhor Parâmetros</h2>
  <p>{grid.best_params_}</p>
  <h2>Distribuição de Acurácias em CV</h2>
  <img src="images/acc_dist.png" alt="Distribuição de Acurácias">
  <h2>Matriz de Confusão no Teste</h2>
  <img src="images/cm_test.png" alt="Matriz de Confusão Teste">
</body>
</html>"""
        with open("results/report.html", "w", encoding="utf-8") as f:
            f.write(html)

#========================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-dataset', default='heart_scale', help='arquivo .libsvm')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    SVMParametersSklearn().main(args.dataset)
