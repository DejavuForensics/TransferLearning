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
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

# Para HalvingGridSearchCV:
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

#========================================================================
class SVMParametersSklearn:
    def __init__(self, args):
        self.args = args

    def main(self):
        # 0) Criar pasta de resultados        
        os.makedirs(self.args.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.args.results_dir, "images"), exist_ok=True)
        print(f"[INFO] Diretórios '{self.args.results_dir}' e 'images/' prontos.")

        # 1) Carrega dados
        print(f"[INFO] Carregando dados do arquivo: {self.args.dataset}")
        X, y = load_svmlight_file(self.args.dataset)
        X = X.toarray()

        # 2) Podar por correlação
        y, X = self.pruning_dataset(y, X, self.args.threshold)

        # 3) Split hold-out
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.args.test_size,
            stratify=y,
            random_state=self.args.random_state
        )

        # 4) Construir grade de parâmetros dinamicamente
        exponents_C = np.linspace(self.args.C_min, self.args.C_max, self.args.C_steps)
        exponents_gamma = np.linspace(self.args.g_min, self.args.g_max, self.args.g_steps)
        param_grid = {
            'kernel': self.args.kernels,
            'C': [10 ** e for e in exponents_C],
            'gamma': [10 ** e for e in exponents_gamma]
        }

        # 5) Escolher estratégia de busca
        cv = StratifiedKFold(n_splits=self.args.cv, shuffle=True, random_state=self.args.random_state)
        if self.args.search == 'grid':
            searcher = GridSearchCV(
                SVC(),
                param_grid,
                cv=cv,
                return_train_score=True,
                n_jobs=self.args.n_jobs,
                verbose=self.args.verbose
            )
        elif self.args.search == 'random':
            searcher = RandomizedSearchCV(
                SVC(),
                param_grid,
                n_iter=self.args.n_iter,
                cv=cv,
                return_train_score=True,
                n_jobs=self.args.n_jobs,
                verbose=self.args.verbose,
                random_state=self.args.random_state
            )
        else:  # halving
            searcher = HalvingGridSearchCV(
                SVC(),
                param_grid,
                cv=cv,
                return_train_score=True,
                factor=self.args.factor,
                random_state=self.args.random_state,
                n_jobs=self.args.n_jobs,
                verbose=self.args.verbose
            )

        # 6) Executa busca
        print(f"[INFO] Iniciando {self.args.search}-search: cv={self.args.cv}, "
              f"n_jobs={self.args.n_jobs}, n_iter={getattr(self.args, 'n_iter', '—')}")
        searcher.fit(X_train, y_train)

        # 7) Salvar resultados
        os.makedirs(self.args.results_dir, exist_ok=True)
        images_dir = os.path.join(self.args.results_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 7a) Estatísticas de CV
        with open(os.path.join(self.args.results_dir, "svm_parameters_results.txt"), "w", encoding="utf-8") as pf:
            pf.write("kernel\tC\tgamma\tmean_train\tstd_train\tmean_test\tstd_test\n")
            cv_results = searcher.cv_results_
            for tr, st, te, se, params in zip(
                cv_results['mean_train_score'], cv_results['std_train_score'],
                cv_results['mean_test_score'],  cv_results['std_test_score'],
                cv_results['params']
            ):
                pf.write(f"{params['kernel']}\t"
                         f"{params['C']:.3e}\t"
                         f"{params['gamma']:.3e}\t"
                         f"{tr*100:.2f}\t"
                         f"{st*100:.2f}\t"
                         f"{te*100:.2f}\t"
                         f"{se*100:.2f}\n")

        # 7b) Matriz de confusão no teste
        best = searcher.best_estimator_
        y_pred = best.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize='true') * 100
        with open(os.path.join(self.args.results_dir, "svm_confusion_matrix_test.txt"), "w", encoding="utf-8") as cf:
            cf.write(f"Best params: {searcher.best_params_}\n")
            cf.write("Confusion Matrix (Teste, %):\n")
            np.savetxt(cf, cm, fmt="%.2f", delimiter="\t")

        # 8) Relatório HTML e gráficos
        self.generate_html_report(searcher, cm, images_dir)
        print(f"[DONE] Relatório gerado em {self.args.results_dir}/report.html")

    def pruning_dataset(self, y, X, threshold):
        stds = X.std(axis=0)
        nonzero = stds > 0
        X_nz = X[:, nonzero]
        with np.errstate(divide='ignore', invalid='ignore'):
            cors = np.array([
                np.corrcoef(X_nz[:, i], y)[0, 1]
                if np.count_nonzero(~np.isnan(X_nz[:, i])) > 1 else 0
                for i in range(X_nz.shape[1])
            ])
        cors = np.nan_to_num(cors)
        keep = np.abs(cors) >= threshold
        orig_idx = np.where(nonzero)[0][keep]
        df = pd.DataFrame({
            'Original_Index': orig_idx + 1,
            'Correlation': cors[keep]
        })
        df.to_csv(os.path.join(self.args.results_dir, 'selected_features.csv'),
                  index=False)
        return y, X[:, orig_idx]

    def generate_html_report(self, searcher, cm_test, images_dir):
        # histograma de acurácias
        scores = searcher.cv_results_['mean_test_score'] * 100
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=20, alpha=0.7)
        plt.xlabel('Acurácia CV (%)')
        plt.ylabel('Frequência')
        plt.title('Distribuição de Acurácias em CV')
        plt.savefig(os.path.join(images_dir, 'acc_dist.png'))
        plt.close()

        # matriz de confusão
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
  <pre>{searcher.best_params_}</pre>
  <h2>Distribuição de Acurácias em CV</h2>
  <img src="images/acc_dist.png" alt="Distribuição de Acurácias">
  <h2>Matriz de Confusão no Teste</h2>
  <img src="images/cm_test.png" alt="Matriz de Confusão Teste">
</body>
</html>"""
        with open(os.path.join(self.args.results_dir, "report.html"), "w", encoding="utf-8") as f:
            f.write(html)

#========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Busca de Hiperparâmetros para SVM")
    p.add_argument('-d', '--dataset',    default='heart_scale', help='arquivo .libsvm')
    p.add_argument('--results-dir',      default='results',      help='pasta de saída')
    p.add_argument('--threshold',  type=float, default=0.1,   help='limiar correlação')
    p.add_argument('--test-size',  type=float, default=0.2,   help='proporção do hold-out')
    p.add_argument('--random-state',type=int,   default=1,     help='seed')
    p.add_argument('--cv',          type=int,   default=10,    help='nº folds CV')
    p.add_argument('--search',      choices=['grid','random','halving'], default='grid',
                   help='tipo de busca')
    p.add_argument('--n-iter',      type=int,   default=50,    help='iterações RandomizedSearch')
    p.add_argument('--factor',      type=int,   default=3,     help='fator HalvingGridSearch')
    p.add_argument('--n-jobs',      type=int,   default=-1,    help='n_jobs para paralelismo')
    p.add_argument('--verbose',     type=int,   default=1,     help='verbose das buscas')
    p.add_argument('--C-min',       type=float, default=-3,    help='expoente C inicial')
    p.add_argument('--C-max',       type=float, default=3,     help='expoente C final')
    p.add_argument('--C-steps',     type=int,   default=7,     help='pontos na faixa C')
    p.add_argument('--g-min',       type=float, default=-3,    help='expoente γ inicial')
    p.add_argument('--g-max',       type=float, default=3,     help='expoente γ final')
    p.add_argument('--g-steps',     type=int,   default=7,     help='pontos na faixa γ')
    p.add_argument('--kernels',     nargs='+',
                   default=['linear','poly','rbf','sigmoid'],
                   help='lista de kernels')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    SVMParametersSklearn(args).main()