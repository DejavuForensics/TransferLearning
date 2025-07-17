import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import os
import json
from pathlib import Path

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = '.0f'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def generate_html_report(results, output_dir='SVM/results'):
    """
    Generate an HTML report with SVM results including confusion matrices and metrics.
    
    Args:
        results (dict): Dictionary containing results for each fold and overall metrics
        output_dir (str): Directory to save the report and images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get number of folds from results
    n_folds = len([key for key in results.keys() if key.startswith('fold_')])
    
    # Save confusion matrix plots
    for fold in range(n_folds):
        plt = plot_confusion_matrix(
            results[f'fold_{fold}']['confusion_matrix'],
            classes=['Benign', 'Malware'],
            title=f'Confusion Matrix - Fold {fold+1}'
        )
        plt.savefig(f'{output_dir}/confusion_matrix_fold_{fold+1}.png')
        plt.close()
        
        # Normalized confusion matrix
        plt = plot_confusion_matrix(
            results[f'fold_{fold}']['confusion_matrix'],
            classes=['Benign', 'Malware'],
            normalize=True,
            title=f'Normalized Confusion Matrix - Fold {fold+1}'
        )
        plt.savefig(f'{output_dir}/confusion_matrix_norm_fold_{fold+1}.png')
        plt.close()

    # Calculate average confusion matrix
    avg_confusion_matrix = np.zeros((2, 2))
    for fold in range(n_folds):
        avg_confusion_matrix += results[f'fold_{fold}']['confusion_matrix']
    avg_confusion_matrix = avg_confusion_matrix / n_folds

    # Plot average confusion matrix
    plt = plot_confusion_matrix(
        avg_confusion_matrix,
        classes=['Benign', 'Malware'],
        title='Average Confusion Matrix (All Folds)'
    )
    plt.savefig(f'{output_dir}/confusion_matrix_average.png')
    plt.close()

    # Normalized average confusion matrix
    plt = plot_confusion_matrix(
        avg_confusion_matrix,
        classes=['Benign', 'Malware'],
        normalize=True,
        title='Normalized Average Confusion Matrix (All Folds)'
    )
    plt.savefig(f'{output_dir}/confusion_matrix_average_norm.png')
    plt.close()

    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SVM Classification Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .metrics-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            .metrics-table th { background-color: #f2f2f2; }
            .metrics-table tr:nth-child(even) { background-color: #f9f9f9; }
            .confusion-matrix { display: flex; flex-wrap: wrap; justify-content: space-around; }
            .matrix-container { margin: 10px; text-align: center; }
            .matrix-container img { max-width: 400px; }
            h1, h2 { color: #333; }
            .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .avg-metrics { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SVM Classification Results</h1>
            
            <div class="summary">
                <h2>Overall Performance (Percentages)</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Metric</th>
                        <th>Mean (%)</th>
                        <th>Standard Deviation (%)</th>
                    </tr>
                    <tr>
                        <td>Accuracy</td>
                        <td>{{ "%.2f"|format(overall_metrics.accuracy_mean * 100) }}</td>
                        <td>{{ "%.2f"|format(overall_metrics.accuracy_std * 100) }}</td>
                    </tr>
                    <tr>
                        <td>Precision</td>
                        <td>{{ "%.2f"|format(overall_metrics.precision_mean * 100) }}</td>
                        <td>{{ "%.2f"|format(overall_metrics.precision_std * 100) }}</td>
                    </tr>
                    <tr>
                        <td>Recall</td>
                        <td>{{ "%.2f"|format(overall_metrics.recall_mean * 100) }}</td>
                        <td>{{ "%.2f"|format(overall_metrics.recall_std * 100) }}</td>
                    </tr>
                    <tr>
                        <td>F1-Score</td>
                        <td>{{ "%.2f"|format(overall_metrics.f1_mean * 100) }}</td>
                        <td>{{ "%.2f"|format(overall_metrics.f1_std * 100) }}</td>
                    </tr>
                    <tr>
                        <td>AUC</td>
                        <td>{{ "%.2f"|format(overall_metrics.auc_mean * 100) }}</td>
                        <td>{{ "%.2f"|format(overall_metrics.auc_std * 100) }}</td>
                    </tr>
                </table>
            </div>

            <div class="avg-metrics">
                <h2>Average Confusion Matrix Values</h2>
                <table class="metrics-table">
                    <tr>
                        <th></th>
                        <th>Predicted Benign</th>
                        <th>Predicted Malware</th>
                    </tr>
                    <tr>
                        <td><strong>Actual Benign</strong></td>
                        <td>{{ "%.2f"|format(avg_confusion_matrix[0][0]) }}</td>
                        <td>{{ "%.2f"|format(avg_confusion_matrix[0][1]) }}</td>
                    </tr>
                    <tr>
                        <td><strong>Actual Malware</strong></td>
                        <td>{{ "%.2f"|format(avg_confusion_matrix[1][0]) }}</td>
                        <td>{{ "%.2f"|format(avg_confusion_matrix[1][1]) }}</td>
                    </tr>
                </table>
                
                <h3>Average Confusion Matrix Visualization</h3>
                <div style="text-align: center;">
                    <img src="confusion_matrix_average.png" alt="Average Confusion Matrix" style="max-width: 500px;">
                    <img src="confusion_matrix_average_norm.png" alt="Normalized Average Confusion Matrix" style="max-width: 500px;">
                </div>
            </div>

            <h2>Per-Fold Results (Percentages)</h2>
            <table class="metrics-table">
                <tr>
                    <th>Fold</th>
                    <th>Accuracy (%)</th>
                    <th>Precision (%)</th>
                    <th>Recall (%)</th>
                    <th>F1-Score (%)</th>
                    <th>AUC (%)</th>
                </tr>
                {% for fold in range(n_folds) %}
                <tr>
                    <td>{{ fold + 1 }}</td>
                    <td>{{ "%.2f"|format(results['fold_' + fold|string].accuracy * 100) }}</td>
                    <td>{{ "%.2f"|format(results['fold_' + fold|string].precision * 100) }}</td>
                    <td>{{ "%.2f"|format(results['fold_' + fold|string].recall * 100) }}</td>
                    <td>{{ "%.2f"|format(results['fold_' + fold|string].f1 * 100) }}</td>
                    <td>{{ "%.2f"|format(results['fold_' + fold|string].auc * 100) }}</td>
                </tr>
                {% endfor %}
            </table>

            <h2>Individual Fold Confusion Matrices</h2>
            <div class="confusion-matrix">
                {% for fold in range(n_folds) %}
                <div class="matrix-container">
                    <h3>Fold {{ fold + 1 }}</h3>
                    <img src="confusion_matrix_fold_{{ fold + 1 }}.png" alt="Confusion Matrix Fold {{ fold + 1 }}">
                    <img src="confusion_matrix_norm_fold_{{ fold + 1 }}.png" alt="Normalized Confusion Matrix Fold {{ fold + 1 }}">
                </div>
                {% endfor %}
            </div>
        </div>
    </body>
    </html>
    """

    # Calculate overall metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    overall_metrics = {}
    for metric in metrics:
        values = [results[f'fold_{i}'][metric] for i in range(n_folds)]
        overall_metrics[f'{metric}_mean'] = np.mean(values)
        overall_metrics[f'{metric}_std'] = np.std(values)

    # Render HTML
    template = Template(html_template)
    html_content = template.render(
        results=results,
        overall_metrics=type('Metrics', (), overall_metrics),
        avg_confusion_matrix=avg_confusion_matrix,
        n_folds=n_folds
    )

    # Save HTML file
    with open(f'{output_dir}/svm_results_report.html', 'w') as f:
        f.write(html_content)

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate all relevant metrics for a single fold.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary containing all metrics
    """
    return {
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob)
    }

if __name__ == "__main__":
    # Example usage
    results = {}
    for fold in range(5):
        # Load your results for each fold here
        # This is just an example - you'll need to replace this with your actual data loading
        results[f'fold_{fold}'] = {
            'confusion_matrix': np.array([[100, 20], [10, 100]]),  # Example confusion matrix
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.88,
            'f1': 0.85,
            'auc': 0.92
        }
    
    generate_html_report(results) 