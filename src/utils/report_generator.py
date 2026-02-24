import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
import pandas as pd
import os

class ReportGenerator:
    """
    Generates training/evaluation reports for the Multi-Task Model.
    Covers Shape (Classification), Features (Multi-label), and Skin Tone (Regression).
    """
    def __init__(self, output_dir="results/reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_classification_report(self, y_true, y_pred, labels, task_name="Face_Shape"):
        """Generates heatmap and metrics for single-label classification."""
        # 1. Metrics
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(os.path.join(self.output_dir, f"{task_name}_metrics.csv"))
        
        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.title(f"{task_name} Confusion Matrix")
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        
        cm_path = os.path.join(self.output_dir, f"{task_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        return cm_path

    def generate_training_summary(self, history):
        """Generates loss/accuracy curves from training history."""
        df_history = pd.DataFrame(history)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_history)
        plt.title("Multi-Task Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        
        history_path = os.path.join(self.output_dir, "training_history.png")
        plt.savefig(history_path)
        plt.close()
        return history_path

    def generate_skin_tone_metrics(self, ita_true, ita_pred):
        """Generates MAE and Error Distribution for ITA regression."""
        mae = mean_absolute_error(ita_true, ita_pred)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(ita_true, ita_pred, alpha=0.5)
        plt.plot([min(ita_true), max(ita_true)], [min(ita_true), max(ita_true)], 'r--')
        plt.title(f"ITA Regression Analysis (MAE: {mae:.2f})")
        plt.xlabel("True ITA")
        plt.ylabel("Predicted ITA")
        
        ita_path = os.path.join(self.output_dir, "ita_regression_analysis.png")
        plt.savefig(ita_path)
        plt.close()
        return {"mae": mae, "plot": ita_path}
