from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_bar_importance(importances, feature_names, path, title, top_n=30):
    imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp = imp.sort_values("importance", ascending=False).head(top_n)
    plt.figure(figsize=(10, max(6, 0.28 * len(imp))))
    plt.barh(imp["feature"][::-1], imp["importance"][::-1])
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return imp


def predicted_vs_actual(y_true, y_pred, path, title="Predicted vs Actual"):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75)
    mn = min(np.nanmin(y_true), np.nanmin(y_pred))
    mx = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def residual_plot(y_true, y_pred, path, title="Residuals"):
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual = Actual - Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def confusion_matrix_plot(cm, labels, path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(max(7, 0.6*len(labels)), max(6, 0.5*len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def pca_scatter(pca_df, color_col, path, title):
    plt.figure(figsize=(8, 6))
    labels = pca_df[color_col].astype(str).fillna("NA")
    for lab in sorted(labels.unique()):
        mask = labels == lab
        plt.scatter(pca_df.loc[mask, "PC1"], pca_df.loc[mask, "PC2"], label=lab, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    if labels.nunique() <= 12:
        plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
