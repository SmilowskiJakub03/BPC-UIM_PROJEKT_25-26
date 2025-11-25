# === Import knihoven === #
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import numpy as np

# Import z main.py
from main import data_preprocessing

def roc_krivka(model_path: str, test_data_path: str, scaler_path: str = None):
    """
    Načte model a testovací data, vypočítá ROC křivku a vykreslí ji.

    Parametry
    ----------
    model_path : str
        Cesta k uloženému modelu (.pkl).
    test_data_path : str
        Cesta k testovacím datům (.csv).
    scaler_path : str, optional
        Cesta ke scaleru (.pkl) – pokud model vyžaduje škálování.

    Návratové hodnoty
    -------
    roc_auc : float
        Hodnota plochy pod ROC křivkou (AUC).
    """

    #Načtení modelu
    model = joblib.load(model_path)

    # Načtení dat
    df_test = pd.read_csv(test_data_path)

    # preprocessing
    df_test = data_preprocessing(df_test)

    # rozdělení dat na X a y
    X_test = df_test.drop(columns=["Outcome"])
    y_test = df_test["Outcome"]

    # Aplikace scaleru, pokud existuje
    if scaler_path is not None:
        try:
            scaler = joblib.load(scaler_path)
            X_test = scaler.transform(X_test)
        except FileNotFoundError:
            print("Scaler nebyl nalezen – pokračuji bez škálování.")

    # Výpočet skóre (pravděpodobností)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
        # převod na 0–1 pomocí logistické funkce
        y_scores = 1 / (1 + np.exp(-y_scores))
    else:
        raise ValueError("Model nepodporuje ani predict_proba(), ani decision_function().")

    # Výpočet ROC křivky a AUC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Vykreslení ROC křivky
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="red", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="blue", lw=1.5, linestyle="--", label="Náhodná klasifikace")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC křivka")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nROC AUC: {roc_auc:.4f}")
    return roc_auc


# Spouštěcí blok
if __name__ == "__main__":

    roc_krivka(
        model_path="trained_model_xgb.pkl",
        test_data_path="test_preprocessed_xgb.csv",
        scaler_path=""
    )

    # #trained model
    # trained_model_svc.pkl
    # trained_model_logreg.pkl
    # trained_model_rf.pkl
    # trained_model_xgb.pkl

    # test_preprocessed
    # test_preprocessed_svc.csv
    # test_preprocessed_logreg.csv
    # test_preprocessed_rf.csv
    # test_preprocessed_xgb.csv

    # scaler
    # scaler_svc.pkl
    # scaler_logreg.pkl
