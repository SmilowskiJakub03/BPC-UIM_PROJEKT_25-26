# -*- coding: utf-8 -*-

"""
Created on 11. 09. 2025 at 11:33:18

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Tento script slouží k testování vašeho modelu.
    V rámci testování byste měli vytvořit funkci, která načte data a model,
    provede testování a vypíše výsledky včetně matice záměny a mathews correlation coefficient.
"""
# Import necessary modules
import pandas as pd
import joblib
import numpy as np
import os

# Import funkcí z mainu
from main import compute_statistics
from main import data_preprocessing

# při odevzdání trénování modelu na celém datasetu
# testing skript upravit aby nahrál model, přidat preproccesinf na test_data_path

def test_model(model_path: str, scaler_path: str, test_data_path: str):
    """
    Otestuje natrénovaný model na zadaném testovacím datasetu.

    Parameters
    ----------
    model_path : str
        Cesta k uloženému modelu (.pkl soubor).
    test_data_path : str
        Cesta k CSV souboru s testovacími daty.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix (TN, FP, FN, TP).
    mcc : float
        Matthews correlation coefficient.
    """

    #Načtení uloženého modelu
    model = joblib.load(model_path)

    #Načtení testovacích dat
    df_test = pd.read_csv(test_data_path)

    # preprocessing
    df_test = data_preprocessing(df_test)

    #Rozdělení na vstupy (X) a cílovou proměnnou
    X_test = df_test.drop(columns=["Outcome"])
    y_test = df_test["Outcome"]

    # Načtení scaleru (pokud existuje)
    if scaler_path is not None and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test = scaler.transform(X_test)
    else:
        print("Scaler nebyl nalezen — pokračuji bez škálování.")

    #Predikce na testovacích datech
    y_pred = model.predict(X_test)

    # Vyhodnocení (z mainu)
    cm, mcc = compute_statistics(y_test, y_pred, plot=True) # TRUE pro plot cm

    return cm, mcc


if __name__ == "__main__":
    # Spustí test modelu
    test_model(
        model_path="trained_model_svc.pkl",
        scaler_path="scaler_svc.pkl", # scaler_svc.pkl, scaler_logreg.pkl
        test_data_path="test_preprocessed_svc.csv",
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