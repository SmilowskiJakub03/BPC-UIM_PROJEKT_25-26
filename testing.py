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

# Import funkcí z mainu
from main import data_preprocessing, compute_statistics

def test_model(model_path="trained_model_xgb.pkl", test_data_path="test_preprocessed_xgb.csv"):
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
    test_df = pd.read_csv(test_data_path)

    #Rozdělení na vstupy (X) a cílovou proměnnou
    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]

    # 5. Predikce na testovacích datech
    y_pred = model.predict(X_test)

    # Vyhodnocení (z mainu)
    cm, mcc = compute_statistics(y_test, y_pred, plot=True) # TRUE pro plot cm

    return cm, mcc


if __name__ == "__main__":
    # Spustí test modelu
    test_model()
