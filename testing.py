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
# Import potřebných modulů
import pandas as pd
import joblib
import os

# Import funkcí z mainu
from main import compute_statistics
from main import data_preprocessing

def test_model(model_path: str, scaler_path: str, test_data_path: str):
    """
    Otestuje uložený klasifikační model na zadaném testovacím datasetu.

    Parametry
    ---------
    model_path : str
        Cesta k uloženému modelu ve formátu .pkl (uloženému např. pomocí joblib).
    scaler_path : str
        Cesta ke scaleru použitěnému při trénování modelu. Pokud scaler existuje,
        bude aplikován i na testovací data. Pokud není nalezen, test proběhne bez škálování.
    test_data_path : str
        Cesta k CSV souboru obsahujícímu testovací dataset

    Návratové hodnoty
    -----------------
    cm : np.ndarray
        Matice záměny kde TN = True Negative, FP = False Positive, FN = False Negative, TP = True Positive.
    mcc : float
        Matthews correlation coefficient

    Poznámky
    --------
    Funkce provede kompletní testovací pipeline:
        - načte uložený model a testovací data,
        - provede stejné předzpracování jako při trénování (`data_preprocessing`),
        - pokud je dostupný scaler, aplikuje jej na vstupní proměnné,
        - provede predikci modelu na testovací sadě,
        - vyhodnotí výsledky pomocí `compute_statistics`,
          včetně výpočtu MCC a volitelného vykreslení matice záměny.

    Historie použití
    ----------------
    V předchozích verzích projektu byl tento testovací skript určen k porovnávání
    a testování **více různých modelů**.
    V aktuální verzi však slouží výhradně k testování **jediného finálního modelu**,
    který byl vybrán a uložen během trénovací fáze.

    Funkce tak nyní umožňuje samostatné, čisté a reprodukovatelné otestování
    finálního modelu na externím nebo validačním datasetu.
    """

    #Načtení uloženého modelu
    model = joblib.load(model_path)

    #Načtení testovacích dat
    df_test = pd.read_csv(test_data_path)

    #Preprocessing
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
        model_path="trained_model_svc_final.pkl",
        scaler_path="scaler_svc_final.pkl.pkl",
        test_data_path="", # <------------------------------ zadání cesty k testovacímu souboru
    )
