# -*- coding: utf-8 -*-

"""
Created on 11. 09. 2025 at 11:13:56

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
    Tento script slouží jako hlavní spouštěcí bod pro projekt.
    Skript berte jako volný rámec, který můžete upravit dle svých potřeb.    
"""
# === Import necessary modules === #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import seaborn as sns
import joblib

# My model
# ==== LOAD DATA === #
def load_data(path: str = "diabetes_data.csv", test_size: float = 0.3, random_state: int = 42):
    """
    Načte dataset z CSV a rozdělí ho na trénovací a testovací část.

    Parameters
    ----------
    path : str
        Cesta k CSV souboru s daty.
    test_size : float
        Procento dat určených na testování (0.3 = 30%).
    random_state : int
        Seed pro reprodukovatelnost rozdělení. random_state=42 zajistí, že rozpad dat i model budou pokaždé stejné

    Returns
    -------
    train_df : pd.DataFrame
        Trénovací dataset.
    test_df : pd.DataFrame
        Testovací dataset.
    """
    df = pd.read_csv(path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["Outcome"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# === Preprocessing ====
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Předzpracování dat: odstranění chybných hodnot, imputace pomocí KNN.

    Parameters
    ----------
    df : pd.DataFrame
        Vstupní DataFrame s neupravenými daty.

    Returns
    -------
    df_imputed : pd.DataFrame
        Předzpracovaný dataset po KNN imputaci.
    """

    # Pregnancies
    df.loc[(df["Pregnancies"] < 0) | (df["Pregnancies"] > 15), "Pregnancies"] = np.nan
    df["Pregnancies"] = df["Pregnancies"].round(0).astype("Int64")

    # Glucose
    df.loc[(df["Glucose"] < 50) | (df["Glucose"] > 500), "Glucose"] = np.nan

    # BloodPressure
    df.loc[(df["BloodPressure"] < 40) | (df["BloodPressure"] > 200), "BloodPressure"] = np.nan
    df["BloodPressure"] = df["BloodPressure"].round(0).astype("Int64")

    # SkinThickness
    df.loc[
        (df["SkinThickness"] <= 0) | (df["SkinThickness"] < 5) | (df["SkinThickness"] > 80), "SkinThickness"] = np.nan
    df["SkinThickness"] = df["SkinThickness"].round(0).astype("Int64")

    # Insulin
    df.loc[(df["Insulin"] <= 0) | (df["Insulin"] < 10) | (df["Insulin"] > 300), "Insulin"] = np.nan
    df["Insulin"] = df["Insulin"].round(0).astype("Int64")

    # BMI
    df.loc[(df["BMI"] <= 0) | (df["BMI"] < 10) | (df["BMI"] > 70), "BMI"] = np.nan

    # DiabetesPedigreeFunction
    df.loc[df["DiabetesPedigreeFunction"] < 0, "DiabetesPedigreeFunction"] = np.nan

    # Age
    df.loc[(df["Age"] <= 0) | (df["Age"] < 1) | (df["Age"] > 80), "Age"] = np.nan
    df["Age"] = df["Age"].round(0).astype("Int64")

    # KNN imputace
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Přetypování zpět u integer sloupců
    for col in ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "Age"]:
        df_imputed[col] = df_imputed[col].round(0).astype(int)

    return df_imputed


# === MODEL ===
def my_model(model_type: str = "rf", random_state: int = 42):
    """
    Vytvoří a vrátí klasifikační model podle zadaného typu.

    Parameters
    ----------
    model_type : str
        Typ modelu: 'logreg' = Logistic Regression, 'rf' = Random Forest, 'xgb' = XGBoost. Výchozí hodnota 'rf' = Random Forest
    random_state : int
        Seed pro reprodukovatelnost.

    Returns
    -------
    model : object
        Inicializovaný model připravený k trénování.
    """
    if model_type == "logreg":
        return LogisticRegression(solver="liblinear", random_state=random_state, max_iter=1000)
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight="balanced"
        )
    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=random_state, use_label_encoder=False, eval_metric="logloss"
        )
    else:
        raise ValueError("Neznámý model_type. Použij 'logreg', 'rf' nebo 'xgb'.")

# ========== STATISTICS ==========
def compute_statistics(y_true, y_pred, plot: bool = True):
    """
    Vyhodnotí predikce modelu pomocí confusion matrix a Matthews correlation coefficient.

    Parameters
    ----------
    y_true : array-like
        Skutečné hodnoty (0/1).
    y_pred : array-like
        Predikované hodnoty (0/1).
    plot : bool
        Pokud True, vykreslí confusion matrix jako heatmapu.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix [[TN, FP], [FN, TP]].
    mcc : float
        Matthews correlation coefficient.
    """
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    #True Positive (TP): Počet správně klasifikovaných kladných případů (model předpověděl „ANO“, skutečnost byla „ANO“).
    #True Negative (TN): Počet správně klasifikovaných záporných případů (model předpověděl „NE“, skutečnost byla „NE“).
    #False Positive (FP): Počet chybně klasifikovaných kladných případů (model předpověděl „ANO“, ale skutečnost byla „NE“). Také se označuje jako chyba 1. typu.
    #False Negative (FN): Počet chybně klasifikovaných záporných případů (model předpověděl „NE“, ale skutečnost byla „ANO“). Také se označuje jako chyba 2. typu.

    print("\n=== Confusion Matrix ===")
    print(cm)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # Pokud True, vykreslí confusion matrix jako heatmapu
    if plot:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"])
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    return cm, mcc

# ===MAIN ===
def main():
    """
    Hlavní spouštěcí funkce:
    - načte a předzpracuje data
    - rozdělí na train/test
    - vytrénuje model
    - vyhodnotí predikce na testovací sadě
    - uloží vytrénovaný model pro pozdější testování
    """

    # Načtení dat
    train_df, test_df = load_data("diabetes_data.csv")

    # Preprocessing
    train_df = data_preprocessing(train_df)
    test_df = data_preprocessing(test_df)

    # Rozdělení X a y
    #X → matice vstupních proměnných (nezávislé proměnné, prediktory).
    #y → vektor cílových hodnot (závislá proměnná, co chceme předpovědět).
    X_train, y_train = train_df.drop(columns=["Outcome"]), train_df["Outcome"]
    X_test, y_test = test_df.drop(columns=["Outcome"]), test_df["Outcome"]

    # Inicializace modelu (výchozí – Random Forest)
    model = my_model("xgb")
    # Typ modelu: 'logreg' = Logistic Regression, 'rf' = Random Forest, 'xgb' = XGBoost

    # Trénink modelu
    model.fit(X_train, y_train)

    # Predikce
    y_pred = model.predict(X_test)

    # Vyhodnocení
    compute_statistics(y_test, y_pred,plot=True)

    # Uložení modelu
    #joblib.dump(model, "trained_model_rf.pkl")
    #joblib.dump(model, "trained_model_logreg.pkl")
    joblib.dump(model, "trained_model_xgb.pkl")

if __name__ == "__main__":
    main()
