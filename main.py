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
# === Import potřebných modulů === #
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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# My model
# ==== LOAD DATA === #
def load_data(path: str = "diabetes_data.csv", random_state: int = 42):
    """
    Načte dataset z CSV souboru a vrátí jej ve formě pandas DataFrame.

    Parametry
    ---------
    path : str
        Cesta k CSV souboru obsahujícímu vstupní data.
    random_state : int
        Seed pro reprodukovatelnost (zde se nevyužívá, ale zachováno pro kompatibilitu).

    Návratová hodnota
    -----------------
    df : pd.DataFrame
        Načtený dataset bez dalších úprav.
    """
    df = pd.read_csv(path)
    return df

# === Preprocessing ====
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provede kompletní předzpracování vstupního datasetu:
        - detekci a označení chybných hodnot jako NaN,
        - zaokrouhlení hodnot tam, kde je to vhodné,
        - imputaci chybějících dat pomocí KNN,
        - obnovení celočíselných datových typů u příslušných sloupců.

    Parametry
    ---------
    df : pd.DataFrame
        Vstupní DataFrame obsahující neupravená a potenciálně chybná data.

    Návratová hodnota
    -----------------
    df_imputed : pd.DataFrame
        Předzpracovaná a imputovaná tabulka připravená pro modelování.

    Poznámka
    --------
    Funkce aplikuje doménově specifické kontroly rozsahů (např. fyziologické limity
    glykemie, BMI, krevního tlaku atd.). Chybné hodnoty jsou převedeny na NaN a následně
    doplněny pomocí KNNImputer.
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
def my_model(model_type: str = "svc", random_state: int = 42):
    """
    Inicializuje a vrátí klasifikační model zvoleného typu.

    Parametry
    ---------
    model_type : str
        Identifikátor modelu. Možné hodnoty:
            - "logreg" : logistická regrese
            - "rf"     : Random Forest
            - "xgb"    : XGBoost classifier
            - "svc"    : Support Vector Classifier
    random_state : int
        Seed pro reprodukovatelnost výsledků.

    Návratová hodnota
    -----------------
    model : objekt scikit-learn nebo XGBoost
        Inicializovaný model připravený k trénování na trénovacích datech.

    Výjimky
    -------
    ValueError
        Pokud je zadán neznámý typ modelu.
    """

    if model_type == "logreg":
        return LogisticRegression(
            solver="liblinear",
            random_state=random_state,
            max_iter=1000,
            C=0.001,
            class_weight='balanced',
            penalty = 'l2'
        )
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            min_samples_split=2,
            min_samples_leaf=2,
            max_depth=None
        )
    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=4,
            subsample=1,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss"
        )
    elif model_type == "svc":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            C=1,
            gamma=0.1,
            random_state=random_state
        )

    else:
        raise ValueError("Neznámý model_type.")

# ========== STATISTICS ==========
def compute_statistics(y_true, y_pred, plot: bool = True):
    """
    Vypočítá a zobrazí základní metriky klasifikační úspěšnosti modelu.

    Parametry
    ---------
    y_true : array-like
        Skutečné (pravdivé) štítky tříd 0/1.
    y_pred : array-like
        Predikované štítky tříd 0/1.
    plot : bool
        Pokud True, vykreslí matici záměny jako heatmapu pomocí Seaborn.

    Návratové hodnoty
    -----------------
    cm : np.ndarray
        Matice záměny ve formátu [[TN, FP], [FN, TP]].
    mcc : float
        Matthews correlation coefficient — robustní metrika kvality modelu pro nevyvážené datasety.

    Poznámka
    --------
    Funkce vypíše hlavní metriky do konzole a volitelně vykreslí heatmapu,
    která vizuálně zobrazuje správné a chybné klasifikace.
    """
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

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
    Hlavní řídicí funkce celého skriptu.

    Funkce provádí následující kroky:
        1. načte dataset z CSV,
        2. provede jeho předzpracování,
        3. rozdělí data na train / validation / test (poměr 70 / 15 / 15),
        4. inicializuje vybraný model,
        5. provede případné škálování dat (pro logreg a SVC),
        6. natrénuje model na trénovacích datech,
        7. vyhodnotí výkon modelu na validačních datech,
        8. uloží vytrénovaný model i scaler (pokud byl použit),
        9. uloží předzpracovaná testovací data pro další použití.

    Návratová hodnota
    -------
        None

    Poznámka
    --------
    Tato funkce neslouží pouze jako pipeline pro trénování modelu,
    ale také jako centrální místo pro definici a export důležitých funkcí*
    (např. `data_preprocessing`, `compute_statistics').
    Finální verze modelu je dotrénována na celém trénovacím datasetu,
    aby se maximalizoval jeho výkon před uložením.

    Historie použití
    ----------------
    Tento skript původně sloužil k rozdělení dat původního datasetu,
    jeho předzpracování a k výpočtu hodnotích metrik na validační sadě.
    Skript nyní slouží zejména pro definici a export důležitých funkcí.
    """

    # Načtení dat
    df = load_data("diabetes_data.csv")

    # Preprocessing
    df = data_preprocessing(df)

    # Rozdělení dat: 70/15/15
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["Outcome"]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["Outcome"]
    )

    # Rozdělení X a y
    #X → matice vstupních proměnných (nezávislé proměnné, prediktory).
    #y → vektor cílových hodnot (závislá proměnná, co chceme předpovědět).
    X_train = train_df.drop(columns=["Outcome"])
    y_train = train_df["Outcome"]

    X_valid = valid_df.drop(columns=["Outcome"])
    y_valid = valid_df["Outcome"]

    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]

    # Inicializace modelu (výchozí – Random Forest)
    model_type = "rf"  # logreg / rf / xgb / svc
    model = my_model(model_type)

    # volitelné škálování
    scaler = None
    if model_type in ["logreg", "svc"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) # fit + transform (učíme se z tréninkových dat)
        X_valid = scaler.transform(X_valid) # pouze transform (stejné měřítko)
        X_test = scaler.transform(X_test)  # pouze transform (stejné měřítko)

        # ulož scaler
        joblib.dump(scaler, f"scaler_{model_type}.pkl")
        print(f"Scaler uložen jako scaler_{model_type}.pkl")

    # Trénink modelu
    model.fit(X_train, y_train)

    # Predikce
    y_pred_valid = model.predict(X_valid)

    # Vyhodnocení
    compute_statistics(y_valid, y_pred_valid,plot=True)

    # Uložení modelu
    joblib.dump(model, f"trained_model_{model_type}.pkl")

    # Uložení testovacích dat do CSV
    test_df.to_csv(f"test_preprocessed_{model_type}.csv", index=False)

if __name__ == "__main__":
    main()
