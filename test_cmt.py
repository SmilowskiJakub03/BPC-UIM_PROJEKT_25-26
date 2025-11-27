import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_data(path: str = "diabetes_data.csv", test_size: float = 0.3, random_state: int = 42):
    """
    Načte dataset z CSV souboru a provede stratifikované rozdělení na trénovací a testovací sadu.
    Funkce zajišťuje konzistentní přípravu dat při experimentálním testování preprocessing kroků,
    přičemž obě vrácené množiny mají resetované indexy pro snadnější další zpracování.

    Parametry
    ----------
    path : str, default="diabetes_data.csv"
        Cesta k CSV souboru obsahujícímu vstupní dataset s proměnnými potřebnými pro klasifikaci diabetu.
    test_size : float, default=0.3
        Procentuální poměr (0–1), určující velikost testovací sady při rozdělení datasetu.
    random_state : int, default=42
        Seed pro reprodukovatelnost náhodného rozdělení dat.

    Returns
    -------
    train_df : pandas.DataFrame
        Trénovací část původního datasetu po rozdělení.
    test_df : pandas.DataFrame
        Testovací část datasetu se stejným rozložením cílové proměnné díky stratifikaci.
    """


    # Načtení dat
    df = pd.read_csv(path)

    # Rozdělení na train/test
    # při odevzdávání se model natrénuje na celé sadě, nebude potřeba rozdělit dataset na train a test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["Outcome"])

    # Reset indexů, aby byly čisté
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def data_preprocessing(df: pd.DataFrame, save_path: str ) -> pd.DataFrame:
    """
    Provádí komplexní předzpracování dat podle doménových pravidel pro diabetologický dataset.
    Funkce aplikuje fyziologické limity pro jednotlivé atributy, převádí chybné hodnoty na NaN,
    provádí imputaci pomocí KNNImputer a připravuje také Z-score škálovanou verzi dat.
    Primárně ukládá dataset ve fázi po základním čištění (s NaN hodnotami) a vrací jej ke
    skladování nebo analýze; alternativně může vrátit i imputovanou či škálovanou variantu.

    Parametry
    ----------
    df : pandas.DataFrame
        Vstupní dataset obsahující originální hodnoty atributů, včetně cílové proměnné "Outcome".
        Očekává se struktura typická pro diabetes dataset (např. Pregnancies, Glucose, BMI atd.).
    save_path : str
        Cesta k výstupnímu CSV souboru, do kterého bude uložena verze datasetu po provedení
        základního čištění (nahrazení chybných hodnot NaN a úpravě rozsahů).

    Returns
    -------
    pandas.DataFrame
        DataFrame obsahující dataset po aplikaci kontrol rozsahů a označení chybných hodnot
        jako NaN. Slouží jako základ pro následné imputace nebo škálování.
    """

    # === Pregnancies ===
    df.loc[(df["Pregnancies"] < 0) | (df["Pregnancies"] > 15), "Pregnancies"] = np.nan
    df["Pregnancies"] = df["Pregnancies"].round(0).astype("Int64")

    # === Glucose ===
    df.loc[(df["Glucose"] < 50) | (df["Glucose"] > 500), "Glucose"] = np.nan

    # === BloodPressure ===
    df.loc[(df["BloodPressure"] < 40) | (df["BloodPressure"] > 200), "BloodPressure"] = np.nan
    df["BloodPressure"] = df["BloodPressure"].round(0).astype("Int64")

    # === SkinThickness ===
    df.loc[(df["SkinThickness"] <= 0) | (df["SkinThickness"] < 5) | (df["SkinThickness"] > 80), "SkinThickness"] = np.nan
    df["SkinThickness"] = df["SkinThickness"].round(0).astype("Int64")

    # === Insulin ===
    df.loc[(df["Insulin"] <= 0) | (df["Insulin"] < 10) | (df["Insulin"] > 300), "Insulin"] = np.nan
    df["Insulin"] = df["Insulin"].round(0).astype("Int64")

    # === BMI ===
    df.loc[(df["BMI"] <= 0) | (df["BMI"] < 10) | (df["BMI"] > 70), "BMI"] = np.nan

    # === DiabetesPedigreeFunction ===
    df.loc[df["DiabetesPedigreeFunction"] < 0, "DiabetesPedigreeFunction"] = np.nan

    # === Age ===
    df.loc[(df["Age"] <= 0) | (df["Age"] < 1) | (df["Age"] > 80), "Age"] = np.nan
    df["Age"] = df["Age"].round(0).astype("Int64")

    # === KNN imputace ===
    imputer = KNNImputer(n_neighbors=5, weights="uniform")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Přetypování zpět u integer sloupců
    for col in ["Pregnancies", "BloodPressure", "SkinThickness", "Insulin", "Age"]:
        df_imputed[col] = df_imputed[col].round(0).astype(int)

    # Z-score škálování
    features = df_imputed.drop(columns=["Outcome"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled["Outcome"] = df_imputed["Outcome"]

    #  === Uložení datasetu s NaN ===
    df.to_csv(save_path, index=False)

    # === Uložení datasetu s IMPUTACÍ ===
    #df_imputed.to_csv(save_path, index=False)

    # === Uložení datasetu se Z-score ===
    #df_scaled.to_csv(save_path, index=False)

    # === Výpočet počtu NaN hodnot ===
    # nan_counts = df.isna().sum()
    # print("\nPočet NaN hodnot v každém sloupci:")
    # print(nan_counts)

    # === Vizualizace počtu NaN hodnot ===
    # plt.figure(figsize=(8,5))
    # nan_counts.plot(kind="bar", color="steelblue", edgecolor="black")
    # plt.title("Počet NaN hodnot ve sloupcích")
    # plt.ylabel("Počet NaN")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.show()

    return  df #data_preprocessing hodnoty s NAN
    #return df_imputed #data_preprocessing hodnoty s IMPUTEM
    #return df_scaled #data_preprocessing se Z - score


# === Spuštění ===
if __name__ == "__main__":
    train_data, test_data = load_data("diabetes_data.csv")

    pd.set_option("display.max_columns",None)
    pd.set_option("display.width",None)
    pd.set_option("display.max_rows",None)

    # raw data
    print("\nPo preprocessingu (train):")
    print("Train set:", train_data.shape)
    print("Test set:", test_data.shape)

    print(train_data.head())
    print(test_data.head())

    # Preprocessing uložení
    train_data = data_preprocessing(train_data, save_path="train_preprocessed_NAN.csv") #train NAN
    test_data = data_preprocessing(test_data, save_path="test_preprocessed_NAN.csv") #test NAN
    #train_data = data_preprocessing(train_data, save_path="train_preprocessed_IMPUTE.csv") # train IMPUTE
    #test_data = data_preprocessing(test_data, save_path="test_preprocessed_IMPUTE.csv") # test IMPUTE
    #train_data = data_preprocessing(train_data, save_path="train_preprocessed_ZSCORE.csv") # train ZSCORE
    #test_data = data_preprocessing(train_data, save_path="test_preprocessed_ZSCORE.csv") # test ZCORE


    # Preprocessing

    print("\nPo preprocessingu:")
    print(train_data.head())
    print(test_data.head())

