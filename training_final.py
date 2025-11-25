# === Import potřebných knihoven === #
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from main import load_data, data_preprocessing, my_model

# === Finální trénink SVC === #
def training_final_svc(data_path: str = "diabetes_data.csv"):
    """
    Provede finální trénink SVC modelu na celém dostupném datasetu.

    Tato funkce slouží k vytvoření definitivního modelu, který bude následně
    použit pro externí testování ve skriptu `testing.py`. Oproti hlavnímu
    trénovacímu skriptu (`main.py`) zde neprobíhá rozdělení na trénovací,
    validační a testovací sadu – model je natrénován na všech dostupných datech
    pro maximalizaci výkonu finální verze.

    Postup:
        1. načtení surových dat pomocí `load_data`,
        2. předzpracování dat stejným způsobem jako při trénování (`data_preprocessing`),
        3. rozdělení na vstupní proměnné X a cílovou hodnotu y,
        4. inicializace SVC modelu pomocí `my_model("svc")`,
        5. škálování vstupních proměnných pomocí `StandardScaler`,
        6. uložení použitého scaleru pro pozdější použití při testování,
        7. natrénování modelu na kompletním datasetu,
        8. uložení finálního modelu ve formátu `.pkl`.

    Parametry
    ----------
    data_path : str
        Cesta k CSV souboru obsahujícímu vstupní dataset.

    Výstupy
    -------
    Uloží dva soubory:
        - trained_model_svc_final.pkl – finální natrénovaný SVC model,
        - scaler_svc_final.pkl – scaler použitý při trénování.
    """

    # Načtení dat
    df = load_data(data_path)

    # Preprocessing dat
    df = data_preprocessing(df)

    # Rozdělení na X a y
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Inicializace modelu SVC s finálními parametry
    model = my_model("svc")  # z main

    # Normalizace dat pomocí StandardScaleru (Z-score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Uložení scaleru
    joblib.dump(scaler, "scaler_svc_final.pkl")

    # Trénink modelu na celém datasetu
    model.fit(X_scaled, y)

    # Uložení modelu
    joblib.dump(model, "trained_model_svc_final.pkl")

# === Spouštěcí blok === #
if __name__ == "__main__":
    training_final_svc(data_path="diabetes_data.csv")
