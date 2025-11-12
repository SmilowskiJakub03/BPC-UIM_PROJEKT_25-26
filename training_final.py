
# === Import potřebných knihoven === #
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from main import load_data, data_preprocessing, my_model

# === Finální trénink SVC === #
def training_final_svc(data_path: str = "diabetes_data.csv"):
    """
    Finální trénink modelu SVC na celém datasetu.

    Parameters
    ----------
    data_path : str
        Cesta k CSV souboru s daty.
    """
    # Načtení dat
    df = load_data(data_path)

    # Preproccesing dat
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
