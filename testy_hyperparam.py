# === Import knihoven === #
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
import joblib

# import vlastních funkcí z main.py
from main import load_data, data_preprocessing

def hyper_param_test(model_type: str):
    """
    Provede ladění (GridSearchCV) pro zadaný model pomocí metriky MCC.

    Parameters
    ----------
    model_type : str
        Typ modelu ("logreg", "rf", "xgb", "svc")

    Returns
    -------
    grid_search : objekt GridSearchCV
        Vytrénovaný GridSearch s nejlepšími parametry (dle MCC).
    """

    # 1. Načtení a předzpracování dat
    df = load_data("diabetes_data.csv")

    # logreg a svc potřebují škálování
    scale_option = model_type if model_type in ["logreg", "svc"] else None
    df = data_preprocessing(df, scale_for_model=scale_option)

    # 2. Rozdělení na X a y
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # 3. Definice modelu a mřížky parametrů
    if model_type == "logreg":
        params = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"]
        }
        model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)

    elif model_type == "rf":
        params = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True, False]
        }
        model = RandomForestClassifier(class_weight="balanced", random_state=42)

    elif model_type == "xgb":
        params = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 6],
            "subsample": [0.6, 0.8, 1],
            "colsample_bytree": [0.6, 0.8, 1]
        }
        model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    elif model_type == "svc":
        params = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["linear", "rbf"]
        }
        model = SVC(probability=True, random_state=42, class_weight="balanced")

    else:
        raise ValueError(" Neplatný model_type.")

    # 4. Definice metriky (MCC)
    mcc_scorer = make_scorer(matthews_corrcoef)

    # 5. Spuštění ladění
    print(f"\n Spouštím ladění hyperparametrů pro model: {model_type.upper()} ...")

    grid_search = GridSearchCV(
        model,
        param_grid=params,
        cv=5,
        scoring=mcc_scorer,  # optimalizace podle MCC
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    # Výpis výsledků
    print("\n Ladění dokončeno!")
    print(f"Nejlepší parametry: {grid_search.best_params_}")
    print(f"Nejlepší MCC skóre: {grid_search.best_score_:.4f}")
    print(f"Nejlepší model:\n{grid_search.best_estimator_}")

    # === 7. Uložení nejlepšího modelu === #
    best_model_filename = f"best_model_{model_type}.pkl"
    joblib.dump(grid_search.best_estimator_, best_model_filename)
    print(f"\nNejlepší model uložen do: {best_model_filename}")

    return grid_search


# === Spouštěcí blok === #
if __name__ == "__main__":
    # Vyber model, který chceš ladit
    model_type = "logreg"
    #model_type = "rf"
    #model_type = "xgb"
    #model_type = "svc"

    hyper_param_test(model_type)
