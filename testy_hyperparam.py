# === Import knihoven === #
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib

# import vlastních funkcí z main.py
from main import load_data, data_preprocessing

def hyper_param_test(model_type: str):
    """
    Provede ladění hyperparametrů (GridSearchCV) pro zadaný klasifikační model
    s použitím Matthews Correlation Coefficient (MCC) jako optimalizační metriky.

    Funkce provádí následující kroky:
        1. načtení originálního datasetu pomocí `load_data()`,
        2. kompletní předzpracování dat prostřednictvím `data_preprocessing()`,
        3. rozdělení dat na trénovací a validační sadu (85 % / 15 %) se stratifikací,
        4. sestavení pipeline (StandardScaler → model) pro modely vyžadující škálování,
        5. definici gridu hyperparametrů podle vybraného modelu,
        6. provedení GridSearchCV s 5-fold cross-validací,
        7. optimalizaci na základě metriky MCC,
        8. vypsání nejlepších nalezených parametrů a dosaženého MCC výkonu,
        9. uložení nejlepšího nalezeného modelu

    Podporované modely (`model_type`):
        - "logreg" : Logistic Regression
        - "rf"     : Random Forest Classifier
        - "xgb"    : XGBoost Classifier
        - "svc"    : Support Vector Classifier

    Parametry
    ----------
    model_type : str
        Identifikátor modelu, který má být laděn. Musí být jednou z hodnot:
        {"logreg", "rf", "xgb", "svc"}.

    Návratová hodnota
    -----------------
    grid_search : GridSearchCV
        Objekt GridSearchCV obsahující kompletní výsledky ladění, včetně:
            - nejlepších hyperparametrů (`best_params_`),
            - nejlepší dosažené hodnoty MCC (`best_score_`),
            - nejlepší pipeline (`best_estimator_`).

    Poznámky
    --------
    - Funkce používá MCC
    - Ve výsledném .pkl souboru je uložen kompletní pipeline objekt,
      včetně scaleru (je-li použit) a optimálních hyperparametrů.
    - Funkce je určena pro explorativní ladění modelů, nikoli pro finální trénink.
    """

    # Načtení a předzpracování dat
    df = load_data("diabetes_data.csv")

    df = data_preprocessing(df)

    # rozdělení na train a valid
    train_df, valid_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["Outcome"]
    )

    X_train = train_df.drop(columns=["Outcome"])
    y_train = train_df["Outcome"]

    # Definice modelů a param gridů
    if model_type == "logreg":
        params = {
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__penalty": ["l1", "l2"]
        }
        base_model = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=42
        )

    elif model_type == "rf":
        params = {
            "model__n_estimators": [100, 150],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
        base_model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42
        )

    elif model_type == "xgb":
        params = {
            "model__n_estimators": [200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 4],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
        base_model = XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

    elif model_type == "svc":
        params = {
            "model__C": [1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1],
            "model__kernel": ["linear", "rbf"]
        }
        base_model = SVC(
            probability=True,
            class_weight="balanced",
            random_state=42
        )

    else:
        raise ValueError("Neplatný model_type (logreg / rf / xgb / svc).")

    pipe_steps = []

    if model_type in ["logreg", "svc"]:
        pipe_steps.append(("scaler", StandardScaler()))

    pipe_steps.append(("model", base_model))

    model_pipeline = Pipeline(pipe_steps)

    # MCC jako skóre
    mcc_scorer = make_scorer(matthews_corrcoef)

    print(f"\n=== Spouštím GridSearchCV pro {model_type.upper()} ===")

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=params,
        cv=5,
        scoring=mcc_scorer,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\n=== LADĚNÍ DOKONČENO ===")
    print("Nejlepší parametry:", grid_search.best_params_)
    print("Nejlepší MCC:", grid_search.best_score_)

    filename = f"best_model_{model_type}.pkl"
    joblib.dump(grid_search.best_estimator_, filename)

    print(f"\nModel uložen do: {filename}")
    return grid_search


# Spouštěcí blok
if __name__ == "__main__":
    model_type = "rf"  # logreg / rf / xgb / svc
    hyper_param_test(model_type)
