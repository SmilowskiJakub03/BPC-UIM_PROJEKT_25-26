from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



def hyper_param_test(model_type:str,x_train,y_train):

    if model_type == "logreg":

        params = {
                "C":[0.001, 0.01, 0.1, 1, 10, 100],
                "penalty":["l1", "l2"]
        }

        model = LogisticRegression(solver="liblinear", max_iter= 1000)

    elif model_type == "xgb":

        params = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 6],
            "subsample": [0.6, 0.8, 1],
            "colsample_bytree": [0.6, 0.8, 1]
        }

        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")


    elif model_type == "rf":

        params = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True, False]
        }

        model = RandomForestClassifier(class_weight="balanced", random_state=42)

    elif model_type == "svc":

        params = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1 ,1],
            "kernel": ["linear", "rbf", "poly", "sigmoid"]
        }

        model = SVC(probability=True, random_state=42)

    else:
        raise ValueError("Invalid model type")


    grid_search = GridSearchCV(model, params, cv=5)
    grid_search.fit(x_train,y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best estimators:", grid_search.best_estimator_)



# test zpráva