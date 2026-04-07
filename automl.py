import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, f1_score, classification_report,
                              confusion_matrix)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")


def run_automl(df: pd.DataFrame) -> dict | None:
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None

    possible_targets = ["target","label","class","outcome","y","survived","price"]
    target_col = None

    for col in df.columns:
        if col.lower() in possible_targets:
            target_col = col
            break

    if target_col is None:
        target_col = numeric_cols[-1]
    feature_cols = [c for c in numeric_cols if c != target_col]
    feature_cols = [c for c in feature_cols if "id" not in c.lower()]
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
    feature_cols = feature_cols[:8]

    if len(feature_cols) == 0:
        return None

    target_series = df[target_col].dropna()
    n_unique = target_series.nunique()
    is_classification = (n_unique <= 10 and n_unique >= 2 and
                         set(target_series.unique()).issubset(set(range(-1000, 1001))) and
                         all(float(v).is_integer() for v in target_series.unique()))
    task = "classification" if is_classification else "regression"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        return None
    valid_idx = y.notna()
    X, y = X[valid_idx], y[valid_idx]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    if is_classification:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(int))
    else:
        y_encoded = y.values

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42
    )

    if len(X_train) < 10:
        return None
    model_results = []
    best_score = -np.inf
    best_model_name = ""
    best_predictions = None
    best_actuals = None
    best_cm = None

    if task == "regression":
        models = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge(alpha=1.0)),
            ("Decision Tree", DecisionTreeRegressor(max_depth=5, random_state=42)),
            ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
        ]
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = round(r2_score(y_test, preds), 4)
                rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
                mae = round(mean_absolute_error(y_test, preds), 4)
                model_results.append({"name": name, "r2": r2, "rmse": rmse, "mae": mae})
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    best_predictions = preds.tolist()
                    best_actuals = y_test.tolist()
            except Exception:
                continue

    else:  
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ]
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = round(accuracy_score(y_test, preds), 4)
                f1 = round(f1_score(y_test, preds, average="weighted", zero_division=0), 4)
                model_results.append({"name": name, "accuracy": acc, "f1": f1})
                if acc > best_score:
                    best_score = acc
                    best_model_name = name
                    best_predictions = preds.tolist()
                    best_actuals = y_test.tolist()
                    best_cm = confusion_matrix(y_test, preds).tolist()
            except Exception:
                continue

    if not model_results:
        return None

    feature_importance = []
    for col in feature_cols:
        try:
            col_vals = df[col].fillna(df[col].median())
            valid = y.notna() & col_vals.notna()
            corr = np.corrcoef(col_vals[valid].values, y[valid].values)[0, 1]
            feature_importance.append({"col": col, "importance": round(abs(corr), 4)})
        except Exception:
            feature_importance.append({"col": col, "importance": 0.0})

    try:
        rf_model = [m for _, m in models if "Random Forest" in _][0] if task == "regression" else None
        if rf_model and hasattr(rf_model, "feature_importances_"):
            rf_imp = rf_model.feature_importances_
            feature_importance = [{"col": feature_cols[i], "importance": round(float(rf_imp[i]), 4)}
                                   for i in range(len(feature_cols))]
    except Exception:
        pass

    feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    result = {
        "task": task,
        "target": target_col,
        "features": feature_cols,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "models": model_results,
        "best_model": best_model_name,
        "best_score": best_score,
        "feature_importance": feature_importance,
    }

    if task == "regression":
        result["predictions"] = best_predictions
        result["actuals"] = best_actuals
        result["residuals"] = True  
    else:
        if best_cm:
            result["confusion_matrix"] = best_cm

    return result
