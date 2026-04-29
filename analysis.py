import pandas as pd
import numpy as np
def run_eda(df: pd.DataFrame) -> dict:
    n = len(df)
    column_info = {}
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        missing = series.isna().sum()
        missing_pct = round((missing/n)*100, 1)
        unique_vals = non_null.nunique()
        is_numeric = col in numeric_cols
        is_id = unique_vals==n and not is_numeric
        is_constant = unique_vals<=1
        is_high_card = (not is_numeric) and (unique_vals>n*0.5) and (unique_vals>20)
        info = {
            "type": "numeric" if is_numeric else "categorical",
            "total": n,
            "missing": int(missing),
            "missing_pct": missing_pct,
            "unique": int(unique_vals),
            "is_id": is_id,
            "is_constant": is_constant,
            "is_high_cardinality": is_high_card,
        }
        if is_numeric and len(non_null)>0:
            vals = non_null.astype(float)
            q1 = vals.quantile(0.25)
            q3 = vals.quantile(0.75)
            iqr = q3-q1
            lo, hi = q1-1.5*iqr, q3+1.5*iqr
            outliers = int(((vals<lo)|(vals>hi)).sum())
            skew = round(float(vals.skew()), 3)
            info.update({
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "q1": round(float(q1), 4),
                "median": round(float(vals.median()), 4),
                "q3": round(float(q3), 4),
                "outliers": outliers,
                "skew": skew,
            })
        else:
            vc = non_null.value_counts()
            info["top_values"] = vc.head(10).to_dict()
        column_info[col] = info
        if missing_pct>30:
            issues.append({"sev": "high", "msg": f'"{col}" has {missing_pct}% missing values — impute or drop'})
        elif missing_pct>10:
            issues.append({"sev": "med", "msg": f'"{col}" has {missing_pct}% missing values'})
        if is_id:
            issues.append({"sev": "info", "msg": f'"{col}" appears to be an ID column — drop before modeling'})
        if is_constant:
            issues.append({"sev": "high", "msg": f'"{col}" has zero variance (constant) — drop it'})
        if is_high_card:
            issues.append({"sev": "med", "msg": f'"{col}" has very high cardinality ({unique_vals} unique) — needs encoding strategy'})
        if is_numeric and len(non_null)>0:
            if info.get("outliers", 0)>0:
                issues.append({"sev": "info", "msg": f'"{col}" has {info["outliers"]} outlier(s) detected via IQR method'})
            if abs(info.get("skew", 0))>1:
                direction = "right" if info["skew"]>0 else "left"
                issues.append({"sev": "info", "msg": f'"{col}" is {direction}-skewed (skew={info["skew"]}) — consider log/sqrt transform'})
    missing_pct_series = pd.Series({col: column_info[col]["missing_pct"] for col in df.columns})
    avg_missing = missing_pct_series.mean()
    high_issues = sum(1 for i in issues if i["sev"]=="high")
    med_issues = sum(1 for i in issues if i["sev"]=="med")
    quality_score = max(0, round(100-avg_missing*1.5-high_issues*8-med_issues*3))
    return {
        "column_info": column_info,
        "missing_pct": missing_pct_series,
        "issues": issues,
        "quality_score": quality_score,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
    }
