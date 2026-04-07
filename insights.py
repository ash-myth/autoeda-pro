import time
from google import genai


def generate_ai_insights(
    df,
    eda: dict,
    ml_result: dict | None,
    api_key: str,
    filename: str
) -> str:

    try:
        client = genai.Client(api_key=api_key)

        column_summary = []

        for col, info in list(eda["column_info"].items())[:14]:

            if info["type"] == "numeric":
                summary = (
                    f"{col} (numeric): "
                    f"mean={info.get('mean')}, "
                    f"std={info.get('std')}, "
                    f"skew={info.get('skew')}, "
                    f"outliers={info.get('outliers')}, "
                    f"missing={info.get('missing_pct')}%"
                )
            else:
                summary = (
                    f"{col} (categorical): "
                    f"{info.get('unique')} unique, "
                    f"missing={info.get('missing_pct')}%"
                )

            column_summary.append(summary)

        issues_text = "\n".join(
            f"[{i['sev'].upper()}] {i['msg']}"
            for i in eda["issues"]
        ) or "None"

        ml_text = "AutoML not run"

        if ml_result:
            top_features = ", ".join(
                f["col"]
                for f in ml_result["feature_importance"][:5]
            )

            ml_text = (
                f"Task: {ml_result['task']}\n"
                f"Target: {ml_result['target']}\n"
                f"Best Model: {ml_result['best_model']}\n"
                f"Best Score: {ml_result['best_score']}\n"
                f"Top Features: {top_features}"
            )

        prompt = f"""You are a senior data scientist. Analyze this dataset and provide structured insights.

Dataset: {filename}
Rows: {len(df)}
Columns: {len(df.columns)}
Quality Score: {eda['quality_score']}/100

Column Summary:
{chr(10).join(column_summary)}

Issues Detected:
{issues_text}

ML Results:
{ml_text}

Write a professional analysis with these exact sections using markdown headers:

## 1. Dataset Overview
Brief description of the dataset size, structure, and domain.

## 2. Key Findings
The 3-5 most important insights from the data.

## 3. Data Quality Assessment
Interpretation of the quality score and what the detected issues mean in practice.

## 4. Feature Relationships
Notable correlations or patterns between features.

## 5. Modeling Results
Interpretation of the AutoML results — what the best model and score mean, which features matter most.

## 6. Recommendations
Concrete next steps for data cleaning, feature engineering, and modeling improvements.

Be specific, actionable, and concise. Use bullet points within sections where appropriate."""

        models_to_try = [
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ]

        last_err = ""
        for model_name in models_to_try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    return response.text
                except Exception as e:
                    last_err = str(e)
                    if "429" in last_err or "quota" in last_err.lower() or "rate" in last_err.lower():
                        if attempt == 0:
                            time.sleep(5)  
                            continue
                        else:
                            break  
                    else:
                        raise 

        return "ERROR: Rate limit exceeded on all available models. Please wait a few minutes and try again, or set up billing at aistudio.google.com."

    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "invalid api key" in err.lower() or "400" in err:
            return "ERROR: Invalid API key. Check your key at aistudio.google.com."
        elif "quota" in err.lower() or "429" in err or "rate" in err.lower():
            return "ERROR: Rate limit exceeded on all available models. Please wait a few minutes and try again."
        elif "SAFETY" in err:
            return "ERROR: Response blocked by Gemini safety filters."
        else:
            return f"ERROR: {err}"
