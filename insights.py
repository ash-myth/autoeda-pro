import json
import numpy as np
import requests
import os

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

def _build_context(df, eda: dict, ml_result: dict | None) -> str:
    lines = []

    lines.append(f"DATASET: {len(df)} rows × {len(df.columns)} columns")
    lines.append(f"QUALITY SCORE: {eda['quality_score']}/100")

    lines.append("\nCOLUMN STATS:")
    for col, info in eda["column_info"].items():
        if info["type"] == "numeric":
            lines.append(
                f"  {col} [numeric]: mean={info.get('mean')}, std={info.get('std')}, "
                f"skew={info.get('skew')}, outliers={info.get('outliers')}, "
                f"missing={info['missing_pct']}%, unique={info['unique']}"
            )
        else:
            top = list(info.get("top_values", {}).items())[:5]
            lines.append(
                f"  {col} [categorical]: {info['unique']} unique values, "
                f"top={top}, missing={info['missing_pct']}%"
            )

    numeric_cols = [c for c in df.columns if eda["column_info"][c]["type"] == "numeric"]
    if len(numeric_cols) >= 2:
        num_df = df[numeric_cols].select_dtypes(include=np.number)
        corr = num_df.corr()
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], round(corr.iloc[i, j], 3)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        lines.append("\nPAIRWISE CORRELATIONS (top 10 by |r|):")
        for a, b, r in pairs[:10]:
            lines.append(f"  {a} ↔ {b}: r={r}")

    if ml_result:
        lines.append(f"\nAUTOML TASK: {ml_result['task']}")
        lines.append(f"TARGET: {ml_result['target']}")
        lines.append(f"FEATURES USED: {ml_result['features']}")
        lines.append(f"TRAIN/TEST: {ml_result['train_size']}/{ml_result['test_size']}")
        lines.append(f"BEST MODEL: {ml_result['best_model']} (score={ml_result['best_score']})")
        lines.append("ALL MODEL SCORES:")
        for m in ml_result["models"]:
            if ml_result["task"] == "regression":
                lines.append(f"  {m['name']}: R²={m.get('r2')}, RMSE={m.get('rmse')}, MAE={m.get('mae')}")
            else:
                lines.append(f"  {m['name']}: Accuracy={m.get('accuracy')}, F1={m.get('f1')}")
        lines.append("FEATURE IMPORTANCE (ranked):")
        for fi in ml_result["feature_importance"]:
            lines.append(f"  {fi['col']}: {fi['importance']}")
    else:
        lines.append("\nAUTOML: Not run (insufficient numeric columns).")

    return "\n".join(lines)

def _leakage_suspects(df, eda: dict, ml_result: dict | None) -> list:
    if not ml_result:
        return []
    target = ml_result.get("target")
    if not target or target not in df.columns:
        return []

    suspects = []
    y = df[target].dropna()

    for col in ml_result.get("features", []):
        if col not in df.columns:
            continue
        try:
            x = df[col].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 10:
                continue
            r = float(np.corrcoef(x[common].values, y[common].values)[0, 1])
            if abs(r) >= 0.95:
                suspects.append({"col": col, "r": round(r, 3), "level": "critical"})
            elif abs(r) >= 0.85:
                suspects.append({"col": col, "r": round(r, 3), "level": "warning"})
        except Exception:
            continue

    return suspects

def _call_groq(api_key: str, messages: list, max_tokens: int = 2000, response_format: dict | None = None) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _parse_json(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")
    return json.loads(raw[start:end + 1])

def generate_report(api_key: str, df, eda: dict, ml_result: dict | None) -> dict:
    """
    Returns dict with keys:
      domain        — real-world domain inference + what the target means
      cross_feature — cross-feature reasoning, multicollinearity, model spread interpretation
      leakage       — leakage analysis using pre-computed correlation suspects
      action_plan   — ordered, specific, executable steps (named columns + transforms)
      _leakage_suspects — raw list for UI badge rendering
    """
    context = _build_context(df, eda, ml_result)
    suspects = _leakage_suspects(df, eda, ml_result)
    leakage_note = (
        f"PRE-COMPUTED LEAKAGE SUSPECTS: {json.dumps(suspects)}"
        if suspects else
        "PRE-COMPUTED LEAKAGE SUSPECTS: none found (no feature correlated ≥0.85 with target)"
    )

    system = """You are a senior ML engineer doing a pre-modeling review of a dataset.

Your job is NOT to repeat statistics the user can already see in charts.
Your job is to reason across the data and catch things a junior analyst would miss.

Respond ONLY with a valid JSON object with exactly these four keys:

"domain" — 2-3 sentences. Infer what this dataset likely represents in the real world.
  What domain is it from? What does the target column probably mean in practice?
  What kind of decision would this model support?

"cross_feature" — 3-5 sentences. Reason about RELATIONSHIPS between features, not individual columns.
  Which correlated feature pairs might cause multicollinearity?
  If two top-importance features are also strongly correlated with each other, flag that.
  Are there groups of features that seem to measure the same underlying thing?
  What does the spread between the best and worst model scores tell you about the data structure?

"leakage" — 2-4 sentences. Use the PRE-COMPUTED LEAKAGE SUSPECTS provided.
  If suspects exist, explain WHY each one is suspicious and what the consequence would be if left in.
  If none exist, briefly confirm the feature-target correlations look healthy and explain
  what that implies about expected model difficulty.

"action_plan" — A numbered list of 5-7 concrete, ordered steps. Each step must name the exact
  column and transformation. No vague advice.
  BAD: "handle missing values"
  GOOD: "Impute Age with median (not mean — right-skewed at 0.8); then bin into 4 age bands before encoding"
  Order: data cleaning → feature engineering → encoding → modeling choices → validation strategy.

Be specific. Reference actual column names and numbers from the context. No filler."""

    user = (
        f"Dataset analysis:\n\n{context}\n\n"
        f"{leakage_note}\n\n"
        "Write the four-section review as a JSON object."
    )

    raw = _call_groq(api_key, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], response_format={"type": "json_object"})

    result = _parse_json(raw)
    result["_leakage_suspects"] = suspects
    return result

def ask_analyst(
    api_key: str,
    question: str,
    df,
    eda: dict,
    ml_result: dict | None,
    history: list,
) -> str:
    context = _build_context(df, eda, ml_result)

    system = (
        "You are a senior ML engineer. The user is asking follow-up questions about their dataset.\n"
        "Answer concisely and specifically — cite actual column names, correlation values, "
        "model scores, or feature importances from the context.\n"
        "Do NOT restate what the user can already see in charts. "
        "Reason about implications, causes, and what to do next.\n"
        "Keep answers under 120 words.\n\n"
        f"FULL DATASET CONTEXT:\n{context}"
    )

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    return _call_groq(api_key, messages, max_tokens=400)
