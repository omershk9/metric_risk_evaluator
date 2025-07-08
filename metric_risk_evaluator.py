from sqlglot import parse_one
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
xgb_model = xgb.Booster()
xgb_model.load_model("ftc_risk_model.json")

def extract_features(ast):
    group_fields = ast.find_all("Group")
    join_count = len(ast.find_all("Join"))
    sensitive_terms = ["zip", "gender", "dob"]
    sensitive_count = sum(1 for node in ast.find_all("Column") if any(term in node.sql().lower() for term in sensitive_terms))
    return np.array([len(group_fields), join_count, sensitive_count])

def embed_sql(sql_query):
    tokens = tokenizer(sql_query, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].detach().numpy()

def detect_overexposure(sql_query):
    try:
        ast = parse_one(sql_query)
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

    semantic = embed_sql(sql_query)
    syntax = extract_features(ast)
    full_features = np.concatenate([semantic[0], syntax])
    dmatrix = xgb.DMatrix(full_features.reshape(1, -1))
    risk_score = xgb_model.predict(dmatrix)[0]

    if risk_score > 0.85:
        return {
            "status": "BLOCKED",
            "risk_score": float(risk_score),
            "explanation": "Metric may overexpose sensitive groupings like gender or ZIP; revise group-by logic."
        }
    else:
        return {
            "status": "APPROVED",
            "risk_score": float(risk_score)
        }
