import io
from typing import Tuple

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import Logit


app = Flask(__name__)

FEATURES = ["height_right", "margin_low", "margin_up", "length"]
TARGET_COLUMN = "is_genuine"
ID_COLUMN = "id"


def run_detection(train_csv_bytes: bytes, test_csv_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    Billet_df = pd.read_csv(io.BytesIO(train_csv_bytes))
    Billet_test_df = pd.read_csv(io.BytesIO(test_csv_bytes))

    # Equivalent à Billet_test_df.info() pour vérifier 'id' présent
    if ID_COLUMN not in Billet_test_df.columns:
        raise ValueError(f"La colonne '{ID_COLUMN}' est absente du testing set.")

    # y (authenticité)
    y_billet = Billet_df.loc[:, Billet_df.columns == TARGET_COLUMN]

    # X (variables explicatives)
    X_billet = Billet_df[FEATURES]
    X_billet = sm.add_constant(X_billet)

    # Régression logistique
    reg_log = Logit(endog=y_billet, exog=X_billet)
    model_reg_log = reg_log.fit(disp=False)

    # Prédiction sur données inconnues
    X_test = Billet_test_df[FEATURES]
    X_test = sm.add_constant(X_test)
    Billet_test_df["proba"] = model_reg_log.predict(X_test)
    Billet_test_df["pred"] = (model_reg_log.predict(X_test) >= 0.5).astype(int)

    # Texte comme le print original
    lines = ["Indetification des billets:"]
    for i, k in zip(Billet_test_df["pred"], Billet_test_df[ID_COLUMN]):
        if i == 1:
            lines.append(f"Le billet {k} est vrai")
        else:
            lines.append(f"Le billet {k} est faux")
    text_result = "\n".join(lines)

    return Billet_test_df[[ID_COLUMN, "proba", "pred"]], text_result


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/detect")
def detect():
    train_file = request.files.get("train_csv")
    test_file = request.files.get("test_csv")
    if not train_file or not test_file:
        return jsonify({"error": "Veuillez fournir les deux fichiers CSV."}), 400
    try:
        out_df, text = run_detection(train_file.read(), test_file.read())
        # Save CSV to memory for download
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        return jsonify(
            {
                "text": text,
                "table": out_df.to_dict(orient="records"),
                "csv": csv_bytes.decode("utf-8"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import os
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)

