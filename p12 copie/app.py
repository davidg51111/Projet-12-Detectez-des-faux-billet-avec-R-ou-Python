import io
from typing import Tuple, Optional

import gradio as gr
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import Logit


FEATURES = ["height_right", "margin_low", "margin_up", "length"]
TARGET_COLUMN = "is_genuine"
ID_COLUMN = "id"


def read_csv_file(file_obj: gr.File) -> pd.DataFrame:
    if file_obj is None:
        raise ValueError("Aucun fichier fourni.")
    # gr.File may provide a path or bytes; handle both
    if hasattr(file_obj, "name") and file_obj.name:
        return pd.read_csv(file_obj.name)
    if hasattr(file_obj, "read"):
        return pd.read_csv(io.BytesIO(file_obj.read()))
    raise ValueError("Format de fichier non pris en charge.")


def validate_columns(df: pd.DataFrame, required_cols: list, context: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {context}: {', '.join(missing)}. "
            f"Colonnes requises: {', '.join(required_cols)}"
        )


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    validate_columns(train_df, FEATURES + [TARGET_COLUMN], "le training set")
    validate_columns(test_df, FEATURES + [ID_COLUMN], "le testing set")

    y_billet = train_df.loc[:, train_df.columns == TARGET_COLUMN]
    X_billet = train_df[FEATURES]
    X_billet = sm.add_constant(X_billet)

    reg_log = Logit(endog=y_billet, exog=X_billet)
    model_reg_log = reg_log.fit(disp=False)

    X_test = test_df[FEATURES]
    X_test = sm.add_constant(X_test)
    proba = model_reg_log.predict(X_test)
    pred = (proba >= 0.5).astype(int)

    out_df = pd.DataFrame(
        {
            ID_COLUMN: test_df[ID_COLUMN].values,
            "proba": proba.values,
            "pred": pred.values,
        }
    )

    # Build textual result similar to the original print loop
    lines = ["Indetification des billets:"]
    for i, k in zip(out_df["pred"], out_df[ID_COLUMN]):
        if i == 1:
            lines.append(f"Le billet {k} est vrai")
        else:
            lines.append(f"Le billet {k} est faux")
    text_result = "\n".join(lines)
    return out_df, text_result


def detect_interface(train_file: gr.File, test_file: gr.File) -> Tuple[pd.DataFrame, str, Optional[str]]:
    try:
        train_df = read_csv_file(train_file)
        test_df = read_csv_file(test_file)
        out_df, text_result = train_and_predict(train_df, test_df)
        # Provide a downloadable CSV
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        csv_io = io.BytesIO(csv_bytes)
        csv_io.name = "billets_identification.csv"
        return out_df, text_result, csv_io
    except Exception as e:
        return pd.DataFrame(), f"Erreur: {e}", None


with gr.Blocks(title="Détecteur de billets") as demo:
    gr.Markdown(
        "## Détecteur de billets\n"
        "Glissez-déposez votre training set (avec la colonne 'is_genuine') et votre testing set (avec la colonne 'id').\n\n"
        f"Colonnes de caractéristiques requises (dans les deux fichiers): {', '.join(FEATURES)}"
    )
    with gr.Row():
        train_input = gr.File(label="Training set (CSV)", file_types=[".csv"])
        test_input = gr.File(label="Testing set (CSV)", file_types=[".csv"])
    run_btn = gr.Button("Lancer la détection")
    with gr.Row():
        table_out = gr.Dataframe(label="Résultats (id, proba, pred)", wrap=True)
        text_out = gr.Textbox(label="Résumé", lines=15)
    download_out = gr.File(label="Télécharger les résultats (CSV)")

    run_btn.click(
        fn=detect_interface,
        inputs=[train_input, test_input],
        outputs=[table_out, text_out, download_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

