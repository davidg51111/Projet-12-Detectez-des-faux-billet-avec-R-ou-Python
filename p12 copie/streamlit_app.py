import io
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import Logit
import streamlit as st


FEATURES: List[str] = ["height_right", "margin_low", "margin_up", "length"]
TARGET_COLUMN = "is_genuine"
ID_COLUMN = "id"


def validate_columns(df: pd.DataFrame, required_cols: list, context: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {context}: {', '.join(missing)}. "
            f"Colonnes requises: {', '.join(required_cols)}"
        )


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
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
    return out_df


st.set_page_config(page_title="Détecteur de billets", layout="wide")
st.title("Détecteur de billets")
st.write(
    "Glissez-déposez votre training set (avec la colonne 'is_genuine') et votre testing set (avec la colonne 'id')."
)
st.write(f"Caractéristiques utilisées: {', '.join(FEATURES)}")

col1, col2 = st.columns(2)
with col1:
    train_file = st.file_uploader("Training set (CSV)", type=["csv"], key="train")
with col2:
    test_file = st.file_uploader("Testing set (CSV)", type=["csv"], key="test")

run = st.button("Lancer la détection")

if run:
    if train_file is None or test_file is None:
        st.error("Veuillez fournir les deux fichiers CSV.")
    else:
        try:
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
            out_df = train_and_predict(train_df, test_df)

            st.subheader("Résultats")
            st.dataframe(out_df, use_container_width=True)

            st.subheader("Résumé")
            lines = ["Indetification des billets:"]
            for i, k in zip(out_df["pred"], out_df[ID_COLUMN]):
                if i == 1:
                    lines.append(f"Le billet {k} est vrai")
                else:
                    lines.append(f"Le billet {k} est faux")
            st.code("\n".join(lines))

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les résultats (CSV)",
                data=csv_bytes,
                file_name="billets_identification.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erreur: {e}")


