import io
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.api import Logit
import streamlit as st


def fit_logit(train_df: pd.DataFrame) -> Logit:
    required_cols = ["is_genuine", "height_right", "margin_low", "margin_up", "length"]
    missing = [c for c in required_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le training set: {missing}")

    y_billet = train_df.loc[:, train_df.columns == "is_genuine"]
    X_billet = train_df[["height_right", "margin_low", "margin_up", "length"]]
    X_billet = sm.add_constant(X_billet)
    reg_log = Logit(endog=y_billet, exog=X_billet)
    model = reg_log.fit(disp=False)
    return model


def predict(model: Logit, test_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    required_cols = ["id", "height_right", "margin_low", "margin_up", "length"]
    missing = [c for c in required_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le testing set: {missing}")

    X_test = test_df[["height_right", "margin_low", "margin_up", "length"]]
    X_test = sm.add_constant(X_test)
    proba = model.predict(X_test)
    result = test_df.copy()
    result["proba"] = proba
    result["pred"] = (proba >= threshold).astype(int)
    return result


def render_text_results(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    buffer.write("\nIndetification des billets:\n\n")
    for i, k in zip(df["pred"], df["id"]):
        if i == 1:
            buffer.write(f"Le billet {k} est vrai\n")
        else:
            buffer.write(f"Le billet {k} est faux\n")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Détecteur de billets", page_icon="💶", layout="centered")
    st.title("💶 Détecteur de billets")
    st.write("Glissez-déposez vos deux fichiers CSV: training (avec `is_genuine`) et testing (avec `id`).")

    with st.sidebar:
        st.header("Paramètres")
        threshold = st.slider("Seuil de classification (≥ probabilité)", 0.0, 1.0, 0.5, 0.01)

    col1, col2 = st.columns(2)
    with col1:
        train_file = st.file_uploader(
            "Training set (contient `is_genuine`)", type=["csv"], key="train_uploader"
        )
    with col2:
        test_file = st.file_uploader(
            "Testing set (contient `id`)", type=["csv"], key="test_uploader"
        )

    run_btn = st.button("Lancer la détection", type="primary", use_container_width=True)

    if run_btn:
        if not train_file or not test_file:
            st.error("Veuillez fournir les deux fichiers CSV (training et testing).")
            st.stop()
        try:
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
        except Exception as e:
            st.exception(e)
            st.stop()

        try:
            model = fit_logit(train_df)
        except Exception as e:
            st.error("Erreur pendant l'entraînement du modèle.")
            st.exception(e)
            st.stop()

        try:
            results_df = predict(model, test_df, threshold=threshold)
        except Exception as e:
            st.error("Erreur pendant la prédiction sur le testing set.")
            st.exception(e)
            st.stop()

        st.success("Détection terminée.")

        st.subheader("Résultats (table)")
        st.dataframe(
            results_df[["id", "proba", "pred"]].sort_values("id").reset_index(drop=True),
            use_container_width=True,
        )

        st.subheader("Résultats (texte)")
        st.code(render_text_results(results_df))

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger les résultats (CSV)",
            data=csv_bytes,
            file_name="resultats_detection.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()


