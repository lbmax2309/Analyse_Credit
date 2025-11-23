import streamlit as st
import script_rating as sr 

page = st.sidebar.radio(
    "Navigation",
    ["Agences", "Radar", "Données", "Indicateurs dans le temps", "Tous les pays"]
)

# Tout ce qui charge / calcule est dans le spinner
with st.spinner("Chargement…"):

    df = sr.countries10_Zscore()
    latest = df[df["Annee"] == df["Annee"].max()]

    if page == "Agences":
        st.header("Comparaison agences")
        st.pyplot(sr.compare_agencies_ratings())

    elif page == "Radar":
        st.header("Radar par pays")
        pays = st.selectbox("Choisir un pays", latest["Pays"].unique())
        
        # Récupérer la ligne du pays
        df_country = latest[latest["Pays"] == pays].iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Notation modèle", df_country["Rating_modele"])
        with col2:
            st.metric("Score de solvabilité", round(df_country["Score_solvabilite"], 2))

        # Slopes + outlook + commentaire
        slopes = sr.compute_slopes()
        country_slopes = slopes[slopes["Pays"] == pays].iloc[0]

        outlook = sr.compute_outlook({**df_country, **country_slopes})
        st.write("Outlook :", outlook)

        comment = sr.make_comment({**df_country, **country_slopes})
        st.write(comment)

        st.pyplot(sr.radar_country(pays))

    elif page == "Indicateurs dans le temps":
        st.header("Évolution d’un indicateur dans le temps")

        ind = st.selectbox(
            "Choisir un indicateur",
            sr.valid_indicators,
            key="selectbox_time_series"
        )

        st.pyplot(sr.time_series(ind))

    elif page == "Données":
        st.header("Données 2024")
        st.dataframe(latest)

        # Télécharger juste l'année la plus récente
        csv = latest.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les données 2024 (CSV)",
            csv,
            "donnees_2024.csv",
            "text/csv"
        )

        st.divider()
        st.header("Données complètes (1984–2024)")

        df_all = sr.df_10countries()
        csv_all = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger toutes les données (1984–2024)",
            csv_all,
            "donnees_1984_2024.csv",
            "text/csv"
        )

        st.divider()
        st.header("Données enrichies avec notation")

        df_model = sr.compute_Zscore()
        csv_model = df_model.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger dataset notation (CSV)",
            csv_model,
            "dataset_notation.csv",
            "text/csv"
        )

    elif page == "Tous les pays":
        st.header("Tous les pays notés par le modèle")

        # DataFrame complet (tous les pays, année end_year dans compute_Zscore)
        df_all_model = sr.compute_Zscore()

        # Tri par score de solvabilité décroissant
        df_all_model_sorted = df_all_model.sort_values(
            "Score_solvabilite",
            ascending=False
        )

        st.dataframe(df_all_model_sorted)

        # Bouton de téléchargement CSV
        csv_all_model = df_all_model_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger toutes les notations (CSV)",
            csv_all_model,
            "notations_tous_pays.csv",
            "text/csv"
        )
