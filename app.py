import streamlit as st
import script_rating as sr 
from PIL import Image

# ========== CONFIG GLOBALE ==========
st.set_page_config(
    page_title="Modèle de notation souveraine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# style pour la page
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #0f172a; /* bleu nuit */
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .card {
        padding: 1.2rem 1.4rem;
        border-radius: 0.7rem;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== SIDEBAR (sans logo) ==========
st.sidebar.title("🏦 Modèle de notation souveraine")

page = st.sidebar.radio(
    "📌 Navigation",
    ["Accueil","Agences", "Analyse par pays", "Données", "Indicateurs dans le temps", "Tous les pays"]
)

# ========== CONTENU ==========
with st.spinner("Chargement des données…"):

    df = sr.countries10_Zscore()
    latest = df[df["Annee"] == df["Annee"].max()]

    # ========== PAGE ACCUEIL ==========
    if page == "Accueil":
        st.markdown('<div class="main-title">Tableau de bord souverain</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="main-subtitle">'
            "Outil interne pour explorer les notations souveraines, "
            "les scores de solvabilité et les indicateurs macroéconomiques."
            "</div>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.markdown(
                """
                ### Que permet ce modèle ?

                - 📊 Comparer la notation du modèle avec les agences
                - 📍 Analyser un pays en détail (radar des facteurs, Outlook)
                - ⏱ Suivre l’évolution des indicateurs dans le temps
                - 📂 Télécharger les jeux de données pour analyse externe
                """
            )

        with col2:
            st.subheader("🔎 Accès rapide")
            st.markdown("• **Agences** : comparer modèle vs agences")
            st.markdown("• **Analyse par pays** : vue détaillée par pays")
            st.markdown("• **Données** : export CSV")
            st.markdown('</div>', unsafe_allow_html=True)

    # ========== PAGE AGENCES ==========
    elif page == "Agences":
        st.header("📊 Comparaison avec les agences de notation")
        st.caption("Écart entre la notation du modèle et celles des principales agences.")
        st.pyplot(sr.compare_agencies_ratings(), use_container_width=True)
        st.caption("*Echelle de notation transposée allant de 1(meilleur) à 22(moins bon)"
                   "  \n Correspond à la note de notre modèle moins la moyenne des de notes de S&P, Moody's et Fitch")

    # ========== PAGE Pays ==========
    elif page == "Analyse par pays":
        st.header("📍 Analyse par pays")

        col_select, col_info = st.columns([1.2, 2])
        with col_select:
            pays = st.selectbox("Choisir un pays :", latest["Pays"].unique())

        df_country = latest[latest["Pays"] == pays].iloc[0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Notation modèle", df_country["Rating_modele"])
        with col2:
            st.metric("Score de solvabilité", round(df_country["Score_solvabilite"], 2))
        with col3:
            st.metric("Année", int(df_country["Annee"]))

        # Slopes + outlook + commentaire
        slopes = sr.compute_slopes()
        country_slopes = slopes[slopes["Pays"] == pays].iloc[0]

        inputs = {**df_country.to_dict(), **country_slopes.to_dict()}
        outlook = sr.compute_outlook(inputs)
        comment = sr.make_comment(inputs)

        st.subheader("🧭 Outlook du modèle")
        st.write("**Outlook :**", outlook)
        st.write(comment)
        st.markdown('</div>', unsafe_allow_html=True)

        radar_col, imf_col = st.columns([1.3, 1.7])

        with radar_col:
            st.subheader("Radar des facteurs")
            st.pyplot(sr.radar_country(pays), use_container_width=True)

        with imf_col:
            st.subheader("📈 Outlook IMF — séries historiques")
            try:
                fig_dette, fig_epargne, fig_autres, score_imf, class_imf = sr.outlook_imf(pays)

                st.info(f"**Score Outlook IMF :** {score_imf:.3f} ({class_imf})")

                if fig_dette is not None:
                    st.pyplot(fig_dette, use_container_width=True)
                if fig_epargne is not None:
                    st.pyplot(fig_epargne, use_container_width=True)
                if fig_autres is not None:
                    st.pyplot(fig_autres, use_container_width=True)

            except FileNotFoundError:
                st.info("Fichier Outlook IMF introuvable (vérifie le chemin dans outlook_imf).")
            except ValueError as e:
                st.info(str(e))

    # ========== PAGE INDICATEURS DANS LE TEMPS ==========
    elif page == "Indicateurs dans le temps":
        st.header("⏱ Évolution d’un indicateur dans le temps")
        ind = st.selectbox(
            "Choisir un indicateur",
            sr.valid_indicators,
            key="selectbox_time_series",
        )
        st.caption("Série historique pour l’ensemble des pays (ou selon le paramétrage de la fonction).")
        st.pyplot(sr.time_series(ind), use_container_width=True)

    # ========== PAGE DONNÉES ==========
    elif page == "Données":
        st.header("📂 Données")

        #page avec 3 onglets
        tab1, tab2, tab3 = st.tabs(["Données 2024", "1984–2024", "Dataset notation"])

        with tab1:
            st.subheader("Données les plus récentes")
            st.dataframe(latest, use_container_width=True)
            csv = latest.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger les données 2024 (CSV)",
                csv,
                "donnees_2024.csv",
                "text/csv",
            )

        with tab2:
            st.subheader("Historique complet 1984–2024")
            df_all = sr.df_10countries()
            st.dataframe(df_all, use_container_width=True, height=400)
            csv_all = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger toutes les données (1984–2024)",
                csv_all,
                "donnees_1984_2024.csv",
                "text/csv",
            )

        with tab3:
            st.subheader("Données enrichies avec notation du modèle")
            df_model = sr.compute_Zscore()
            st.dataframe(df_model, use_container_width=True, height=400)
            csv_model = df_model.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Télécharger dataset notation (CSV)",
                csv_model,
                "dataset_notation.csv",
                "text/csv",
            )

    # ========== PAGE TOUS LES PAYS ==========
    elif page == "Tous les pays":
        st.header("🌍 Tous les pays notés par le modèle")

        df_all_model = sr.compute_Zscore()
        df_all_model_sorted = df_all_model.sort_values(
            "Score_solvabilite",
            ascending=False,
        )

        #bouton pour télécharger le dataframe avec tous les pays
        csv_all_model = df_all_model_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger toutes les notations (CSV)",
            csv_all_model,
            "notations_tous_pays.csv",
            "text/csv",
        )

        #plot de la distribution des Z score
        st.subheader("📈 Distribution des scores de solvabilité")
        st.pyplot(sr.plot_score_distribution(), use_container_width=True)

        st.caption("Triés par score de solvabilité décroissant.")
        st.dataframe(df_all_model_sorted, use_container_width=True, height=500)

# ========== PETIT FOOTER ==========
st.markdown("---")
st.caption("📌 Tout investissement présente un risque de perte partielle ou totale en capital. Sauf le monéro, le monéro c'est génial.")