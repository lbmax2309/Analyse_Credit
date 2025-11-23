import pandas as pd
import requests
import unicodedata
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from functools import reduce
import streamlit as st

# ===================== PARAMÈTRES =====================

data_path = r"./data.csv"
start_year = 2019
end_year   = 2024
years = [str(y) for y in range(start_year, end_year+1)]

# ------------------ Mapping IMF → ISO3 ------------------

mapping_imf_to_iso = {
    "united states": "USA","canada": "CAN","mexico": "MEX","guatemala": "GTM",
    "honduras": "HND","costa rica": "CRI","panama": "PAN","brazil": "BRA",
    "argentina": "ARG","chile": "CHL","colombia": "COL","peru": "PER",
    "venezuela": "VEN","ecuador": "ECU","bolivia": "BOL","uruguay": "URY",
    "paraguay": "PRY","germany": "DEU","france": "FRA","italy": "ITA",
    "spain": "ESP","netherlands": "NLD","belgium": "BEL","switzerland": "CHE",
    "austria": "AUT","sweden": "SWE","norway": "NOR","denmark": "DNK",
    "finland": "FIN","ireland": "IRL","united kingdom": "GBR","luxembourg": "LUX",
    "iceland": "ISL","portugal": "PRT","poland": "POL","czech republic": "CZE",
    "hungary": "HUN","romania": "ROU","bulgaria": "BGR","slovak republic": "SVK",
    "estonia": "EST","latvia": "LVA","lithuania": "LTU","serbia": "SRB",
    "south africa": "ZAF","egypt": "EGY","nigeria": "NGA","kenya": "KEN",
    "morocco": "MAR","tunisia": "TUN","ghana": "GHA","cameroon": "CMR",
    "ethiopia": "ETH","uganda": "UGA","côte d'ivoire": "CIV","senegal": "SEN",
    "togo": "TGO","burkina faso": "BFA","mali": "MLI","tanzania": "TZA",
    "mozambique": "MOZ","zambia": "ZMB","sudan": "SDN","namibia": "NAM",
    "zimbabwe": "ZWE","turkey": "TUR","saudi arabia": "SAU","israel": "ISR",
    "jordan": "JOR","lebanon": "LBN","qatar": "QAT","united arab emirates": "ARE",
    "kuwait": "KWT","japan": "JPN","korea": "KOR","china": "CHN","singapore": "SGP",
    "indonesia": "IDN","thailand": "THA","philippines": "PHL","malaysia": "MYS",
    "vietnam": "VNM","australia": "AUS","india": "IND","pakistan": "PAK",
    "bangladesh": "BGD","sri lanka": "LKA","nepal": "NPL","maldives": "MDV",
    "cuba": "CUB","haiti": "HTI","jamaica": "JAM","papua new guinea": "PNG"
}

# Fonction nettoyage noms IMF
def clean_imf_country(x):
    if pd.isna(x):
        return None
    x = x.lower().strip()
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii")
    return mapping_imf_to_iso.get(x)

# ===================== 1) EXTRACTION IMF =====================
@st.cache_data(show_spinner=True)
def process_dataframe ():
    df_imf = pd.read_csv(
        data_path,
        low_memory=False
    )

    codes_imf = {
        "Solde_budgetaire_PIB": "GGXCNL_NGDP",
        "Dette_publique_PIB":   "GGXWDG_NGDP",
        "Recettes_publiques":   "GGR_NGDP",
        "Depenses_publiques":   "GGXONLB_NGDP",
        "Balance_courante_PIB": "BCA_NGDPD",
        "Taux_change":          "PPPEX",
        "Reserves_change":      "TMG_RPCH",
        "Balance_commerciale":  "BCA"
    }

    rows_imf = []

    for name, code in codes_imf.items():
        subset = df_imf[df_imf["SERIES_CODE"].str.contains(code, case=False, na=False)]
        subset = subset[["COUNTRY", "SERIES_CODE"] + years].copy()

        # nettoyage + conversion ISO3
        subset["Pays"] = subset["COUNTRY"].apply(clean_imf_country)
        subset = subset.dropna(subset=["Pays"])

        long_df = subset.melt(
            id_vars=["Pays", "SERIES_CODE"],
            value_vars=years,
            var_name="Annee",
            value_name=name
        )

        long_df = long_df[["Pays", "Annee", name]]
        rows_imf.append(long_df)

    df_imf_final = reduce(
        lambda left, right: pd.merge(left, right, on=["Pays", "Annee"], how="outer"),
        rows_imf
    )
    # ===================== SUPPRESSION DES DOUBLONS IMF =====================

    # Il peut y avoir plusieurs lignes pour un même pays/année à cause de SERIES_CODE multiples.
    # On garde la première valeur disponible pour chaque indicateur.

    df_imf_final = (
        df_imf_final
        .groupby(["Pays", "Annee"], as_index=False)
        .first()
    )

    df_imf_final["Annee"] = df_imf_final["Annee"].astype(int)


    # ===================== 2) EXTRACTION WDI + WGI =====================

    session = requests.Session()
    rows_wdi = []

    # Codes WDI/WGI (comme avant)
    wdi_indicators = {
        "NY.GDP.MKTP.CD":      "PIB_total_$",
        "NY.GDP.MKTP.KD.ZG":   "Croissance_PIB",
        "NY.GDP.PCAP.CD":      "PIB_par_habitant",
        "FP.CPI.TOTL.ZG":      "Inflation",
        "GC.BAL.CASH.GD.ZS":   "Deficit_budgetaire_PIB",
        "GC.REV.XGRT.GD.ZS":   "Recettes_publiques_PIB",
        "GC.XPN.TOTL.GD.ZS":   "Depenses_publiques_PIB",
        "BN.CAB.XOKA.GD.ZS":   "BalanceCourante_PIB",
        "FI.RES.TOTL.CD":      "Reserves_change_$",
        "NE.IMP.GNFS.CD":      "Importations_$",
        "GC.DOD.TOTL.GD.ZS":   "Dette_publique_PIB",
    }

    wgi_indicators = {
        "PV.EST": "Stabilite_Politique",
        "GE.EST": "Efficacite_Gouvernement",
        "CC.EST": "Corruption",
        "RL.EST": "Etat_de_droit",
        "VA.EST": "Voix_responsabilisation"
    }

    countries_iso = list(mapping_imf_to_iso.values())
    countries_str = ";".join(countries_iso)

    def fetch_indicator(indicator, name):
        url = (
            f"https://api.worldbank.org/v2/country/{countries_str}/indicator/{indicator}"
            f"?format=json&per_page=20000&date={start_year}:{end_year}"
        )

        r = session.get(url)
        try:
            data = r.json()
        except:
            return

        if not data or len(data) < 2 or not isinstance(data[1], list):
            return

        for e in data[1]:
            country = e.get("countryiso3code")
            year    = e.get("date")
            value   = e.get("value")

            if country in countries_iso and value is not None:
                rows_wdi.append([country, int(year), name, value])


    for ind, name in tqdm(wdi_indicators.items(), desc="WDI"):
        fetch_indicator(ind, name)

    for ind, name in tqdm(wgi_indicators.items(), desc="WGI"):
        fetch_indicator(ind, name)

    df_wdi = pd.DataFrame(rows_wdi, columns=["Pays","Annee","Indicateur","Valeur"])
    df_wdi_pivot = df_wdi.pivot_table(
        index=["Pays","Annee"],
        columns="Indicateur",
        values="Valeur"
    ).reset_index()

    # ===================== 3) FUSION IMF + WDI/WGI =====================

    df_final = pd.merge(
        df_imf_final, df_wdi_pivot,
        on=["Pays", "Annee"], how="outer"
    )

    df_final.sort_values(["Pays","Annee"], inplace=True)
    # ===================== Nettoyage des doublons : priorité IMF =====================

    cols_to_drop = [
        "Dette_publique_PIB_y",      # doublon WDI
        "Depenses_publiques_PIB",    # doublon WDI
        "Recettes_publiques_PIB",    # doublon WDI
        "BalanceCourante_PIB"        # doublon WDI
    ]

    # On supprime seulement si la colonne existe
    cols_to_drop = [c for c in cols_to_drop if c in df_final.columns]

    df_final = df_final.drop(columns=cols_to_drop)

    # On renomme proprement les variables restantes si nécessaire
    df_final = df_final.rename(columns={
        "Dette_publique_PIB_x": "Dette_publique_PIB"  # IMF garde le nom canonique
    })

    return df_final

@st.cache_data(show_spinner=True)
def compute_Zscore():
    df_clean = process_dataframe()

    df_clean = df_clean.sort_values(["Pays", "Annee"]).reset_index(drop=True)

    # ===================== 2) Interpolation =====================

    df_clean = df_clean.groupby("Pays", group_keys=False).apply(lambda x: x.interpolate()).reset_index()

    # ===================== 3) Ratios & volatilités =====================

    # Ratio réserves / importations
    if "Reserves_change_$" in df_clean.columns and "Importations_$" in df_clean.columns:
        df_clean["Reserves_sur_Importations"] = df_clean["Reserves_change_$"] / df_clean["Importations_$"]
    else:
        df_clean["Reserves_sur_Importations"] = np.nan

    # Volatilité croissance 5 ans
    df_clean["Volatilite_Croissance"] = df_clean.groupby("Pays")["Croissance_PIB"].transform(
        lambda x: x.rolling(5, min_periods=2).std()
    )

    # Volatilité inflation 5 ans
    df_clean["Volatilite_Inflation"] = df_clean.groupby("Pays")["Inflation"].transform(
        lambda x: x.rolling(5, min_periods=2).std()
    )

    # ===================== 4) Dataset final pour l’année la plus récente =====================

    df_last = df_clean[df_clean["Annee"] == end_year].copy()

    # ===================== 5) Normalisation des variables =====================

    all_features = [
        "PIB_par_habitant","Croissance_PIB","Inflation","Deficit_budgetaire_PIB",
        "Recettes_publiques_PIB","Depenses_publiques_PIB","BalanceCourante_PIB",
        "Reserves_sur_Importations","Stabilite_Politique","Efficacite_Gouvernement",
        "Corruption","Etat_de_droit","Voix_responsabilisation","Volatilite_Croissance",
        "Volatilite_Inflation","Dette_publique_PIB", "Solde_budgetaire_PIB", "Balance_commerciale", "PIB_total_$"
    ]

    # Ajouter colonnes manquantes
    for f in all_features:
        if f not in df_last.columns:
            df_last[f] = np.nan

    # Colonnes exploitables
    features_effective = [f for f in all_features if not df_last[f].isna().all()]

    # Normalisation
    df_feat = df_last[features_effective].copy()
    df_feat = df_feat.fillna(df_feat.mean())
    Z = StandardScaler().fit_transform(df_feat)

    # Ajouter les Z-scores au dataset
    df_model = df_last.copy()
    for i, f in enumerate(features_effective):
        df_model[f + "_z"] = Z[:, i]
    for f in all_features:
        if f + "_z" not in df_model.columns:
            df_model[f + "_z"] = 0

    # ===================== 6) Variables structurelles =====================

    df_model["Monnaie_reserve"] = (df_model["Pays"] == "USA").astype(int)
    df_model["Safe_haven"] = df_model["Pays"].isin(["CHE","NOR","DNK","SGP","DEU"]).astype(int)
    df_model["Euro_core"] = df_model["Pays"].isin(["DEU","FRA","NLD","FIN","IRL"]).astype(int)
    df_model["Developpe"] = df_model["Pays"].isin([
        "USA","DEU","FRA","JPN","CAN","GBR","ITA","ESP","NLD","AUS","CHE",
        "SWE","NOR","DNK","FIN","IRL","KOR","SGP","CZE","PRT","ISR"
    ]).astype(int)

    # ===================== 7) Score de solvabilité =====================

    df_model["Score_solvabilite"] = (
        + 0.50 * df_model["PIB_par_habitant_z"]
        + 0.30 * df_model["Croissance_PIB_z"]
        - 0.20 * df_model["Volatilite_Croissance_z"]
        - 0.20 * df_model["Inflation_z"]
        - 0.25 * df_model["Volatilite_Inflation_z"]
        - 0.15 * df_model["Deficit_budgetaire_PIB_z"]
        + 0.25 * df_model["Recettes_publiques_PIB_z"]
        - 0.35 * df_model["Dette_publique_PIB_z"]
        + 0.25 * df_model["BalanceCourante_PIB_z"]
        + 0.60 * df_model["Reserves_sur_Importations_z"]
        + 1.2 * df_model["Stabilite_Politique_z"]
        + 1.0 * df_model["Efficacite_Gouvernement_z"]
        + 1.1 * df_model["Etat_de_droit_z"]
        + 0.8 * df_model["Voix_responsabilisation_z"]
        - 0.6 * df_model["Corruption_z"]
        + 0.4 * df_model["Developpe"]
        + 0.3 * df_model["PIB_total_$_z"]
        + 0.3 * df_model["Balance_commerciale_z"]
    )

    df_model["Score_solvabilite"] += (
        1.5 * df_model["Monnaie_reserve"]
        + 0.2 * df_model["Safe_haven"]
        + 0.3 * df_model["Euro_core"]
    )

    # ===================== 8) Notation =====================

    def map_rating(score):
        if score > 7.4: return "AAA"
        if score > 5.6: return "AA+"
        if score > 4.7: return "AA"
        if score > 4.0: return "AA-"
        if score > 3.5: return "A+"
        if score > 2.7: return "A"
        if score > 2.0: return "A-"
        if score > 1.3: return "BBB+"
        if score > 0.7: return "BBB"
        if score > 0.0: return "BBB-"
        if score > -0.3: return "BB+"
        if score > -0.8: return "BB"
        if score > -1.5: return "BB-"
        if score > -2.5: return "B+"
        if score > -3.5: return "B"
        if score > -4.5: return "B-"
        if score > -6.0: return "CCC+"
        if score > -7.5: return "CCC"
        if score > -9.0: return "CCC-"
        if score > -12.0: return "CC"
        if score > -15.0: return "C"
        return "D"

    df_model["Rating_modele"] = df_model["Score_solvabilite"].apply(map_rating)

    # ===================== 9) Export =====================

    # df_clean.to_excel("data_complete_2019_2024.xlsx", index=False)
    # df_model.sort_values("Score_solvabilite", ascending=False).to_excel(
    #     "ratings_2019_2024.xlsx", index=False
    # )

    return df_model

@st.cache_data(show_spinner=True)
def df_10countries():

    countries = ["USA", "DEU", "FRA", "JPN", "CAN", "IND", "BRA", "ZAF", "IDN", "MAR"]

    countries_str = ";".join(countries)

    # WDI : fondamentaux macro + budget + externe + dette
    wdi_indicators = {
        "NY.GDP.MKTP.CD":      "PIB_total_$",
        "NY.GDP.MKTP.KD.ZG":   "Croissance_PIB",
        "NY.GDP.PCAP.CD":      "PIB_par_habitant",
        "FP.CPI.TOTL.ZG":      "Inflation",
        "GC.BAL.CASH.GD.ZS":   "Deficit_budgetaire_PIB",
        "GC.REV.XGRT.GD.ZS":   "Recettes_publiques_PIB",
        "GC.XPN.TOTL.GD.ZS":   "Depenses_publiques_PIB",
        "BN.CAB.XOKA.GD.ZS":   "BalanceCourante_PIB",
        "FI.RES.TOTL.CD":      "Reserves_change_$",
        "NE.IMP.GNFS.CD":      "Importations_$",
        # variables de dette :
        "GC.DOD.TOTL.GD.ZS":   "Dette_publique_PIB",       # dette publique (% PIB)
    }

    # WGI : gouvernance
    wgi_indicators = {
        "PV.EST": "Stabilite_Politique",
        "GE.EST": "Efficacite_Gouvernement",
        "CC.EST": "Corruption",
        "RL.EST": "Etat_de_droit",
        "VA.EST": "Voix_responsabilisation"
    }

    start_year = 1984
    end_year   = 2024

    session = requests.Session()

    # ===================== Téléchargement groupé WDI + WGI =====================

    rows = []

    def fetch_indicator(indicator, name):
        url = (
            f"https://api.worldbank.org/v2/country/{countries_str}"
            f"/indicator/{indicator}?format=json&per_page=20000&date={start_year}:{end_year}"
        )
        r = session.get(url)
        try:
            data = r.json()
        except Exception:
            return

        if not data or len(data) < 2 or not isinstance(data[1], list):
            return

        for entry in data[1]:
            country = entry.get("countryiso3code")
            year    = entry.get("date")
            value   = entry.get("value")
            if country in countries and value is not None:
                try:
                    year = int(year)
                except Exception:
                    continue
                rows.append([country, year, name, value])

    # WDI
    for ind, name in tqdm(wdi_indicators.items(), desc="WDI groupés"):
        fetch_indicator(ind, name)

    # WGI
    for ind, name in tqdm(wgi_indicators.items(), desc="WGI groupés"):
        fetch_indicator(ind, name)

    df = pd.DataFrame(rows, columns=["Pays","Annee","Indicateur","Valeur"])

    df_pivot = df.pivot_table(
        index=["Pays","Annee"],
        columns="Indicateur",
        values="Valeur"
    ).reset_index()

    df_clean = df_pivot.sort_values(["Pays","Annee"]).copy()

    # Interpolation des séries par pays (on force les colonnes en numériques pour éviter le warning)
    def interpolate_group(x):
        x = x.infer_objects(copy=False)
        return x.interpolate()

    df_clean = df_clean.groupby("Pays", group_keys=False).apply(interpolate_group)
    for col in ["Reserves_change_$", "Importations_$", "Croissance_PIB", "Inflation"]:
        if col not in df_clean.columns:
            df_clean[col] = np.nan

    df_clean["Reserves_sur_Importations"] = df_clean["Reserves_change_$"] / df_clean["Importations_$"]

    df_clean["Volatilite_Croissance"] = (
        df_clean.groupby("Pays")["Croissance_PIB"]
        .transform(lambda x: x.rolling(5, min_periods=2).std())
    )

    df_clean["Volatilite_Inflation"] = (
        df_clean.groupby("Pays")["Inflation"]
        .transform(lambda x: x.rolling(5, min_periods=2).std()))
    df_clean
    
    return df_clean

@st.cache_data(show_spinner=True)
def countries10_Zscore():
    # Dictionnaire ISO3 → vrai nom pays
    iso3_to_name = {
        "USA": "États-Unis",
        "DEU": "Allemagne",
        "FRA": "France",
        "JPN": "Japon",
        "CAN": "Canada",
        "IND": "Inde",
        "BRA": "Brésil",
        "ZAF": "Afrique du Sud",
        "IDN": "Indonésie",
        "MAR": "Maroc"
    }

    # Copier toutes les années depuis df_clean
    df_manu = df_10countries()

    # Ajouter le vrai nom des pays
    df_manu["Pays_nom"] = df_manu["Pays"].map(iso3_to_name)

    # Colonnes scores depuis df_ratings
    cols_scores = ["Score_solvabilite", "Rating_modele"]

    df_model = compute_Zscore()
    # Colonnes z-scores et bonus depuis df_model
    cols_zscores = [c for c in df_model.columns if c.endswith("_z")]
    cols_bonus   = [c for c in ["Monnaie_reserve","Safe_haven","Euro_core","Developpe"] if c in df_model.columns]

    # Sélection des données 2024
    df_2024 = df_model[df_model["Annee"] == 2024][["Pays","Annee"] + cols_scores].copy()

    # Ajouter z-scores et bonus pour 2024 depuis df_model
    df_model_2024 = df_model[df_model["Annee"] == 2024][["Pays","Annee"] + cols_zscores + cols_bonus].copy()
    df_2024 = df_2024.merge(df_model_2024, on=["Pays","Annee"], how="left")

    # Fusion dans df_manu (uniquement année 2024 pour les scores, z-scores et bonus)
    df_manu = df_manu.merge(
        df_2024,
        on=["Pays","Annee"],
        how="left",
        suffixes=("", "_2024")
    )

    # Tri pratique
    df_manu = df_manu.sort_values(["Pays","Annee"]).reset_index(drop=True)

    # Export Excel
    df_manu
    return df_manu

def compare_agencies_ratings():
    # Notations agences (2024)
    data_agences = {
        "Pays":      ["USA","DEU","FRA","JPN","CAN","IND","BRA","ZAF","IDN","MAR"],
        "Moody":     ["Aa1","Aaa","Aa3","A1","Aaa","Baa3","Ba1","Ba2","Baa2","Ba1"],
        "Fitch":     ["AA+","AAA","A+","A","AA+","BBB-","BB","BB-","BBB","BB+"],
        "S&P":       ["AA+","AAA","A+","A+","AAA","BBB","BB","BB","BBB","BBB-"]
    }
    df_ag = pd.DataFrame(data_agences)

    # conversion de notation texte en score numérique
    rating_to_num = {
        "AAA": 1,
        "Aaa": 1,
        "AA+": 2,
        "Aa1": 2,
        "AA": 3,
        "Aa2": 3,
        "AA-": 4,
        "Aa3": 4,
        "A+": 5,
        "A1": 5,
        "A": 6,
        "A2": 6,
        "A-": 7,
        "A3": 7,
        "BBB+": 8,
        "Baa1": 8,
        "BBB": 9,
        "Baa2": 9,
        "BBB-": 10,
        "Baa3": 10,
        "BB+": 11,
        "Ba1": 11,
        "BB": 12,
        "Ba2": 12,
        "BB-": 13,
        "Ba3": 13,
        "B+": 14,
        "B1": 14,
        "B": 15,
        "B2": 15,
        "B-": 16,
        "B3": 16,
        "CCC+": 17,
        "CCC": 18,
        "CCC-": 19,
        "CC": 20,
        "C": 21,
        "D": 22
    }

    # Convertir les notations en score numérique
    for agency in ["Moody", "Fitch", "S&P"]:
        df_ag[f"{agency}_num"] = df_ag[agency].map(rating_to_num)

    # Calcul de la moyenne agence (numérique) ---
    df_ag["Moyenne_agences_num"] = df_ag[["Moody_num","Fitch_num","S&P_num"]].mean(axis=1)

    df_compare = countries10_Zscore()
    df_ref = df_compare[df_compare["Annee"] == 2024]  # Données 2024

    df_ref = df_ref.merge(df_ag, on="Pays", how="left")
    df_ref["Model_num"] = df_ref["Rating_modele"].map(rating_to_num)
    df_ref["Ecart_model_vs_agences"] = df_ref["Model_num"] - df_ref["Moyenne_agences_num"]

    # bar chart des écarts ---
    # plt.figure(figsize=(10,6))
    # plt.bar(df_ref["Pays_nom"], df_ref["Ecart_model_vs_agences"], color="skyblue")
    # plt.axhline(0, color="black", linewidth=0.8)
    # plt.xticks(rotation=45)
    # plt.ylabel("Écart (score modèle – moyenne agences)")
    # plt.title("Écart de notation : modèle vs moyenne des agences (2024)")
    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(df_ref["Pays_nom"], df_ref["Ecart_model_vs_agences"], color="skyblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(df_ref)))
    ax.set_xticklabels(df_ref["Pays_nom"], rotation=45)
    ax.set_ylabel("Écart (score modèle – moyenne agences)")
    ax.set_title("Écart de notation : modèle vs moyenne des agences (2024)")
    fig.tight_layout()

    return fig

def radar_country(country_iso3):
    """
    Affiche 2 radars (macro + institutionnel) pour un pays ISO3
    avec conversion z-score → note /10.
    TOUT EST DANS CETTE PUTAIN DE FONCTION.
    """
    # ------------------------------------------------------------
    # 0. Colonnes utilisées
    # ------------------------------------------------------------
    MACRO_COLS = [
        "Croissance_PIB_z",
        "PIB_par_habitant_z",
        "Inflation_z",
        "Dette_publique_PIB_z",
        "BalanceCourante_PIB_z",
        "Reserves_sur_Importations_z"
    ]

    INSTIT_COLS = [
        "Voix_responsabilisation_z",
        "Stabilite_Politique_z",
        "Efficacite_Gouvernement_z",
        "Etat_de_droit_z",
        "Corruption_z"
    ]

    # ------------------------------------------------------------
    # 1. Charger dernière année
    # ------------------------------------------------------------
    df = countries10_Zscore().copy()
    df = df[df["Annee"] == df["Annee"].max()].set_index("Pays")

    if country_iso3 not in df.index:
        raise ValueError(f"Aucune donnée pour {country_iso3}")

    row = df.loc[country_iso3]

    # ------------------------------------------------------------
    # 2. Fonction de conversion z → score/10 (interne)
    # ------------------------------------------------------------
    def z_to_score(z):
        return np.clip(3 * z + 5, 0, 10)

    # ------------------------------------------------------------
    # 3. Construire valeurs MACRO
    # ------------------------------------------------------------
    macro_vals = [z_to_score(row.get(c, 0)) for c in MACRO_COLS]
    macro_angles = np.linspace(0, 2 * np.pi, len(MACRO_COLS), endpoint=False)
    macro_vals = macro_vals + macro_vals[:1]
    macro_angles = np.concatenate([macro_angles, [macro_angles[0]]])

    # ------------------------------------------------------------
    # 4. Construire valeurs INSTIT
    # ------------------------------------------------------------
    instit_vals = [z_to_score(row.get(c, 0)) for c in INSTIT_COLS]
    instit_angles = np.linspace(0, 2 * np.pi, len(INSTIT_COLS), endpoint=False)
    instit_vals = instit_vals + instit_vals[:1]
    instit_angles = np.concatenate([instit_angles, [instit_angles[0]]])

    # ------------------------------------------------------------
    # 5. FIGURE : 2 radars côte à côte
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 6),
        subplot_kw=dict(polar=True)
    )

    ax_macro, ax_instit = axes

    # --- Radar Macro ---
    ax_macro.plot(macro_angles, macro_vals)
    ax_macro.fill(macro_angles, macro_vals, alpha=0.2)
    ax_macro.set_xticks(macro_angles[:-1])
    ax_macro.set_xticklabels(MACRO_COLS, fontsize=8)
    ax_macro.set_title(f"Radar Macro – {country_iso3}")
    ax_macro.set_yticks([0, 2, 4, 6, 8, 10])
    ax_macro.set_ylim(0, 10)

    # --- Radar Instit ---
    ax_instit.plot(instit_angles, instit_vals)
    ax_instit.fill(instit_angles, instit_vals, alpha=0.2)
    ax_instit.set_xticks(instit_angles[:-1])
    ax_instit.set_xticklabels(INSTIT_COLS, fontsize=8)
    ax_instit.set_title(f"Radar Institutionnel – {country_iso3}")
    ax_instit.set_yticks([0, 2, 4, 6, 8, 10])
    ax_instit.set_ylim(0, 10)

    plt.tight_layout()
    return fig


valid_indicators = [
    "BalanceCourante_PIB", "Corruption", "Croissance_PIB",
    "Depenses_publiques_PIB", "Dette_publique_PIB",
    "Efficacite_Gouvernement", "Etat_de_droit",
    "Importations_$", "Inflation", "PIB_par_habitant",
    "PIB_total_$", "Recettes_publiques_PIB",
    "Reserves_change_$", "Stabilite_Politique",
    "Voix_responsabilisation", "Reserves_sur_Importations",
    "Volatilite_Croissance", "Volatilite_Inflation"
]

def time_series(indicator, countries=None):

    if indicator not in valid_indicators:
        raise ValueError(f"Indicateur '{indicator}' non valide.")

    # Charger les données complètes
    df = countries10_Zscore()
    df.columns = df.columns.str.strip()
    df["Annee"] = df["Annee"].astype(int)
    df["Pays"] = df["Pays"].str.strip()

    # Filtre années
    df = df[(df["Annee"] >= 1984) & (df["Annee"] <= 2024)].copy()

    # Liste de pays
    if countries is None:
        country_list = sorted(df["Pays"].unique())
    else:
        country_list = [c for c in countries if c in df["Pays"].unique()]

    years = sorted(df["Annee"].unique())

    # --- création du graphique ---
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_any = False
    for country in country_list:
        ser = (
            df[df["Pays"] == country][["Annee", indicator]]
            .drop_duplicates("Annee")
            .set_index("Annee")[indicator]
            .reindex(years)
        )

        if ser.dropna().empty:
            continue

        ax.plot(years, ser, marker="o", linewidth=1, label=country)
        plotted_any = True

    if not plotted_any:
        raise ValueError(f"Aucune donnée disponible pour l'indicateur {indicator}")

    # Mise en forme
    ax.set_title(f"{indicator} — 1984–2024", fontsize=14)
    ax.set_xlabel("Année")
    ax.set_ylabel(indicator)
    ax.set_xticks(years[::2])
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    return fig

@st.cache_data(show_spinner=True)
def compute_slopes():
    """
    Calcule les pentes (tendances) macro pour chaque pays
    à partir des séries historiques 1984–2024.
    """
    df = df_10countries().copy()
    df["Annee"] = df["Annee"].astype(int)

    slopes = []

    for country, dfc in df.groupby("Pays"):
        years = dfc["Annee"].values

        def slope(col):
            y = pd.to_numeric(dfc[col], errors="coerce").values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return np.nan
            return np.polyfit(years[mask], y[mask], 1)[0]

        slopes.append({
            "Pays": country,
            "slope_Croissance_PIB": slope("Croissance_PIB"),
            "slope_Dette_publique_PIB": slope("Dette_publique_PIB"),
            "slope_Inflation": slope("Inflation"),
            "slope_Reserves_sur_Importations": slope("Reserves_sur_Importations")
        })

    return pd.DataFrame(slopes)

def compute_outlook(row):
    """
    Détermine l'outlook souverain selon quelques règles simples.
    """
    # seuils critiques
    if row.get("Dette_publique_PIB", 0) > 140:
        return "Negative"
    if row.get("Inflation", 0) > 50:
        return "Negative"

    g = row.get("slope_Croissance_PIB", 0)
    d = row.get("slope_Dette_publique_PIB", 0)
    i = row.get("slope_Inflation", 0)
    r = row.get("slope_Reserves_sur_Importations", 0)

    # règles heuristiques
    if g > 0.03 and d < -0.5 and r > 0:
        return "Positive"
    if g < -0.01 and d > 0.5 and i > 0.5:
        return "Negative"

    return "Stable"

def make_comment(row):
    """
    Produit un commentaire automatique type agence :
    score, rating, outlook, points forts, points faibles.
    """
    score = row.get("Score_solvabilite", np.nan)
    rating = row.get("Rating_modele", "N/A")
    outlook = row.get("outlook", "N/A")

    pillars = {
        "Croissance_PIB_z": row.get("Croissance_PIB_z", np.nan),
        "PIB_par_habitant_z": row.get("PIB_par_habitant_z", np.nan),
        "Dette_publique_PIB_z": row.get("Dette_publique_PIB_z", np.nan),
        "Inflation_z": row.get("Inflation_z", np.nan),
        "BalanceCourante_PIB_z": row.get("BalanceCourante_PIB_z", np.nan),
        "Efficacite_Gouvernement_z": row.get("Efficacite_Gouvernement_z", np.nan),
    }

    # Top 1 et Bottom 1
    sorted_p = sorted(
        pillars.items(),
        key=lambda x: np.nan_to_num(x[1], nan=-999),
        reverse=True
    )

    best = sorted_p[0]
    worst = sorted_p[-1]

    return (
        f"Score final : {score:.2f} ({rating}, outlook {outlook}). "
        f"Point fort : {best[0]} ({best[1]:.2f}). "
        f"Point faible : {worst[0]} ({worst[1]:.2f})."
    )


@st.cache_data(show_spinner=True)
def _load_outlook_imf_panel(excel_path: str = r"./outlook datas.xlsx"):
    """
    Charge le fichier IMF Outlook et renvoie le panel CountryCode / COUNTRY / Annee / variables.
    Cette fonction est cachée et réutilisée grâce au cache Streamlit.
    """
    df_raw = pd.read_excel(excel_path)

    # Colonnes d'années
    year_cols = [c for c in df_raw.columns if str(c).isdigit()]

    # Mise en long
    df_long = df_raw.melt(
        id_vars=["DATASET", "SERIES_CODE", "OBS_MEASURE", "COUNTRY",
                 "INDICATOR", "FREQUENCY", "SCALE"],
        value_vars=year_cols,
        var_name="Annee",
        value_name="Value"
    )
    df_long["Annee"] = df_long["Annee"].astype(int)

    # Extraction codes (CountryCode.VarCode)
    split_code = df_long["SERIES_CODE"].str.split(".", expand=True)
    df_long["CountryCode"] = split_code[0]
    df_long["VarCode"] = split_code[1]

    # Mapping IMF → variables
    var_map = {
        "GGXWDG_NGDP": "Dette_publique_PIB",
        "NGSD_NGDP":   "Epargne_nationale_PIB",
        "GGXCNL_NGDP": "Solde_budgetaire_PIB",
        "NGDP_RPCH":   "Croissance_PIB",
        "PCPIPCH":     "Inflation_CPI",
        "LUR":         "Taux_chomage",
        "BCA_NGDPD":   "BalanceCourante_PIB"
    }

    df_long = df_long[df_long["VarCode"].isin(var_map.keys())].copy()
    df_long["Variable"] = df_long["VarCode"].map(var_map)

    # Panel pays/année
    df_panel = df_long.pivot_table(
        index=["CountryCode", "COUNTRY", "Annee"],
        columns="Variable",
        values="Value"
    ).reset_index()

    df_panel = df_panel.sort_values(["CountryCode", "Annee"])
    return df_panel

def outlook_imf(country_code: str, excel_path: str = r"./outlook datas.xlsx"):
    """
    Calcule l'outlook IMF pour un pays (code ISO3/IMF, ex 'USA', 'FRA')
    et renvoie 3 figures matplotlib + score + classification.

    Retourne :
        fig_dette, fig_epargne, fig_autres, outlook_score, outlook_class
    """
    # ---------- paramètres / mappings ----------
    indicators_weights = {
        "Solde_budgetaire_PIB":   +0.35,
        "Dette_publique_PIB":     -0.20,
        "Epargne_nationale_PIB":  +0.10,
        "BalanceCourante_PIB":    +0.10,
        "Croissance_PIB":         +0.10,
        "Inflation_CPI":          -0.07,
        "Taux_chomage":           -0.08
    }

    pretty_names = {
        "Solde_budgetaire_PIB":  "Solde budgétaire (% PIB)",
        "Dette_publique_PIB":    "Dette publique (% PIB)",
        "Epargne_nationale_PIB": "Épargne nationale (% PIB)",
        "BalanceCourante_PIB":   "Balance courante (% PIB)",
        "Croissance_PIB":        "Croissance PIB (%)",
        "Inflation_CPI":         "Inflation CPI (%)",
        "Taux_chomage":          "Chômage (%)"
    }

    # helpers internes
    def slope_last_years(series, n=5):
        s = series.dropna()
        if len(s) < n:
            return np.nan
        y = s.tail(n).values
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    def classify_outlook(score):
        if score > 0.20:
            return "POSITIVE"
        elif score < -0.20:
            return "NEGATIVE"
        return "STABLE"

    # ---------- 1. Chargement du panel via le cache ----------
    df_panel = _load_outlook_imf_panel(excel_path)

    # ---------- 2. Filtre sur le pays demandé ----------
    df_c = df_panel[df_panel["CountryCode"] == country_code].copy()
    if df_c.empty:
        raise ValueError(f"Aucune donnée IMF Outlook pour le pays {country_code}")

    df_c = df_c.set_index("Annee").sort_index()
    country_name = df_c["COUNTRY"].iloc[0]

    # ---------- 3. Score d'outlook ----------
    outlook_score = 0.0
    for ind, weight in indicators_weights.items():
        if ind not in df_c.columns:
            continue
        slope = slope_last_years(df_c[ind], 5)
        if not np.isnan(slope):
            outlook_score += slope * weight

    outlook_class = classify_outlook(outlook_score)

    # ---------- 4. Graphique 1 : dette publique ----------
    fig_dette = None
    if "Dette_publique_PIB" in df_c.columns and not df_c["Dette_publique_PIB"].dropna().empty:
        fig_dette, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_c.index, df_c["Dette_publique_PIB"], marker="o")
        ax.set_title(f"{country_name} — Dette publique (% PIB)")
        ax.set_xlabel("Année")
        ax.set_ylabel("Dette publique (% PIB)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig_dette.tight_layout()

    # ---------- 5. Graphique 2 : épargne nationale ----------
    fig_epargne = None
    if "Epargne_nationale_PIB" in df_c.columns and not df_c["Epargne_nationale_PIB"].dropna().empty:
        fig_epargne, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_c.index, df_c["Epargne_nationale_PIB"], marker="o")
        ax.set_title(f"{country_name} — Épargne nationale (% PIB)")
        ax.set_xlabel("Année")
        ax.set_ylabel("Épargne nationale (% PIB)")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig_epargne.tight_layout()

    # ---------- 6. Graphique 3 : autres indicateurs ----------
    others = [
        "Solde_budgetaire_PIB",
        "Croissance_PIB",
        "Inflation_CPI",
        "Taux_chomage",
        "BalanceCourante_PIB"
    ]

    fig_autres, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False
    for ind in others:
        if ind in df_c.columns and not df_c[ind].dropna().empty:
            ax.plot(df_c.index, df_c[ind], marker="o",
                    label=pretty_names.get(ind, ind))
            plotted_any = True

    if not plotted_any:
        plt.close(fig_autres)
        fig_autres = None
    else:
        ax.set_title(f"{country_name} ({country_code}) — Autres indicateurs Outlook")
        ax.set_xlabel("Année")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig_autres.tight_layout()

    return fig_dette, fig_epargne, fig_autres, outlook_score, outlook_class

