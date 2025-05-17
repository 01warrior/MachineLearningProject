# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Charger les modèles et préprocesseurs sauvegardés ---
@st.cache_resource # Important pour la performance, charge une seule fois
def load_resources():
    model = joblib.load('logistic_regression_dep_model.joblib')
    scaler = joblib.load('scaler_dep.joblib')
    label_mappings = joblib.load('label_mappings_dep.joblib')
    final_training_columns = joblib.load('final_training_columns.joblib')
    original_cols_for_dummies = joblib.load('original_cols_for_dummies.joblib')
    cols_to_scale = joblib.load('cols_to_scale.joblib')
    sleep_mapping = joblib.load('sleep_mapping.joblib')
    
    try:
        top_30_cities = joblib.load('top_30_cities.joblib')
    except FileNotFoundError:
        st.warning("Fichier 'top_30_cities.joblib' non trouvé. Le traitement de 'City' sera simplifié.")
        top_30_cities = None 
            
    return model, scaler, label_mappings, final_training_columns, original_cols_for_dummies, cols_to_scale, sleep_mapping, top_30_cities

model, scaler, label_mappings, final_training_columns, original_cols_for_dummies, cols_to_scale, sleep_mapping, top_30_cities = load_resources()

# --- 2. Définir la fonction de regroupement pour Degree ---
def group_degree(deg):
    deg_str = str(deg).lower()
    if 'class 12' in deg_str or '12th' in deg_str: return 'Class_12'
    elif deg_str.startswith('b.') or deg_str in ['ba', 'bsc', 'bhm', 'bba', 'bca', 'be', 'llb', 'mbbs']: return 'Bachelor'
    elif deg_str.startswith('m.') or deg_str in ['mba', 'msc', 'mpharm', 'ma', 'mhm', 'me', 'md', 'llm']: return 'Master_MD'
    elif 'phd' in deg_str: return 'PhD'
    elif deg_str == 'others': return 'Other_Degree'
    else: return 'Other_Degree' # Catégorie par défaut

# --- 3. Définir la fonction de prétraitement ---
def preprocess_input_streamlit(data_dict_raw):
    # Créer une copie pour éviter de modifier le dictionnaire original si passé par référence
    df_input = pd.DataFrame([data_dict_raw.copy()])

    # 1. Traitement 'Sleep Duration'
    if 'Sleep Duration' in df_input.columns:
        if df_input['Sleep Duration'].iloc[0] == 'Others_Sleep_Input': # Nom distinct pour l'option Streamlit
            # Gérer 'Others' comme vous l'avez fait à l'entraînement (ex: supprimer ou imputer)
            # Ici, pour la prédiction, on ne peut pas supprimer. Il faut soit retourner une erreur,
            # soit l'imputer avec une valeur neutre (ex: moyenne des autres) ou la valeur la plus fréquente
            # Pour l'instant, on retourne un message d'erreur si 'Others' est sélectionné
            # car vous avez supprimé ces lignes à l'entraînement.
            return "Erreur: La catégorie 'Others' pour Sleep Duration n'est pas gérée pour la prédiction."
        df_input['Sleep_Duration_Numeric'] = df_input['Sleep Duration'].map(sleep_mapping)
        df_input.drop('Sleep Duration', axis=1, inplace=True)

    # 2. Label Encoding pour Gender, Suicidal Thoughts, Family History
    # Les clés dans label_mappings sont 'Gender_Encoded', 'Suicidal_Thoughts_Encoded', etc.
    # Les clés dans data_dict_raw sont les noms originaux des colonnes
    if 'Gender' in df_input.columns:
        df_input['Gender_Encoded'] = df_input['Gender'].map(label_mappings['Gender_Encoded'])
        df_input.drop('Gender', axis=1, inplace=True)
    
    if 'Have you ever had suicidal thoughts ?' in df_input.columns:
        df_input['Suicidal_Thoughts_Encoded'] = df_input['Have you ever had suicidal thoughts ?'].map(label_mappings['Suicidal_Thoughts_Encoded'])
        df_input.drop('Have you ever had suicidal thoughts ?', axis=1, inplace=True)

    if 'Family History of Mental Illness' in df_input.columns:
        df_input['Family_History_Encoded'] = df_input['Family History of Mental Illness'].map(label_mappings['Family_History_Encoded'])
        df_input.drop('Family History of Mental Illness', axis=1, inplace=True)
        
    # 3. Traitement 'Profession' (Is_Student)
    if 'Profession' in df_input.columns:
        df_input['Is_Student'] = df_input['Profession'].apply(lambda x: 1 if x == 'Student' else 0)
        df_input.drop('Profession', axis=1, inplace=True)
            
    # 4. Traitement 'City' (Regroupement en 'City_Processed')
    if 'City' in df_input.columns:
        if top_30_cities is not None:
            df_input['City_Processed'] = df_input['City'].apply(lambda x: x if x in top_30_cities else 'Other_City')
        else: # Gestion simplifiée si top_30_cities n'est pas disponible
            df_input['City_Processed'] = df_input['City'] # On espère que la ville est connue
        df_input.drop('City', axis=1, inplace=True)

    # 5. Traitement 'Degree' (Regroupement en 'Degree_Grouped')
    if 'Degree' in df_input.columns:
        # Gérer le cas 'Others_Degree_Input' venant du selectbox
        if df_input['Degree'].iloc[0] == 'Others_Degree_Input':
            df_input['Degree_Grouped'] = 'Other_Degree' # Mapper à la catégorie 'Other_Degree'
        else:
            df_input['Degree_Grouped'] = df_input['Degree'].apply(group_degree)
        df_input.drop('Degree', axis=1, inplace=True)

    # 6. One-Hot Encoding avec pd.get_dummies
    # original_cols_for_dummies = ['City_Processed', 'Dietary Habits', 'Degree_Grouped']
    cols_to_get_dummies_present_in_df = [col for col in original_cols_for_dummies if col in df_input.columns]
    if cols_to_get_dummies_present_in_df:
        df_input = pd.get_dummies(df_input, columns=cols_to_get_dummies_present_in_df, dtype=int)
    
    # 7. Réindexer pour correspondre exactement aux colonnes d'entraînement
    df_processed = df_input.reindex(columns=final_training_columns, fill_value=0)
    
    # 8. Mise à l'échelle
    # cols_to_scale doit contenir les noms après transformation (ex: 'Sleep_Duration_Numeric')
    cols_to_scale_present_final_in_df = [col for col in cols_to_scale if col in df_processed.columns]

    if cols_to_scale_present_final_in_df:
         df_processed[cols_to_scale_present_final_in_df] = scaler.transform(df_processed[cols_to_scale_present_final_in_df])
    
    return df_processed

    # --- 4. Interface Utilisateur Streamlit ---
st.set_page_config(page_title="Prédiction de Dépression Étudiante", layout="centered", initial_sidebar_state="expanded")
st.title("🎓 Prédiction de Dépression chez les Étudiants")
st.markdown("Entrez les informations de l'étudiant pour obtenir une prédiction sur son état de santé mentale.")
st.markdown("---")

# Noms de colonnes originaux (pour les labels des widgets)
# CES NOMS DOIVENT CORRESPONDRE AUX CLÉS QUE VOUS UTILISEREZ DANS `input_data`
feature_labels = {
    "Age": "Âge",
    "Gender": "Genre",
    "City": "Ville",
    "Profession": "Profession",
    "Academic Pressure": "Pression Académique (1-5)",
    "Work Pressure": "Pression Professionnelle (0-5)",
    "CGPA": "Moyenne Cumulative (CGPA 0-10)",
    "Study Satisfaction": "Satisfaction des Études (1-5)",
    "Job Satisfaction": "Satisfaction Professionnelle (0-5)",
    "Sleep Duration": "Durée du Sommeil",
    "Dietary Habits": "Habitudes Alimentaires",
    "Degree": "Diplôme",
    "Have you ever had suicidal thoughts ?": "Pensées Suicidaires Antérieures ?",
    "Work/Study Hours": "Heures de Travail/Étude par jour",
    "Financial Stress": "Stress Financier (1-5)",
    "Family History of Mental Illness": "Antécédents Familiaux de Maladie Mentale"
}

input_data = {} # Dictionnaire pour stocker les entrées

# Créer des formulaires pour une meilleure organisation
with st.form(key='prediction_form'):
    st.subheader("Informations Personnelles et Académiques")
    col1, col2 = st.columns(2)
    with col1:
        input_data["Age"] = st.number_input(feature_labels["Age"], min_value=15, max_value=50, value=22, step=1)
        input_data["Gender"] = st.selectbox(feature_labels["Gender"], options=list(label_mappings['Gender_Encoded'].keys()), index=0)
        
        if top_30_cities is not None:
            city_options_display = sorted(list(top_30_cities) + ['Autre Ville (Other_City)'])
            selected_city_display = st.selectbox(feature_labels["City"], options=city_options_display, index=0)
            input_data["City"] = 'Other_City' if selected_city_display == 'Autre Ville (Other_City)' else selected_city_display
        else:
            input_data["City"] = st.text_input(f"{feature_labels['City']} (ex: Kalyan, Srinagar, ...)", "Kalyan")

        profession_options = ['Student', 'Architect', 'Teacher', 'Digital Marketer', 'Content Writer', 'Chef', 'Doctor', 'Pharmacist', 'Civil Engineer', 'UX/UI Designer', 'Educational Consultant', 'Manager', 'Lawyer', 'Entrepreneur', 'Autre Profession']
        selected_profession = st.selectbox(feature_labels["Profession"], options=profession_options, index=0)
        input_data["Profession"] = 'Other' if selected_profession == 'Autre Profession' else selected_profession


    with col2:
        input_data["Degree"] = st.selectbox(feature_labels["Degree"], 
                                        options=['Class 12', 'B.Ed', 'B.Com', 'B.Arch', 'BCA', 'MSc', 'B.Tech', 'MCA', 
                                                 'M.Tech', 'BHM', 'BSc', 'M.Ed', 'B.Pharm', 'M.Com', 'BBA', 'MBBS', 
                                                 'LLB', 'BE', 'BA', 'M.Pharm', 'MD', 'MBA', 'MA', 'PhD', 'LLM', 'MHM', 'ME', 'Others_Degree_Input'], index=0)
        input_data["CGPA"] = st.slider(feature_labels["CGPA"], 0.0, 10.0, 7.5, step=0.01)
        input_data["Work/Study Hours"] = st.slider(feature_labels["Work/Study Hours"], 0.0, 18.0, 6.0, step=0.5)

    st.markdown("---")
    st.subheader("Pressions et Satisfactions")
    col3, col4 = st.columns(2)
    with col3:
        input_data["Academic Pressure"] = st.slider(feature_labels["Academic Pressure"], 1.0, 5.0, 3.0, step=0.1)
        input_data["Study Satisfaction"] = st.slider(feature_labels["Study Satisfaction"], 1.0, 5.0, 3.0, step=0.1)
    with col4:
        input_data["Work Pressure"] = st.slider(feature_labels["Work Pressure"], 0.0, 5.0, 2.0, step=0.1) # Noter que 0 est une option
        input_data["Job Satisfaction"] = st.slider(feature_labels["Job Satisfaction"], 0.0, 5.0, 2.0, step=0.1) # Noter que 0 est une option

    st.markdown("---")
    st.subheader("Habitudes de Vie et Santé Mentale")
    col5, col6 = st.columns(2)
    with col5:
        sleep_duration_options = list(sleep_mapping.keys())
        input_data["Sleep Duration"] = st.selectbox(feature_labels["Sleep Duration"], options=sleep_duration_options, index=0)
        input_data["Dietary Habits"] = st.selectbox(feature_labels["Dietary Habits"], options=['Healthy', 'Moderate', 'Unhealthy', 'Others'], index=0)
        input_data["Financial Stress"] = st.slider(feature_labels["Financial Stress"], 1.0, 5.0, 2.0, step=0.1)
    with col6:
        input_data["Have you ever had suicidal thoughts ?"] = st.radio(
            feature_labels["Have you ever had suicidal thoughts ?"],
            options=list(label_mappings['Suicidal_Thoughts_Encoded'].keys()), index=0, horizontal=True
        )
        input_data["Family History of Mental Illness"] = st.radio(
            feature_labels["Family History of Mental Illness"],
            options=list(label_mappings['Family_History_Encoded'].keys()), index=0, horizontal=True
        )
    
    st.markdown("---")
    submit_button = st.form_submit_button(label="Obtenir la Prédiction ✨", use_container_width=True)

# Logique de prédiction après soumission du formulaire
if submit_button:
    # Vérifier si l'utilisateur a choisi "Others_Sleep_Input" pour Sleep Duration
    if input_data["Sleep Duration"] == 'Others_Sleep_Input': # Doit correspondre à l'option du selectbox
        st.error("La catégorie 'Others' pour la durée du sommeil n'est pas directement gérée. Veuillez estimer une catégorie proche.")
    else:
        with st.spinner("Analyse en cours..."):
            try:
                processed_df = preprocess_input_streamlit(input_data)
                
                if isinstance(processed_df, str): # Si preprocess_input retourne un message d'erreur
                    st.error(processed_df)
                else:
                    prediction = model.predict(processed_df)
                    prediction_proba = model.predict_proba(processed_df)

                    st.subheader("Résultat de la Prédiction :")
                    if prediction[0] == 1:
                        st.error("Le modèle prédit un statut de **Dépression**.", icon="😟")
                    else:
                        st.success("Le modèle prédit un statut de **Non-Dépression**.", icon="😊")

                    st.metric(label="Probabilité de Dépression (Classe 1)", value=f"{prediction_proba[0][1]:.2%}")
                    
                    # Optionnel: Afficher un graphique des probabilités
                    # prob_data = {'Classe': ['Non-Déprimé (0)', 'Déprimé (1)'], 
                    #              'Probabilité': [prediction_proba[0][0], prediction_proba[0][1]]}
                    # prob_chart_df = pd.DataFrame(prob_data)
                    # st.bar_chart(prob_chart_df.set_index('Classe'))

                    # Pour le débogage,:
                    # st.subheader("Données brutes entrées (dictionnaire):")
                    # st.json(input_data)
                    # st.subheader("Données prétraitées envoyées au modèle (DataFrame):")
                    # st.dataframe(processed_df)

            except Exception as e:
                st.error(f"Une erreur est survenue lors de la prédiction : {e}")
                st.error("Veuillez vérifier les valeurs d'entrée et la configuration du prétraitement. Consultez la console pour plus de détails si vous exécutez localement.")
                print(f"Erreur détaillée: {e}") # Pour le log serveur/console

st.markdown("---")
st.sidebar.header("ℹ️ À Propos de l'Application")
st.sidebar.info(
    "Cette application utilise un modèle de **Régression Logistique** pour estimer "
    "la probabilité de dépression chez les étudiants. Elle est basée sur un dataset "
    "comprenant divers facteurs socio-démographiques, académiques et de style de vie."
    "\n\n**Note:** Ceci est un outil de démonstration et ne doit pas remplacer un diagnostic médical professionnel."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Développé par [Vos Noms/Noms du Groupe]")