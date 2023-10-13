import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Glass Identification",
    page_icon=":mirror:",
    layout="wide"  # Set the layout to wide mode
)

# Load the dataset
@st.cache_data
def load_data(): 
    column_headers = [
        'Id',
        'RI',
        'Na',
        'Mg',
        'Al',
        'Si',
        'K',
        'Ca',
        'Ba',
        'Fe',
        'Type of glass'
    ]
    df = pd.read_csv("glass.data", names=column_headers, index_col='Id')
    return df

df = load_data()

st.markdown("<h1 style='text-align: center;'>Glass Type Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>✨Welcome to my awesome AI task where I code a make machine learning models to predict types of glass! :)✨</p>", unsafe_allow_html=True)
# Create a two-column layout
col1, col2 = st.columns(2)

# Data Exploration Section
with col1:
    st.header("EDA")
    st.write("Choose what info you would like to view about the dataset")
    sample_data_button = st.button("Sample Data")
    dataset_summary_button = st.button("Dataset Summary")
    dataset_information_button = st.button("Dataset Information")
    distribution_button = st.button("Distribution of Glass Types")

    if sample_data_button:
        st.subheader("Sample Data")
        st.markdown("The first 10 rows of the dataset")
        st.write(df.head(10))
        st.markdown("The last 10 rows of the dataset")
        st.write(df.tail(10))

    if dataset_summary_button:
        st.subheader("Dataset Summary (describe)")
        st.write(df.describe())

    if dataset_information_button:
        st.subheader("Dataset Information")
        info_text = f"Number of Rows: {df.shape[0]}\n"
        info_text += f"Number of Columns: {df.shape[1]}\n\n"
        info_text += "Data Types and Non-Null Counts:\n"
        for column in df.columns:
            info_text += f"{column}: {df[column].dtype} ({df[column].count()} non-null)\n"
        info_text += "\nMissing (Null) Values:\n"
        info_text += df.isnull().sum().to_string()
        st.text(info_text)

    if distribution_button:
        st.subheader("Distribution of Glass Types")
        glass_type_dict = {
            1: "building_windows_float_processed",
            2: "building_windows_non_float_processed",
            3: "vehicle_windows_float_processed",
            4: "vehicle_windows_non_float_processed",
            5: "containers",
            6: "tableware",
            7: "headlamps"
        }
        all_class_attributes = list(glass_type_dict.keys())
        x_labels = [glass_type_dict[i] for i in all_class_attributes]
        type_counts = df['Type of glass'].value_counts()
        st.bar_chart(type_counts, use_container_width=True)  # Use container width for better visualization

# Model Training Section
with col2:
    st.header("Model Training")
    st.write("Choose a Model")
    selected_model = st.selectbox("Select a Model", ["Random Forest", "AdaBoost", "K-Nearest Neighbors"])

    # Define hyperparameter controls
    if selected_model == "Random Forest":
        st.subheader("Random Forest Hyperparameters")
        n_estimators = st.slider("Number of Trees (n_estimators)", 1, 200, 100)

    elif selected_model == "AdaBoost":
        st.subheader("AdaBoost Hyperparameters")
        n_estimators = st.slider("Number of Estimators (n_estimators)", 1, 200, 50)

    elif selected_model == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors Hyperparameters")
        n_neighbors = st.slider("Number of Neighbors (n_neighbors)", 1, 20, 5)


    feature_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    random_seed = 52
    x = df[feature_cols]
    y = df['Type of glass']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_seed)

    if selected_model == "Random Forest":
        st.subheader("Random Forest Model")
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        sns.heatmap(cm, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()

    elif selected_model == "AdaBoost":
        st.subheader("AdaBoost Model with Random Forest Base Estimator")
        clf1 = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=n_estimators)
        clf1.fit(x_train, y_train)
        y_pred1 = clf1.predict(x_test)
        accuracy1 = metrics.accuracy_score(y_test, y_pred1)
        st.write(f"Accuracy: {accuracy1}")
        cm1 = confusion_matrix(y_test, y_pred1)
        st.subheader("Confusion Matrix")
        sns.heatmap(cm1, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()

    elif selected_model == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors Model")
        clf2 = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf2.fit(x_train, y_train)
        y_pred2 = clf2.predict(x_test)
        accuracy2 = metrics.accuracy_score(y_test, y_pred2)
        st.write(f"Accuracy: {accuracy2}")
        cm2 = confusion_matrix(y_test, y_pred2)
        st.subheader("Confusion Matrix")
        sns.heatmap(cm2, annot=True, fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        st.pyplot()
