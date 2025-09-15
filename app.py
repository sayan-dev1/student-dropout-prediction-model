import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Dropout Prediction System", layout="wide")

# --- Custom CSS for Layout ---
st.markdown("""
<style>
/* This CSS targets the main content block and limits its width */
.block-container {
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI-Based Dropout Prediction & Counseling System")
st.markdown("This application predicts the risk of a student dropping out and provides personalized counseling messages.")

# --- Load the trained model ---
try:
    with open('dropout_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'dropout_model.pkl' not found. Please run 'model_training.py' first.")
    st.stop()

# --- Section 1: Interactive Prediction for a Single Student ---
st.header("1. Predict Dropout Risk for a Single Student")
st.write("Enter the following student details to get a prediction:")

# Create input columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at enrollment", min_value=17, max_value=80, value=20)
    marital_status = st.selectbox("Marital status", [1, 2, 3, 4, 5, 6], format_func=lambda x: {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto union", 6: "Legally separated"}.get(x, x))
    scholarship_holder = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}.get(x, x))
    tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}.get(x, x))
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: {0: "Female", 1: "Male"}.get(x, x))
    previous_qualification = st.selectbox("Previous qualification", list(range(1, 18)), format_func=lambda x: str(x))
    unemployment_rate = st.number_input("Unemployment rate", min_value=0.0, value=10.0)

with col2:
    sem1_approved = st.number_input("Curricular units 1st sem (approved)", min_value=0, value=10)
    sem1_grade = st.number_input("Curricular units 1st sem (grade)", min_value=0.0, value=13.0)
    sem2_approved = st.number_input("Curricular units 2nd sem (approved)", min_value=0, value=10)
    sem2_grade = st.number_input("Curricular units 2nd sem (grade)", min_value=0.0, value=13.0)
    mother_qualification = st.number_input("Mother's qualification", min_value=1, max_value=44, value=1)
    father_qualification = st.number_input("Father's qualification", min_value=1, max_value=44, value=1)
    daytime_attendance = st.selectbox("Daytime/evening attendance", [0, 1], format_func=lambda x: {0: "Evening", 1: "Daytime"}.get(x, x))

# The 'Predict' button is placed after all the inputs
if st.button("Predict Dropout Risk"):
    # Create a DataFrame with the user's input.
    input_data = pd.DataFrame([[
        age, marital_status, scholarship_holder, tuition_fees_up_to_date, gender, 
        previous_qualification, sem1_approved, sem1_grade, sem2_approved, sem2_grade, 
        mother_qualification, father_qualification, daytime_attendance, unemployment_rate
    ]],
    columns=[
        'Age at enrollment', 'Marital status', 'Scholarship holder', 'Tuition fees up to date', 'Gender',
        'Previous qualification', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
        "Mother's qualification", "Father's qualification", 'Daytime/evening attendance', 'Unemployment rate'
    ])
    
    # Get the prediction probability
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    
    # --- Display the results ---
    st.subheader("Prediction Result")
    st.metric(label="Predicted Dropout Probability", value=f"{prediction_proba * 100:.2f}%")
    
    if prediction_proba > 0.5:
        st.error("High Risk of Dropout")
        st.write("It is highly recommended that you seek immediate academic counseling to discuss strategies for success and support systems.")
    else:
        st.success("Low Risk of Dropout")
        st.write("Keep up the excellent work! Continue to engage with your studies and campus community.")

st.markdown("---")

# --- Section 2: Batch Prediction and Visualization from CSV/Excel ---
st.header("2. Batch Prediction & Visualization")
st.write("Upload a CSV or Excel file to get predictions and insights for multiple students at once.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("File uploaded successfully!")
        
        # --- Make Predictions ---
        # The model expects a specific list of columns. We check if they exist.
        required_features = [
            'Age at enrollment', 'Marital status', 'Scholarship holder', 'Tuition fees up to date', 'Gender',
            'Previous qualification', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
            "Mother's qualification", "Father's qualification", 'Daytime/evening attendance', 'Unemployment rate'
        ]

        if not set(required_features).issubset(df.columns):
            st.error("The uploaded file is missing some of the required columns. Please check your data.")
            st.write("Expected columns:", required_features)
            st.stop()

        # Get prediction probabilities for all students
        df['Predicted Dropout Probability'] = model.predict_proba(df[required_features])[:, 1]
        
        # Add a final prediction column
        df['Predicted Status'] = np.where(df['Predicted Dropout Probability'] > 0.5, 'High Risk', 'Low Risk')
        
        st.subheader("Prediction Results")
        st.dataframe(df)

        st.subheader("Visualizations from Your Data")

        # --- Visualization 1: Predicted Status Distribution ---
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Predicted Status', data=df, ax=ax)
        plt.title('Distribution of Predicted Dropout Risk')
        plt.xlabel('Predicted Status')
        plt.ylabel('Number of Students')
        st.pyplot(fig)

        # --- Visualization 2: Relationship between Grade and Dropout Risk ---
        fig, ax = plt.subplots(figsize=(7, 4)) # Adjusted figsize
        sns.histplot(data=df, x='Curricular units 2nd sem (grade)', hue='Predicted Status', kde=True, ax=ax)
        plt.title('Grade Distribution by Predicted Dropout Risk')
        plt.xlabel('Curricular units 2nd sem (grade)')
        plt.ylabel('Count')
        st.pyplot(fig)

        # --- Additional Visualizations from User Request ---
        
        # 3. Bar chart of predicted dropout status
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Predicted Status', data=df, ax=ax)
        plt.title('Distribution of Predicted Dropout Risk')
        plt.xticks(ticks=[0, 1], labels=['Low Risk', 'High Risk'])
        plt.xlabel('Predicted Status')
        plt.ylabel('Number of Students')
        st.pyplot(fig)

        # 4. Histograms for a few key numeric columns
        numeric_cols_to_plot = ['Age at enrollment', 'Curricular units 1st sem (grade)', 'Unemployment rate']
        
        for c in numeric_cols_to_plot:
            if c in df.columns:
                fig, ax = plt.subplots(figsize=(7, 4)) # Adjusted figsize
                sns.histplot(df[c].dropna(), kde=False, ax=ax)
                plt.title(f'Distribution of {c}')
                st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")