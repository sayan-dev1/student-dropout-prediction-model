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

# --- Qualification mapping for the dropdown ---
qualification_mapping = {
    1: 'Secondary Education', 
    2: 'Higher Education - Bachelor', 
    3: 'Higher Education - Master',
    4: 'Higher Education - Doctorate', 
    5: 'Technical or Professional Training', 
    6: 'Other',
    7: 'Secondary Education - Vocational', 
    8: 'Higher Education - Polytechnic Bachelor',
    9: 'Higher Education - Polytechnic Master', 
    10: 'Technical Certificate',
    11: 'Higher Education - Unspecified', 
    12: 'No Formal Qualification', 
    13: 'Middle School',
    14: 'Primary School', 
    15: 'Further Vocational Training', 
    16: 'High School - Academic',
    17: 'High School - Technical'
}

with col1:
    age = st.number_input("Age at enrollment", min_value=17, max_value=80, value=20)
    marital_status = st.selectbox("Marital status", [1, 2, 3, 4, 5, 6], format_func=lambda x: {1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto union", 6: "Legally separated"}.get(x, x))
    scholarship_holder = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}.get(x, x))
    tuition_fees_up_to_date = st.selectbox("Tuition fees up to date", [0, 1], format_func=lambda x: {0: "No", 1: "Yes"}.get(x, x))
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: {0: "Female", 1: "Male"}.get(x, x))
    # UPDATED: Use the new qualification mapping
    previous_qualification = st.selectbox(
        "Previous qualification",
        options=list(qualification_mapping.keys()),
        format_func=lambda x: qualification_mapping.get(x, x)
    )
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

    # --- Visualization for single student ---
    fig, ax = plt.subplots(figsize=(6, 1.0))
    ax.barh(['Predicted Risk'], [prediction_proba], color=['red' if prediction_proba > 0.5 else 'green'])
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color='gray', linestyle='--', label='50% Threshold')
    ax.set_xticks(ticks=[0, 0.25, 0.5, 0.75, 1.0], labels=['0%', '25%', '50%', '75%', '100%'])
    ax.legend()
    ax.set_xlabel('Probability')
    ax.set_title('Visualizing Dropout Probability')
    st.pyplot(fig)
    # --- END NEW VISUALIZATION ---
    
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
        
        # --- FIX: Convert columns to numeric before plotting ---
        for col in required_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Get prediction probabilities for all students
        df['Predicted Dropout Probability'] = model.predict_proba(df[required_features])[:, 1]
        
        # Add a final prediction column
        df['Predicted Status'] = np.where(df['Predicted Dropout Probability'] > 0.5, 'High Risk', 'Low Risk')
        
        st.subheader("Prediction Results")
        st.dataframe(df)
        
        # --- NEW SECTION: Analysis and Insights (Now wrapped in an expander) ---
        with st.expander("Expand to view Analysis and Insights 📈"):
            st.write("Based on the uploaded dataset and the visualizations, here are some key insights:")
        
            # Get the total number of students and high-risk students
            total_students = len(df)
            high_risk_students = len(df[df['Predicted Status'] == 'High Risk'])
            low_risk_students = len(df[df['Predicted Status'] == 'Low Risk'])
            
            # Display key metrics
            st.markdown(f"**Total Students Analyzed:** {total_students}")
            st.markdown(f"**High Risk of Dropout:** {high_risk_students} ({high_risk_students / total_students:.1%})")
            st.markdown(f"**Low Risk of Dropout:** {low_risk_students} ({low_risk_students / total_students:.1%})")
            
            # Provide insights based on data
            st.subheader("Key Findings:")
            
            # Insight 1: Based on the overall dropout risk percentage
            risk_percentage = high_risk_students / total_students
            if risk_percentage > 0.4:
                st.write("💡 **High-Level Concern:** The data suggests a significantly high percentage of students are at risk of dropping out. This indicates a potential systemic issue that requires a focused intervention strategy.")
            elif risk_percentage > 0.2:
                st.write("💡 **Moderate Concern:** There is a notable portion of students at risk. Targeted support programs could be highly effective in reducing this number.")
            else:
                st.write("💡 **Low-Level Concern:** The overall dropout risk is low. The system appears to be working well, but a few targeted interventions could further improve student retention.")
            
            # Insight 2: Based on the grade distribution chart
            st.write("📊 **Grade Performance:** The 'Grade Distribution by Predicted Dropout Risk' chart shows a clear inverse relationship. Students with lower grades in the second semester are much more likely to be predicted as 'High Risk'. This reinforces the importance of academic support and tutoring programs for struggling students.")
            
            # Insight 3: Based on the correlation heatmap
            st.write("🔗 **Feature Relationships:** The correlation heatmap provides a deeper look into the data. Look for features with strong positive or negative correlations to identify potential drivers of dropout. For example, a strong negative correlation between 'Tuition fees up to date' and 'Predicted Dropout Probability' could highlight a financial barrier to student success.")
            
            st.markdown("---")
            st.markdown("✨ **Next Steps:** Use these insights to create actionable plans, such as developing academic support programs, providing financial aid counseling, or running mentorship initiatives.")
        
        st.markdown("---")
        st.subheader("Visualizations from Your Data")

        # --- Visualization 1: Predicted Status Distribution ---
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Predicted Status', data=df, ax=ax)
        plt.title('Distribution of Predicted Dropout Risk')
        plt.xlabel('Predicted Status')
        plt.ylabel('Number of Students')
        st.pyplot(fig)

        # --- Visualization 2: Relationship between Grade and Dropout Risk ---
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x='Curricular units 2nd sem (grade)', hue='Predicted Status', kde=True, ax=ax)
        plt.title('Grade Distribution by Predicted Dropout Risk')
        plt.xlabel('Curricular units 2nd sem (grade)')
        plt.ylabel('Count')
        st.pyplot(fig)

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
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.histplot(df[c].dropna(), kde=False, ax=ax)
                plt.title(f'Distribution of {c}')
                st.pyplot(fig)

        # --- NEW: Visualization 5: Correlation Heatmap ---
        st.subheader("Correlation Heatmap")
        
        # Select only the numeric columns for the heatmap
        numeric_df = df.select_dtypes(include=['number'])

        if not numeric_df.empty:
            # Calculate the correlation matrix
            corr_matrix = numeric_df.corr()

            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Matrix of Features', fontsize=16)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found to generate the correlation heatmap.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

st.markdown("---")
st.header("3. Model Evaluation & Explainability")
st.write("Understand the model's overall performance and which features are most important.")

# --- LOAD METRICS ---
try:
    with open('model_metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
except FileNotFoundError:
    accuracy = "--"
    precision = "--"
    recall = "--"
    st.warning("`model_metrics.pkl` not found. Please run your `model_training.py` script to generate this file.")
except Exception as e:
    accuracy = "--"
    precision = "--"
    recall = "--"
    st.error(f"Error loading metrics: {e}")

# --- DISPLAY METRICS ---
st.subheader("Model Performance Metrics")
st.write("This section shows the model's performance on a held-out test set.")
col_acc, col_prec, col_rec = st.columns(3)

with col_acc:
    st.info("💡 **Accuracy:** Percentage of correct predictions.")
    st.metric(label="Accuracy", value=f"{accuracy:.2f}" if isinstance(accuracy, (int, float)) else accuracy)
with col_prec:
    st.info("💡 **Precision:** Of all predicted high-risk students, how many were actually high-risk?")
    st.metric(label="Precision", value=f"{precision:.2f}" if isinstance(precision, (int, float)) else precision)
with col_rec:
    st.info("💡 **Recall:** Of all actual high-risk students, how many did the model find?")
    st.metric(label="Recall", value=f"{recall:.2f}" if isinstance(recall, (int, float)) else recall)


# --- LOAD FEATURE IMPORTANCE ---
try:
    with open('feature_importance.pkl', 'rb') as f:
        importance_df = pickle.load(f)
        if importance_df is None:
            st.warning("Feature importance data is `None`. Your model type may not support it or it was not saved correctly.")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax, palette='viridis') # Display top 10 features
            ax.set_title('Top 10 Feature Importance')
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Features')
            st.pyplot(fig)
except FileNotFoundError:
    st.warning("`feature_importance.pkl` not found. Please run your `model_training.py` script to generate this file.")
except Exception as e:
    st.error(f"An error occurred while plotting feature importance: {e}")
