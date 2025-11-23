import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('diabetes_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'diabetes_model.pkl' not found. Please ensure it's in the same directory.")
        return None
    except ModuleNotFoundError as e:
        st.error(f"⚠️ Missing required module: {str(e)}")
        st.info("""
        **To fix this error, install the required package:**
        ```
        pip install scikit-learn
        ```
        Then restart the Streamlit app.
        """)
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.info("This might be due to missing dependencies. Try installing: `pip install scikit-learn pandas numpy`")
        return None

# App header
st.title("🏥 Diabetes Prediction App")
st.markdown("""
This application uses machine learning to predict the likelihood of diabetes based on medical information.
Please enter your medical information below.
""")

st.divider()

# Create input form
st.subheader("📋 Enter Medical Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Number of times pregnant"
    )
    
    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0,
        max_value=300,
        value=120,
        step=1,
        help="Plasma glucose concentration"
    )
    
    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=200,
        value=70,
        step=1,
        help="Diastolic blood pressure"
    )
    
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Triceps skin fold thickness"
    )

with col2:
    insulin = st.number_input(
        "Insulin (μU/mL)",
        min_value=0,
        max_value=900,
        value=80,
        step=1,
        help="2-Hour serum insulin"
    )
    
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index (weight in kg/(height in m)^2)"
    )
    
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        help="Diabetes pedigree function (genetic influence)"
    )
    
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=30,
        step=1,
        help="Age in years"
    )

st.divider()

# Predict button
if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
    model = load_model()
    
    if model is not None:
        # Prepare input data
        input_data = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]])
        
        # Create a DataFrame for better compatibility
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        input_df = pd.DataFrame(input_data, columns=feature_names)
        
        try:
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Try to get probability if available
            try:
                probability = model.predict_proba(input_df)[0]
                prob_diabetes = probability[1] * 100
                prob_no_diabetes = probability[0] * 100
            except:
                probability = None
            
            # Display results
            st.divider()
            st.subheader("📊 Prediction Results")
            
            if prediction == 1:
                st.error("⚠️ **High Risk: Diabetes Detected**")
                st.markdown("""
                The model predicts a **positive result** for diabetes based on the provided information.
                
                **Important:** This is a prediction tool and not a medical diagnosis. 
                Please consult with a healthcare professional for proper medical advice and testing.
                """)
            else:
                st.success("✅ **Low Risk: No Diabetes Detected**")
                st.markdown("""
                The model predicts a **negative result** for diabetes based on the provided information.
                
                **Important:** Continue maintaining a healthy lifestyle and regular check-ups with your healthcare provider.
                """)
            
            # Show probabilities if available
            if probability is not None:
                st.divider()
                st.subheader("🎯 Confidence Levels")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Diabetes", f"{prob_no_diabetes:.1f}%")
                with col2:
                    st.metric("Diabetes", f"{prob_diabetes:.1f}%")
                
                # Progress bar for visualization
                st.progress(prob_diabetes / 100)
            
        except Exception as e:
            st.error(f"⚠️ Error making prediction: {str(e)}")
            st.info("Please check that your model is compatible with the input features.")

# Footer
st.divider()
st.markdown("""
### ⚕️ Medical Disclaimer
This application is for educational and informational purposes only. It should not be used as a substitute 
for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other 
qualified health provider with any questions you may have regarding a medical condition.

### 📊 About the Features
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (μU/mL)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **Diabetes Pedigree Function**: A function that represents genetic influence
- **Age**: Age in years
""")
