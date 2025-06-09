import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
try:
    lgbm_model = joblib.load('lgbm_model.joblib')
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    lgbm_model = None

# Add title and description
st.title("Prediksi Dropout Mahasiswa")
st.write("Aplikasi ini memprediksi apakah seorang mahasiswa berisiko dropout berdasarkan data yang dimasukkan.")

if lgbm_model:
    st.header("Masukkan Data Mahasiswa")

    # Create input fields for features (replace with actual feature names)
    # Example:
    marital_status = st.number_input("Marital Status", min_value=1, max_value=6, value=1)
    application_mode = st.number_input("Application Mode", min_value=1, max_value=57, value=1)
    application_order = st.number_input("Application Order", min_value=0, max_value=9, value=1)
    course = st.number_input("Course", min_value=33, max_value=9991, value=171)
    daytime_evening_attendance = st.number_input("Daytime Evening Attendance (1=Daytime, 0=Evening)", min_value=0, max_value=1, value=1)
    previous_qualification = st.number_input("Previous Qualification", min_value=1, max_value=43, value=1)
    previous_qualification_grade = st.number_input("Previous Qualification Grade", min_value=0.0, max_value=200.0, value=120.0)
    nacionality = st.number_input("Nacionality", min_value=1, max_value=109, value=1)
    mothers_qualification = st.number_input("Mothers Qualification", min_value=1, max_value=44, value=1)
    fathers_qualification = st.number_input("Fathers Qualification", min_value=1, max_value=44, value=1)
    mothers_occupation = st.number_input("Mothers Occupation", min_value=0, max_value=195, value=1)
    fathers_occupation = st.number_input("Fathers Occupation", min_value=0, max_value=195, value=1)
    admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, value=120.0)
    displaced = st.number_input("Displaced (1=Yes, 0=No)", min_value=0, max_value=1, value=1)
    educational_special_needs = st.number_input("Educational Special Needs (1=Yes, 0=No)", min_value=0, max_value=1, value=0)
    debtor = st.number_input("Debtor (1=Yes, 0=No)", min_value=0, max_value=1, value=0)
    tuition_fees_up_to_date = st.number_input("Tuition Fees Up to Date (1=Yes, 0=No)", min_value=0, max_value=1, value=1)
    gender = st.number_input("Gender (1=Male, 0=Female)", min_value=0, max_value=1, value=1)
    scholarship_holder = st.number_input("Scholarship Holder (1=Yes, 0=No)", min_value=0, max_value=1, value=0)
    age_at_enrollment = st.number_input("Age at Enrollment", min_value=17, max_value=90, value=20)
    international = st.number_input("International (1=Yes, 0=No)", min_value=0, max_value=1, value=0)
    curricular_units_1st_sem_credited = st.number_input("Curricular Units 1st Sem Credited", min_value=0, max_value=30, value=0)
    curricular_units_1st_sem_enrolled = st.number_input("Curricular Units 1st Sem Enrolled", min_value=0, max_value=30, value=6)
    curricular_units_1st_sem_evaluations = st.number_input("Curricular Units 1st Sem Evaluations", min_value=0, max_value=40, value=8)
    curricular_units_1st_sem_approved = st.number_input("Curricular Units 1st Sem Approved", min_value=0, max_value=30, value=6)
    curricular_units_1st_sem_grade = st.number_input("Curricular Units 1st Sem Grade", min_value=0.0, max_value=20.0, value=12.0)
    curricular_units_1st_sem_without_evaluations = st.number_input("Curricular Units 1st Sem Without Evaluations", min_value=0, max_value=20, value=0)
    curricular_units_2nd_sem_credited = st.number_input("Curricular Units 2nd Sem Credited", min_value=0, max_value=30, value=0)
    curricular_units_2nd_sem_enrolled = st.number_input("Curricular Units 2nd Sem Enrolled", min_value=0, max_value=30, value=6)
    curricular_units_2nd_sem_evaluations = st.number_input("Curricular Units 2nd Sem Evaluations", min_value=0, max_value=40, value=8)
    curricular_units_2nd_sem_approved = st.number_input("Curricular Units 2nd Sem Approved", min_value=0, max_value=30, value=6)
    curricular_units_2nd_sem_grade = st.number_input("Curricular Units 2nd Sem Grade", min_value=0.0, max_value=20.0, value=12.0)
    curricular_units_2nd_sem_without_evaluations = st.number_input("Curricular Units 2nd Sem Without Evaluations", min_value=0, max_value=20, value=0)
    unemployment_rate = st.number_input("Unemployment Rate", min_value=0.0, max_value=20.0, value=10.0)
    inflation_rate = st.number_input("Inflation Rate", min_value=-5.0, max_value=5.0, value=1.0)
    gdp = st.number_input("GDP", min_value=-5.0, max_value=5.0, value=0.0)


    # Create a dictionary from the input values
    input_data = {
        'Marital_status': marital_status,
        'Application_mode': application_mode,
        'Application_order': application_order,
        'Course': course,
        'Daytime_evening_attendance': daytime_evening_attendance,
        'Previous_qualification': previous_qualification,
        'Previous_qualification_grade': previous_qualification_grade,
        'Nacionality': nacionality,
        'Mothers_qualification': mothers_qualification,
        'Fathers_qualification': fathers_qualification,
        'Mothers_occupation': mothers_occupation,
        'Fathers_occupation': fathers_occupation,
        'Admission_grade': admission_grade,
        'Displaced': displaced,
        'Educational_special_needs': educational_special_needs,
        'Debtor': debtor,
        'Tuition_fees_up_to_date': tuition_fees_up_to_date,
        'Gender': gender,
        'Scholarship_holder': scholarship_holder,
        'Age_at_enrollment': age_at_enrollment,
        'International': international,
        'Curricular_units_1st_sem_credited': curricular_units_1st_sem_credited,
        'Curricular_units_1st_sem_enrolled': curricular_units_1st_sem_enrolled,
        'Curricular_units_1st_sem_evaluations': curricular_units_1st_sem_evaluations,
        'Curricular_units_1st_sem_approved': curricular_units_1st_sem_approved,
        'Curricular_units_1st_sem_grade': curricular_units_1st_sem_grade,
        'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_sem_without_evaluations,
        'Curricular_units_2nd_sem_credited': curricular_units_2nd_sem_credited,
        'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_sem_enrolled,
        'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_sem_evaluations,
        'Curricular_units_2nd_sem_approved': curricular_units_2nd_sem_approved,
        'Curricular_units_2nd_sem_grade': curricular_units_2nd_sem_grade,
        'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_sem_without_evaluations,
        'Unemployment_rate': unemployment_rate,
        'Inflation_rate': inflation_rate,
        'GDP': gdp
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    if st.button("Prediksi Status"):
        prediction = lgbm_model.predict(input_df)

        if prediction[0] == 1:
            st.error("Prediksi: Mahasiswa berisiko Dropout")
        else:
            st.success("Prediksi: Mahasiswa kemungkinan Tidak Dropout")

else:
    st.warning("Model tidak dapat dimuat. Prediksi tidak tersedia.")