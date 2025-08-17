import pandas as pd 
import numpy as np 
import streamlit as st
import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Loading the trained model
model = tf.keras.models.load_model('model.h5')

#Loding encoder and scaler
with open('Label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


#strimlit app
st.title('Customer Churn Prediction')

#User Input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
cradit_Score = st.number_input('Credit Score')
Estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox('Has Cr Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [cradit_Score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [Estimated_salary]
})


# One-hot encoder "Geography"
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# combined one-hot encoded column with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

## Scaling the data 
input_scaled = scaler.transform(input_data)

##predict churn
prediction = model.predict(input_scaled)
prediction_prob = prediction[0][0]

# Display the result to the user
if prediction_prob > 0.5:
    st.error(f"The customer is likely to churn. (Prediction Probability: {prediction_prob:.2f})")
else:
    st.success(f"The customer is not likely to churn. (Prediction Probability: {prediction_prob:.2f})")

st.info("The prediction is based on the input features and the trained model.")