# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 01:57:14 2022

@author: Anon
"""

import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('C:/DEPLOYMENT/trained_model.sav', 'rb'))


# creating a function for prediction
def wine_quality_prediction(input_data):
    

    # changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the data as we are predicting the label for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):
      return 'Good Wine'
    else:
      return 'Bad Wine'
  
    
  
def main():
    
    
    # giving a title
    st.title('Wine Quality Prediction Web App')
    
    
    # getting the input data from the user

    
    fixedacidity = st.text_input('fixed acidity value')
    volatileacidity = st.text_input('volatile acidity value')
    citricacid = st.text_input('citric acid value')
    residualsugar = st.text_input('residual sugar level')
    chlorides = st.text_input('chloride level')
    freesulfurdioxide = st.text_input('free sulfur dioxide value')
    totalsulfurdioxide = st.text_input('total sulfur dioxide value')
    density = st.text_input('density value')
    pH = st.text_input('ph level')
    sulphates = st.text_input('sulphate value')
    alcohol = st.text_input('alcohol level')
    
    
    # code for prediction
    testing = ''
    
    # creating a button for Prediction
    
    if st.button('Wine Quality Prediction Result'):
        testing = wine_quality_prediction([fixedacidity, volatileacidity, citricacid, residualsugar, chlorides, freesulfurdioxide, totalsulfurdioxide, density, pH, sulphates, alcohol])
    
    
    st.success(testing)
    
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    