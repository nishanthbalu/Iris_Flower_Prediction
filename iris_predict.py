import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import time


model_dict = pk.load(open("iris_prediction_model.pkl", "rb"))
scaler = model_dict["scaler"]
model_knn = model_dict["model_knn"]
model_rf = model_dict["model_rf"]
model_nv = model_dict["model_nv"]
model_svm = model_dict["model_svm"]


md_accuracy = {'KNN Model':' Model Accuracy - 97 %', 
                'Random Forest Model':' Model Accuracy - 95 %', 
                'Naive Bayes Model':' Model Accuracy - 93 %',
                'Support Vector Machines Model':' Model Accuracy - 95 %'}




def main():

    st.title("Iris Flower Prediction")

    option = st.selectbox(
    'Choose model for prediction',
    ('KNN Model', 'Random Forest Model', 'Naive Bayes Model','Support Vector Machines Model'))
    
    a = float(st.number_input("Sepal length in cm",step= 0.1))
    b = float(st.number_input("Sepal width in cm",step= 0.1))
    c = float(st.number_input("Petal length in cm",step= 0.1))
    d = float(st.number_input("Petal width in cm",step= 0.1))

    btn = st.button("Predict")

    input_data = [[a,b,c,d]]
    input_data_scaled = scaler.transform(input_data)

    if btn:
        my_bar = st.progress(0)

        for percent_complete in range(100):
            time.sleep(0.03)
            my_bar.progress(percent_complete + 1)
        if my_bar.progress(100):    
            st.success('Prediction success!', icon="✅") 

        try:

            if option == 'KNN Model':
                prediction = model_knn.predict(input_data_scaled)[0]
                st.header(prediction)
                st.caption(md_accuracy[option])
                st.image(prediction+'.jpg')  

        except KeyError:

            st.error("Error Occured")
            st.stop()


        try:

            if option == 'Random Forest Model':
                prediction = model_rf.predict(input_data_scaled)[0]
                st.header(prediction)
                st.caption(md_accuracy[option])
                st.image(prediction+'.jpg')

        except KeyError:

            st.error("Error Occured")
            st.stop()


        try:

            if option == 'Naive Bayes Model':
                prediction = model_nv.predict(input_data_scaled)[0]
                st.header(prediction)
                st.caption(md_accuracy[option])
                st.image(prediction+'.jpg')
                
        except KeyError:

            st.error("Error Occured")
            st.stop()


        try:

            if option == 'Support Vector Machines Model':
                prediction = model_svm.predict(input_data_scaled)[0]
                st.header(prediction)
                st.caption(md_accuracy[option])
                st.image(prediction+'.jpg')
                
        except KeyError:

            st.error("Error Occured")
            st.stop()
    







if __name__ == "__main__":
    main()


