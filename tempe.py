import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción del la temperatura  ''')
st.image("Imtempe.jpeg", caption="Temperatura.")

st.header('Datos personales')

def user_input_features():
  # Entrada
  Mes = st.number_input('Mes:', min_value=0, max_value=12, value = 0, step = 1)
  Año = st.number_input('Año:',  min_value=0, max_value=3000, value = 0, step = 1)
  Ciudad = st.number_input('Ciudad (Acapulco= 0, Acuña=1, Aguascalientes=2):', min_value=0, max_value=5, value = 0, step = 1)


  user_input_data = {'month': Mes,
                     'year': Año,
                     'City': Ciudad,
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

datos =  pd.read_csv('Temperatura.csv', encoding='latin-1')
X = datos.drop(columns='AverageTemperature')
y = datos['AverageTemperature']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1614372)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['year'] + b1[1]*df['month'] + b1[2]*df['City']

st.subheader('Predición de la temperatura')
st.write('La temperatura es ', prediccion)
