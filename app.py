import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import streamlit as st
ad = pd.read_csv(
    filepath_or_buffer="./Advertising Budget and Sales.csv",index_col=[0],)
columns = ["tv", "radio", "news", "sales"]
ad.columns=columns
dict_scaler = dict()
for col in columns:
    dict_scaler[col] = StandardScaler().fit(ad[[col]])
df = ad.copy()
for col in columns:
    df[col] = dict_scaler[col].transform(X=ad[[col]])
target = "sales"
features = df.columns.to_list()
features.remove(target)
X = df[features].values
y = df[target].values
robot = LinearRegression().fit(X=X, y=y)
st.title(body="Predict Sales by Advertisement Types")
st.number_input(label="tv")
user_input = dict()
for col in columns:
   user_input[col] = st.number_input(label=col, value=ad[col].mean())
#st.write(user_input)
df_input = pd.DataFrame(data=[user_input],columns=features)
df_scaled = df_input.copy()
for col in features:
    df_scaled[col] = dict_scaler[col].transform(X=df_input[[col]])
X_test = df_scaled.values
st.dataframe(data=df_input)
y_pred = robot.predict(X=X_test)
y_pred = dict_scaler[target].inverse_transform(y_pred.reshape(-1,1))
st.write(y_pred)