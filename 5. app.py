import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Impacto del Clima en la Agricultura Española",page_icon="🌾",layout="wide")

@st.cache_data

def cargar_datos():
    return pd.read_parquet("dataset_analitico_filtrado.parquet")

df=cargar_datos()

cereales = ["TRIGO DURO", "TRIGO BLANDO Y SEMIDURO", "CEBADA DE 2 CARRERAS",
            "CEBADA DE 6 CARRERAS", "AVENA", "MAIZ", "TRITICALE"]
aceitunas = ["ACEITUNA DE MESA", "ACEITUNA DE DOBLE APTITU", "ACEITUNA DE ALMAZARA"]
vinos = ["UVA DE MESA NO ESPECIFIC", "UVA DE TRANSFORMACION"]
citricos = ["NARANJO", "MANDARINO", "LIMONERO", "NARANJO AMARGO"]

st.title("🌾 Impacto del Clima en la Agricultura Española")
st.markdown("Análisis histórico y predicción de superficie cultivada 2005-2029")

st.sidebar.title("Filtros")

grupo_seleccionado = st.sidebar.selectbox(
    "Grupo de cultivo",
    ["Cereales", "Olivar", "Viñedo", "Cítricos"]
)

grupos = {
    "Cereales": cereales,
    "Olivar": aceitunas,
    "Viñedo": vinos,
    "Cítricos": citricos
}

st.header("📊 Evolución histórica")

df_grupo = df[df["Cultivo"].isin(grupos[grupo_seleccionado])].copy()
df_evolucion = df_grupo.groupby("Anio")["Hectareas"].sum().reset_index()

fig = px.line(
    df_evolucion,
    x="Anio",
    y="Hectareas",
    title=f"Evolución de hectáreas - {grupo_seleccionado}",
    labels={"Anio": "Año", "Hectareas": "Hectáreas"}
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    df_temp = df_grupo.groupby("Anio")["Temp_media"].mean().reset_index()
    fig_temp = px.line(
        df_temp,
        x="Anio",
        y="Temp_media",
        title="Temperatura media por año",
        labels={"Anio": "Año", "Temp_media": "°C"}
    )
    st.plotly_chart(fig_temp, use_container_width=True)

with col2:
    df_prec = df_grupo.groupby("Anio")["Precipitacion"].mean().reset_index()
    fig_prec = px.line(
        df_prec,
        x="Anio",
        y="Precipitacion",
        title="Precipitación media por año",
        labels={"Anio": "Año", "Precipitacion": "mm"}
    )
    st.plotly_chart(fig_prec, use_container_width=True)

st.header("🔮 Predicción hasta 2029")

@st.cache_data
def predecir(cultivos_grupo):
    df_prophet = df[df["Cultivo"].isin(cultivos_grupo)].groupby("Anio").agg({
        "Hectareas": "sum",
        "Temp_media": "mean",
        "Precipitacion": "mean"
    }).reset_index()
    
    df_prophet.columns = ["ds", "y", "Temp_media", "Precipitacion"]
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")
    
    modelo = Prophet(yearly_seasonality=False)
    modelo.add_regressor("Temp_media")
    modelo.add_regressor("Precipitacion")
    modelo.fit(df_prophet)
    
    futuro = modelo.make_future_dataframe(periods=5, freq="YE")
    futuro["Temp_media"] = df_prophet["Temp_media"].mean()
    futuro["Precipitacion"] = df_prophet["Precipitacion"].mean()
    
    return modelo.predict(futuro), df_prophet

prediccion, df_hist = predecir(grupos[grupo_seleccionado])

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=df_hist["ds"], y=df_hist["y"], 
                               mode="markers", name="Histórico"))
fig_pred.add_trace(go.Scatter(x=prediccion["ds"], y=prediccion["yhat"], 
                               mode="lines", name="Predicción"))
fig_pred.add_trace(go.Scatter(x=prediccion["ds"], y=prediccion["yhat_upper"],
                               fill=None, mode="lines", line_color="lightblue", 
                               showlegend=False))
fig_pred.add_trace(go.Scatter(x=prediccion["ds"], y=prediccion["yhat_lower"],
                               fill="tonexty", mode="lines", line_color="lightblue",
                               name="Intervalo de confianza"))

fig_pred.update_layout(title=f"Predicción de hectáreas - {grupo_seleccionado}",
                        xaxis_title="Año", yaxis_title="Hectáreas")
st.plotly_chart(fig_pred, use_container_width=True)