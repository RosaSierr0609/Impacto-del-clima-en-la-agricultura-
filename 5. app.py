import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Impacto del Clima en la Agricultura Española",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap');
    h1, h2, h3, h4, h5, h6, p, li, label, div, span.st-emotion-cache, 
    button, input, select, textarea {
        font-family: 'Montserrat', sans-serif !important;
    }
    [data-testid="stSidebarCollapseButton"] * {
    font-family: 'Material Symbols Rounded' !important;
    }
    [data-testid="stSidebarCollapseButton"] span {
        font-family: 'Material Symbols Rounded' !important;
    }
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }
    [data-testid="stSidebar"] {
        background-color: #c8dfc0;
    }
    .stApp {
        background-color: #fffff0;
    }
    [data-testid="stSidebar"] * {
        color: #4a3728 !important;
    }
    h1, h2, h3, p, li, div {
        color: #4a3728 !important;
    }
    header[data-testid="stHeader"] {
        background-color: #c8dfc0;
    }
    [data-testid="stSidebar"] .stRadio {
    margin-top: 100px;
    }
    [data-testid="stSidebar"] .stRadio label p {
        font-size: 18px !important;
        padding: 8px 0px !important;
    }   
    [data-testid="stSidebar"] .stRadio label {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_datos():
    return pd.read_parquet("dataset_analitico_filtrado.parquet")

df = cargar_datos()

cereales = ["TRIGO DURO", "TRIGO BLANDO Y SEMIDURO", "CEBADA DE 2 CARRERAS",
            "CEBADA DE 6 CARRERAS", "AVENA", "MAIZ", "TRITICALE"]
aceitunas = ["ACEITUNA DE MESA", "ACEITUNA DE DOBLE APTITU", "ACEITUNA DE ALMAZARA"]
vinos = ["UVA DE MESA NO ESPECIFIC", "UVA DE TRANSFORMACION"]
citricos = ["NARANJO", "MANDARINO", "LIMONERO", "NARANJO AMARGO"]

grupos = {
    "Cereales": cereales,
    "Olivar": aceitunas,
    "Viñedo": vinos,
    "Cítricos": citricos
}

st.sidebar.markdown("""
    <div style='position: fixed; bottom: 40px; font-size: 14px; color: #4a3728; opacity: 0.6; text-align: center;'>
        🌿 Agricultura & Clima · 2026
    </div>
""", unsafe_allow_html=True)

pagina = st.sidebar.radio(
    "Secciones",
    ["Sobre el análisis", "Tendencias históricas", "¿Qué nos espera?", "Factores climáticos"]
)

if pagina == "Sobre el análisis":
    st.title("🌾 Impacto del Clima en la Agricultura Española")
    st.markdown("""
    *Este proyecto analiza cómo el cambio climático ha afectado a los principales 
    cultivos de España entre 2005 y 2024, y predice la evolución de la superficie 
    cultivada hasta 2029.*
    
    **Cultivos analizados:** Cereales, Olivar, Viñedo y Cítricos  
    **Provincias:** Valladolid, Valencia, Jaén y La Rioja  
    **Fuentes:** 
    - **ESYRCE** (Encuesta sobre Superficies y Rendimientos de Cultivos): datos de hectáreas cultivadas por provincia y cultivo en España
    - **AEMET** (Agencia Estatal de Meteorología): registros históricos de temperatura y precipitación de estaciones meteorológicas españolas
    - **NASA** (POWER Project): datos climáticos satelitales a nivel nacional
    - **FAOSTAT** (Organización de las Naciones Unidas para la Alimentación): estadísticas de producción agrícola a nivel mundial
    """)

elif pagina == "Tendencias históricas":
    st.header("Evolución histórica")
    st.markdown("*Descubre cómo ha cambiado la superficie dedicada a cada cultivo en España a lo largo de los últimos 20 años, " \
    "y cómo han evolucionado la temperatura y las lluvias en ese mismo período.*")
    grupo_seleccionado = st.selectbox(
        "Grupo de cultivo",
        ["Cereales", "Olivar", "Viñedo", "Cítricos"]
    )
    df_grupo = df[df["Cultivo"].isin(grupos[grupo_seleccionado])].copy()

    df_evolucion = df_grupo.groupby("Anio")["Hectareas"].sum().reset_index()
    fig = px.line(df_evolucion, x="Anio", y="Hectareas",
              title=f"Evolución de hectáreas - {grupo_seleccionado}",
              labels={"Anio": "Año", "Hectareas": "Hectáreas"},
              height=300)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        df_temp = df_grupo.groupby("Anio")["Temp_media"].mean().reset_index()
        fig_temp = px.line(df_temp, x="Anio", y="Temp_media",
                   title="Temperatura media por año",
                   labels={"Anio": "Año", "Temp_media": "°C"},
                   height=250)
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        df_prec = df_grupo.groupby("Anio")["Precipitacion"].mean().reset_index()
        fig_prec = px.line(df_prec, x="Anio", y="Precipitacion",
                   title="Precipitación media por año",
                   labels={"Anio": "Año", "Precipitacion": "mm"},
                   height=250)
        st.plotly_chart(fig_prec, use_container_width=True)

elif pagina == "¿Qué nos espera?":
    st.header("Pronóstico hasta 2029")
    st.markdown("*¿Cuántas hectáreas habrá de cada cultivo en los próximos años? Este modelo aprende de los datos históricos" \
    " y del clima para estimar la tendencia futura hasta 2029. La zona sombreada muestra el rango de posibles valores.*")
    grupo_seleccionado = st.selectbox(
        "Grupo de cultivo",
        ["Cereales", "Olivar", "Viñedo", "Cítricos"]
    )

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

    prediccion, df_hist = predecir(tuple(grupos[grupo_seleccionado]))

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
    fig_pred.update_layout(title=f"Predicción - {grupo_seleccionado}",
                            xaxis_title="Año", yaxis_title="Hectáreas")
    st.plotly_chart(fig_pred, use_container_width=True)

elif pagina == "Factores climáticos":
    st.header("Importancia de variables climáticas")
    st.markdown("*No todos los factores climáticos afectan igual a cada cultivo. Aquí puedes ver qué variables del clima, como la temperatura o las lluvias, " \
    "tienen más peso a la hora de explicar los cambios en la superficie cultivada.*")
    grupo_seleccionado = st.selectbox(
        "Grupo de cultivo",
        ["Cereales", "Olivar", "Viñedo", "Cítricos"]
    )

    features = ["Temp_media", "Temp_min", "Temp_max", "Precipitacion"]

    @st.cache_data
    def calcular_importancia(cultivos_grupo):
        df_rf = df[df["Cultivo"].isin(cultivos_grupo)].dropna(subset=["Hectareas"])
        X = df_rf[features]
        y = df_rf["Hectareas"].values
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        return pd.DataFrame({
            "Variable": features,
            "Importancia": rf.feature_importances_
        }).sort_values("Importancia", ascending=True)

    importancia = calcular_importancia(tuple(grupos[grupo_seleccionado]))

    nombres_variables = {
        "Temp_media": "Temperatura media (AEMET)",
        "Temp_min": "Temperatura mínima (AEMET)",
        "Temp_max": "Temperatura máxima (AEMET)",
        "Precipitacion": "Precipitación (AEMET)"
    }
    importancia["Variable"] = importancia["Variable"].map(nombres_variables)

    fig_imp = px.bar(
        importancia,
        x="Importancia",
        y="Variable",
        orientation="h",
        title=f"Importancia de variables climáticas - {grupo_seleccionado}"
    )
    st.plotly_chart(fig_imp, use_container_width=True)