# 🌾 Impacto del Clima en la Agricultura Española

> Análisis y predicción de la superficie cultivada de los principales cultivos españoles mediante variables climáticas (2005-2024)

## 🌐 Aplicación

Accede a la aplicación interactiva en:
👉 [https://k5vk467mndl4bcv9vsshsc.streamlit.app]

## 📋 Descripción

Este proyecto analiza cómo el cambio climático ha afectado a los principales cultivos de España entre 2005 y 2024, e integra datos de múltiples fuentes oficiales para construir un pipeline completo de extracción, procesamiento, análisis y predicción, culminando en una aplicación interactiva desarrollada con Streamlit.

## 🎯 Objetivos

- Recopilar y procesar datos climáticos históricos de 4 provincias españolas representativas
- Integrar datos de superficie cultivada con variables meteorológicas
- Realizar un análisis exploratorio (EDA) para identificar patrones y correlaciones entre clima y agricultura
- Desarrollar modelos de serie temporal para predecir la evolución de la superficie cultivada hasta 2029
- Crear una aplicación interactiva para visualización y análisis

## 📊 Fuentes de datos

| Fuente | Descripción | Acceso | Cobertura |
|--------|-------------|--------|-----------|
| **AEMET** | Agencia Estatal de Meteorología - Datos climáticos históricos | API oficial (api.aemet.es) | Jaén, Logroño, Valencia, Valladolid |
| **NASA POWER** | Datos climáticos globales (temperatura, precipitación) | Descarga directa (power.larc.nasa.gov) | Datos nacionales |
| **FAOSTAT** | Base de datos FAO - Producción agrícola España | Descarga directa (fao.org) | Nacional por cultivo |
| **ESYRCE** | Encuesta sobre Superficies y Rendimientos de Cultivos | Descarga directa (mapa.gob.es) | Provincial por cultivo |

> **Nota**: FAOSTAT y NASA se analizaron en el EDA pero no se incorporaron al modelo predictivo. FAOSTAT tiene datos solo a nivel nacional (no provincial) y NASA resultó menos representativa que AEMET para el análisis agrícola provincial en España.

## 🌿 Cultivos analizados

- **Cereales**: Trigo, Cebada, Avena, Maíz, Triticale
- **Olivar**: Aceituna de mesa, de almazara y doble aptitud
- **Viñedo**: Uva de mesa y de transformación
- **Cítricos**: Naranjo, Mandarino, Limonero

## 🏗️ Estructura del proyecto

```
.
├── 0. Creación_CSVs_con_API_AEMET.ipynb    # Extracción de datos AEMET via API
├── 1. Exploración_limpieza.ipynb           # Limpieza y preprocesamiento
├── 2. SQL.ipynb                            # Carga y consultas en TiDB
├── 3. EDA.ipynb                            # Análisis exploratorio
├── 4. Machine Learning.ipynb               # Modelos predictivos
├── 5. app.py                               # Aplicación Streamlit
│
├── data/
│   ├── FAOSTAT_data_es_*.csv
│   ├── POWER_Point_Monthly_*.csv
│   └── AEMET/
│       ├── jaen/
│       ├── logroño/
│       ├── tablas_limpias/
│       ├── valencia/
│       └── valladolid/
│
├── DF_procesados/
│   ├── AEMET_procesado.csv
│   ├── ESYRCE_procesado.csv
│   ├── FAOSTAT_procesado.csv
│   └── NASA_procesado.csv
│
├── requirements.txt                        # Dependencias del proyecto
├── credenciales.py                         # Config BD (no compartir)
├── apikey.env                              # API keys (no compartir)
└── LICENSE
```

## 🚀 Instalación y uso

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/Impacto-del-clima-en-la-agricultura-.git
cd Impacto-del-clima-en-la-agricultura-
```

2. Crear entorno virtual e instalar dependencias:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Si necesitas regenerar el archivo `requirements.txt`:
> ```bash
> pip freeze > requirements.txt
> ```

3. Configurar credenciales:
   - Obtener API key en [AEMET OpenData](https://www.aemet.es/es/datos_abiertos) y añadirla en `apikey.env`
   - Actualizar `credenciales.py` con los datos de conexión a TiDB

4. Ejecutar los notebooks en orden (0 → 4)

5. Lanzar la aplicación:
```bash
streamlit run "5. app.py"
```

## 📈 Resultados del modelo

Se desarrollaron modelos de serie temporal con Prophet, uno por grupo de cultivo, incorporando temperatura y precipitación como variables externas:

| Cultivo | R² | Tendencia 2005-2024 | Predicción 2029 |
|---------|-----|---------------------|-----------------|
| Cereales | 0.595 | Bajista | Continúa bajando |
| Olivar | 0.993 | Alcista | Continúa subiendo |
| Viñedo | 0.682 | Recuperación desde 2012 | Continúa subiendo |
| Cítricos | 0.865 | Bajista | Continúa bajando |

## 🔍 Hallazgos clave

- La **temperatura máxima** es el factor climático más determinante para el olivar y los cítricos
- Las **heladas** (temperatura mínima) son el factor crítico para los cereales
- El **olivar** es el cultivo más estable y predecible (R²=0.993)
- Las **temperaturas en España han subido** de forma sostenida desde 2015
- La fuente **AEMET es más representativa** que NASA para el análisis agrícola provincial

## 🔐 Seguridad

⚠️ Los archivos `credenciales.py` y `apikey.env` contienen información sensible y no deben subirse al repositorio. Están incluidos en `.gitignore`.

## 📚 Dependencias principales

- `pandas`, `numpy` - Manipulación de datos
- `scikit-learn` - Random Forest para importancia de variables
- `prophet` - Modelos de serie temporal
- `plotly`, `streamlit` - Visualización interactiva
- `mysql-connector-python` - Conexión a TiDB
- `requests` - Extracción de datos via API

## 📝 Licencia

MIT License. Ver archivo [LICENSE](LICENSE) para detalles.

## ✍️ Autor

Rosa Sierra· Junio 2026

---

**Última actualización**: Junio 2026
