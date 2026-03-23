"""
Frontend para el Asistente Inteligente de Seguridad Urbana
Tema visual inspirado en el OIJ.
Ejecutar: streamlit run src/app/streamlit_app.py
"""
import datetime
from operator import truediv

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import sys
import requests
import io
from PIL import Image
from pathlib import Path

from keras.src.saving import load_model

sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from src import config
except ImportError:
    config = None

# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Asistente de Seguridad Urbana",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ESTILO PERSONALIZADO
st.markdown("""
<style>
/* VARIABLES TEMA OIJ */
:root {
    --oij-blue: #002B5C;
    --oij-gold: #D4AF37;
    --oij-dark-gray: #1E2A3A;
    --oij-medium-gray: #4A5B6E;
    --oij-light-gray: #F8F9FC;
    --primary-color: var(--oij-blue);
}

/* RESET Y MEJORAS DE LEGIBILIDAD */
body, .stApp {
    background-color: #F5F7FA; 
    color: var(--oij-dark-gray);
}
h1, h2, h3, h4, h5, h6 {
    color: var(--oij-blue);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: var(--oij-blue);
    border-right: 3px solid var(--oij-gold);
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--oij-gold) !important;
}
[data-testid="stSidebar"] .stInfo {
    background-color: rgba(255,255,255,0.1);
    border-left-color: var(--oij-gold);
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] a {
    color: var(--oij-gold) !important;
    text-decoration: none;
}
[data-testid="stSidebar"] a:hover {
    text-decoration: underline;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--oij-gold);
}
.stTabs [data-baseweb="tab"] {
    background-color: var(--oij-light-gray);
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: 0.2s;
    color: var(--oij-dark-gray);
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: var(--oij-gold);
    color: var(--oij-blue);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: var(--oij-blue);
    color: white;
    border-bottom: 2px solid var(--oij-gold);
}

/* SUB-HEADER (usado en los tabs) */
.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--oij-blue);
    border-left: 4px solid var(--oij-gold);
    padding-left: 1rem;
    margin-top: 1rem;
}
/* FOOTER */
footer, div:has(> div[style*="text-align: center"]) {
    border-top: 1px solid var(--oij-gold);
    margin-top: 2rem;
    padding-top: 1rem;
    font-size: 0.9rem;
    color: var(--oij-medium-gray);
}

[data-testid="stWidgetLabel"] p {
    color: #002B5C !important; /* Azul OIJ para los títulos */
    font-weight: bold !important;
}

div[data-baseweb="select"] * {
    color: #1E2A3A !important; /* Gris oscuro/Negro para las opciones */
}

div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    border-radius: 4px;
}

/* 1. Cambia el color del texto de la opción seleccionada */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    color: #1E2A3A !important; /* Azul oscuro/Gris */
    background-color: #FFFFFF !important; /* Fondo blanco */
}



</style>
""", unsafe_allow_html=True)



# SIDEBAR
with st.sidebar:
    st.title("⚖️ SmartCityIA")
    st.markdown("### Asistente de Seguridad Urbana para Costa Rica")
    st.markdown("---")

    st.markdown("### 👥 Equipo")
    st.info("""
    - Fernando Contreras Artavia
    - Marisol Viquez Rivera
    - Claudio Poveda Sánchez
    - Monica Mendoza Morales
    - Víctor Rojas Navarro

    *IA Aplicada - CUC - 2026*
    """)

    st.markdown("---")
    st.markdown("### 🔗 Recursos")
    st.markdown("[📖 Documentación API](http://localhost:8000/docs)")
    st.markdown("[📁 GitHub del Proyecto](#)")
    st.markdown("[📊 Dataset UCF Crime](https://www.kaggle.com/datasets/mission-ai/crimeucfdataset)")

# PÁGINA PRINCIPAL
st.title("⚖️ Asistente Inteligente para Seguridad Urbana en Costa Rica")
st.markdown("---")

st.markdown("""
Este sistema utiliza inteligencia artificial para apoyar a las autoridades de seguridad pública:
- **Detección de actividad sospechosa** en imágenes de cámaras CCTV mediante CNN.
- **Predicción temporal de incidentes** por zona y hora usando RNN/LSTM.
""")

# TABS
tab_cnn, tab_rnn, tab_info = st.tabs(["📸 Detección en Imágenes", "📈 Predicción Temporal", "📊 Acerca del Proyecto"])

# TAB 1: CNN
with tab_cnn:
    st.markdown('<div class="sub-header">📸 Detección de Actividad Sospechosa (CNN)</div>', unsafe_allow_html=True)

    st.markdown("""
    Sube una imagen de una cámara CCTV y el modelo clasificará la escena en:
    - 🟢 **Normal**  
    - 🔴 **Robo / Asalto**  
    - ⚔️ **Pelea**  
    - 👤 **Merodeo**  
    - 🚮 **Vandalismo**
    """)
    #Llamada del Api

    api_cctv = "http://localhost:8000/api/predict/cctv"
    img_carga = st.file_uploader("Selecciona una imagen de una CCTV", type=["png", "jpg", "jpeg"])

    if img_carga is not None:
        image = Image.open(img_carga)

        col1,col2 = st.columns([1,1])

        with col1:
            st.image(image, caption="Imagen de CCTV", use_container_width=True)

        with col2:
            msj = st.empty()

            msj.info("Se esta realizando el análisis de la escena..")
            #st.info("Se esta realizando el análisis de la escena..")

            if st.button("Predecir"):
                try:
                    img_byte = io.BytesIO()
                    image.save(img_byte, format=image.format)
                    img_byte = img_byte.getvalue()

                    #Peticiones de Post
                    files = {"file": (img_carga.name, img_byte, img_carga.type)}
                    response = requests.post(api_cctv, files=files)

                    if response.status_code == 200:
                        msj.empty()

                        rest =response.json()["resultado"]
                        categoria = rest["categoria"]
                        confianza = rest["confiaPrediccion"]
                        alerta = rest["alerta"]

                        st.subheader(f"Resultado obtenido: {categoria}")
                        st.write(f"Confianza del modelo: {confianza}")

                        if alerta:
                            st.error(f"🚨 ¡ALERTA DETECTADA! Posible acto de {categoria}")
                        else:
                            st.success("✅ Escena normal. No se detectan anomalías.")

                    else:
                        st.error(f"Erroren el servidor: {response.status_code}")

                except Exception as e:
                    st.error(f"No se pudo realizar la conexión a la API, tenemos este error: {e}")




# TAB 2: RNN
@st.cache_resource
def cargar_modelornn(prov):
    base_path = Path(__file__).resolve().parent.parent
    folder_models = base_path / "models"

    archivos_modelos = {
        "SAN JOSE": "LSTM-SAN JOSE.h5",
        "ALAJUELA": "LSTM-ALAJUELA.h5",
        "CARTAGO": "LSTM-CARTAGO.h5",
        "HEREDIA": "LSTM-HEREDIA.h5",
        "GUANACASTE": "LSTM-GUANACASTE.h5",
        "PUNTARENAS": "LSTM-PUNTARENAS.h5",
        "LIMON": "LSTM-LIMON.h5"
    }

    nombre_archivo = archivos_modelos.get(prov)
    if nombre_archivo:
        ruta_h5 = folder_models / nombre_archivo

        if ruta_h5.exists():
            return load_model(str(ruta_h5), compile=False)
        else:
            st.error(f"No se encontró el modelo .h5 en: {ruta_h5}")
            return None
    return None

with tab_rnn:
    st.markdown('<div class="sub-header">📈 Predicción Temporal de Incidencias (RNN/LSTM)</div>', unsafe_allow_html=True)

    st.markdown("""
    Predice el número de incidentes esperados en las próximas 4 horas para una zona específica,
    basado en datos históricos del OIJ y condiciones contextuales.
    """)

    #with st.form("form_incidencias"):
    col1, col2 = st.columns(2)
    #ZONAS
    zonas_provincia = {
            "SAN JOSE": ["CENTRO", "ESCAZU", "DESAMPARADOS", "PURISCAL", "TARRAZU", "ASSERRI", "MORA", "GOICOECHEA",
                         "SANTA ANA", "ALAJUELITA", "V VAZQUEZ DE CORONADO", "ACOSTA", "TIBAS", "MORAVIA",
                         "MONTES DE OCA", "TURRUBARES", "DOTA", "CURRIDABAT", "PEREZ ZELEDON", "LEON CORTES"],
            "CARTAGO": ["CENTRO", "PARAISO", "LA UNION", "JIMENEZ", "TURRIALBA", "ALVARADO", "OREAMUNO", "EL GUARCO"],
            "ALAJUELA": ["CENTRO", "SAN RAMON", "GRECIA", "SAN MATEO", "ATENAS", "NARANJO", "PALMARES", "POAS",
                         "OROTINA", "SAN CARLOS", "ZARCERO", "SARCHI", "UPALA", "LOS CHILES", "GUATUSO", "RIO CUARTO"],
            "HEREDIA": ["CENTRO", "BARVA", "SANTO DOMINGO", "SANTA BARBARA", "SAN RAFAEL", "ISIDRO", "BELEN", "FLORES",
                        "PABLO DE HEREDIA", "SARAPIQUI"],
            "GUANACASTE": ["LIBERIA", "NICOYA", "SANTA CRUZ", "BAGACES", "CAÑAS", "ABANGARES", "TILARAN", "NANDAYURE",
                           "LA CRUZ", "HOJANCHA"],
            "PUNTARENAS": ["CENTRO", "ESPARZA", "BUENOS AIRES", "MONTES DE ORO", "OSA", "QUEPOS", "GOLFITO",
                           "COTO BRUS", "PARRITA", "CORREDORES", "GARABITO", "MONTEVERDE", "PUERTO JIMENEZ"],
            "LIMON": ["CENTRO", "POCOCI", "SIQUIRRES", "TALAMANCA", "MATINA", "GUACIMO"]
    }

    with col1:
            provincia = st.selectbox("📍 Selecciona una provincia", list(zonas_provincia.keys()))
            zona_sel = st.selectbox("🏘️ Selecciona la Zona/Cantón", zonas_provincia[provincia])
            fecha = st.date_input("📅 Fecha", datetime.datetime.now(), disabled=True)
            hora_mostrada = st.slider("🕒 Hora actual ", 0, 23, datetime.datetime.now().hour, disabled=True)
            hora_pred = (hora_mostrada + 4) % 24

    with col2:
            clima = st.selectbox("☁️ Clima", ["Despejado", "Nublado", "Lluvioso"])
            delito = st.selectbox("📌 Tipo de Delito", [
                                    "ASALTO",
                                    "HOMICIDIO",
                                    "HURTO",
                                    "ROBO",
                                    "ROBO DE VEHICULO",
                                    "TACHA DE VEHICULO"
                                ])
            festivo = st.selectbox("¿Hoy es un día festivo?", ["Sí", "No"])
            festivo_b = True if festivo == "Sí" else False

    btn_prediccion_rnn = st.button(" Predecir..")

    if btn_prediccion_rnn:
        try:
            with st.spinner(f'Cargando modelo de {provincia} y calculando...'):

                modelo_actual = cargar_modelornn(provincia)

                if modelo_actual:
                    n_feat = modelo_actual.input_shape[-1]
                    input_total = np.zeros(n_feat, dtype=np.float32)

                    input_total[0] = float(hora_pred)
                    input_total[1] = float(festivo_b)

                    lista_zonas = zonas_provincia[provincia]
                    if zona_sel in lista_zonas:
                        idx_zona = lista_zonas.index(zona_sel) + 2
                        if idx_zona < n_feat:
                            input_total[idx_zona] = 1.0

                    lista_delitos = ["ASALTO", "HOMICIDIO", "HURTO", "ROBO", "ROBO DE VEHICULO", "TACHA DE VEHICULO"]
                    if delito in lista_delitos:
                        idx_delito = lista_delitos.index(delito) + 2 + len(lista_zonas)
                        if idx_delito < n_feat:
                            input_total[idx_delito] = 1

                    inpuRe = np.reshape(input_total, (1, 1, n_feat))
                    prediccionR = modelo_actual.predict(inpuRe)
                    prediccion = float(prediccionR[0][0])

                    riesgo = "Alto" if prediccion > 0.5 else "Bajo"

                    with col1:
                        st.markdown(f"""
                            <div style="
                                background-color: white; 
                                padding: 20px; 
                                border-radius: 10px; 
                                border-left: 5px solid #002B5C;
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <p style="color: #002B5C; margin: 0; font-weight: bold; font-size: 1.1rem;">Resultados</p>
                                <h2 style="color: black; margin: 0; font-weight: 900; font-size: 2.2rem;"></h2>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        color_riesgo = "#d9534f" if riesgo == "Alto" else "#5cb85c"
                        st.markdown(f"""
                            <div style="
                                background-color: white; 
                                padding: 20px; 
                                border-radius: 10px; 
                                border-left: 5px solid {color_riesgo};
                                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                <p style="color: #002B5C; margin: 0; font-weight: bold; font-size: 1.1rem;">Nivel de Riesgo</p>
                                <h2 style="color: {color_riesgo}; margin: 0; font-weight: 900; font-size: 2.2rem;">{riesgo}</h2>
                                <p style="color: {color_riesgo}; margin: 0; font-weight: bold; font-size: 0.9rem;">
                                    {"▲ Riesgo detectado" if riesgo == "Alto" else "▼ Sin riesgo"}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    #if riesgo == "Alto":
                        #st.warning("Se recomienda precaución y vigilancia en la zona.")
                    #else:
                        #st.success("Condiciones normales para la zona.")
                else:
                    st.error(f"No se encontró el archivo del modelo para {provincia}")

        except Exception as e:
            st.error(f"Error en la predicción directa: {str(e)}")



# TAB 3: Información
with tab_info:
    st.markdown('<div class="sub-header">📊 Acerca del Proyecto</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Objetivos")
        st.markdown("""
        - Detectar automáticamente actividades sospechosas en tiempo real a partir de imágenes CCTV.
        - Predecir zonas y horarios de mayor incidencia delictiva mediante series temporales.
        - Apoyar la toma de decisiones de seguridad pública con alertas preventivas.
        """)
        st.markdown("---")
        st.markdown("### 🧠 Modelos Implementados")
        st.markdown("""
        **CNN (Clasificación de imágenes)**
        - Arquitectura: Conv2D → MaxPool → Dropout → Softmax
        - Clases: Normal, Robo/Asalto, Pelea, Merodeo, Vandalismo
        - Dataset: UCF-Crime Dataset (frames extraídos)
        - Métrica objetivo: Accuracy ≥ 88%

        **RNN/LSTM (Predicción temporal)**
        - Arquitectura: LSTM → Dropout → Dense(1)
        - Ventana histórica: 30 días
        - Datos: Series de incidentes por zona/hora (OIJ)
        - Métricas: RMSE, MAE
        """)

    with col2:
        st.markdown("### 🔧 Tecnologías")
        st.markdown("""
        - **Python**
        - **TensorFlow / Keras** (CNN y RNN)
        - **OpenCV** (preprocesamiento de imágenes)
        - **FastAPI** (backend REST)
        - **Streamlit** (frontend interactivo)
        - **Docker** (para despliegue)
        """)
        st.markdown("---")
        st.markdown("### 📁 Estructura del Proyecto")
        st.code("""
SmartCityIA/
├── data/
│   ├── processed/
│   └── raw/
├── notebooks/
├── src/
│   ├── api/           # FastAPI endpoints
│   ├── app/           # Streamlit app
│   └── utils/         # Utilidades
├── models/            # Modelos .h5
├── .gitignore
├── README.md
└── requirements.txt
        """)

    st.markdown("---")
    st.markdown("### 🎓 Conclusiones Esperadas")
    st.markdown("""
    - Reducción en tiempos de respuesta ante incidentes.
    - Asignación proactiva de recursos policiales en zonas de alto riesgo.
    - Mejora continua mediante actualización con nuevos datos.
    """)

# FOOTER
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; font-size: 0.9rem; color: #4A5B6E;'>
© 2026 Asistente Inteligente para Seguridad Urbana | Colegio Universitario de Cartago<br>
</div>
""", unsafe_allow_html=True)