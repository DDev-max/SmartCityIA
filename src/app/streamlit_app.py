"""
Frontend para el Asistente Inteligente de Seguridad Urbana
Tema visual inspirado en el OIJ.
Ejecutar: streamlit run src/app/streamlit_app.py
"""
import streamlit as st
import sys
from pathlib import Path

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
# TAB 2: RNN
with tab_rnn:
    st.markdown('<div class="sub-header">📈 Predicción Temporal de Incidencias (RNN/LSTM)</div>', unsafe_allow_html=True)

    st.markdown("""
    Predice el número de incidentes esperados en las próximas 4 horas para una zona específica,
    basado en datos históricos del OIJ y condiciones contextuales.
    """)

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