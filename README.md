# SmartCityIA – Asistente Inteligente para Seguridad Urbana en Costa Rica
👥 Equipo
- **Fernando Contreras Artavia**
- **Marisol Viquez Rivera**
- **Claudio Poveda Sánchez**
- **Monica Mendoza Morales**
- **Víctor Rojas Navarro**

*IA Aplicada - CUC - 2026*

📋 Descripción del Proyecto
Este sistema utiliza inteligencia artificial para apoyar a las autoridades de seguridad pública:
- **Detección de actividad sospechosa** en imágenes de cámaras CCTV mediante CNN.
- **Predicción temporal de incidentes** por zona y hora usando RNN/LSTM.

🎯 Objetivos
- Detectar automáticamente actividades sospechosas en tiempo real a partir de imágenes CCTV.
- Predecir zonas y horarios de mayor incidencia delictiva mediante series temporales.
- Apoyar la toma de decisiones de seguridad pública con alertas preventivas.

🧠 Modelos Implementados

CNN (Clasificación de imágenes)
- Arquitectura: Conv2D → MaxPool → Dropout → Softmax
- Clases: Normal, Robo/Asalto, Pelea, Merodeo, Vandalismo
- Dataset: UCF-Crime Dataset (frames extraídos)
- Métrica objetivo: Accuracy ≥ 88%

RNN/LSTM (Predicción temporal)
- Arquitectura: LSTM → Dropout → Dense(1)
- Ventana histórica: 30 días
- Datos: Series de incidentes por zona/hora (OIJ)
- Métricas: RMSE, MAE

📊 Datasets

UCF-Crime Dataset (CNN)
- **Fuente**: University of Central Florida / Kaggle
- **URL**: [Kaggle](https://www.kaggle.com/datasets/mission-ai/crimeucfdataset) | [UCF](https://crcv.ucf.edu/research/real-world-anomaly-detection)
- **Contenido**: Más de 1,900 videos de vigilancia.

Datos OIJ (RNN/LSTM)
- **Fuente**: Poder Judicial de Costa Rica – Organismo de Investigación Judicial
- **URL**: [Estadísticas OIJ](https://pjenlinea3.poder-judicial.go.cr/estadisticasoij)
- **Variables**: Fecha, hora, zona, tipo de delito, condición climática, día festivo.

🔧 Instalación

Requisitos Previos
- Python 3.9+

Ejecutar Frontend (Streamlit)
streamlit run src/app/streamlit_app.py

📁 Estructura del Proyecto
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

🛠️ Tecnologías
Python
TensorFlow / Keras (CNN y RNN)
OpenCV (preprocesamiento de imágenes)
FastAPI (backend REST)
Streamlit (frontend interactivo)
Docker (para despliegue)

🎓 Conclusiones Esperadas
Reducción en tiempos de respuesta ante incidentes.
Asignación proactiva de recursos policiales en zonas de alto riesgo.
Mejora continua mediante actualización con nuevos datos.

Proyecto desarrollado para el curso de Inteligencia Artificial Aplicada
Colegio Universitario de Cartago (CUC) – 2026
Fecha de entrega: 23 de marzo




- pip

