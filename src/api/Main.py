import os
import numpy as np
import io
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

app = FastAPI("SmartCity IA - Deteccion de actividades sospechosas")

class SmartCity:
    def __init__(self):
        self.path_cctv = "./src/models/modelo_CNN_CCTV.h5"
        self_path_rnn = "./src/models/modelo_CNN_RNN.h5" # cambiar

        self.labels_cctv = ["Normal","Pelea","Robo","Vandalismo"]

        #Carga de modelos
        self.model_cctv = tf.keras.models.load_model(self.path_cctv)
        #self.model_rnn = tf.keras.models.load_model(self.self_path_rnn)

    def process_image(self, image):
        img = Image.open(io.BytesIO(image)).convert("RGB")
        img = img.resize((150, 150))
        img_array = np.array(img)/255.0
        return np.expand_dims(img_array, axis=0)

    def predict_cctv(self, image):
        data = self.process_image(image)
        prediccion = self.model_cctv.predict(data)
        idx = np.argmax(prediccion[0])

        return {
            "categoria": self.labels_cctv[idx],
            "confiaPrediccion": float(prediccion[0][idx]),
            "alerta" : self.labels_cctv[idx] != "Normal"
        }

instancia = SmartCity()

@app.post("/api/predict/cctv")
async def predict_cctv(file: UploadFile = File(...)):
    img = await file.read()
    resultado = instancia.predict_cctv(img)
    return {"tipo": "CCTV", "resultado": resultado}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

