from .agrupar_infrecuentes import agrupar_infrecuentes
from .fechas import crear_ciclos, crear_variables_temporales
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preparar_features(df, umbrales=None):
    df = agrupar_infrecuentes(df, umbrales)
    df = crear_variables_temporales(df)
    df = crear_ciclos(df)
    
    df = df.drop(columns=['Fecha'])
    df['Conteo'] = df.pop('Conteo') # Para poner la columna al final del df. Se necesita para crear las ventanas

    return  df


class LSTMPipeline:
    def __init__(self, umbrales=None, n_pasos=9): # 9 = 24h
        self.umbrales = umbrales
        self.n_pasos = n_pasos
        self.col_transformer = None
        self.scaler_y = MinMaxScaler(clip=True)
        self.feature_cols = None

    def fit(self, df):
        X = preparar_features(df, self.umbrales)
        y = X.pop('Conteo').values.reshape(-1, 1)

        cat_cols = X.select_dtypes(include=['str', 'bool', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include='number').columns.tolist()
        self.feature_cols = {'cat': cat_cols, 'num': num_cols}

        self.col_transformer = ColumnTransformer([
            ('num', MinMaxScaler(clip=True), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ])

        self.col_transformer.fit(X)
        self.scaler_y.fit(y)
        return self

    def transform(self, df):
        X = preparar_features(df, self.umbrales)
        y = X.pop('Conteo').values.reshape(-1, 1)

        X_arr = self.col_transformer.transform(X)
        y_arr = self.scaler_y.transform(y)

        Xs, ys = [], []
        for i in range(self.n_pasos, len(X_arr)):
            Xs.append(X_arr[i - self.n_pasos:i])
            ys.append(y_arr[i])

        return np.array(Xs), np.array(ys)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def transform_predict(self, df):
        X = preparar_features(df, self.umbrales)

        X_arr = self.col_transformer.transform(X)
        Xs = [X_arr[i - self.n_pasos:i] for i in range(self.n_pasos, len(X_arr) + 1)]
        return np.array(Xs)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

    def save(self, path='pipeline_lstm.pkl'):
        joblib.dump(self, path)

    @staticmethod
    def load(path='pipeline_lstm.pkl'):
        return joblib.load(path)