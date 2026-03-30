import numpy as np
import pandas as pd
import holidays


def crear_ciclos(df):
    df = df.copy()
    n_bloques = 24 / 8 #  df['Hora'].unique() => 8 
    df['Bloque_hora'] = df['Hora'] // n_bloques
    df['hora_sin'] = np.sin(2 * np.pi * df['Bloque_hora'] / 6)
    df['hora_cos'] = np.cos(2 * np.pi * df['Bloque_hora'] / 6)
    df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
    df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)
    df = df.drop(columns=['Bloque_hora', 'Hora', 'Mes'])
    return df


def crear_variables_temporales(df):
    df = df.copy()
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Mes'] = df['Fecha'].dt.month
    df['Anio'] = df['Fecha'].dt.year
    df['dia_sem'] = df['Fecha'].dt.day_of_week
    df['fin_semana'] = (df['dia_sem'] >= 5).astype(int)

    lista_anios = np.sort(df['Anio'].unique())
    cr_holidays = holidays.CountryHoliday('CR', years=lista_anios)
    df['es_festivo'] = df['Fecha'].apply(lambda x: x.date() in cr_holidays)

    return df


def rellenar_zona(df_zona):
    df_zona['Fecha'] = pd.to_datetime(df_zona['Fecha'])

    horas_posibles = df_zona['Hora'].unique()
    fechas_completas = pd.date_range(df_zona["Fecha"].min(), df_zona["Fecha"].max(), freq="D")

    idx_completo = pd.MultiIndex.from_product(
        [fechas_completas, horas_posibles],
        names=["Fecha", "Hora"]
    ).to_frame(index=False)

    df_completo = idx_completo.merge(
        df_zona,
        on=["Fecha", "Hora"],
        how="left"
    ).fillna({"Conteo": 0})

    df_completo["Conteo"] = df_completo["Conteo"].astype(int)
    df_completo = df_completo.fillna('Sin delitos')


    return df_completo.sort_values(["Fecha", "Hora"]).reset_index(drop=True)