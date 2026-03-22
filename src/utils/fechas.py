import numpy as np
import pandas as pd
import holidays


def crear_ciclos(df_completo):
    n_bloques = 24 / len(df_completo['Hora'].unique())
    df_completo['Bloque_hora'] = df_completo['Hora'] // n_bloques
    df_completo['hora_sin'] = np.sin(2 * np.pi * df_completo['Bloque_hora'] / 6)
    df_completo['hora_cos'] = np.cos(2 * np.pi * df_completo['Bloque_hora'] / 6)
    df_completo['Mes_sin'] = np.sin(2 * np.pi * df_completo['Mes'] / 12)
    df_completo['Mes_cos'] = np.cos(2 * np.pi * df_completo['Mes'] / 12)
    df_completo = df_completo.drop(columns=['Bloque_hora', 'Hora', 'Mes'])

    return df_completo


def crear_variables_temporales(df_rellenado, shift = 9):
    df_rellenado['Mes'] = df_rellenado['Fecha'].dt.month
    df_rellenado['Anio'] = df_rellenado['Fecha'].dt.year
    df_rellenado['dia'] = df_rellenado['Fecha'].dt.day
    df_rellenado['dia_sem'] = df_rellenado['Fecha'].dt.day_of_week
    df_rellenado['fin_semana'] = df_rellenado['dia'] >= 5

    lista_anios = np.sort(df_rellenado['Anio'].unique())
    cr_holidays = holidays.CountryHoliday('CR', years=lista_anios)
    df_rellenado['es_festivo'] = df_rellenado['Fecha'].apply(lambda x: x.date() in cr_holidays)

    df_rellenado['lag_conteo1'] = df_rellenado['Conteo'].shift(1)
    df_rellenado[f'lag_conteo{shift}'] = df_rellenado['Conteo'].shift(shift)
    df_rellenado = df_rellenado.dropna()
    
    return df_rellenado


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