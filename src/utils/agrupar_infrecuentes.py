def agrupar_infrecuentes(df, umbrales_pct):
    for columna, umbral in umbrales_pct.items():
        frecuencias = df[columna].value_counts(normalize=True)
        infrecuentes = frecuencias[frecuencias < umbral].index
        df[columna] = df[columna].apply(lambda x: 'Otros' if x in infrecuentes else x)
    
    return df