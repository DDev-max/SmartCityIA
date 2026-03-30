def agrupar_infrecuentes(df, umbrales_pct=None):
    df = df.copy()
    
    if umbrales_pct is None:
        umbrales_pct = {
            'Nacionalidad': 0.03,
            'Canton': 0.05,
        }
    
    for columna, umbral in umbrales_pct.items():
        frecuencias = df[columna].value_counts(normalize=True)
        infrecuentes = frecuencias[frecuencias < umbral].index
        
        df[columna] = df[columna].apply(
            lambda x: 'Otros' if x in infrecuentes else x
        )
    
    return df