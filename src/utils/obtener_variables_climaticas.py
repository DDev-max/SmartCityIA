import requests
import pandas as pd

def obtener_variables_climaticas(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str],
    interval_hours: int = 4
):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    df_avg = df.resample(f"{interval_hours}h").mean()

    return df_avg
