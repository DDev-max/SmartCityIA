from pydantic import BaseModel

class Incidencia(BaseModel):
    provincia: str
    fecha: str
    hora: int
    zona: str
    tipo_delito: str
    clima: str
    festivo : bool