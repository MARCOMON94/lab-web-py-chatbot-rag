import logging
import re
import time

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from chatbot import chat, obtener_documentos_indexados, obtener_historial_sesion


app = FastAPI(
    title="Chatbot RAG sobre documentos propios",
    description="API para hacer preguntas sobre documentos usando RAG",
    version="1.0.0"
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


MAX_PREGUNTA_CARACTERES = 500
MAX_PETICIONES_POR_MINUTO = 10
VENTANA_SEGUNDOS = 60

peticiones_por_ip = {}


class ChatRequest(BaseModel):
    pregunta: str
    session_id: str


def obtener_ip_cliente(request: Request) -> str:
    if request.client:
        return request.client.host

    return "ip_desconocida"


def comprobar_rate_limit(ip: str) -> None:
    ahora = time.time()

    peticiones_recientes = peticiones_por_ip.get(ip, [])

    peticiones_recientes = [
        momento for momento in peticiones_recientes
        if ahora - momento < VENTANA_SEGUNDOS
    ]

    if len(peticiones_recientes) >= MAX_PETICIONES_POR_MINUTO:
        raise HTTPException(
            status_code=429,
            detail="Demasiadas peticiones. Inténtalo de nuevo en un minuto."
        )

    peticiones_recientes.append(ahora)
    peticiones_por_ip[ip] = peticiones_recientes


def validar_pregunta(pregunta: str) -> None:
    if len(pregunta) > MAX_PREGUNTA_CARACTERES:
        raise HTTPException(
            status_code=400,
            detail=f"La pregunta no puede superar los {MAX_PREGUNTA_CARACTERES} caracteres."
        )

    if not pregunta.strip():
        raise HTTPException(
            status_code=400,
            detail="La pregunta no puede estar vacía."
        )


def detectar_datos_personales(texto: str) -> list[str]:
    datos_detectados = []

    patron_email = r"\b[\w\.-]+@[\w\.-]+\.\w+\b"
    patron_nombre = r"\b(me llamo|mi nombre es)\s+[a-záéíóúñA-ZÁÉÍÓÚÑ]+"

    if re.search(patron_email, texto):
        datos_detectados.append("email")

    if re.search(patron_nombre, texto, re.IGNORECASE):
        datos_detectados.append("nombre")

    return datos_detectados


@app.post("/chat")
def responder_chat(datos: ChatRequest, request: Request) -> dict:
    ip = obtener_ip_cliente(request)

    comprobar_rate_limit(ip)
    validar_pregunta(datos.pregunta)

    logging.info(
        "POST /chat | ip=%s | session_id=%s | longitud_pregunta=%s",
        ip,
        datos.session_id,
        len(datos.pregunta)
    )

    datos_personales = detectar_datos_personales(datos.pregunta)

    if datos_personales:
        return {
            "advertencia_privacidad": (
                "La pregunta parece contener datos personales. "
                "Por privacidad, no se ha enviado al modelo. "
                "Reformula la pregunta eliminando esos datos."
            ),
            "datos_detectados": datos_personales,
            "enviado_al_modelo": False,
            "session_id": datos.session_id
        }

    resultado = chat(
        pregunta=datos.pregunta,
        session_id=datos.session_id
    )

    resultado["enviado_al_modelo"] = True

    return resultado


@app.get("/chat/history/{session_id}")
def ver_historial(session_id: str, request: Request) -> dict:
    ip = obtener_ip_cliente(request)

    comprobar_rate_limit(ip)

    logging.info(
        "GET /chat/history/%s | ip=%s",
        session_id,
        ip
    )

    historial = obtener_historial_sesion(session_id)

    return {
        "session_id": session_id,
        "historial": historial
    }


@app.get("/documentos")
def listar_documentos(request: Request) -> dict:
    ip = obtener_ip_cliente(request)

    comprobar_rate_limit(ip)

    logging.info(
        "GET /documentos | ip=%s",
        ip
    )

    documentos = obtener_documentos_indexados()

    return {
        "documentos": documentos,
        "total": len(documentos)
    }