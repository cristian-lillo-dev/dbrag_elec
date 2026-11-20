import html
import json
import os
import re
import time
import urllib.error
import urllib.request
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastembed import TextEmbedding
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
TOP_K = int(os.getenv("TOP_K", "6"))
FETCH_K = int(os.getenv("FETCH_K", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1400"))

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

if not GROQ_API_KEY:
    raise RuntimeError("Falta la variable de entorno GROQ_API_KEY")

RIC_PATTERN = re.compile(
    r"\bric\s*[-_–—]?\s*n?\s*[°º]?\s*[-_–—]?\s*0*(\d+)\b",
    re.IGNORECASE,
)

GLOSARIO_TEMAS_RIC: dict[str, str] = {
    # RIC-N01: Empalmes
    "empalme": "empalme electrico acometida caja de empalme medidor limitador limite de propiedad RIC-N01",
    "empalmes": "empalmes medidores acometida concentrado distribuido alimentador RIC-N01",
    "acometida": "acometida red de distribucion caja de empalme medidor RIC-N01",
    "medidor": "equipo de medida medidor de energia empalme concentrado distribuido RIC-N01 RIC-N14",
    "medidores": "equipo de medida empalme concentrado distribuido shaft RIC-N01",
    # RIC-N02: Tableros Eléctricos
    "tablero": "tablero electrico tablero general tablero de distribucion barra colectora envolvente RIC-N02",
    "tableros": "tableros electricos TG TGA TD envolvente gabinete IP IK barras RIC-N02",
    "barra": "barra colectora barra omnibus busbar tablero electrico RIC-N02",
    "barras": "barra colectora barra omnibus busbar tablero electrico RIC-N02",
    "gabinete": "gabinete electrico envolvente tablero IP IK RIC-N02",
    "envolvente": "envolvente gabinete barrera proteccion IP IK RIC-N02 RIC-N05",
    "interruptor general": "interruptor general disyuntor general cabecera tablero RIC-N02",
    # RIC-N03: Alimentadores y Demanda
    "alimentador": "alimentador subalimentador caida de tension 3% 5% demanda maxima RIC-N03",
    "alimentadores": "alimentadores subalimentadores caida de tension factor de demanda diversidad RIC-N03",
    "caida de tension": "caida de tension regulacion de tension 3% alimentador 5% acumulado RIC-N03",
    "demanda": "demanda maxima carga conectada factor de demanda factor de diversidad RIC-N03",
    "factor de potencia": "factor de potencia cos phi potencia activa reactiva aparente 0.93 RIC-N03 RIC-N14",
    "armonicos": "armonicos distorsion harmonica cargas no lineales neutro RIC-N03 RIC-N14",
    # RIC-N04: Conductores y Canalizaciones
    "conductor": "conductor activo neutro PE PEN seccion nominal mm2 codigo de colores RIC-N04",
    "conductores": "conductores aislados cable seccion nominal mm2 codigo de colores canalizacion RIC-N04",
    "cable": "cable cubierta chaqueta LSZH libre de halogenos resistente al fuego RIC-N04",
    "cables": "cables canalizacion bandeja portaconductores escalerilla ductos RIC-N04",
    "lszh": "cable libre de halogenos LSZH HFFR baja emision de humos lugares de reunion RIC-N04 RIC-N10",
    "halogenos": "cable libre de halogenos LSZH HFFR vias de evacuacion RIC-N04 RIC-N10",
    "canalizacion": "canalizacion electrica ducto bandeja portaconductores escalerilla moldura RIC-N04",
    "canalizaciones": "canalizaciones ductos tuberias bandeja portaconductores escalerilla camara de inspeccion RIC-N04",
    "ducto": "ducto tuberia electrica conduit PVC HDPE acero galvanizado RIC-N04",
    "ductos": "ductos tuberias canalizacion subterranea camara de inspeccion RIC-N04",
    "bandeja": "bandeja portaconductores BPC canalizacion cables RIC-N04",
    "escalerilla": "escalerilla portaconductores EPC cables potencia control RIC-N04",
    "colores": "codigo de colores obligatorio fase neutro PE verde amarillo RIC-N04",
    "codigo de colores": "codigo de colores obligatorio fase neutro PE verde amarillo RIC-N04",
    "seccion": "seccion nominal conductor mm2 IEC 60228 seccion minima RIC-N04 RIC-N10",
    "secciones": "seccion nominal conductor mm2 seccion minima alumbrado enchufes alimentadores RIC-N04 RIC-N10",
    # RIC-N05: Medidas de Protección Contra Tensiones Peligrosas
    "diferencial": "interruptor diferencial protector diferencial RCD RCCB sensibilidad 30mA tipo A tipo B RIC-N05",
    "diferenciales": "interruptores diferenciales RCD RCBO 30mA tipo A tipo B enchufes alumbrado RIC-N05 RIC-N10",
    "rcd": "interruptor diferencial protector diferencial RCCB RCBO corriente diferencial residual RIC-N05",
    "rcbo": "interruptor diferencial con proteccion contra sobrecorrientes incorporada RCBO RIC-N05 RIC-N02",
    "contacto directo": "contacto directo proteccion contra contactos directos barrera envolvente IP2X 30mA RIC-N05",
    "contactos directos": "proteccion contra contactos directos barrera envolvente IP2X 30mA RIC-N05",
    "contacto indirecto": "contacto indirecto desconexion automatica puesta a tierra clase II MBTS RIC-N05",
    "contactos indirectos": "proteccion contra contactos indirectos desconexion automatica puesta a tierra RIC-N05",
    "tension peligrosa": "tension peligrosa tension limite convencional UL 50V CA 24V humedos RIC-N05",
    "arco electrico": "arco electrico arc flash AFDD proteccion falla por arco energia incidente RIC-N02 RIC-N05 RIC-N17",
    "afdd": "dispositivo de proteccion de falla por arco electrico AFDD RIC-N02 RIC-N05",
    "dps": "dispositivo de proteccion contra sobretensiones transitorias DPS SPD tipo 1 tipo 2 tipo 3 RIC-N02 RIC-N05",
    "sobretension": "sobretension transitoria sobretension permanente POP DPS RIC-N05",
    "sobretensiones": "sobretensiones transitorias permanentes DPS POP RIC-N05",
    "cortocircuito": "cortocircuito corriente de cortocircuito capacidad de ruptura poder de corte Icu Icn RIC-N02 RIC-N05",
    "selectividad": "coordinacion de protecciones selectividad total parcial curva B C D RIC-N02 RIC-N05",
    "curva": "curva de disparo termomagnetico curva B curva C curva D IEC 60898 RIC-N02 RIC-N05",
    "esquema": "esquema de conexion a tierra TN-S TN-C-S TT IT regimen de neutro RIC-N05 RIC-N06",
    # RIC-N06: Puesta a Tierra y Enlace Equipotencial
    "tierra": "puesta a tierra electrodo malla linea de tierra conductor PE resistencia puesta a tierra RIC-N06",
    "puesta a tierra": "puesta a tierra de proteccion de servicio malla electrodo resistencia de tierra RIC-N06",
    "malla": "malla de puesta a tierra electrodo cobre desnudo resistencia de puesta a tierra RIC-N06",
    "electrodo": "electrodo de puesta a tierra malla resistividad del terreno resistencia RT RIC-N06",
    "equipotencial": "enlace equipotencial union equipotencial suplementaria masas conductoras RIC-N05 RIC-N06 RIC-N10",
    "equipotencialidad": "enlace equipotencial union equipotencial suplementaria puesta a tierra RIC-N06 RIC-N10",
    "resistividad": "resistividad del terreno ohm metro medicion puesta a tierra RIC-N06",
    "pararrayos": "pararrayos SPDA proteccion contra descargas atmosfericas puesta a tierra RIC-N06",
    # RIC-N07: Instalaciones de Equipos
    "motor": "instalacion de equipos motores proteccion diferencial sobrecarga cortocircuito RIC-N07",
    "motores": "instalacion de equipos motores proteccion diferencial curva D aparamenta RIC-N07",
    "transformador": "transformador de distribucion transformador de aislacion subestacion boveda RIC-N07 RIC-N13",
    "transformadores": "transformadores de distribucion aislacion boveda subestacion RIC-N07 RIC-N13",
    # RIC-N08: Sistemas de Emergencia
    "emergencia": "sistemas de emergencia alumbrado de emergencia grupo electrogeno UPS transferencia automatica RIC-N08",
    "alumbrado de emergencia": "alumbrado de emergencia evacuacion antipanico lux servicio de seguridad RIC-N08",
    "grupo electrogeno": "grupo electrogeno fuente de energia de emergencia transferencia automatica TTA RIC-N08",
    "ups": "sistema de alimentacion ininterrumpida UPS respaldo transferencia RIC-N08",
    "transferencia": "tablero de transferencia automatica TTA ATS fuente de emergencia RIC-N08",
    "tta": "tablero de transferencia automatica TTA ATS grupo electrogeno emergencia RIC-N08",
    # RIC-N09: Sistemas de Autogeneración
    "autogeneracion": "sistemas de autogeneracion generacion distribuida net billing inversor fotovoltaico TE4 RIC-N09",
    "fotovoltaico": "inversor fotovoltaico autogeneracion generacion distribuida net billing proteccion anti-isla RIC-N09",
    "inversor": "inversor fotovoltaico autogeneracion proteccion anti-isla sincronismo RIC-N09",
    "anti-isla": "proteccion anti-isla anti-islanding inversor autogeneracion RIC-N09",
    "net billing": "generacion distribuida net billing ley 20571 21118 autogeneracion TE4 RIC-N09",
    # RIC-N10: Instalaciones de Uso General
    "enchufe": "enchufe hembra tomacorriente circuito de enchufes 10A 16A 30mA RIC-N10",
    "enchufes": "enchufes tomacorrientes circuito de enchufes 10A 16A 30mA volumenes bano RIC-N10",
    "tomacorriente": "enchufe hembra tomacorriente circuito de enchufes 10A 16A RIC-N10",
    "tomacorrientes": "enchufes tomacorrientes circuito de enchufes 10A 16A RIC-N10",
    "alumbrado": "alumbrado circuitos de alumbrado luminaria lux iluminancia RIC-N10",
    "iluminacion": "alumbrado iluminacion luminaria lux UGR IRC RIC-N10",
    "bano": "cuarto de bano volumenes de proteccion 0 1 2 union equipotencial suplementaria 30mA RIC-N10",
    "banos": "banos duchas volumenes de proteccion 0 1 2 union equipotencial suplementaria 30mA RIC-N10",
    "ducha": "ducha banera volumenes de proteccion 0 1 2 IPX4 IPX5 IPX7 30mA RIC-N10",
    "vivienda": "instalaciones de uso general vivienda circuitos enchufes alumbrado diferencial 30mA RIC-N10",
    "local comercial": "local comercial circuitos enchufes alumbrado diferencial 30mA RIC-N10",
    # RIC-N11: Instalaciones Especiales
    "ambiente humedo": "ambiente humedo condensacion humedad relativa grados IP RIC-N10 RIC-N11",
    "ambiente mojado": "ambiente mojado lluvia chorros de agua IP54 IP65 IP66 IP67 IP68 RIC-N10 RIC-N11",
    "ambiente corrosivo": "ambiente corrosivo vapores gases agentes quimicos canalizaciones envolventes RIC-N11",
    "provisional": "instalacion provisional de faenas 30mA IK08 IP55 RIC-N11",
    "faena": "instalacion provisional de faenas proteccion diferencial 30mA IK08 IP55 RIC-N11",
    "reunion de personas": "local de reunion de personas LSZH alumbrado de emergencia verificacion periodica RIC-N10 RIC-N11",
    # RIC-N12: Ambientes Explosivos
    "explosivo": "ambientes explosivos zona clasificada 0 1 2 20 21 22 certificacion Ex RIC-N12",
    "explosivos": "instalaciones en ambientes explosivos zona clasificada 0 1 2 20 21 22 Ex RIC-N12",
    "zona clasificada": "zona clasificada ambientes explosivos gases polvos combustibles Ex RIC-N12",
    # RIC-N13: Subestaciones y Salas Eléctricas
    "subestacion": "subestacion electrica sala electrica celda de media tension transformador RIC-N13",
    "sala electrica": "sala electrica recinto electrico exclusivo resistencia al fuego ventilacion pasillos RIC-N13",
    "media tension": "media tension MT 1kV a 23kV celda de media tension subestacion RIC-N01 RIC-N13",
    # RIC-N14: Eficiencia Energética
    "eficiencia energetica": "eficiencia energetica submedicion factor de potencia gestion energetica RIC-N14",
    # RIC-N15: Vehículos Eléctricos
    "vehiculo electrico": "infraestructura de recarga de vehiculos electricos EVSE modo de carga conector TE6 RIC-N15",
    "vehiculos electricos": "recarga de vehiculos electricos EVSE modo 1 2 3 4 tipo B RDC-DD TE6 RIC-N15",
    "evse": "estacion de carga de vehiculos electricos EVSE hilo piloto modo 3 4 RIC-N15",
    "cargador": "estacion de carga EVSE vehiculo electrico diferencial tipo B RDC-DD TE6 RIC-N15",
    # RIC-N16: Subsistemas de Distribución
    "subsistema": "subsistema de distribucion red interior condominio parque industrial RIC-N16",
    # RIC-N17: Operación y Mantenimiento
    "mantenimiento": "mantenimiento electrico preventivo operacion segura cinco reglas de oro LOTO RIC-N17",
    "loto": "bloqueo y etiquetado LOTO cinco reglas de oro operacion segura RIC-N17",
    "cinco reglas de oro": "cinco reglas de oro operacion segura bloqueo etiquetado ausencia de tension RIC-N17",
    # RIC-N18: Presentación de Proyectos
    "te1": "declaracion de instalacion electrica interior TE1 planos memoria explicativa RIC-N18 RIC-N19",
    "te2": "declaracion de alumbrado publico TE2 presentacion de proyectos RIC-N18",
    "te4": "declaracion de sistemas de autogeneracion TE4 net billing RIC-N09 RIC-N18",
    "te6": "declaracion de infraestructura de recarga de vehiculos electricos TE6 RIC-N15 RIC-N18",
    "memoria explicativa": "memoria explicativa cuadros de cargas calculos justificativos planos TE1 RIC-N18",
    "planos": "planos diagrama unilineal plano as-built presentacion de proyectos TE1 RIC-N18",
    "unilineal": "diagrama unilineal planos presentacion de proyectos RIC-N18",
    # RIC-N19: Puesta en Servicio
    "puesta en servicio": "puesta en servicio verificacion inicial protocolo resistencia de aislacion bucle de falla RIC-N19",
    "verificacion inicial": "verificacion inicial inspeccion visual resistencia de aislacion impedancia de bucle RIC-N19",
    "aislacion": "resistencia de aislacion megometro megohms verificacion inicial RIC-N19",
    "bucle de falla": "impedancia del bucle de falla zs verificacion inicial puesta en servicio RIC-N05 RIC-N19",
    # Grados de protección y normativas transversales
    "ip": "grado de proteccion IP polvos liquidos IP20 IP44 IP54 IP65 IP67 IP68 RIC-N02 RIC-N10 RIC-N11",
    "ik": "grado de proteccion IK resistencia a impactos mecanicos IK00 a IK10 RIC-N02 RIC-N04",
    "lux": "lux iluminancia nivel de iluminacion alumbrado de emergencia RIC-N08 RIC-N10",
    "ds 8": "decreto supremo 8 2019 reglamento de seguridad de instalaciones de consumo pliegos RIC",
    "ds n° 8": "decreto supremo 8 2019 reglamento de seguridad de instalaciones de consumo pliegos RIC",
}


def es_consulta_corta(question: str) -> bool:
    palabras = [p for p in re.findall(r"\w+", question.lower()) if len(p) > 1]
    return len(palabras) <= 4


def expandir_consulta_para_busqueda(question: str) -> str:
    """Enriquece la consulta solo para el retriever usando alias RIC y el glosario técnico."""
    partes = [question]

    numeros = sorted({match.group(1) for match in RIC_PATTERN.finditer(question)})
    if numeros:
        aliases: list[str] = []
        for num in numeros:
            aliases.extend([f"RIC-N{num}", f"RIC N° {num}", f"RIC {num}"])
        partes.append(f"({' / '.join(dict.fromkeys(aliases))})")

    q_lower = re.sub(r"\s+", " ", question.lower().strip())
    tokens = set(re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9°º_-]+", q_lower))
    extras_agregados: list[str] = []

    for clave, expansion in GLOSARIO_TEMAS_RIC.items():
        if " " in clave and clave in q_lower:
            extras_agregados.append(expansion)
        elif " " not in clave and clave in tokens:
            extras_agregados.append(expansion)

    if extras_agregados:
        partes.append(" ".join(extras_agregados[:2]))

    return " ".join(partes)


class LocalFastEmbedEmbeddings(Embeddings):

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    def _prepare(self, texts: list[str], kind: str) -> list[str]:
        if "e5" in self.model_name.lower():
            prefix = "passage: " if kind == "doc" else "query: "
            return [f"{prefix}{t}" for t in texts]
        return texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared = self._prepare(texts, "doc")
        return [vec.tolist() for vec in self.model.embed(prepared)]

    def embed_query(self, text: str) -> list[float]:
        prepared = self._prepare([text], "query")
        first = next(self.model.embed(prepared))
        return first.tolist()


def _recortar_limpio(texto: str, limite: int = MAX_CONTEXT_CHARS) -> str:
    texto = texto.strip()
    if len(texto) <= limite:
        return texto
    recorte = texto[:limite]
    ultimo_espacio = recorte.rfind(" ")
    if ultimo_espacio > int(limite * 0.7):
        recorte = recorte[:ultimo_espacio]
    return recorte + "..."


def _limpiar_html_respuesta(raw_text: str) -> str:
    texto = raw_text.strip()
    texto = re.sub(r"^```(?:html)?\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s*```$", "", texto)

    texto = re.sub(
        r"</?(?:script|style|iframe|object|embed|link|meta|html|head|body)[^>]*>",
        "",
        texto,
        flags=re.IGNORECASE,
    )
    texto = re.sub(
        r"\s+on[a-zA-Z0-9_-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
        "",
        texto,
    )

    if not re.search(r"<(?:h3|h4|p|ul|ol|li|strong|div)\b", texto, re.IGNORECASE):
        escapado = html.escape(texto)
        escapado = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escapado)
        parrafos = [
            f"<p>{p.replace(chr(10), '<br>')}</p>"
            for p in escapado.split("\n\n")
            if p.strip()
        ]
        return "".join(parrafos) if parrafos else "<p>Sin respuesta del modelo.</p>"

    return texto


def generate_groq_answer(
    prompt: str,
    consulta_abierta: bool = False,
    model: str | None = None,
    max_retries: int = 3,
) -> str:
    active_model = model or GROQ_CHAT_MODEL
    guia_abierta = (
        "Si la pregunta es una palabra o tema amplio (por ejemplo 'enchufes'), resume de forma ordenada "
        "las condiciones, circuitos, recintos o requisitos que sí aparezcan en el contexto recuperado. "
        if consulta_abierta
        else ""
    )
    payload = {
        "model": active_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un asistente técnico de los Pliegos Técnicos Normativos RIC de la SEC (Chile). "
                    "Responde ÚNICAMENTE con la información del contexto entregado, de forma clara, técnica y estructurada. "
                    + guia_abierta
                    + "Devuelve EXCLUSIVAMENTE un fragmento HTML válido usando solo las etiquetas: "
                    "<h3>, <p>, <ul>, <ol>, <li>, <strong> y <br>. "
                    "NO uses Markdown (nada de **, #, -, ni ```html). "
                    "NO incluyas <html>, <head> ni <body>. "
                    "NO repitas la fuente ni el nombre del PDF dentro de cada viñeta, porque las fuentes se muestran aparte. "
                    "Organiza la respuesta con un <h3> breve del tema y listas <ul>/<ol> con <strong> en los conceptos clave. "
                    "Si el dato no está en el contexto o depende de una tabla/anexo incompleto, indícalo en un <p> sin inventar."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    raw_payload = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(
            GROQ_CHAT_URL,
            data=raw_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "User-Agent": "RIC-RAG-MVP/1.3",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            choices = data.get("choices", [])
            if not choices:
                return "<p>No se obtuvo respuesta del modelo.</p>"
            message = choices[0].get("message", {})
            contenido = (message.get("content") or "").strip()
            if not contenido:
                return "<p>No se obtuvo respuesta del modelo.</p>"
            return _limpiar_html_respuesta(contenido)
        except urllib.error.HTTPError as exc:
            err_text = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503) and attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise HTTPException(
                status_code=502,
                detail=f"Error al consultar Groq ({exc.code}): {err_text}",
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise HTTPException(
                status_code=504,
                detail=f"Tiempo de espera agotado al consultar Groq: {exc}",
            ) from exc

    raise HTTPException(
        status_code=502,
        detail="No se pudo obtener respuesta de Groq tras varios reintentos.",
    )


app = FastAPI(title="RIC RAG MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

embeddings = LocalFastEmbedEmbeddings(model_name=EMBED_MODEL)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)
retriever_similarity = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": FETCH_K},
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/ric/docs")
def list_ingested_docs():
    registro_path = os.path.join(PERSIST_DIR, "registro_ingesta.json")
    if not os.path.exists(registro_path):
        return {
            "mensaje": (
                "Aún no existe registro_ingesta.json. Ejecuta ingest.py"
                " primero."
            ),
            "archivos": {},
        }
    with open(registro_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/ric/ask")
def ask_ric(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="La pregunta no puede estar vacía."
        )

    corta = es_consulta_corta(question)
    consulta_busqueda = expandir_consulta_para_busqueda(question)
    active_retriever = retriever_similarity if corta else retriever_mmr

    docs = active_retriever.invoke(consulta_busqueda)
    if not docs:
        return {
            "answer": (
                "<p>No encontré información relevante en la base de datos RIC"
                " para esa consulta.</p>"
            ),
            "sources": [],
        }

    context = "\n\n---\n\n".join([
        (
            f"[{d.metadata.get('archivo', 'RIC')} - Pág."
            f" {d.metadata.get('page', 0) + 1}]\n{_recortar_limpio(d.page_content, MAX_CONTEXT_CHARS)}"
        )
        for d in docs
    ])

    prompt = f"""Contexto normativo:
{context}

Pregunta del usuario: {question}"""

    answer_html = generate_groq_answer(prompt, consulta_abierta=corta)

    sources = []
    seen = set()
    for d in docs:
        archivo = d.metadata.get("archivo", "RIC")
        pagina = d.metadata.get("page", 0) + 1
        key = (archivo, pagina)
        if key not in seen:
            seen.add(key)
            sources.append({"archivo": archivo, "pagina": pagina})

    return {"answer": answer_html, "sources": sources}