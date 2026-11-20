import json
import os

PERSIST_DIR = os.getenv("CHROMA_DIR", "/data/db_ric")
REGISTRO_PATH = os.path.join(PERSIST_DIR, "registro_ingesta.json")


def main():
    if not os.path.exists(REGISTRO_PATH):
        print(f"No se encontró registro en {REGISTRO_PATH}")
        return

    with open(REGISTRO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Última actualización (UTC): {data.get('ultima_actualizacion_utc')}")
    print(f"Modelo embeddings:        {data.get('embed_model')}")
    print(f"Total archivos:           {data.get('total_archivos')}")
    print(f"Total fragmentos:         {data.get('total_fragmentos')}\n")

    print("=== ARCHIVOS CARGADOS ===")
    for nombre, info in data.get("archivos", {}).items():
        print(
            f"- {nombre} | Págs: {info['paginas']} | "
            f"Chunks: {info['fragmentos']} | Estado: {info['estado_ultima_ingesta']}"
        )


if __name__ == "__main__":
    main()