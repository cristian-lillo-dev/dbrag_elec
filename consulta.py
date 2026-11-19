import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://dbragelec-production.up.railway.app/api/ric/ask"


def main():
    if len(sys.argv) < 2 or not " ".join(sys.argv[1:]).strip():
        print('Uso: python consulta.py "tu pregunta aquí"')
        sys.exit(1)

    question = " ".join(sys.argv[1:]).strip()
    url = os.getenv("RIC_API_URL", DEFAULT_URL)
    payload = json.dumps({"question": question}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status = resp.status
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} desde {url}:\n{body}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Error de conexión hacia {url}: {exc.reason}")
        sys.exit(1)

    print(f"HTTP {status}\n")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(raw)
        sys.exit(0)

    print("=== RESPUESTA ===")
    print(data.get("answer", "(sin campo 'answer')"))

    sources = data.get("sources", [])
    if sources:
        print("\n=== FUENTES ===")
        for s in sources:
            print(f"- {s.get('archivo', 'RIC')} | Pág. {s.get('pagina', '?')}")


if __name__ == "__main__":
    main()