import argparse
import json
import os
import sqlite3
import sys
import time

from colorama import Fore, Style, init as colorama_init
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.traceback import install as install_rich_traceback
from rich.tree import Tree

# Inicializa Colorama para compatibilidad ANSI en Windows y terminales clásicas
colorama_init(autoreset=True)

# Activa tracebacks legibles de Rich
install_rich_traceback(show_locals=False)

console = Console()


def banner_estado(texto: str, nivel: str = "info") -> None:
    colores = {
        "ok": Fore.GREEN + Style.BRIGHT,
        "warn": Fore.YELLOW + Style.BRIGHT,
        "err": Fore.RED + Style.BRIGHT,
        "info": Fore.CYAN + Style.BRIGHT,
    }
    prefijo = colores.get(nivel, Fore.CYAN)
    print(f"{prefijo}[SQLite-CLI]{Style.RESET_ALL} {texto}")


def conectar(db_path: str) -> sqlite3.Connection:
    if not os.path.exists(db_path):
        banner_estado(f"No se encontró la base de datos: {db_path}", nivel="err")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def obtener_tablas(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name ASC;
        """
    )
    return [row["name"] for row in cursor.fetchall()]


def contar_filas(conn: sqlite3.Connection, tabla: str) -> int:
    cursor = conn.execute(f'SELECT COUNT(*) AS total FROM "{tabla}";')
    row = cursor.fetchone()
    return int(row["total"]) if row else 0


def formatear_valor(valor, max_len: int) -> str:
    if valor is None:
        return "[dim italic]NULL[/dim italic]"
    if isinstance(valor, bytes):
        return f"[magenta]<BLOB {len(valor)} B>[/magenta]"
    if isinstance(valor, (int, float)):
        return f"[bold yellow]{valor}[/bold yellow]"

    texto = str(valor).replace("\r\n", " ").replace("\n", " ").strip()
    if len(texto) > max_len:
        return texto[: max_len - 1] + "…"
    return texto


def _obtener_archivos_desde_sqlite(conn: sqlite3.Connection) -> dict[str, dict]:
    tablas = obtener_tablas(conn)
    if "embedding_metadata" not in tablas:
        return {}

    cols = {
        row["name"]
        for row in conn.execute('PRAGMA table_info("embedding_metadata");').fetchall()
    }
    if not {"key", "string_value"}.issubset(cols):
        return {}

    rows = conn.execute(
        """
        SELECT string_value AS archivo, COUNT(*) AS fragmentos
        FROM embedding_metadata
        WHERE key = 'archivo' AND string_value IS NOT NULL
        GROUP BY string_value
        ORDER BY string_value ASC;
        """
    ).fetchall()

    archivos: dict[str, dict] = {}
    for r in rows:
        archivos[r["archivo"]] = {
            "paginas": "-",
            "fragmentos": r["fragmentos"],
            "estado_ultima_ingesta": "en_chroma",
        }
    return archivos


def mostrar_documentos_ingresados(conn: sqlite3.Connection, db_path: str) -> bool:
    carpeta_db = os.path.dirname(os.path.abspath(db_path))
    registro_path = os.path.join(carpeta_db, "registro_ingesta.json")

    archivos: dict[str, dict] = {}
    ultima_act = None
    modelo_emb = None

    if os.path.exists(registro_path):
        try:
            with open(registro_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            archivos = data.get("archivos", {}) or {}
            ultima_act = data.get("ultima_actualizacion_utc")
            modelo_emb = data.get("embed_model")
        except (OSError, json.JSONDecodeError) as exc:
            banner_estado(f"No se pudo leer {registro_path}: {exc}", nivel="warn")

    if not archivos:
        archivos = _obtener_archivos_desde_sqlite(conn)

    if not archivos:
        return False

    tabla_docs = Table(
        title="Documentos ingresados",
        box=box.ROUNDED,
        border_style="green",
        header_style="bold green",
    )
    tabla_docs.add_column("#", style="dim", justify="right", width=4)
    tabla_docs.add_column("Documento / Archivo", style="bold cyan")
    tabla_docs.add_column("Páginas", justify="right", style="yellow")
    tabla_docs.add_column("Fragmentos", justify="right", style="bold white")
    tabla_docs.add_column("Estado", justify="center")

    estilos_estado = {
        "nuevo": "[bold green]nuevo[/bold green]",
        "modificado": "[bold yellow]modificado[/bold yellow]",
        "sin_cambios": "[dim cyan]sin_cambios[/dim cyan]",
        "en_chroma": "[cyan]en_chroma[/cyan]",
    }

    for idx, (nombre, info) in enumerate(sorted(archivos.items()), start=1):
        paginas = info.get("paginas", "-")
        fragmentos = info.get("fragmentos", "-")
        estado_raw = str(info.get("estado_ultima_ingesta", "-"))
        estado_fmt = estilos_estado.get(estado_raw, f"[white]{estado_raw}[/white]")

        pag_str = f"{paginas:,}" if isinstance(paginas, int) else str(paginas)
        frag_str = f"{fragmentos:,}" if isinstance(fragmentos, int) else str(fragmentos)

        tabla_docs.add_row(str(idx), nombre, pag_str, frag_str, estado_fmt)

    subtitulo = []
    if ultima_act:
        subtitulo.append(f"Actualizado: {ultima_act}")
    if modelo_emb:
        subtitulo.append(f"Embeddings: {modelo_emb}")

    console.print(tabla_docs)
    if subtitulo:
        console.print(f"[dim]{' | '.join(subtitulo)}[/dim]")
    return True


def mostrar_resumen_y_arbol(conn: sqlite3.Connection, db_path: str) -> None:
    tablas = obtener_tablas(conn)
    if not tablas:
        console.print(
            Panel(
                f"La base [cyan]{db_path}[/cyan] no contiene tablas de usuario.",
                title="Base vacía",
                border_style="yellow",
            )
        )
        return

    resumen = Table(
        title=f"Resumen de tablas — {os.path.basename(db_path)}",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
    )
    resumen.add_column("#", style="dim", justify="right", width=4)
    resumen.add_column("Tabla", style="bold green")
    resumen.add_column("Columnas", justify="right", style="yellow")
    resumen.add_column("Filas", justify="right", style="bold white")

    arbol = Tree(f"[bold bright_blue]📁 {os.path.basename(db_path)}[/bold bright_blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Analizando estructura SQLite...", total=len(tablas))

        for idx, nombre in enumerate(tablas, start=1):
            cols = conn.execute(f'PRAGMA table_info("{nombre}");').fetchall()
            total_filas = contar_filas(conn, nombre)
            resumen.add_row(str(idx), nombre, str(len(cols)), f"{total_filas:,}")

            rama_tabla = arbol.add(
                f"[bold green]🗂 {nombre}[/bold green] [dim]({total_filas:,} filas)[/dim]"
            )
            for c in cols:
                pk_tag = " [bold green][PK][/bold green]" if c["pk"] else ""
                nn_tag = " [red]*[/red]" if c["notnull"] else ""
                tipo = c["type"] or "ANY"
                rama_tabla.add(
                    f"[cyan]{c['name']}[/cyan] [yellow]({tipo})[/yellow]{nn_tag}{pk_tag}"
                )

            progress.advance(task)
            time.sleep(0.02)

    console.print(resumen)
    console.print()
    mostro_docs = mostrar_documentos_ingresados(conn, db_path)
    if mostro_docs:
        console.print()
    console.print(Panel(arbol, title="Árbol de esquema", border_style="cyan", expand=False))


def mostrar_esquema_tabla(conn: sqlite3.Connection, tabla: str) -> None:
    cols = conn.execute(f'PRAGMA table_info("{tabla}");').fetchall()
    if not cols:
        banner_estado(f"La tabla '{tabla}' no existe.", nivel="err")
        return

    ddl_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?;",
        (tabla,),
    ).fetchone()

    schema_table = Table(
        title=f"Esquema de '{tabla}'",
        box=box.SIMPLE_HEAVY,
        border_style="magenta",
        header_style="bold magenta",
    )
    schema_table.add_column("CID", justify="right", style="dim", width=4)
    schema_table.add_column("Columna", style="bold cyan")
    schema_table.add_column("Tipo", style="yellow")
    schema_table.add_column("Not Null", justify="center")
    schema_table.add_column("Default", style="dim")
    schema_table.add_column("PK", justify="center", style="bold green")

    for c in cols:
        schema_table.add_row(
            str(c["cid"]),
            str(c["name"]),
            str(c["type"] or "ANY"),
            "[green]SÍ[/green]" if c["notnull"] else "[dim]no[/dim]",
            str(c["dflt_value"]) if c["dflt_value"] is not None else "-",
            "[bold green]PK[/bold green]" if c["pk"] else "",
        )

    console.print(schema_table)

    if ddl_row and ddl_row["sql"]:
        syntax = Syntax(ddl_row["sql"], "sql", theme="monokai", line_numbers=False, word_wrap=True)
        console.print(Panel(syntax, title="DDL CREATE TABLE", border_style="dim", expand=False))


def mostrar_filas(
    conn: sqlite3.Connection,
    tabla: str,
    limit: int = 20,
    offset: int = 0,
    max_len: int = 60,
    query_sql: str | None = None,
) -> None:
    tablas = obtener_tablas(conn)
    if not query_sql and tabla not in tablas:
        banner_estado(f"La tabla '{tabla}' no existe en la base de datos.", nivel="err")
        return

    if query_sql:
        syntax = Syntax(query_sql.strip(), "sql", theme="monokai", word_wrap=True)
        console.print(Panel(syntax, title="Consulta ejecutada", border_style="blue", expand=False))
        cursor = conn.execute(query_sql)
        titulo = "Resultados de consulta SQL"
    else:
        total = contar_filas(conn, tabla)
        cursor = conn.execute(
            f'SELECT * FROM "{tabla}" LIMIT ? OFFSET ?;',
            (limit, offset),
        )
        inicio = offset + 1 if total > 0 else 0
        fin = min(offset + limit, total)
        titulo = f"Tabla '{tabla}' ({inicio}..{fin} de {total:,} filas)"

    rows = cursor.fetchall()
    if not rows:
        banner_estado(f"Sin registros para mostrar en '{tabla}'.", nivel="warn")
        return

    columnas = rows[0].keys()
    grid = Table(
        title=titulo,
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold white on blue",
        row_styles=["", "on #161b22"],
        expand=False,
    )

    for col in columnas:
        grid.add_column(str(col), overflow="fold")

    for r in rows:
        grid.add_row(*[formatear_valor(r[col], max_len) for col in columnas])

    console.print(grid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualizador SQLite en consola usando Rich y Colorama."
    )
    parser.add_argument("db_path", help="Ruta al archivo .sqlite o .db")
    parser.add_argument("-t", "--table", help="Nombre de la tabla a inspeccionar")
    parser.add_argument("-s", "--schema", action="store_true", help="Mostrar columnas y DDL de la tabla")
    parser.add_argument("-d", "--docs", action="store_true", help="Mostrar lista de documentos ingresados")
    parser.add_argument("-n", "--limit", type=int, default=20, help="Filas máximas a mostrar (default: 20)")
    parser.add_argument("--offset", type=int, default=0, help="Desplazamiento de filas (default: 0)")
    parser.add_argument("--max-len", type=int, default=60, help="Máximo de caracteres por celda (default: 60)")
    parser.add_argument("-q", "--sql", help="Consulta SELECT personalizada con resaltado de sintaxis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tamano_kb = os.path.getsize(args.db_path) / 1024 if os.path.exists(args.db_path) else 0
    banner_estado(f"Conectando a '{args.db_path}' ({tamano_kb:,.1f} KB)...", nivel="ok")

    conn = conectar(args.db_path)
    try:
        if args.sql:
            if not args.sql.strip().lower().startswith("select"):
                banner_estado("Solo se permiten consultas de lectura (SELECT).", nivel="err")
                sys.exit(1)
            mostrar_filas(conn, args.table or "query", max_len=args.max_len, query_sql=args.sql)
            return

        if args.docs:
            if not mostrar_documentos_ingresados(conn, args.db_path):
                banner_estado("No se encontraron documentos ingresados.", nivel="warn")
            return

        if not args.table:
            mostrar_resumen_y_arbol(conn, args.db_path)
            return

        if args.schema:
            mostrar_esquema_tabla(conn, args.table)

        mostrar_filas(
            conn,
            args.table,
            limit=args.limit,
            offset=args.offset,
            max_len=args.max_len,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()