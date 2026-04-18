

from pathlib import Path
import platform
import subprocess
import shutil

from docx import Document


def extraer_docx(path_archivo: Path) -> str:
	doc = Document(path_archivo)
	return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def ejecutar_comando_texto(comando: list[str]) -> str:
	resultado = subprocess.run(
		comando,
		capture_output=True,
		text=True,
		encoding="latin-1",
		errors="ignore"
	)

	if resultado.returncode != 0:
		raise RuntimeError(resultado.stderr.strip() or "Falló el comando externo")

	return resultado.stdout


def extraer_doc_linux(path_archivo: Path) -> str:
	# if shutil.which("antiword"):
	# 	return ejecutar_comando_texto(["antiword", str(path_archivo)])

	# if shutil.which("catdoc"):
	# 	return ejecutar_comando_texto(["catdoc", str(path_archivo)])

	# raise RuntimeError("No encuentro ni 'antiword' ni 'catdoc'.")
	if not shutil.which("antiword"):
		raise RuntimeError("No encuentro 'antiword'.")

	resultado = subprocess.run(
		["antiword", str(path_archivo)],
		capture_output=True
	)

	if resultado.returncode != 0:
		raise RuntimeError(resultado.stderr.decode("utf-8", errors="ignore"))

	for enc in ("utf-8", "cp1252", "latin-1"):
		try:
			return resultado.stdout.decode(enc)
		except UnicodeDecodeError:
			pass

	return resultado.stdout.decode("utf-8", errors="replace")

def extraer_doc_windows(path_archivo: Path) -> str:
	import win32com.client  # pyright: ignore[reportMissingImports,reportMissingModuleSource]

	word = win32com.client.Dispatch("Word.Application")
	word.Visible = False

	try:
		doc = word.Documents.Open(str(path_archivo))
		texto = doc.Content.Text
		doc.Close(False)
		return texto
	finally:
		word.Quit()

def extraer_texto(path_archivo: Path, sistema: str) -> str:
	ext = path_archivo.suffix.lower()

	if ext == ".docx":
		return extraer_docx(path_archivo)

	if ext == ".doc":
		if sistema == "Linux":
			return extraer_doc_linux(path_archivo)
		if sistema == "Windows":
			return extraer_doc_windows(path_archivo)

	raise RuntimeError(f"No se soporta {ext} en {sistema}")


def recorrer_y_mostrar(carpeta: str) -> None:
	base = Path(carpeta)
	sistema = platform.system()

	print(f"SO detectado: {sistema}")
	print(f"Carpeta: {base.resolve()}")
	print()

	for archivo in base.iterdir():
		if archivo.is_file():
			print(f"Archivo encontrado: {archivo.name}")

	print("\n" + "=" * 60 + "\n")

	for archivo in base.iterdir():
		if archivo.is_file() and archivo.suffix.lower() in {".doc", ".docx"}:
			print(f"Procesando: {archivo.name}")
			try:
				texto = extraer_texto(archivo, sistema)
				print(texto[:1500])
				print("-" * 40)
			except Exception as e:
				print(f"Error al leer {archivo.name}: {e}")
				print("-" * 40)