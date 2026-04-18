from bible_doc_extractor import recorrer_y_mostrar

from pathlib import Path
import platform
import subprocess
import shutil
import re
import importlib

from docx import Document


def extraer_docx(path_archivo: Path) -> str:
	doc = Document(path_archivo)
	return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extraer_doc_linux(path_archivo: Path) -> str:
	if shutil.which("antiword"):
		resultado = subprocess.run(
			["antiword", str(path_archivo)],
			capture_output=True
		)

		if resultado.returncode != 0:
			raise RuntimeError("Falló antiword")

		for enc in ("utf-8", "cp1252", "latin-1"):
			try:
				return resultado.stdout.decode(enc)
			except UnicodeDecodeError:
				pass

		return resultado.stdout.decode("utf-8", errors="replace")

	if shutil.which("catdoc"):
		resultado = subprocess.run(
			["catdoc", str(path_archivo)],
			capture_output=True
		)

		if resultado.returncode != 0:
			raise RuntimeError("Falló catdoc")

		for enc in ("utf-8", "cp1252", "latin-1"):
			try:
				return resultado.stdout.decode(enc)
			except UnicodeDecodeError:
				pass

		return resultado.stdout.decode("utf-8", errors="replace")

	raise RuntimeError("No encuentro ni 'antiword' ni 'catdoc'.")


def extraer_doc_windows(path_archivo: Path) -> str:
	win32_client = importlib.import_module("win32com.client")

	word = win32_client.Dispatch("Word.Application")
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


def contar_palabras(texto: str) -> int:
	palabras = re.findall(r"\b\w+\b", texto, flags=re.UNICODE)
	return len(palabras)


def concatenar_y_contar(carpeta: str) -> None:
	base = Path(carpeta)
	sistema = platform.system()
	textos = []

	for archivo in base.iterdir():
		if archivo.is_file() and archivo.suffix.lower() in {".doc", ".docx"}:
			print(f"Procesando: {archivo.name}")
			try:
				texto = extraer_texto(archivo, sistema)
				textos.append(texto)
			except Exception as e:
				print(f"Error al leer {archivo.name}: {e}")

	texto_total = "\n\n".join(textos)
	total_palabras = contar_palabras(texto_total)

	print("\n" + "=" * 60)
	print(f"Cantidad total de archivos leídos: {len(textos)}")
	print(f"Cantidad total de palabras: {total_palabras}")
	print("=" * 60)

	# Vista previa del texto concatenado
	print(texto_total[:2000])


if __name__ == "__main__":
	ruta_carpeta = "/home/felpipe/Datasets/I Vinya Vére - The New Testament in Neo-Quenya"
	# recorrer_y_mostrar(ruta_carpeta)
	concatenar_y_contar(ruta_carpeta)
	# Windows:
	# ruta_carpeta = r"C:\ruta\a\tu\carpeta"

	# Linux:
	# ruta_carpeta = "/home/tu_usuario/documentos"

	#ruta_carpeta = r"C:\ruta\a\tu\carpeta"

	import platform
	so = platform.system()
	print(so)

	if so == "Windows":
		print("Estoy en Windows")
	elif so == "Linux":
		print("Estoy en Linux")
	elif so == "Darwin":
		print("Estoy en macOS")
	else:
		print(f"SO no reconocido: {so}")

	import sys

	print(sys.platform)
