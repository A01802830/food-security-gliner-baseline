"""
src/ner/gliner_ner.py

Extrae entidades candidatas usando GLiNER.
Misma interfaz que SpacyNER: recibe texto, devuelve EntitySpan / DocumentResult.

Modelos recomendados:
  - urchade/gliner_multi-v2.1			(multi-lingual information extraction)
  - urchade/gliner_multi_pii-v1			(detect and mask personal data across documents)
  - knowledgator/gliner-bi-small-v1.0	(best for handling many entity types, lighter but less accurate)

Instalación:
  pip install gliner

Para más información: https://urchade.github.io/GLiNER/
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

# Importación lazy para dar un error claro si gliner no está instalado
try:
	from gliner import GLiNER
except ImportError:
	raise ImportError(
		"GLiNER no está instalado.\n"
		"Instálalo con:  pip install gliner\n"
		"Documentación:  https://github.com/urchade/GLiNER"
	)

# Reutilizamos los mismos tipos que spacy_ner para interfaz uniforme
from .spacy_ner import EntitySpan, DocumentResult


# ---------------------------------------------------------------------------
# Etiquetas que pedimos a GLiNER
# A diferencia de spaCy, GLiNER recibe etiquetas en lenguaje natural.
# Podemos ser muy específicos con el dominio.
# ---------------------------------------------------------------------------

DEFAULT_LABELS: list[str] = [
	"persona",           # nombres propios de personas
	"lugar",             # ciudades, estados, comunidades, ranchos
	"organización",      # ONGs, programas de gobierno, empresas
]

# Mapa de etiqueta GLiNER (lowercase) → etiqueta interna del proyecto
LABEL_MAP: dict[str, str] = {
	"persona":       "PERSONA",
	"lugar":         "LUGAR",
	"organización":  "ORGANIZACION",
	"organizacion":  "ORGANIZACION",   # por si el modelo omite la tilde
	"org":           "ORGANIZACION",
}

# Umbral de confianza por defecto (GLiNER devuelve scores 0-1)
DEFAULT_THRESHOLD = 0.40   # más bajo que el default de GLiNER (0.5)
						   # para capturar más candidatos; el LLM filtra FP

# Modelo por defecto
DEFAULT_GLINER_MODEL = "urchade/gliner_multi_pii-v1"


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class GlinerNER:
	"""
	Wrapper de GLiNER para extracción de entidades candidatas.

	Uso básico:
		ner = GlinerNER()
		result = ner.process("Mi vecina Esperanza vive en San Cristóbal.")
		for ent in result.entities:
			print(ent.text, ent.label, ent.score)

	Uso en lote:
		results = ner.process_batch(textos, doc_ids=ids)

	Nota sobre chunks:
		GLiNER tiene un límite de tokens por llamada (~512 tokens).
		Esta clase parte automáticamente textos largos en fragmentos
		solapados para no perder entidades en los cortes.
	"""

	def __init__(
		self,
		model: str = DEFAULT_GLINER_MODEL,	# Por default
		labels: Optional[list[str]] = None,
		threshold: float = DEFAULT_THRESHOLD,
		chunk_size: int = 300,      # tokens aproximados por chunk
		chunk_overlap: int = 40,    # tokens de solapamiento entre chunks
		device: Optional[str] = None,
	):
		"""
		Args:
			model: Nombre del modelo en HuggingFace Hub.
			labels: Lista de etiquetas en lenguaje natural para pedir a GLiNER.
			threshold: Score mínimo para incluir una entidad (0-1).
			chunk_size: Tamaño de chunk en palabras (aproximación a tokens).
			chunk_overlap: Solapamiento en palabras entre chunks consecutivos.
			device: "cpu", "cuda", "mps" o None (autodetecta).
		"""
		self.model_name   = model
		self.labels       = labels or DEFAULT_LABELS
		self.threshold    = threshold
		self.chunk_size   = chunk_size
		self.chunk_overlap = chunk_overlap
		self.device       = device or self._detect_device()

		self.model: GLiNER = self._load_model()

	# ------------------------------------------------------------------
	# Setup
	# ------------------------------------------------------------------

	@staticmethod
	def _detect_device() -> str:
		if torch.cuda.is_available():
			device = "cuda"
		else:
			device = "cpu"
		print(f"[GlinerNER] Dispositivo detectado: {device}")
		return device

	def _load_model(self) -> GLiNER:
		print(f"[GlinerNER] Cargando '{self.model_name}' "
			  f"(primera vez descarga ~500MB)…")
		try:
			model = GLiNER.from_pretrained(self.model_name)
			# GLiNER no expone .to() directamente en todas las versiones;
			# movemos el modelo interno si es posible
			if hasattr(model, "model"):
				model.model = model.model.to(self.device)
			print(f"[GlinerNER] Modelo listo en {self.device}.")
			return model
		except Exception as e:
			raise RuntimeError(
				f"No se pudo cargar el modelo '{self.model_name}'.\n"
				f"Error: {e}\n"
				f"Verifica tu conexión o prueba con:\n"
				f"  knowledgator/gliner-bi-small-v1.0  (más ligero)"
			)

	# ------------------------------------------------------------------
	# Chunking — maneja el límite de tokens de GLiNER
	# ------------------------------------------------------------------

	def _split_into_chunks(self, text: str) -> list[tuple[str, int]]:
		"""
		Parte el texto en chunks de palabras con solapamiento.

		Returns:
			Lista de (chunk_text, offset) donde offset es
			la posición en el texto original donde empieza el chunk.
		"""
		words = text.split(" ")
		chunks: list[tuple[str, int]] = []
		step = self.chunk_size - self.chunk_overlap

		i = 0
		while i < len(words):
			chunk_words = words[i: i + self.chunk_size]
			chunk_text  = " ".join(chunk_words)

			# Calcular offset real de caracteres para este chunk
			# (reconstruimos cuántos chars hay antes del chunk i)
			prefix = " ".join(words[:i])
			offset = len(prefix) + (1 if prefix else 0)

			chunks.append((chunk_text, offset))
			i += step

		return chunks

	# ------------------------------------------------------------------
	# Procesamiento de un texto
	# ------------------------------------------------------------------

	def process(self, text: str, doc_id: str = "doc_0") -> DocumentResult:
		"""
		Procesa un texto y devuelve sus entidades candidatas.

		Para textos largos, hace chunking automático y reconcilia
		las posiciones de caracteres con el texto original.
		"""
		words = text.split()

		if len(words) <= self.chunk_size:
			# Texto corto: procesar directamente
			raw_entities = self.model.predict_entities(
				text, self.labels, threshold=self.threshold
			)
			entities = self._convert_entities(raw_entities, char_offset=0)
		else:
			# Texto largo: chunking con solapamiento
			entities = self._process_long_text(text)

		# Deduplicar (el solapamiento puede generar spans repetidos)
		entities = self._deduplicate(entities)

		return DocumentResult(doc_id=doc_id, text=text, entities=entities)

	def _process_long_text(self, text: str) -> list[EntitySpan]:
		"""Procesa texto largo por chunks y reconcilia posiciones."""
		all_entities: list[EntitySpan] = []
		chunks = self._split_into_chunks(text)

		for chunk_text, char_offset in chunks:
			try:
				raw = self.model.predict_entities(
					chunk_text, self.labels, threshold=self.threshold
				)
				chunk_entities = self._convert_entities(raw, char_offset=char_offset)
				all_entities.extend(chunk_entities)
			except Exception as e:
				warnings.warn(f"Error en chunk (offset {char_offset}): {e}")

		return all_entities

	def _convert_entities(
		self,
		raw_entities: list[dict],
		char_offset: int = 0,
	) -> list[EntitySpan]:
		"""
		Convierte la salida cruda de GLiNER a EntitySpan.

		GLiNER devuelve dicts con: text, label, start, end, score
		donde start/end son índices de caracteres dentro del chunk.
		"""
		spans: list[EntitySpan] = []
		for ent in raw_entities:
			label_raw = ent.get("label", "").lower().strip()
			label     = LABEL_MAP.get(label_raw, label_raw.upper())
			text      = ent.get("text", "").strip()

			if not text:
				continue

			spans.append(EntitySpan(
				text   = text,
				label  = label,
				start  = ent.get("start", 0) + char_offset,
				end    = ent.get("end",   0) + char_offset,
				source = "gliner",
				score  = round(float(ent.get("score", 0.0)), 4),
			))
		return spans

	@staticmethod
	def _deduplicate(entities: list[EntitySpan]) -> list[EntitySpan]:
		"""
		Elimina entidades duplicadas por (start, end).
		En caso de duplicado, conserva la de mayor score.
		"""
		best: dict[tuple[int, int], EntitySpan] = {}
		for ent in entities:
			key = (ent.start, ent.end)
			if key not in best or ent.score > best[key].score:
				best[key] = ent
		return sorted(best.values(), key=lambda e: e.start)

	# ------------------------------------------------------------------
	# Procesamiento en lote
	# ------------------------------------------------------------------

	def process_batch(
		self,
		texts: list[str],
		doc_ids: Optional[list[str]] = None,
	) -> list[DocumentResult]:
		"""
		Procesa múltiples textos.

		Nota: GLiNER no tiene un pipe() nativo como spaCy;
		iteramos uno a uno pero podrías paralelizar con ThreadPoolExecutor
		si el cuello de botella es la I/O de disco.
		"""
		if doc_ids is None:
			doc_ids = [f"doc_{i}" for i in range(len(texts))]

		if len(texts) != len(doc_ids):
			raise ValueError("'texts' y 'doc_ids' deben tener la misma longitud.")

		results = []
		for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
			print(f"  [{i+1}/{len(texts)}] {doc_id}…", end=" ", flush=True)
			result = self.process(text, doc_id=doc_id)
			print(f"{len(result.entities)} entidades")
			results.append(result)

		return results

	# ------------------------------------------------------------------
	# Utilidades
	# ------------------------------------------------------------------

	def print_entities(self, result: DocumentResult) -> None:
		"""Imprime entidades con score (útil para calibrar threshold)."""
		print(f"\n=== {result.doc_id} — {len(result.entities)} entidades ===")
		if not result.entities:
			print("  (ninguna entidad encontrada)")
			return
		for ent in result.entities:
			snippet = result.text[max(0, ent.start - 20): ent.end + 20]	# Texto original de donde proviene la entidad
			snippet = snippet.replace("\n", " ")
			print(
				f"  [{ent.label:14}] '{ent.text}'"	# Se imprime la entidad, placeholder,
				f"  (chars {ent.start}–{ent.end})"	# posicion en el texto (start, end) y el extracto de texto
				f"  (score {ent.score})"	# posicion en el texto (start, end) y el extracto de texto
				f"  \"{snippet}\""
			)

	def model_info(self) -> dict:
		return {
			"model":     self.model_name,
			"labels":    self.labels,
			"threshold": self.threshold,
			"device":    self.device,
			"chunk_size": self.chunk_size,
		}
