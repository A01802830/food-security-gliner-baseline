"""
src/ner/spacy_ner.py

Extrae entidades candidatas de texto usando spaCy.
Salida estandarizada: lista de EntitySpan listos para pasar al LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc


# ---------------------------------------------------------------------------
# Tipos de entidades que nos interesan para anonimización
# ---------------------------------------------------------------------------

# Mapa de etiquetas spaCy → etiqueta interna del proyecto
# es_core_news_lg usa estas etiquetas:
#   PER  → personas
#   LOC  → lugares (ciudad, estado, rancho, comunidad)
#   ORG  → organizaciones (programa gobierno, ONG, empresa)
#   MISC → misceláneos (a veces captura apodos o gentilicios)
LABEL_MAP: dict[str, str] = {
	"PER": "PERSONA",
	"LOC": "LUGAR",
	"ORG": "ORGANIZACION",
	"MISC": "MISC",
}

# Etiquetas que incluimos por defecto (MISC es ruidoso; se puede activar)
DEFAULT_LABELS: set[str] = {"PER", "LOC", "ORG"}


# ---------------------------------------------------------------------------
# Dataclass de salida — interfaz común con GLiNER después
# ---------------------------------------------------------------------------

@dataclass
class EntitySpan:
	text: str           # texto tal cual aparece en el documento
	label: str          # etiqueta interna: PERSONA, LUGAR, ORGANIZACION, MISC
	start: int          # índice de carácter de inicio (en el texto original)
	end: int            # índice de carácter de fin
	source: str = "spacy"
	score: float = 1.0  # spaCy no da score; GLiNER sí — por eso el campo

	def to_dict(self) -> dict:
		return asdict(self)


@dataclass
class DocumentResult:
	"""Resultado de procesar un documento completo."""
	doc_id: str
	text: str
	entities: list[EntitySpan] = field(default_factory=list)

	def to_dict(self) -> dict:
		return {
			"doc_id": self.doc_id,
			"text": self.text,
			"entities": [e.to_dict() for e in self.entities],
		}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class SpacyNER:
	"""
	Wrapper de spaCy para extracción de entidades candidatas.

	Uso básico:
		ner = SpacyNER()
		result = ner.process("Hola, soy Ana García y vivo en Oaxaca.")
		for ent in result.entities:
			print(ent.text, ent.label, ent.start, ent.end)

	Uso en lote:
		results = ner.process_batch(textos, doc_ids=ids)
	"""

	def __init__(
		self,
		model: str = "es_core_news_lg",
		labels: Optional[set[str]] = None,
		disable_pipes: list[str] | None = None,
	):
		"""
		Args:
			model: Modelo spaCy a cargar.
				   Preferido: es_core_news_lg (más preciso).
				   Alternativa ligera: es_core_news_sm.
			labels: Conjunto de etiquetas spaCy a considerar.
					Por defecto: {"PER", "LOC", "ORG"}.
			disable_pipes: Componentes del pipeline a deshabilitar
						   para acelerar (ej. ["parser", "senter"]).
		"""
		self.model_name = model
		self.labels = labels or DEFAULT_LABELS
		self._disable = disable_pipes or ["parser", "senter"]

		self.nlp: Language = self._load_model()

	# ------------------------------------------------------------------
	# Carga del modelo
	# ------------------------------------------------------------------

	def _load_model(self) -> Language:
		try:
			nlp = spacy.load(self.model_name, disable=self._disable)
			print(f"[SpacyNER] Modelo '{self.model_name}' cargado OK.")
			return nlp
		except OSError:
			raise OSError(
				f"Modelo '{self.model_name}' no encontrado.\n"
				f"Instálalo con:\n"
				f"  python -m spacy download {self.model_name}\n"
				f"Modelos disponibles para español:\n"
				f"  es_core_news_sm  (rápido, menos preciso)\n"
				f"  es_core_news_md\n"
				f"  es_core_news_lg  (recomendado)"
			)

	# ------------------------------------------------------------------
	# Procesamiento de un texto
	# ------------------------------------------------------------------

	def process(self, text: str, doc_id: str = "doc_0") -> DocumentResult:
		"""
		Procesa un texto y devuelve sus entidades candidatas.

		Args:
			text: Texto de la entrevista.
			doc_id: Identificador del documento (para trazabilidad).

		Returns:
			DocumentResult con lista de EntitySpan.
		"""
		doc: Doc = self.nlp(text)
		entities = self._extract_entities(doc)
		return DocumentResult(doc_id=doc_id, text=text, entities=entities)

	def _extract_entities(self, doc: Doc) -> list[EntitySpan]:
		"""Filtra y convierte entidades de spaCy al formato interno."""
		seen: set[tuple[int, int]] = set()  # evita duplicados por span
		spans: list[EntitySpan] = []

		for ent in doc.ents:
			if ent.label_ not in self.labels:
				continue

			key = (ent.start_char, ent.end_char)
			if key in seen:
				continue
			seen.add(key)

			cleaned_text = self._clean_entity_text(ent.text)
			if not cleaned_text:
				continue

			spans.append(
				EntitySpan(
					text=cleaned_text,
					label=LABEL_MAP.get(ent.label_, ent.label_),
					start=ent.start_char,
					end=ent.end_char,
					source="spacy",
					score=1.0,
				)
			)

		return spans

	@staticmethod
	def _clean_entity_text(text: str) -> str:
		"""
		Limpieza mínima del texto de la entidad:
		- Quita espacios al inicio/fin
		- Quita saltos de línea internos
		- Descarta entidades de 1 carácter o solo puntuación
		"""
		cleaned = text.strip().replace("\n", " ")
		cleaned = re.sub(r"\s+", " ", cleaned)
		if len(cleaned) <= 1 or re.fullmatch(r"[^\w]+", cleaned):
			return ""
		return cleaned

	# ------------------------------------------------------------------
	# Procesamiento en lote
	# ------------------------------------------------------------------

	def process_batch(
		self,
		texts: list[str],
		doc_ids: Optional[list[str]] = None,
		batch_size: int = 32,
	) -> list[DocumentResult]:
		"""
		Procesa múltiples textos eficientemente con nlp.pipe().

		Args:
			texts: Lista de textos.
			doc_ids: IDs para cada documento. Si no se pasan, se generan
					 automáticamente (doc_0, doc_1, ...).
			batch_size: Tamaño de lote para spaCy.

		Returns:
			Lista de DocumentResult, uno por texto.
		"""
		if doc_ids is None:
			doc_ids = [f"doc_{i}" for i in range(len(texts))]

		if len(texts) != len(doc_ids):
			raise ValueError("'texts' y 'doc_ids' deben tener la misma longitud.")

		results: list[DocumentResult] = []

		for i, (doc, doc_id) in enumerate(
			zip(self.nlp.pipe(texts, batch_size=batch_size), doc_ids)
		):
			print(f"  [{i+1}/{len(texts)}] {doc_id}…", end=" ", flush=True)
			result = self._extract_entities(doc)
			print(f"{len(result)} entidades")
			results.append(
				DocumentResult(doc_id=doc_id, text=doc.text, entities=result)
			)

		return results

	# ------------------------------------------------------------------
	# Utilidades de inspección
	# ------------------------------------------------------------------

	def print_entities(self, result: DocumentResult) -> None:
		"""Imprime entidades encontradas en formato legible."""
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
				f"  \"{snippet}\""
			)

	def model_info(self) -> dict:
		"""Devuelve metadatos del modelo cargado."""
		return {
			"model": self.model_name,
			"labels_active": sorted(self.labels),
			"pipeline": self.nlp.pipe_names,
			"language": self.nlp.lang,
		}
