"""
src/anonymization/replacer.py

Construye el diccionario de placeholders y anonimiza el texto.

Flujo:
  1. Leer decisiones del LLM (LLMOutput JSON validado)
  2. Agrupar variantes por forma canónica → asignar placeholder
	   "Ana García", "doña Ana", "Ana" → [PERSONA_1]
  3. Encontrar TODOS los spans de cada entidad en el texto
	 (el LLM solo vio menciones candidatas; puede haber más en el texto)
  4. Sustituir de atrás hacia adelante para no desplazar índices
  5. Guardar texto anonimizado + diccionario de sustituciones

Formato de placeholders: [PERSONA_1], [LUGAR_1], [ORGANIZACION_1], [MISC_1]

Output por documento:
  data/processed/entrevistas_anonimizadas/
	  doc1_anonimizado.txt
	  doc1_diccionario.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

# Prefijos de placeholder por label
LABEL_PREFIX: dict[str, str] = {
	"PERSONA":      "PERSONA",
	"LUGAR":        "LUGAR",
	"ORGANIZACION": "ORGANIZACION",
	"MISC":         "MISC",
}

# Símbolo a usar para anonimizar
DEFAULT_SYMBOL = "brackets"

VALID_SYMBOLS = {
	"brackets": ("[", "]"),
	"hash": ("#", "#"),
	"asterisk": ("**", "**"),
	"angle": ("<", ">"),
}



@dataclass
class PlaceholderEntry:
	"""Una entrada del diccionario: todas las variantes que mapean al mismo placeholder."""
	placeholder:  str          # ej. "[PERSONA_1]"
	label:        str          # ej. "PERSONA"
	canonical:    str          # forma canónica elegida por el LLM
	variants:     list[str]    # todas las formas del texto que se reemplazan
	span_count:   int = 0      # cuántas sustituciones se hicieron en el texto


@dataclass
class AnonymizationResult:
	doc_id:            str
	original_text:     str
	anonymized_text:   str
	dictionary:        list[PlaceholderEntry] = field(default_factory=list)
	total_replacements: int = 0

	def to_dict(self) -> dict:
		return {
			"doc_id":             self.doc_id,
			"total_replacements": self.total_replacements,
			"dictionary": [
				{
					"placeholder": e.placeholder,
					"label":       e.label,
					"canonical":   e.canonical,
					"variants":    e.variants,
					"span_count":  e.span_count,
				}
				for e in self.dictionary
			],
		}


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class EntityReplacer:
	"""
	Anonimiza texto reemplazando entidades confirmadas con placeholders.

	 Parameters
    ----------
    symbol : {'brackets', 'hash', 'asterisk', 'angle'}, default='brackets'

	Uso:
		replacer = EntityReplacer()

		result = replacer.anonymize_from_files(
			validated_json = Path("data/processed/entidades_validadas/doc1/doc1_validated.json"),
			original_txt   = Path("data/raw/entrevistas_originales/doc1.txt"),
		)

	En lote:
		results = replacer.anonymize_directory(
			validated_dir = Path("data/processed/entidades_validadas/"),
			originals_dir = Path("data/raw/entrevistas_originales/"),
			output_dir    = Path("data/processed/entrevistas_anonimizadas/"),
		)
	"""

	def __init__(
		self,
		encoding: str = "utf-8",
		symbol: str = DEFAULT_SYMBOL
	):
		self.encoding = encoding
		
		# Validación
		if symbol not in VALID_SYMBOLS:
			valid = ", ".join(VALID_SYMBOLS.keys())
			raise ValueError(
				f"Invalid symbol '{symbol}'. Valid options are: {valid}"
			)

		self.symbol = symbol
		self.symbol_start, self.symbol_end = self._get_symbol_pair()

	# ------------------------------------------------------------------
	# Métodos
	# ------------------------------------------------------------------

	def anonymize_from_files(
		self,
		validated_json: Path,
		original_txt:   Path,
		output_dir:     Optional[Path] = None,
	) -> AnonymizationResult:
		"""
		Anonimiza un documento a partir de su JSON validado y el TXT original.
		"""
		validated = json.loads(validated_json.read_text(encoding=self.encoding))
		text      = original_txt.read_text(encoding=self.encoding)
		doc_id    = validated.get("doc_id", original_txt.stem)

		result = self.anonymize(doc_id=doc_id, text=text, validated=validated)

		if output_dir:
			self._save(result, output_dir)

		return result

	def anonymize(
		self,
		doc_id:    str,
		text:      str,
		validated: dict,      # contenido del JSON validado por el LLM
	) -> AnonymizationResult:
		"""
		Núcleo de la anonimización.

		Args:
			doc_id:    Identificador del documento.
			text:      Texto original de la entrevista.
			validated: Dict con "decisions" del LLMOutput.
		"""
		# 1. Construir diccionario canónico → placeholder
		dictionary = self._build_dictionary(validated.get("decisions", []))

		if not dictionary:
			return AnonymizationResult(
				doc_id          = doc_id,
				original_text   = text,
				anonymized_text = text,
				dictionary      = [],
				total_replacements = 0,
			)

		# 2. Encontrar todos los spans a reemplazar en el texto
		spans = self._find_all_spans(text, dictionary)

		# 3. Sustituir de atrás hacia adelante
		anonymized_text, replacement_count = self._replace_spans(text, spans)

		# Actualizar span_count en el diccionario
		placeholder_counts: dict[str, int] = defaultdict(int)
		for _, placeholder in spans:
			placeholder_counts[placeholder] += 1
		for entry in dictionary:
			entry.span_count = placeholder_counts.get(entry.placeholder, 0)

		return AnonymizationResult(
			doc_id             = doc_id,
			original_text      = text,
			anonymized_text    = anonymized_text,
			dictionary         = dictionary,
			total_replacements = replacement_count,
		)

	def anonymize_directory(
		self,
		validated_dir: Path,
		originals_dir: Path,
		output_dir:    Path,
	) -> list[AnonymizationResult]:
		"""
		Anonimiza todos los documentos en un directorio.

		Espera que validated_dir tenga subcarpetas por documento:
			validated_dir/doc1/doc1_validated.json
		O bien JSONs directamente en validated_dir:
			validated_dir/doc1_validated.json
		"""
		output_dir.mkdir(parents=True, exist_ok=True)
		results = []

		json_files = self._find_validated_jsons(validated_dir)
		if not json_files:
			raise FileNotFoundError(
				f"No se encontraron JSONs validados en {validated_dir}"
			)

		for json_path in sorted(json_files):
			doc_id = self._infer_doc_id(json_path)
			txt_path = originals_dir / f"{doc_id}.txt"

			if not txt_path.exists():
				print(f"  x  {doc_id}: TXT no encontrado en {originals_dir}")
				continue

			try:
				result = self.anonymize_from_files(
					validated_json = json_path,
					original_txt   = txt_path,
					output_dir     = output_dir,
				)
				results.append(result)
				print(
					f"  * {doc_id:<40} "
					f"{result.total_replacements} sustituciones  "
					f"({len(result.dictionary)} entidades únicas)"
				)
			except Exception as e:
				print(f"  x {doc_id}: {e}")

		return results

	# ------------------------------------------------------------------
	# Construcción del diccionario
	# ------------------------------------------------------------------

	def _build_dictionary(self, decisions: list[dict]) -> list[PlaceholderEntry]:
		"""
		Agrupa decisiones confirmadas por forma canónica y asigna placeholders.

		Lógica:
		  - Solo procesa decisiones con confirmed=True
		  - Agrupa por (canonical_form, label) → mismo placeholder
		  - Ordena por primera aparición para numeración consistente
		"""
		# Agrupar variantes por (canonical, label)
		groups: dict[tuple[str, str], list[str]] = defaultdict(list)

		for d in decisions:
			if not d.get("confirmed", False):
				continue

			label     = d.get("final_label", "MISC")
			canonical = d.get("canonical_form") or d.get("text", "")
			text_var  = d.get("text", "")

			if not canonical or not text_var:
				continue

			# Normalizar label al prefijo conocido
			label = LABEL_PREFIX.get(label, "MISC")
			key   = (canonical.strip(), label)

			if text_var not in groups[key]:
				groups[key].append(text_var)
			# La forma canónica también es variante si difiere del text
			if canonical != text_var and canonical not in groups[key]:
				groups[key].append(canonical)

		# Asignar placeholders numerados por label
		counters: dict[str, int] = defaultdict(int)
		dictionary: list[PlaceholderEntry] = []

		for (canonical, label), variants in groups.items():
			counters[label] += 1
			placeholder = f"{self.symbol_start}{label}_{counters[label]}{self.symbol_end}"	# [PERSON_n], <PLACE_n>...
			dictionary.append(PlaceholderEntry(
				placeholder = placeholder,
				label       = label,
				canonical   = canonical,
				variants    = variants,
			))

		return dictionary

	# ------------------------------------------------------------------
	# Búsqueda de spans
	# ------------------------------------------------------------------

	def _find_all_spans(
		self,
		text:       str,
		dictionary: list[PlaceholderEntry],
	) -> list[tuple[tuple[int, int], str]]:
		"""
		Encuentra TODOS los spans de cada variante en el texto.

		Busca con regex case-insensitive y límites de palabra para no
		reemplazar subcadenas (ej. "Ana" dentro de "Banana").

		Returns:
			Lista de ((start, end), placeholder), ordenada por start desc
			para sustituir de atrás hacia adelante.
		"""
		spans: list[tuple[tuple[int, int], str]] = []
		seen: set[tuple[int, int]] = set()   # evita solapamientos

		# Ordenar variantes por longitud desc para que "Ana García"
		# tenga prioridad sobre "Ana" en caso de solapamiento
		entries_sorted = [
			(variant, entry.placeholder)
			for entry in dictionary
			for variant in sorted(entry.variants, key=len, reverse=True)
		]

		for variant, placeholder in entries_sorted:
			if not variant.strip():
				continue

			pattern = r'(?<!\w)' + re.escape(variant.strip()) + r'(?!\w)'
			for match in re.finditer(pattern, text, re.IGNORECASE):
				start, end = match.start(), match.end()
				span = (start, end)

				# Saltar si ya está cubierto por un span previo (más largo)
				if any(
					s <= start and end <= e
					for (s, e) in seen
				):
					continue

				seen.add(span)
				spans.append((span, placeholder))

		# Ordenar de atrás hacia adelante para no desplazar índices
		spans.sort(key=lambda x: x[0][0], reverse=True)
		return spans

	# ------------------------------------------------------------------
	# Sustitución
	# ------------------------------------------------------------------

	@staticmethod
	def _replace_spans(
		text:  str,
		spans: list[tuple[tuple[int, int], str]],
	) -> tuple[str, int]:
		"""
		Reemplaza los spans en el texto de atrás hacia adelante.
		Devuelve (texto_anonimizado, número_de_sustituciones).
		"""
		text_list = list(text)   # lista de chars para sustitución eficiente
		count = 0

		for (start, end), placeholder in spans:
			text_list[start:end] = list(placeholder)
			count += 1

		return "".join(text_list), count

	# ------------------------------------------------------------------
	# I/O helpers
	# ------------------------------------------------------------------

	def _get_symbol_pair(self) -> tuple[str, str]:
		"""Returns the opening and closing delimiters."""
		return VALID_SYMBOLS[self.symbol]

	@staticmethod
	def _find_validated_jsons(validated_dir: Path) -> list[Path]:
		"""Busca JSONs validados tanto en raíz como en subcarpetas."""
		jsons = list(validated_dir.glob("**/*_validated.json"))
		# Excluir archivos de batch y error
		return [
			p for p in jsons
			if "_batch_" not in p.name and "_ERROR" not in p.name
		]

	@staticmethod
	def _infer_doc_id(json_path: Path) -> str:
		"""Infiere el doc_id del nombre del archivo: doc1_validated.json → doc1"""
		return json_path.stem.replace("_validated", "")

	def _save(self, result: AnonymizationResult, output_dir: Path) -> None:
		output_dir.mkdir(parents=True, exist_ok=True)

		# Texto anonimizado
		txt_path = output_dir / f"{result.doc_id}_anonimizado.txt"
		txt_path.write_text(result.anonymized_text, encoding=self.encoding)

		# Diccionario de sustituciones
		dict_path = output_dir / f"{result.doc_id}_diccionario.json"
		dict_path.write_text(
			json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
			encoding=self.encoding,
		)
