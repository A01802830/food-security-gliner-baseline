"""
src/evaluation/metrics.py

Evalúa la anonimización automática contra el ground truth humano.

Ground truth: el humano marcó entidades con **ETIQUETA** en el texto.
Sistema:      el pipeline produce [PERSONA_1], [LUGAR_1], etc.

Como las etiquetas son distintas, la evaluación es puramente de COBERTURA:
  ¿El sistema cubrió los mismos spans que el humano anonimizó?

Métricas:
  - Precision  = spans_sistema ∩ spans_humano / spans_sistema
				 "De lo que anonimicé, ¿cuánto debía anonimizarse?"
  - Recall     = spans_sistema ∩ spans_humano / spans_humano
				 "De lo que el humano anonimizó, ¿cuánto capturé yo?"
  - F1         = media armónica de precision y recall

Modos de matching:
  - exact:   los spans deben coincidir exactamente (mismo texto)
  - partial:  basta con que haya solapamiento de texto (más justo con
			  variantes como "Ana" vs "Ana García")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Extracción de spans del ground truth
# ---------------------------------------------------------------------------

# Patrón para **CUALQUIER_ETIQUETA** en el texto del humano
HUMAN_PATTERN = re.compile(r'\*\*([^*]+)\*\*')


def extract_human_spans(text: str) -> list[tuple[int, int, str]]:
	"""
	Extrae los spans anonimizados por el humano del texto con **ETIQUETA**.

	Devuelve lista de (start, end, etiqueta) donde start/end son posiciones
	en el texto LIMPIO (sin los asteriscos).

	Ejemplo:
		"Hola **NOMBRE**, vivo en **LUGAR**."
		→ [(5, 11, "NOMBRE"), (22, 27, "LUGAR")]
		  (posiciones en "Hola NOMBRE, vivo en LUGAR.")
	"""
	spans = []
	offset = 0          # diferencia acumulada entre texto original y limpio
	clean_text = text   # iremos construyendo el texto limpio

	for match in HUMAN_PATTERN.finditer(text):
		label       = match.group(1)
		raw_start   = match.start() - offset
		raw_end     = match.end()   - offset

		# En el texto limpio, la etiqueta ocupa len(label) chars
		# sin los 4 chars de ** ** (2 al inicio + 2 al final)
		clean_start = raw_start
		clean_end   = clean_start + len(label)

		spans.append((clean_start, clean_end, label))

		# Actualizar offset: quitamos los 4 asteriscos del match
		offset += 4   # ** al inicio + ** al final

	return spans


def clean_human_text(text: str) -> str:
	"""Quita los marcadores ** del texto humano, dejando solo las etiquetas."""
	return HUMAN_PATTERN.sub(r'\1', text)


# ---------------------------------------------------------------------------
# Extracción de spans del sistema
# ---------------------------------------------------------------------------

# Patrón para [PERSONA_1], [LUGAR_2], etc. en el texto anonimizado
SYSTEM_PATTERN = re.compile(r'\[([A-Z]+)_(\d+)\]')


def extract_system_spans(text: str) -> list[tuple[int, int, str]]:
	"""
	Extrae los spans anonimizados por el sistema del texto con [LABEL_N].

	Devuelve lista de (start, end, label) donde start/end son posiciones
	en el texto LIMPIO (sin los corchetes ni números).

	Para comparar con el humano, normalizamos ambos textos al texto limpio.
	"""
	spans = []
	offset = 0

	for match in SYSTEM_PATTERN.finditer(text):
		label_type  = match.group(1)   # ej. "PERSONA"
		raw_start   = match.start() - offset
		label_repr  = label_type       # en el texto limpio usamos solo el tipo

		clean_start = raw_start
		clean_end   = clean_start + len(label_repr)

		spans.append((clean_start, clean_end, label_type))
		offset += (match.end() - match.start()) - len(label_repr)

	return spans


# ---------------------------------------------------------------------------
# Tipos de resultado
# ---------------------------------------------------------------------------

@dataclass
class SpanMatch:
	human_span:  tuple[int, int, str]
	system_span: Optional[tuple[int, int, str]]
	match_type:  str   # "exact", "partial", "missed"


@dataclass
class DocumentMetrics:
	doc_id:          str
	n_human:         int    # spans anonimizados por el humano
	n_system:        int    # spans anonimizados por el sistema
	true_positives:  int    # correctamente capturados
	false_positives: int    # el sistema anonimizó pero el humano no
	false_negatives: int    # el humano anonimizó pero el sistema no
	precision:       float
	recall:          float
	f1:              float
	missed:          list[str] = field(default_factory=list)   # textos no capturados
	extra:           list[str] = field(default_factory=list)   # textos de más

	def to_dict(self) -> dict:
		return {
			"doc_id":          self.doc_id,
			"n_human":         self.n_human,
			"n_system":        self.n_system,
			"true_positives":  self.true_positives,
			"false_positives": self.false_positives,
			"false_negatives": self.false_negatives,
			"f1":              round(self.f1, 4),
			"precision":       round(self.precision, 4),
			"recall":          round(self.recall, 4),
			"missed":          self.missed,
			"extra":           self.extra,
		}
	
	def print_report(self, complete: bool = False) -> None:
		print("=" * 55)
		print(f"EVALUACIÓN DE ANONIMIZACIÓN - {self.doc_id}")
		print("=" * 55)
		print(f"{'* # NER anotadas x humano':<35} {self.n_human:>5}")
		print(f"{'* # NER anotadas automaticamente':<35} {self.n_system:>5}\n")

		if complete:
			print("-" * 38)
			print("Otras métricas")
			print("-" * 38)
			dictionary = self.to_dict()
			for key, value in dictionary.items():
				if key in ["f1", "precision", "recall", "true_positives", "false_positives", "false_negatives"]:
					print(f"{key.replace('_', ' ').capitalize():<16}: {value:>5}")


@dataclass
class CorpusMetrics:
	documents:        list[DocumentMetrics]
	macro_precision:  float   # promedio de precision por documento
	macro_recall:     float   # promedio de recall por documento
	macro_f1:         float
	micro_precision:  float   # precision sobre todos los spans del corpus
	micro_recall:     float
	micro_f1:         float

	def to_dict(self) -> dict:
		return {
			"macro": {
				"precision": round(self.macro_precision, 4),
				"recall":    round(self.macro_recall,    4),
				"f1":        round(self.macro_f1,        4),
			},
			"micro": {
				"precision": round(self.micro_precision, 4),
				"recall":    round(self.micro_recall,    4),
				"f1":        round(self.micro_f1,        4),
			},
			"documents": [d.to_dict() for d in self.documents],
		}

	def print_report(self) -> None:
		print("=" * 55)
		print("EVALUACIÓN DE ANONIMIZACIÓN")
		print("=" * 55)
		print(f"\n{'Métrica':<20} {'Macro':>8} {'Micro':>8}")
		print("-" * 38)
		print(f"{'Precision':<20} {self.macro_precision:>8.3f} {self.micro_precision:>8.3f}")
		print(f"{'Recall':<20} {self.macro_recall:>8.3f} {self.micro_recall:>8.3f}")
		print(f"{'F1':<20} {self.macro_f1:>8.3f} {self.micro_f1:>8.3f}")

		print(f"\n{'─'*55}")
		print(f"{'Documento':<35} {'P':>5} {'R':>5} {'F1':>5} {'TP':>4} {'FP':>4} {'FN':>4}")
		print("─" * 55)
		for d in self.documents:
			print(
				f"  {d.doc_id:<33} "
				f"{d.precision:>5.2f} "
				f"{d.recall:>5.2f} "
				f"{d.f1:>5.2f} "
				f"{d.true_positives:>4} "
				f"{d.false_positives:>4} "
				f"{d.false_negatives:>4}"
			)

		# Documentos con recall bajo — los más importantes de revisar
		low_recall = [d for d in self.documents if d.recall < 0.80]
		if low_recall:
			print(f"\n⚠️  Documentos con recall < 0.80 (entidades perdidas):")
			for d in sorted(low_recall, key=lambda x: x.recall):
				print(f"  · {d.doc_id}: recall={d.recall:.2f}  "
					  f"perdidas={d.false_negatives}")
				for m in d.missed[:5]:
					print(f"      - '{m}'")


# ---------------------------------------------------------------------------
# Evaluador principal
# ---------------------------------------------------------------------------

class AnonymizationEvaluator:
	"""
	Evalúa la calidad de la anonimización comparando con el ground truth humano.

	Uso:
		evaluator = AnonymizationEvaluator(match_mode="partial")

		# Un documento
		metrics = evaluator.evaluate_document(
			human_file  = Path("data/ground_truth/doc1_corregida.txt"),
			system_file = Path("data/processed/entrevistas_anonimizadas/doc1_anonimizado.txt"),
			original    = Path("data/raw/entrevistas_originales/doc1.txt"),
		)

		# Corpus completo
		corpus = evaluator.evaluate_corpus(
			ground_truth_dir = Path("data/ground_truth/"),
			system_dir       = Path("data/processed/entrevistas_anonimizadas/"),
			originals_dir    = Path("data/raw/entrevistas_originales/"),
		)
		corpus.print_report()
	"""

	def __init__(self, match_mode: str = "partial", encoding: str = "utf-8"):
		"""
		Args:
			match_mode: "exact" (mismo texto) o "partial" (solapamiento).
						"partial" es más justo con variantes del mismo nombre.
			encoding:   Encoding de los archivos TXT.
		"""
		if match_mode not in ("exact", "partial"):
			raise ValueError("match_mode debe ser 'exact' o 'partial'")
		self.match_mode = match_mode
		self.encoding   = encoding

	# ------------------------------------------------------------------
	# Evaluación de un documento
	# ------------------------------------------------------------------

	def evaluate_document(
		self,
		human_file:  Path,    # TXT con **ETIQUETA**
		system_file: Path,    # TXT con [PERSONA_1]
		original:    Path,    # TXT original sin anonimizar
		doc_id:      Optional[str] = None,
	) -> DocumentMetrics:
		"""Evalúa un par (ground truth humano, salida del sistema)."""
		doc_id      = doc_id or human_file.stem
		human_text  = human_file.read_text(encoding=self.encoding)
		system_text = system_file.read_text(encoding=self.encoding)
		orig_text   = original.read_text(encoding=self.encoding)

		return self._compute_metrics(
			doc_id      = doc_id,
			human_text  = human_text,
			system_text = system_text,
			orig_text   = orig_text,
		)

	def evaluate_from_texts(
		self,
		doc_id:      str,
		human_text:  str,
		system_text: str,
		orig_text:   str,
	) -> DocumentMetrics:
		"""Evalúa directamente desde strings (útil en notebooks)."""
		return self._compute_metrics(doc_id, human_text, system_text, orig_text)

	# ------------------------------------------------------------------
	# Evaluación de corpus
	# ------------------------------------------------------------------

	def evaluate_corpus(
		self,
		ground_truth_dir: Path,
		system_dir:       Path,
		originals_dir:    Path,
	) -> CorpusMetrics:
		"""
		Evalúa todos los documentos del corpus.

		Convención de nombres:
		  ground_truth_dir/  doc1_corregida.txt  (o doc1.txt)
		  system_dir/        doc1_anonimizado.txt
		  originals_dir/     doc1.txt
		"""
		doc_metrics: list[DocumentMetrics] = []

		system_files = sorted(system_dir.glob("*_anonimizado.txt"))
		if not system_files:
			raise FileNotFoundError(
				f"No se encontraron archivos *_anonimizado.txt en {system_dir}"
			)

		for sys_path in system_files:
			doc_id = sys_path.stem.replace("_anonimizado", "")

			# Buscar ground truth (acepta doc1_corregida.txt o doc1.txt)
			human_path = self._find_ground_truth(ground_truth_dir, doc_id)
			orig_path  = originals_dir / f"{doc_id}.txt"

			if not human_path:
				print(f"  *  {doc_id}: ground truth no encontrado, saltando")
				continue
			if not orig_path.exists():
				print(f"  *  {doc_id}: original no encontrado, saltando")
				continue

			try:
				dm = self.evaluate_document(
					human_file  = human_path,
					system_file = sys_path,
					original    = orig_path,
					doc_id      = doc_id,
				)
				doc_metrics.append(dm)
				print(
					f"  ✓ {doc_id:<38} "
					f"P={dm.precision:.2f}  R={dm.recall:.2f}  F1={dm.f1:.2f}"
				)
			except Exception as e:
				print(f"  ✗ {doc_id}: {e}")

		if not doc_metrics:
			raise ValueError("No se pudo evaluar ningún documento.")

		return self._aggregate(doc_metrics)

	# ------------------------------------------------------------------
	# Lógica de comparación
	# ------------------------------------------------------------------

	def _compute_metrics(
		self,
		doc_id:      str,
		human_text:  str,
		system_text: str,
		orig_text:   str,
	) -> DocumentMetrics:
		"""
		Estrategia de comparación:
		  1. Recuperar el texto original limpio (sin placeholders ni **)
		  2. Mapear los spans de cada uno al texto original
		  3. Comparar qué rangos de caracteres fueron anonimizados
		"""
		# Extraer spans de ambos en su propio texto con placeholders
		human_spans  = extract_human_spans(human_text)    # (start, end, label) en texto limpio humano
		system_spans = extract_system_spans(system_text)  # (start, end, label) en texto limpio sistema

		# Mapear spans al texto ORIGINAL para comparación directa
		human_orig  = self._map_spans_to_original(clean_human_text(human_text),  orig_text, human_spans)
		system_orig = self._map_spans_to_original(self._clean_system_text(system_text), orig_text, system_spans)

		# Comparar coberturas
		tp, fp, fn, missed, extra = self._compare_spans(
			human_orig  = human_orig,
			system_orig = system_orig,
			orig_text   = orig_text,
		)

		n_human  = len(human_orig)
		n_system = len(system_orig)

		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1        = (2 * precision * recall / (precision + recall)
					 if (precision + recall) > 0 else 0.0)

		return DocumentMetrics(
			doc_id          = doc_id,
			n_human         = n_human,
			n_system        = n_system,
			true_positives  = tp,
			false_positives = fp,
			false_negatives = fn,
			precision       = precision,
			recall          = recall,
			f1              = f1,
			missed          = missed,
			extra           = extra,
		)

	def _compare_spans(
		self,
		human_orig:  list[tuple[int, int]],
		system_orig: list[tuple[int, int]],
		orig_text:   str,
	) -> tuple[int, int, int, list[str], list[str]]:
		"""
		Compara dos listas de spans (start, end) contra el texto original.
		Devuelve (TP, FP, FN, missed_texts, extra_texts).
		"""
		matched_human:  set[int] = set()
		matched_system: set[int] = set()

		for j, s_span in enumerate(system_orig):
			for i, h_span in enumerate(human_orig):
				if i in matched_human:
					continue
				if self._spans_match(s_span, h_span):
					matched_human.add(i)
					matched_system.add(j)
					break

		tp = len(matched_system)
		fp = len(system_orig) - tp
		fn = len(human_orig)  - len(matched_human)

		missed = [
			orig_text[s:e]
			for i, (s, e) in enumerate(human_orig)
			if i not in matched_human
		]
		extra = [
			orig_text[s:e]
			for j, (s, e) in enumerate(system_orig)
			if j not in matched_system
		]

		return tp, fp, fn, missed, extra

	def _spans_match(
		self,
		a: tuple[int, int],
		b: tuple[int, int],
	) -> bool:
		if self.match_mode == "exact":
			return a == b
		# partial: cualquier solapamiento
		return a[0] < b[1] and b[0] < a[1]

	# ------------------------------------------------------------------
	# Alineación de spans al texto original
	# ------------------------------------------------------------------

	@staticmethod
	def _map_spans_to_original(
		clean_text: str,
		orig_text:  str,
		spans:      list[tuple[int, int, str]],
	) -> list[tuple[int, int]]:
		"""
		Mapea spans del texto limpio (con etiquetas) al texto original.

		Construye un índice de alineación carácter a carácter entre
		clean_text y orig_text usando el LCS (longest common subsequence)
		simplificado: avanzamos en paralelo ignorando diferencias.

		Para entrevistas cortas y diferencias pequeñas (solo los placeholders),
		esta alineación es suficientemente precisa.
		"""
		if not spans:
			return []

		# Construir mapa: posición en clean_text → posición en orig_text
		align = _build_alignment(clean_text, orig_text)

		result = []
		for start, end, _ in spans:
			orig_start = align.get(start, start)
			orig_end   = align.get(end - 1, end - 1) + 1 if end > start else orig_start
			result.append((orig_start, orig_end))

		return result

	@staticmethod
	def _clean_system_text(text: str) -> str:
		"""Quita los placeholders del sistema, dejando solo el tipo."""
		return SYSTEM_PATTERN.sub(r'\1', text)

	# ------------------------------------------------------------------
	# Agregación de corpus
	# ------------------------------------------------------------------

	@staticmethod
	def _aggregate(docs: list[DocumentMetrics]) -> CorpusMetrics:
		macro_p  = sum(d.precision for d in docs) / len(docs)
		macro_r  = sum(d.recall    for d in docs) / len(docs)
		macro_f1 = (2 * macro_p * macro_r / (macro_p + macro_r)
					if (macro_p + macro_r) > 0 else 0.0)

		total_tp = sum(d.true_positives  for d in docs)
		total_fp = sum(d.false_positives for d in docs)
		total_fn = sum(d.false_negatives for d in docs)

		micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
		micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
		micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
					if (micro_p + micro_r) > 0 else 0.0)

		return CorpusMetrics(
			documents       = docs,
			macro_precision = macro_p,
			macro_recall    = macro_r,
			macro_f1        = macro_f1,
			micro_precision = micro_p,
			micro_recall    = micro_r,
			micro_f1        = micro_f1,
		)

	@staticmethod
	def _find_ground_truth(ground_truth_dir: Path, doc_id: str) -> Optional[Path]:
		"""Busca el archivo de ground truth con distintos sufijos posibles."""
		for suffix in [f"{doc_id}_corregida.txt", f"{doc_id}.txt",
					   f"{doc_id}_anonimizada.txt", f"{doc_id}_gt.txt"]:
			p = ground_truth_dir / suffix
			if p.exists():
				return p
		return None


# ---------------------------------------------------------------------------
# Alineación de caracteres (helper de módulo)
# ---------------------------------------------------------------------------

def _build_alignment(clean: str, orig: str) -> dict[int, int]:
	"""
	Construye un mapa posición_clean → posición_orig.
	Avanza en paralelo por ambos textos; cuando el carácter coincide,
	registra la alineación. Diseñado para textos que difieren solo en
	los placeholders (texto mayoritariamente igual).
	"""
	align: dict[int, int] = {}
	ci, oi = 0, 0

	while ci < len(clean) and oi < len(orig):
		if clean[ci] == orig[oi]:
			align[ci] = oi
			ci += 1
			oi += 1
		else:
			# El orig tiene texto que el clean no (el texto original de la entidad)
			# Avanzar en orig hasta encontrar coincidencia
			lookahead = 1
			found = False
			while oi + lookahead < len(orig) and lookahead < 60:
				if orig[oi + lookahead] == clean[ci]:
					oi += lookahead
					found = True
					break
				lookahead += 1
			if not found:
				ci += 1   # avanzar en clean si no encontramos coincidencia

	return align
