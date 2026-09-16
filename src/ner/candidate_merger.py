"""
src/ner/candidate_merger.py

Fusiona los candidatos de spaCy y GLiNER en una lista unificada por documento.

Lógica de fusión:
  - Deduplica por solapamiento de spans (no por texto exacto)
  - Cuando dos spans se solapan, conserva el de mayor confianza
  - Registra si la entidad fue detectada por uno o ambos modelos
  - Asigna un confidence_tier para priorizar en el filtro siguiente

Confianza:
  ALTA   → detectada por ambos modelos
  MEDIA  → solo GLiNER con score >= threshold, o solo spaCy
  BAJA   → solo GLiNER con score bajo
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH   = "alta"    # ambos modelos de acuerdo
CONFIDENCE_MEDIUM = "media"   # un solo modelo, score razonable
CONFIDENCE_LOW    = "baja"    # un solo modelo, score bajo

GLINER_MEDIUM_THRESHOLD = 0.50  # score mínimo de GLiNER para tier MEDIA


@dataclass
class MergedEntity:
	text:            str
	label:           str
	start:           int
	end:             int
	sources:         list[str]        # ["spacy"], ["gliner"] o ["spacy", "gliner"]
	score:           float            # max score entre modelos
	confidence_tier: str              # "alta" | "media" | "baja"

	def to_dict(self) -> dict:
		return asdict(self)


@dataclass
class MergedDocument:
	doc_id:   str
	text:     str
	entities: list[MergedEntity] = field(default_factory=list)

	def to_dict(self) -> dict:
		return {
			"doc_id":   self.doc_id,
			"text":     self.text,
			"entities": [e.to_dict() for e in self.entities],
			"stats":    self._stats(),
		}

	def _stats(self) -> dict:
		from collections import Counter
		return {
			"total":            len(self.entities),
			"por_tipo":         dict(Counter(e.label for e in self.entities)),
			"por_confianza":    dict(Counter(e.confidence_tier for e in self.entities)),
			"por_fuente":       {
				"ambos":         sum(1 for e in self.entities if len(e.sources) == 2),
				"solo_spacy":    sum(1 for e in self.entities if e.sources == ["spacy"]),
				"solo_gliner":   sum(1 for e in self.entities if e.sources == ["gliner"]),
			},
		}
	
	def __str__(self):
		# Llamamos al diccionario y lo transformamos en un string bonito
		return f"\n  ── STATS {self.doc_id} ──\n" + json.dumps(self._stats(), indent=4, ensure_ascii=False)



# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class CandidateMerger:
	"""
	Fusiona candidatos de spaCy y GLiNER en una lista unificada.

	Uso:
		merger = CandidateMerger()
		merged = merger.merge_from_files(
			spacy_path  = Path("data/processed/entidades_candidatas_spacy/doc1_candidates.json"),
			gliner_path = Path("data/processed/entidades_candidatas_gliner/doc1_candidates.json"),
		)

	O en lote:
		results = merger.merge_directory(
			spacy_dir  = Path("data/processed/entidades_candidatas/"),
			gliner_dir = Path("data/processed/entidades_candidatas_gliner/"),
		)
	"""

	def __init__(
		self,
		overlap_threshold: float = 0.5,   # IoU mínimo para considerar dos spans como el mismo
		gliner_medium_threshold: float = GLINER_MEDIUM_THRESHOLD,
	):
		self.overlap_threshold        = overlap_threshold
		self.gliner_medium_threshold  = gliner_medium_threshold

	# ------------------------------------------------------------------
	# Métodos
	# ------------------------------------------------------------------

	def merge_from_dicts(
		self,
		doc_id:          str,
		text:            str,
		spacy_candidates:  list[dict],
		gliner_candidates: list[dict],
	) -> MergedDocument:
		"""
		Fusiona dos listas de candidatos (ya cargadas como dicts).
		"""
		spacy_spans  = self._normalize(spacy_candidates,  source="spacy")
		gliner_spans = self._normalize(gliner_candidates, source="gliner")

		merged = self._merge_spans(spacy_spans, gliner_spans)
		merged = sorted(merged, key=lambda e: e.start)

		return MergedDocument(doc_id=doc_id, text=text, entities=merged)

	def merge_from_files(
		self,
		spacy_path:  Optional[Path] = None,
		gliner_path: Optional[Path] = None,
	) -> MergedDocument:
		"""
		Carga los JSONs de spaCy y GLiNER y los fusiona.
		Acepta que uno de los dos no exista (usa lista vacía).
		"""
		spacy_data  = self._load_json(spacy_path)  if spacy_path  else {}
		gliner_data = self._load_json(gliner_path) if gliner_path else {}

		# Inferir doc_id y text del que esté disponible
		doc_id = (spacy_data or gliner_data).get("doc_id", "unknown")
		text   = (spacy_data or gliner_data).get("text",   "")

		spacy_candidates  = spacy_data.get("spacy_candidates",  [])
		gliner_candidates = gliner_data.get("gliner_candidates", [])

		return self.merge_from_dicts(doc_id, text, spacy_candidates, gliner_candidates)

	def merge_directory(
		self,
		spacy_dir:  Path,
		gliner_dir: Path,
		output_dir: Optional[Path] = None,
	) -> list[MergedDocument]:
		"""
		Fusiona todos los documentos en dos directorios.
		Empareja por doc_id (nombre de archivo sin sufijo).

		Convenio de nombres esperado:
		  spacy_dir/   {doc_id}_candidates.json
		  gliner_dir/  {doc_id}_candidates.json
		"""
		# Construir índice de archivos spaCy
		spacy_index: dict[str, Path] = {}
		for p in sorted(spacy_dir.glob("*_candidates.json")):
			doc_id = p.stem.replace("_candidates", "")
			spacy_index[doc_id] = p

		# Construir índice de archivos GLiNER
		gliner_index: dict[str, Path] = {}
		for p in sorted(gliner_dir.glob("*_candidates.json")):
			doc_id = p.stem.replace("_candidates", "")
			gliner_index[doc_id] = p

		all_doc_ids = sorted(set(spacy_index) | set(gliner_index))
		if not all_doc_ids:
			raise FileNotFoundError(
				f"No se encontraron archivos en:\n  {spacy_dir}\n  {gliner_dir}"
			)

		results = []
		for doc_id in all_doc_ids:
			spacy_path  = spacy_index.get(doc_id)
			gliner_path = gliner_index.get(doc_id)

			if not spacy_path:
				print(f"  *  {doc_id}: sin candidatos spaCy")
			if not gliner_path:
				print(f"  *  {doc_id}: sin candidatos GLiNER")

			merged = self.merge_from_files(spacy_path, gliner_path)
			results.append(merged)

			tier_counts = merged._stats()["por_confianza"]
			print(
				f"  * {doc_id:<40} "
				f"{len(merged.entities):>3} entidades  "
				f"[alta={tier_counts.get('alta',0)} "
				f"media={tier_counts.get('media',0)} "
				f"baja={tier_counts.get('baja',0)}]"
			)

		if output_dir:
			output_dir.mkdir(parents=True, exist_ok=True)
			self._export(results, output_dir)

		return results

	# ------------------------------------------------------------------
	# Lógica interna de fusión
	# ------------------------------------------------------------------

	def _normalize(self, candidates: list[dict], source: str) -> list[dict]:
		"""Asegura que cada candidato tenga los campos necesarios y la fuente correcta."""
		normalized = []
		for c in candidates:
			if "start" not in c or "end" not in c:
				continue
			normalized.append({
				"text":   c.get("text", ""),
				"label":  c.get("label", "UNKNOWN"),
				"start":  int(c["start"]),
				"end":    int(c["end"]),
				"source": source,
				"score":  float(c.get("score", 1.0 if source == "spacy" else 0.0)),
			})
		return normalized

	def _merge_spans(
		self,
		spacy_spans:  list[dict],
		gliner_spans: list[dict],
	) -> list[MergedEntity]:
		"""
		Pasos para fusionar:
		1. Para cada span de spaCy, busca el span de GLiNER con mayor solapamiento.
		2. Si el IoU (Intersection over Union) >= overlap_threshold → los fusiona en una entidad ALTA confianza.
		3. Spans sin pareja → confianza MEDIA o BAJA según modelo y score.
		"""
		matched_gliner: set[int] = set()
		merged: list[MergedEntity] = []

		for sp in spacy_spans:
			best_idx, best_iou = self._find_best_overlap(sp, gliner_spans)

			if best_idx is not None and best_iou >= self.overlap_threshold:
				# Fusión: ambos modelos detectaron este span
				gl = gliner_spans[best_idx]
				matched_gliner.add(best_idx)

				# El label de GLiNER es más específico (no tiene MISC)
				# Preferimos GLiNER excepto cuando spaCy tiene algo y GLiNER no
				label = gl["label"] if gl["label"] != "UNKNOWN" else sp["label"]

				merged.append(MergedEntity(
					text            = self._pick_text(sp, gl),
					label           = label,
					start           = min(sp["start"], gl["start"]),
					end             = max(sp["end"],   gl["end"]),
					sources         = ["spacy", "gliner"],
					score           = sp["score"] + gl["score"], 	# puedes ser también max o average (como spacy es 1, entonces siempre se muestra 1 si es max)
					confidence_tier = CONFIDENCE_HIGH,	# ALTA porque fue detectada por los dos
				))
			else:
				# Solo spaCy
				merged.append(MergedEntity(
					text            = sp["text"],
					label           = sp["label"],
					start           = sp["start"],
					end             = sp["end"],
					sources         = ["spacy"],
					score           = sp["score"],
					confidence_tier = CONFIDENCE_MEDIUM, 	# MEDIA porque solo fue detectada por una de las herramientas (spacy)
				))

		# Spans de GLiNER sin pareja
		for i, gl in enumerate(gliner_spans):
			if i in matched_gliner:
				continue
			# BAJA además de solo ser detectada por gliner, el score está abajo del umbral definido
			tier = CONFIDENCE_MEDIUM if gl["score"] >= self.gliner_medium_threshold else CONFIDENCE_LOW

			merged.append(MergedEntity(
				text            = gl["text"],
				label           = gl["label"],
				start           = gl["start"],
				end             = gl["end"],
				sources         = ["gliner"],
				score           = gl["score"],
				confidence_tier = tier
			))

		return merged

	@staticmethod
	def _find_best_overlap(
		span: dict,
		candidates: list[dict],
	) -> tuple[Optional[int], float]:
		"""Encuentra el candidato con mayor IoU respecto a span."""
		best_idx: Optional[int] = None
		best_iou: float = 0.0

		for i, cand in enumerate(candidates):
			iou = CandidateMerger._iou(
				span["start"], span["end"],
				cand["start"], cand["end"],
			)
			if iou > best_iou:
				best_iou = iou
				best_idx = i

		return best_idx, best_iou

	@staticmethod
	def _iou(s1: int, e1: int, s2: int, e2: int) -> float:
		"""Intersection over Union de dos spans de caracteres."""
		inter_start = max(s1, s2)
		inter_end   = min(e1, e2)
		intersection = max(0, inter_end - inter_start)
		if intersection == 0:
			return 0.0
		union = (e1 - s1) + (e2 - s2) - intersection
		return intersection / union if union > 0 else 0.0

	@staticmethod
	def _pick_text(spacy_span: dict, gliner_span: dict) -> str:
		"""
		Cuando ambos modelos detectan el mismo span, elige el texto más largo
		(GLiNER a veces captura el nombre completo, spaCy solo el apellido).
		"""
		t_sp = spacy_span["text"].strip()
		t_gl = gliner_span["text"].strip()
		return t_gl if len(t_gl) >= len(t_sp) else t_sp

	# ------------------------------------------------------------------
	# I/O
	# ------------------------------------------------------------------

	@staticmethod
	def _load_json(path: Path) -> dict:
		with open(path, encoding="utf-8") as f:
			return json.load(f)

	@staticmethod
	def _export(results: list[MergedDocument], output_dir: Path) -> None:
		for doc in results:
			out_path = output_dir / f"{doc.doc_id}_merged.json"
			out_path.write_text(
				json.dumps(doc.to_dict(), ensure_ascii=False, indent=4),
				encoding="utf-8",
			)
		print(f"\n  * {len(results)} archivos exportados a {output_dir}")
