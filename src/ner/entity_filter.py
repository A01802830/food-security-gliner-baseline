"""
src/ner/entity_filter.py

Filtro pre-LLM: elimina falsos positivos evidentes
antes de gastar tokens en el modelo de lenguaje.

Capas de filtrado (en orden):
  1. Longitud mínima
  2. Lista negra del dominio (vocabulario de seguridad alimentaria)
  3. Patrones posesivos ("mi X", "su X", "nuestro X")
  4. Tokens que parecen verbos o palabras funcionales
  5. Score mínimo por confidence_tier
  6. MISC sin respaldo de GLiNER (opcional, configurable)

Diseño conservador: ante la duda, conservar.
El LLM es quien decide los casos ambiguos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .candidate_merger import MergedEntity, MergedDocument, CandidateMerger


# ---------------------------------------------------------------------------
# Listas de filtrado — ajusta según tus entrevistas reales
# ---------------------------------------------------------------------------

# Palabras del dominio de seguridad alimentaria que NER confunde con entidades
DOMAIN_BLACKLIST: set[str] = {
	# Cultivos y prácticas agrícolas
	"maíz", "maiz", "milpa", "milpas", "frijol", "frijoles", "calabaza",
	"ejido", "ejidos", "nopal", "nopales", "tortilla", "tortillas",
	"siembra", "cosecha", "temporal", "riego", "parcela", "parcelas",
	"campo", "campos", "terreno", "terrenos", "rancho", "huerto", "huertos",
	# Roles genéricos (sin nombre propio)
	"productor", "productores", "campesino", "campesinos", "agricultor",
	"agricultores", "jornalero", "jornaleros", "trabajador", "trabajadores",
	"familia", "familias", "comunidad", "comunidades",
	# Términos de gobierno y programas
	"gobierno", "municipio", "municipios", "programa", "programas",
	"proyecto", "proyectos", "apoyo", "apoyos", "subsidio", "subsidios",
	"secretaría", "secretaria", "dependencia",
	# Palabras funcionales que GLiNER/spaCy etiquetan por error
	"año", "años", "mes", "meses", "semana", "semanas", "día", "días",
	"hectárea", "hectáreas", "hectarea", "hectareas", "kilo", "kilos",
	"tonelada", "toneladas", "peso", "pesos",
}

# Prefijos posesivos que indican referencia genérica, no nombre propio
POSSESSIVE_PREFIXES: tuple[str, ...] = (
	"mi ", "mis ", "su ", "sus ", "nuestro ", "nuestra ",
	"nuestros ", "nuestras ", "tu ", "tus ",
)

# Verbos comunes que spaCy/GLiNER a veces etiquetan como entidades
VERB_PATTERNS: list[str] = [
	r"^(sembrar|cosechar|trabajar|vivir|tener|hacer|decir|ir|venir|"
	r"comprar|vender|producir|cultivar|regar|comer|ganar|perder|"
	r"saber|poder|querer|deber|llegar|salir|entrar|usar|necesitar)"
	r"(se|me|te|le|nos|les|lo|la|los|las)?$",
]
VERB_RE = re.compile("|".join(VERB_PATTERNS), re.IGNORECASE)

# Patrones de texto que casi siempre son FP
NOISE_PATTERNS: list[str] = [
	r"^\d+$",                    # solo números
	r"^[\W_]+$",                 # solo puntuación/símbolos
	r"^\w$",                     # un solo carácter
	r"^(el|la|los|las|un|una|unos|unas|y|o|de|del|al|con|por|para|en|a)$",  # artículos/preposiciones
]
NOISE_RE = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Resultado del filtro
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
	kept:    list[MergedEntity] = field(default_factory=list)
	removed: list[dict]         = field(default_factory=list)   # {entity, reason}

	def summary(self) -> str:
		reasons = {}
		for r in self.removed:
			reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
		lines = [
			f"  Total       : {len(self.kept) + len(self.removed)}",
			f"  Conservadas : {len(self.kept)}",
			f"  Eliminadas  : {len(self.removed)}",
		]
		for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
			lines.append(f"    · {reason:<35} {count}")
		return "  ── RESUMEN ──\n" + "\n".join(lines)
	
	def __str__(self) -> str:
		return self.summary()


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class EntityFilter:
	"""
	Filtro determinista de entidades candidatas.

	Uso:
		f = EntityFilter()
		result = f.filter(merged_document)
		print(result.summary())

		# Ver qué se eliminó y por qué
		for r in result.removed:
			print(r["entity"].text, "→", r["reason"])
	"""

	def __init__(
		self,
		min_length:               int   = 3,
		domain_blacklist:         Optional[set[str]] = None,
		possessive_prefixes:      Optional[tuple]    = None,
		filter_misc_solo_spacy:   bool  = True,    # eliminar MISC que solo detectó spaCy
		extra_blacklist:          Optional[set[str]] = None,  # palabras adicionales tuyas
		filter_medium:            bool  = False # por default se convservan las entidades con confianza media
	):
		self.min_length             = min_length
		self.blacklist              = (domain_blacklist or DOMAIN_BLACKLIST).copy()
		self.possessive_prefixes    = possessive_prefixes or POSSESSIVE_PREFIXES
		self.filter_misc_solo_spacy = filter_misc_solo_spacy
		self.filter_medium          = filter_medium

		if extra_blacklist:
			self.blacklist |= {w.lower() for w in extra_blacklist}

	# ------------------------------------------------------------------
	# Métodos
	# ------------------------------------------------------------------

	def filter(self, doc: MergedDocument) -> FilterResult:
		"""Filtra las entidades de un documento y devuelve FilterResult."""
		result = FilterResult()
		for entity in doc.entities:
			reason = self._should_remove(entity)
			if reason:
				result.removed.append({"entity": entity, "reason": reason})
			else:
				result.kept.append(entity)
		return result

	def filter_batch(
		self,
		docs: list[MergedDocument],
		output_dir: Path,
	) -> None:
		"""
		Filtra una lista de documentos y los guarda.
		"""
		results = []
		for doc in docs:
			info_filter = self.filter(doc)
			print(
				f"  {doc.doc_id:<40} "
				f"{len(doc.entities):>3} → {len(info_filter.kept):>3} entidades "
				f"(-{len(info_filter.removed)} eliminadas)"
			)
			fr = self.apply(doc)
			results.append(fr)
		
		if output_dir:
			output_dir.mkdir(parents=True, exist_ok=True)
			CandidateMerger._export(results, output_dir)

	def apply(self, doc: MergedDocument) -> MergedDocument:
		"""
		Aplica el filtro y devuelve un MergedDocument nuevo con solo
		las entidades que pasaron. Útil para encadenar con el merger.
		"""
		fr = self.filter(doc)
		return MergedDocument(
			doc_id   = doc.doc_id,
			text     = doc.text,
			entities = fr.kept,
		)

	# ------------------------------------------------------------------
	# Lógica de filtrado
	# ------------------------------------------------------------------

	def _should_remove(self, entity: MergedEntity) -> Optional[str]:
		"""
		Devuelve el motivo de eliminación, o None si debe conservarse.
		Orden importa: del más barato al más costoso de evaluar.
		"""
		text     = entity.text.strip()
		text_low = text.lower()

		# 1. Longitud mínima
		if len(text) < self.min_length:
			return f"longitud < {self.min_length}"

		# 2. Patrones de ruido (números, puntuación, artículos)
		if NOISE_RE.match(text_low):
			return "patrón de ruido (número/puntuación/artículo)"

		# 3. Lista negra del dominio
		if text_low in self.blacklist:
			return "lista negra del dominio"

		# 4. Prefijo posesivo ("mi parcela", "su comunidad")
		for prefix in self.possessive_prefixes:
			if text_low.startswith(prefix):
				# Excepción: si después del posesivo hay un nombre propio
				# (empieza en mayúscula en el texto original), conservar
				after_prefix = text[len(prefix):]
				if after_prefix and after_prefix[0].isupper():
					break   # "mi Ana" → conservar (referencia a persona con nombre)
				return f"posesivo sin nombre propio ({prefix.strip()}…)"

		# 5. Verbo detectado por error
		if VERB_RE.match(text_low):
			return "verbo detectado por error"

		# 6. MISC sin respaldo de GLiNER (solo spaCy lo marcó como MISC)
		if (
			self.filter_misc_solo_spacy
			and entity.label == "MISC"
			and entity.sources == ["spacy"]
		):
			return "MISC solo spaCy sin respaldo GLiNER"

		# 7. Score muy bajo en entidades de confianza baja (entity.score)
		if entity.confidence_tier == "baja":
			return f"Confidence bajo"
		
		# 8. Entidades de confianza mediua se filtran
		if (
			self.filter_medium
			and entity.confidence_tier == "media"
		):
			return f"Confidence medio"

		return None  # pasa el filtro

	# ------------------------------------------------------------------
	# Utilidades de inspección
	# ------------------------------------------------------------------

	def preview(self, doc: MergedDocument, show_kept: bool = True) -> None:
		"""
		Muestra en consola qué se conserva y qué se elimina,
		útil para calibrar las listas antes de correr en todo el corpus.
		"""
		fr = self.filter(doc)
		print(f"\n{'='*60}")
		print(f"Documento: {doc.doc_id}")
		print(fr.summary())

		if show_kept and fr.kept:
			print(f"\n  ── CONSERVADAS ──")
			for e in fr.kept:
				tier_icon = {"alta": "alta", "media": "media", "baja": "baja"}.get(
					e.confidence_tier, "⚪"
				)
				print(
					f"  {tier_icon:5} [{e.label:14}] '{e.text}'"
					f"  fuentes={e.sources}"
					f"  score={e.score:.2f}"
				)

		if fr.removed:
			print(f"\n  ── ELIMINADAS ──")
			for r in fr.removed:
				e = r["entity"]
				print(f"  x [{e.label:14}] '{e.text}'  → {r['reason']}")
