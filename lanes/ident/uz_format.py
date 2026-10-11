"""Uzbek licence-plate formats, validators and the decoding constraints derived from them.

Sources (see docs/lanes/ident_sources.md and research/data_models.md section 2):
  private persons   ``DD L DDD LL``   e.g. 01 A 123 BC   canonical text ``01A123BC``
  legal entities    ``DD DDD LLL``    e.g. 01 123 ABC    canonical text ``01123ABC``
  electric vehicles same character layouts as the two above, region field on a GREEN background (visual cue only)
  motorcycles       square plate, 3 digits + 2 letters, region at upper right (layout from Wikipedia/goldenpages,
                    NOT verified against real photos in our data); canonical text ``DD DDD LL`` = ``01123AB``
Region codes 01..99 (01-09 Tashkent city ... 95-99 Karakalpakstan). Latin letters only. The letter ``I`` is excluded
by the plate-recognition-uz validators (empirical, soft rule); we keep it as a legal but rare character.

Canonical text = region + series characters without separators, upper case (what the OCR model emits and what
``Det.text`` carries).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
ALPHABET = DIGITS + LETTERS + "_"  # model vocabulary; "_" = pad
PAD = "_"

REGION_NAMES = {
    (1, 9): "Tashkent city", (10, 19): "Tashkent region", (20, 24): "Syrdarya", (25, 29): "Jizzakh",
    (30, 39): "Samarkand", (40, 49): "Fergana", (50, 59): "Namangan", (60, 69): "Andijan", (70, 74): "Kashkadarya",
    (75, 79): "Surkhandarya", (80, 84): "Bukhara", (85, 89): "Navoi", (90, 94): "Khorezm", (95, 99): "Karakalpakstan",
}
BASE_REGION_CODES = (1, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95)
VALID_REGIONS = tuple(f"{i:02d}" for i in range(1, 100))

# Format names are the values of Det.extra["format"].
UZ_PRIVATE, UZ_LEGAL, UZ_EV, UZ_MOTO, FOREIGN, UNKNOWN = "uz_private", "uz_legal", "uz_ev", "uz_moto", "foreign", "unknown"
FORMAT_NAMES = (UZ_PRIVATE, UZ_LEGAL, UZ_EV, UZ_MOTO, FOREIGN, UNKNOWN)

_R = r"(0[1-9]|[1-9]\d)"
PATTERNS = {
    UZ_PRIVATE: re.compile(rf"^{_R}([A-Z])(\d{{3}})([A-Z]{{2}})$"),
    UZ_LEGAL: re.compile(rf"^{_R}(\d{{3}})([A-Z]{{3}})$"),
    UZ_MOTO: re.compile(rf"^{_R}(\d{{3}})([A-Z]{{2}})$"),
}

# Per-slot character classes of each text layout ("d" digit, "l" letter); used by constrained decoding.
LAYOUTS = {
    "private": "ddldddll",
    "legal": "dddddlll",
    "moto": "dddddll",
}
LAYOUT_TO_FORMAT = {"private": UZ_PRIVATE, "legal": UZ_LEGAL, "moto": UZ_MOTO}


@dataclass(frozen=True)
class Parsed:
    format: str  # uz_private / uz_legal / uz_moto / unknown (EV and foreign need visual evidence)
    valid: bool
    region: str | None = None
    series: str | None = None
    number: str | None = None


def normalise(text: str | None) -> str:
    """Upper-case, drop separators / padding."""
    if not text:
        return ""
    return re.sub(r"[^0-9A-Z]", "", text.upper().replace(PAD, ""))


def parse(text: str | None) -> Parsed:
    t = normalise(text)
    for fmt in (UZ_PRIVATE, UZ_LEGAL, UZ_MOTO):
        m = PATTERNS[fmt].match(t)
        if m:
            g = m.groups()
            if fmt == UZ_PRIVATE:
                return Parsed(fmt, True, g[0], g[1] + g[3], g[2])
            if fmt == UZ_LEGAL:
                return Parsed(fmt, True, g[0], g[2], g[1])
            return Parsed(fmt, True, g[0], g[2], g[1])
    return Parsed(UNKNOWN, False)


def is_valid_uz(text: str | None) -> bool:
    return parse(text).valid


def format_of(text: str | None, *, green_region: bool = False, foreign_hint: bool = False) -> tuple[str, bool]:
    """(format, format_valid) for a recognised text. ``green_region`` (visual) upgrades private/legal to uz_ev;
    ``foreign_hint`` (visual/format head) marks non-Uzbek plates."""
    p = parse(text)
    if foreign_hint:
        return FOREIGN, False
    if p.valid and green_region and p.format in (UZ_PRIVATE, UZ_LEGAL):
        return UZ_EV, True
    return p.format, p.valid


def display(text: str) -> str:
    """Human readable spacing: 01 A 123 BC / 01 123 ABC / 01 123 AB."""
    p = parse(text)
    t = normalise(text)
    if p.format == UZ_PRIVATE:
        return f"{t[:2]} {t[2]} {t[3:6]} {t[6:]}"
    if p.format in (UZ_LEGAL, UZ_MOTO):
        return f"{t[:2]} {t[2:5]} {t[5:]}"
    return t


# --------------------------------------------------------------------------------------------- sampling (synthetic)
def sample_region(rng) -> str:
    """Region code with a realistic skew: Tashkent (01) heavy, base codes frequent, the rest uniform."""
    r = rng.random()
    if r < 0.18:
        return "01"
    if r < 0.58:
        return f"{rng.choice(BASE_REGION_CODES):02d}"
    return f"{rng.randint(1, 99):02d}"


def _letter(rng, allow_i: float = 0.004) -> str:
    if rng.random() < allow_i:
        return "I"
    return rng.choice("ABCDEFGHJKLMNOPQRSTUVWXYZ")


def sample_text(rng, fmt: str) -> str:
    region = sample_region(rng)
    if fmt in (UZ_PRIVATE, UZ_EV):
        # EV plates exist in the private layout as well as the legal one; callers pick via ev_layout.
        return f"{region}{_letter(rng)}{rng.randint(0, 999):03d}{_letter(rng)}{_letter(rng)}"
    if fmt == UZ_LEGAL:
        return f"{region}{rng.randint(0, 999):03d}{_letter(rng)}{_letter(rng)}{_letter(rng)}"
    if fmt == UZ_MOTO:
        return f"{region}{rng.randint(0, 999):03d}{_letter(rng)}{_letter(rng)}"
    raise ValueError(fmt)


# ---------------------------------------------------------------------------------------- constrained decoding
def constrained_decode(probs, alphabet: str = ALPHABET, layouts=("private", "legal", "moto"),
                       region_prior: bool = True):
    """Best valid-Uzbek reading under per-slot class constraints.

    ``probs``: array (slots, vocab) of per-slot probabilities (rows sum to 1) as produced by the fast-plate-ocr head.
    Slots are conditionally independent in that head, so the constrained arg-max per layout is exact (no beam
    needed). Returns (text, log_score, layout) for the highest-scoring layout (pad slots after the text must be PAD),
    or ``None`` if the array is too short. Region code 00 is excluded.
    """
    import numpy as np

    p = np.asarray(probs, dtype=np.float64)
    lp = np.log(np.clip(p, 1e-9, 1.0))
    slots = lp.shape[0]
    idx = {c: i for i, c in enumerate(alphabet)}
    dig = np.array([idx[c] for c in DIGITS])
    let = np.array([idx[c] for c in LETTERS])
    pad = idx[PAD]
    best = None
    for name in layouts:
        lay = LAYOUTS[name]
        if len(lay) > slots:
            continue
        text, score = [], 0.0
        for s, kind in enumerate(lay):
            cand = dig if kind == "d" else let
            j = int(cand[np.argmax(lp[s, cand])])
            text.append(alphabet[j])
            score += float(lp[s, j])
        # region 00 is invalid: if both first slots picked '0', take the better alternative
        if "".join(text[:2]) == "00":
            alt = []
            for a in dig:
                for b in dig:
                    if (a, b) != (dig[0], dig[0]):
                        alt.append((lp[0, a] + lp[1, b], a, b))
            sc, a, b = max(alt)
            score += float(sc - (lp[0, dig[0]] + lp[1, dig[0]]))
            text[0], text[1] = alphabet[int(a)], alphabet[int(b)]
        for s in range(len(lay), slots):
            score += float(lp[s, pad])
        if best is None or score > best[1]:
            best = ("".join(text), score, name)
    return best
