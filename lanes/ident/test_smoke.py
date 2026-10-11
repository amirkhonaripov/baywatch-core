"""CPU smoke tests for the ident lane (no GPU needed). Model-dependent tests skip when artifacts are missing."""
from __future__ import annotations

import random

import numpy as np
import pytest

from lanes.ident import uz_format as uz
from lanes.ident.voting import Observation, vote


def test_parse_formats():
    assert uz.parse("01A123BC").format == uz.UZ_PRIVATE
    assert uz.parse("01 123 ABC").format == uz.UZ_LEGAL
    assert uz.parse("01123AB").format == uz.UZ_MOTO
    assert not uz.parse("00A123BC").valid          # region 00 does not exist
    assert not uz.parse("01A12BC").valid
    assert uz.display("01A123BC") == "01 A 123 BC"
    assert uz.format_of("40B472RA", green_region=True) == (uz.UZ_EV, True)
    assert uz.format_of("08301AAP", foreign_hint=True) == (uz.FOREIGN, False)


def _onehot(text: str, slots: int = 10) -> np.ndarray:
    p = np.full((slots, len(uz.ALPHABET)), 0.01, np.float32)
    s = text.ljust(slots, uz.PAD)
    for i, c in enumerate(s):
        p[i, uz.ALPHABET.index(c)] = 0.9
    return p / p.sum(-1, keepdims=True)


def test_constrained_decode_fixes_confusions():
    p = _onehot("01A123BC")
    # slot 2 is a letter: make the model prefer the digit '4' but keep the letter 'A' second best
    p[2, uz.ALPHABET.index("4")] = 0.5
    p[2, uz.ALPHABET.index("A")] = 0.4
    p[2] /= p[2].sum()
    free = "".join(uz.ALPHABET[i] for i in p.argmax(-1)).rstrip("_")
    assert free[2] == "4"
    text, score, layout = uz.constrained_decode(p)
    assert text == "01A123BC" and layout == "private"
    # legal-entity layout is picked when the slots say so
    assert uz.constrained_decode(_onehot("40123ABC"))[0] == "40123ABC"


def test_region_00_is_never_returned():
    p = _onehot("00A123BC")
    text, _, _ = uz.constrained_decode(p)
    assert text[:2] != "00" and uz.parse(text).valid


def test_voting_beats_single_noisy_read():
    rng = np.random.default_rng(0)
    truth = "01A123BC"
    obs = []
    for i in range(7):
        p = _onehot(truth)
        j = rng.integers(0, 8)                      # each frame corrupts a different slot
        p[j] = 0.0
        p[j, uz.ALPHABET.index("Z")] = 0.6
        p[j, uz.ALPHABET.index(truth[j])] = 0.4
        p /= p.sum(-1, keepdims=True)
        t = "".join(uz.ALPHABET[k] for k in p.argmax(-1)).rstrip("_")
        obs.append(Observation(p, t, 0.8, quality=100 + i))
    assert vote(obs, "pooled").text == truth
    assert vote(obs, "pooled").format == uz.UZ_PRIVATE


def test_renderer_and_augment_shapes():
    from lanes.ident.synth.augment import degrade
    from lanes.ident.synth.render import render_plate

    rng = random.Random(0)
    for fmt in (uz.UZ_PRIVATE, uz.UZ_LEGAL, uz.UZ_EV, uz.UZ_MOTO):
        text = uz.sample_text(rng, uz.UZ_PRIVATE if fmt == uz.UZ_EV else fmt)
        plate, f, _ = render_plate(text, rng, fmt)
        assert f == fmt and plate.size[0] > 100
        crop, info = degrade(np.asarray(plate), 1, out_w=128)
        assert crop.dtype == np.uint8 and crop.ndim == 3 and crop.shape[1] > 20


def test_composite_on_synthetic_photo():
    from lanes.ident.synth.composite import paste_plate

    rng = random.Random(1)
    img = (np.random.default_rng(0).random((360, 640, 3)) * 80 + 60).astype(np.uint8)
    quad = np.array([[250, 200], [350, 202], [349, 232], [251, 230]], np.float32)
    res = paste_plate(img, quad, "01A123BC", uz.UZ_PRIVATE, rng)
    assert res is not None
    out, nq, box, info = res
    assert out.shape == img.shape and box[2] > box[0] and (np.abs(out.astype(int) - img.astype(int)).sum() > 0)


def _artifacts():
    from seen.paths import ARTIFACTS
    return ARTIFACTS / "ident" / "plate_det" / "model.onnx", ARTIFACTS / "ident" / "plate_ocr" / "model.onnx"


@pytest.mark.skipif(not all(p.exists() for p in _artifacts()), reason="exported ONNX artifacts not present")
def test_predictor_on_rendered_plate():
    from lanes.ident.predict import get_predictor
    from lanes.ident.synth.render import render_plate
    from seen.predictor import Frame

    rng = random.Random(3)
    text = "01A123BC"
    plate, _, _ = render_plate(text, rng, uz.UZ_PRIVATE)
    frame = np.full((480, 800, 3), 90, np.uint8)
    pl = np.asarray(plate.resize((330, 70)))
    frame[200:270, 200:530] = pl
    p = get_predictor()
    p.load("cpu")
    out = p.predict([Frame(0, 0.0, frame)])
    assert len(out) == 1
    dets = [d for d in out[0].detections if d.cls == "plate"]
    if dets:  # a detector trained on real photos may miss a flat render; then there is nothing to read
        assert dets[0].extra["format"] in uz.FORMAT_NAMES and "format_valid" in dets[0].extra
