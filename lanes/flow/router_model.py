"""Scene router model: SigLIP2 image embeddings -> {gate, bay, wash, street, closeup, motorcycle, unknown}.

Two heads share one frozen encoder:
  * zero-shot: cosine to prompt-ensembled text embeddings, SigLIP logit (scale*cos + bias), softmax over the 7 scenes;
  * linear probe: multinomial logistic regression on the L2-normalised image embedding (trained in router_eval.py).
``SceneRouter.predict`` returns per-frame probabilities to put into ``FramePred.scene``; ``seen.router.decide`` turns them into
a Route (low confidence -> "unknown" with a reason).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from seen.vocab import SCENES

DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"

PROMPTS: dict[str, list[str]] = {
    "gate": ["a photo of a car at a barrier gate", "a vehicle entrance gate with a boom barrier", "a car waiting at a toll booth or parking entrance",
             "a security checkpoint where vehicles enter"],
    "bay": ["a car inside an auto repair workshop", "a car on a lift in a garage service bay", "mechanics working on a car in a workshop",
            "the interior of a car service center"],
    "wash": ["a car being washed at a car wash", "a car wash bay with foam and water", "an automatic car wash tunnel", "washing a car with a hose and sponge"],
    "street": ["a street with cars seen from a traffic camera", "a parking lot full of cars seen from above", "a road with traffic from a surveillance camera",
               "an intersection with many vehicles"],
    "closeup": ["a close-up photo of a car", "a close-up of a car wheel or tyre", "a car bonnet or engine bay up close", "a damaged car body panel up close"],
    "motorcycle": ["a photo of a motorcycle", "a Ducati sport motorbike", "a person riding a motorcycle", "a scooter parked on the street"],
    "unknown": ["a photo with no vehicle in it", "a photo of an animal", "a photo of food", "a person indoors", "a landscape photo",
                "a document or a screenshot", "a room interior with furniture"],
}


# v2: more prompts per scene (written after looking at failures of v1 on the *training* sources only; chosen by training-source accuracy,
# see router_eval.py prompts). Held-out numbers are reported for both.
PROMPTS_V2: dict[str, list[str]] = {
    "gate": ["a car stopped at a toll booth or entrance barrier", "a vehicle checkpoint with a boom barrier and a guard booth", "toll plaza lanes with cars",
             "a parking garage entrance gate with a ticket machine", "a car waiting at a gate with its number plate visible"],
    "bay": ["a workshop with cars on hydraulic lifts", "a tyre fitting shop interior", "an auto repair garage with tools and mechanics", "a car on a lift in a service bay",
            "the interior of a car service center seen from a ceiling camera"],
    "wash": ["a car wash tunnel with rotating brushes", "a hand car wash with foam and a hose", "a self-service car wash bay", "a car being washed",
             "the exterior of a car wash station"],
    "street": ["a street with parked and moving cars", "a busy road with traffic seen from above", "a parking lot with many cars seen from a camera",
               "a city intersection with cars and pedestrians", "cars driving on a highway", "an outdoor scene with several vehicles far from the camera"],
    "closeup": ["a close-up photo of one car", "a car wheel and tyre close-up", "a car engine bay close-up", "a close-up of a car body panel with damage",
                "a photo of a single car with the whole car in frame"],
    "motorcycle": ["a photo of a motorcycle", "a Ducati sport motorbike", "a person riding a motorcycle", "a parked scooter or motorbike"],
    "unknown": ["a photo with no cars or vehicles", "an animal", "food on a plate", "people at an indoor event", "a building with no vehicles", "a landscape", "a document",
                "a screenshot of a computer", "a room interior with furniture"],
}
PROMPT_SETS = {"v1": PROMPTS, "v2": PROMPTS_V2}


class SigLIP2Encoder:
    def __init__(self, name: str = DEFAULT_MODEL, device: str = "cuda:0", dtype: str = "float16"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.name, self.device = name, device
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        if device.startswith("cpu"):
            dt = torch.float32
        self.model = AutoModel.from_pretrained(name, dtype=dt).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(name)
        self.dtype = dt

    @property
    def logit_params(self) -> tuple[float, float]:
        return float(self.model.logit_scale.exp().item()), float(self.model.logit_bias.item())

    def encode_images(self, images: list[np.ndarray], batch: int = 32) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(images), batch):
            inp = self.proc(images=[im for im in images[i:i + batch]], return_tensors="pt")
            pv = inp["pixel_values"].to(self.device, self.dtype)
            kw = {k: v.to(self.device) for k, v in inp.items() if k != "pixel_values"}
            with torch.inference_mode():
                f = self.model.get_image_features(pixel_values=pv, **kw)
                if not torch.is_tensor(f):  # transformers 5 may return a model-output object
                    f = getattr(f, "pooler_output", None) if getattr(f, "pooler_output", None) is not None else f[0]
                f = torch.nn.functional.normalize(f.float(), dim=-1)
            out.append(f.cpu().numpy())
        return np.concatenate(out, 0) if out else np.zeros((0, 1))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        inp = self.proc(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            f = self.model.get_text_features(**inp)
            if not torch.is_tensor(f):
                f = getattr(f, "pooler_output", None) if getattr(f, "pooler_output", None) is not None else f[0]
            f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()

    def class_text_embeddings(self, prompts: dict[str, list[str]] = PROMPTS) -> np.ndarray:
        """(7, D): mean of the normalised prompt embeddings per scene, in ``SCENES`` order, re-normalised."""
        rows = []
        for s in SCENES:
            e = self.encode_texts(prompts[s]).mean(0)
            rows.append(e / np.linalg.norm(e))
        return np.stack(rows)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def zero_shot_probs(emb: np.ndarray, text_emb: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return softmax(scale * emb @ text_emb.T + bias)


class Probe:
    """Linear probe stored as plain arrays: logits = emb @ W.T + b (optionally temperature-scaled)."""

    def __init__(self, W: np.ndarray, b: np.ndarray, classes: list[str], temperature: float = 1.0):
        self.W, self.b, self.classes, self.T = W.astype(np.float32), b.astype(np.float32), list(classes), float(temperature)

    def probs(self, emb: np.ndarray) -> np.ndarray:
        return softmax((emb @ self.W.T + self.b) / self.T)

    def save(self, path: Path, meta: dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W=self.W, b=self.b, classes=np.array(self.classes), T=self.T)
        if meta:
            path.with_suffix(".json").write_text(json.dumps(meta, indent=1))

    @staticmethod
    def load(path: Path) -> "Probe":
        z = np.load(path, allow_pickle=False)
        return Probe(z["W"], z["b"], [str(c) for c in z["classes"]], float(z["T"]))


class SceneRouter:
    """Predictor-compatible (``seen.predictor.Predictor``): fills ``FramePred.scene``."""

    role = "router"
    name = "Scene router (SigLIP2 + linear probe)"

    def __init__(self, encoder_name: str = DEFAULT_MODEL, probe_path: str | Path | None = None, mode: str = "probe", device: str = "cuda:0",
                 blend: float = 0.5, prompt_set: str = "v2"):
        self.encoder_name, self.probe_path, self.mode, self.blend, self.prompt_set = encoder_name, probe_path, mode, blend, prompt_set
        self.version = f"{encoder_name.split('/')[-1]}+{mode}-v1"
        if mode == "zeroshot":
            self.name = "Scene router (SigLIP2 zero-shot)"
        self.enc = None
        self.probe = None
        self.text_emb = None
        self.device = device

    def load(self, device: str = "cuda:0") -> None:
        self.device = device
        self.enc = SigLIP2Encoder(self.encoder_name, device)
        self.text_emb = self.enc.class_text_embeddings(PROMPT_SETS[self.prompt_set])
        self.scale, self.bias = self.enc.logit_params
        if self.mode in ("probe", "blend"):
            self.probe = Probe.load(Path(self.probe_path))

    def probs_from_embeddings(self, emb: np.ndarray) -> np.ndarray:
        zs = zero_shot_probs(emb, self.text_emb, self.scale, self.bias) if self.mode in ("zeroshot", "blend") else None
        pr = self.probe.probs(emb) if self.probe is not None else None
        if self.mode == "zeroshot":
            return zs
        if self.mode == "probe":
            return pr
        return self.blend * pr + (1 - self.blend) * zs

    def predict(self, frames, context=None):
        from seen.predictor import empty_preds

        out = empty_preds(frames)
        if not frames:
            return out
        t0 = time.perf_counter()
        emb = self.enc.encode_images([f.image for f in frames])
        p = self.probs_from_embeddings(emb)
        dt = (time.perf_counter() - t0) * 1000 / len(frames)
        for o, row in zip(out, p):
            o.scene = {s: float(v) for s, v in zip(SCENES, row)}
            o.timings_ms["router"] = dt
        return out
