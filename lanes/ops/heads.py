"""Multi-label linear head over [frozen crop embedding, detector geometry] with masked BCE + per-state Platt scaling.

Learned states: HEAD_STATES. `idle` is derived (car_present x (1 - work)). The head is a single Linear layer, so the
whole thing (standardisation + linear + Platt) folds into one affine map + sigmoid for ONNX export.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

HEAD_STATES = ["car_present", "raised", "wheel_off", "hood_open", "person_at_car", "person_under_car", "work", "washing"]


@dataclass
class HeadModel:
    mean: np.ndarray  # D
    std: np.ndarray  # D
    W: np.ndarray  # D x S
    b: np.ndarray  # S
    platt_a: np.ndarray  # S
    platt_b: np.ndarray  # S
    states: list[str]

    def logits(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std) @ self.W + self.b

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        z = self.logits(X)
        if calibrated:
            z = self.platt_a * z + self.platt_b
        return 1 / (1 + np.exp(-z))

    def folded(self) -> tuple[np.ndarray, np.ndarray]:
        """(W', b') with sigmoid(X @ W' + b') == predict_proba(X)."""
        W = self.W / self.std[:, None] * self.platt_a[None, :]
        b = (self.b - (self.mean / self.std) @ self.W) * self.platt_a + self.platt_b
        return W.astype(np.float32), b.astype(np.float32)

    def save(self, path) -> None:
        np.savez(path, mean=self.mean, std=self.std, W=self.W, b=self.b, platt_a=self.platt_a, platt_b=self.platt_b,
                 states=np.array(self.states))

    @staticmethod
    def load(path) -> "HeadModel":
        z = np.load(path, allow_pickle=False)
        return HeadModel(z["mean"], z["std"], z["W"], z["b"], z["platt_a"], z["platt_b"], [str(s) for s in z["states"]])


def fit_head(X: np.ndarray, Y: np.ndarray, Wt: np.ndarray, *, wd: float = 1e-3, epochs: int = 40, lr: float = 2e-3,
             batch: int = 2048, device: str = "cuda:0", seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X: N x D, Y: N x S in {0,1} or soft [0,1], Wt: N x S sample weights (0 = unknown label). Returns mean,std,W,b."""
    torch.manual_seed(seed)
    mean = X.mean(0)
    std = X.std(0) + 1e-4
    dev = device if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(((X - mean) / std).astype(np.float32))
    Yt = torch.from_numpy(Y.astype(np.float32))
    Wtt = torch.from_numpy(Wt.astype(np.float32))
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    n = len(Xt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ((n + batch - 1) // batch))
    # normalise per-state weight so every state contributes comparably
    colw = Wtt.sum(0).clamp(min=1.0)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            x, y, w = Xt[idx].to(dev), Yt[idx].to(dev), Wtt[idx].to(dev)
            z = lin(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y, weight=w, reduction="none")
            loss = (loss.sum(0) / colw.to(dev) * (n / batch)).sum() / Y.shape[1]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    W = lin.weight.detach().cpu().numpy().T.copy()
    b = lin.bias.detach().cpu().numpy().copy()
    return mean, std, W, b


def fit_platt(z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """Logistic regression of y on the head logit z (1-D). Returns (a, b)."""
    m = ~np.isnan(y)
    z, y = z[m], y[m]
    w = np.ones_like(y) if w is None else w[m]
    if len(np.unique(y.round())) < 2 or len(y) < 20:
        return 1.0, 0.0
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    zt, yt, wt = (torch.from_numpy(v.astype(np.float32)) for v in (z, y, w))
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * zt + b, yt, weight=wt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.detach()), float(b.detach())


def to_onnx(model: HeadModel, path, opset: int = 17) -> None:
    """Folded head as a 2-node ONNX graph: probs = Sigmoid(features @ W + b). Built with onnx.helper (no exporter)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    W, b = model.folded()
    D, S = W.shape
    g = helper.make_graph(
        [helper.make_node("Gemm", ["features", "W", "b"], ["logits"]), helper.make_node("Sigmoid", ["logits"], ["probs"])],
        "ops_bay_state_head",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", D])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["n", S])],
        initializer=[numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)], producer_name="seen-ops")
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    (path.parent / "head_states.json").write_text(json.dumps(model.states))
