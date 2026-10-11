"""List every weight/cache file the deployed pipeline opens (run on the GPU box). Output: JSON list of absolute paths."""
import json, os, sys
opened = set()
KEYS = ("/artifacts/", ".roboflow", "fast-plate-ocr", "open-image-models", "hf-cache", "huggingface", "torch/hub")
def hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if any(k in p for k in KEYS):
            opened.add(os.path.realpath(p))
sys.addaudithook(hook)
import numpy as np
from seen.pipeline import default_pipeline
from seen.predictor import Frame
pipe = default_pipeline(device="cuda:0", vlm=os.environ.get("SEEN_VLM", "local"))
img = (np.random.default_rng(0).random((540, 960, 3)) * 255).astype("uint8")
pipe.analyze([Frame(0, 0.0, img), Frame(1, 1.0, img)])
out = [{"path": p, "bytes": os.path.getsize(p)} for p in sorted(opened) if os.path.isfile(p)]
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(len(out), "files", round(sum(o["bytes"] for o in out) / 1e9, 2), "GB")
