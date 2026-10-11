import os, sys, time
os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np, yaml, keras, torch
from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
from fast_plate_ocr.train.model.model_builders import build_model
from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml
from fast_plate_ocr.train.model.loss import cce_loss
from keras.src.optimizers import AdamW
from seen.paths import RAW
pre = RAW / "ident_pretrained"
pc = yaml.safe_load((pre / "cct_s_v2_global_plate_config.yaml").read_text()); pc["plate_regions"] = ["a", "b", "c", "d", "e"]
open("/tmp/pc_bench.yaml", "w").write(yaml.safe_dump(pc))
cfg = load_plate_config_from_yaml("/tmp/pc_bench.yaml"); mc = load_model_config_from_yaml(pre / "cct_s_v2_global_model_config.yaml")
bs = int(sys.argv[1]); jit = sys.argv[2] == "jit"; ema = sys.argv[3] == "ema"
keras.mixed_precision.set_global_policy("mixed_bfloat16")
m = build_model(mc, cfg, enable_region_head=True)
m.compile(loss={"plate": cce_loss(37, 0.01), "region": keras.losses.CategoricalCrossentropy()}, optimizer=AdamW(1e-4, use_ema=ema), jit_compile=jit)
x = np.random.randint(0, 255, (bs, 64, 128, 3)).astype("float32")
y = {"plate": np.eye(37, dtype="float32")[np.random.randint(0, 37, (bs, 10))], "region": np.eye(5, dtype="float32")[np.random.randint(0, 5, bs)]}
for i in range(3): m.train_on_batch(x, y)
t = time.time()
for i in range(20): m.train_on_batch(x, y)
torch.cuda.synchronize()
print(f"bs={bs} jit={jit} ema={ema}: {(time.time()-t)/20*1000:.0f} ms/step", flush=True)
