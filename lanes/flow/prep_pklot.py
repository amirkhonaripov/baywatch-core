"""Stream PKLot.tar.gz and extract a deterministic 1-in-N subset of the *full* frames (+ their XML), skipping PKLotSegmented
(~700k tiny patches we do not use). Keeps disk I/O small on a shared drive. Selection = md5(stem) % N == 0, so it is reproducible.

    python -m lanes.flow.prep_pklot --keep-every 4
"""
import argparse
import hashlib
import tarfile
from pathlib import Path

from lanes.flow.common import raw_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-every", type=int, default=4)
    a = ap.parse_args()
    root = raw_dir("pklot")
    n = kept = 0
    with tarfile.open(root / "PKLot.tar.gz", "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.startswith("PKLot/PKLot/"):
                continue
            stem = m.name.rsplit(".", 1)[0]
            if int(hashlib.md5(stem.encode()).hexdigest(), 16) % a.keep_every:
                continue
            dst = root / m.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                f = tf.extractfile(m)
                dst.write_bytes(f.read())
            kept += 1
            if kept % 200 == 0:
                print(kept, m.name, flush=True)
    print("done", kept)


if __name__ == "__main__":
    main()
