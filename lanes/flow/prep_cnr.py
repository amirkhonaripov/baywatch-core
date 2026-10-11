"""Stream CNR-EXT_FULL_IMAGE_1000x750.tar (1.1 GB, HDD) once and write every 2nd frame (+ the 9 camera slot CSVs) to the NVMe.

    python -m lanes.flow.prep_cnr --keep-every 2
Selection = md5(frame path) % N == 0 (reproducible). Output: fast/flow/cnrext_full/<tar paths>.
"""
import argparse
import hashlib
import tarfile

from lanes.flow.common import fast_prep, raw_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-every", type=int, default=2)
    a = ap.parse_args()
    out = fast_prep("cnrext_full")
    n = kept = 0
    with tarfile.open(raw_dir("cnrpark") / "CNR-EXT_FULL_IMAGE_1000x750.tar", "r|", bufsize=16 * 1024 * 1024) as tf:
        for m in tf:
            if not m.isfile():
                continue
            n += 1
            if m.name.endswith(".csv") or int(hashlib.md5(m.name.encode()).hexdigest(), 16) % a.keep_every == 0:
                dst = out / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.write_bytes(tf.extractfile(m).read())
                kept += 1
            if n % 500 == 0:
                print(n, kept, flush=True)
    print("done", n, kept)


if __name__ == "__main__":
    main()
