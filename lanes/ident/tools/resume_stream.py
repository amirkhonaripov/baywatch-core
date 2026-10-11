"""Stream a URL to stdout with automatic Range-resume on dropped connections (for `| xz -d | tar -x` pipelines)."""
import sys
import time

import requests

url = sys.argv[1]
out = sys.stdout.buffer
off = 0
size = None
tries = 0
while True:
    try:
        h = {"Range": f"bytes={off}-"} if off else {}
        with requests.get(url, headers=h, stream=True, timeout=(30, 120), allow_redirects=True) as r:
            r.raise_for_status()
            if size is None:
                cr = r.headers.get("Content-Range")
                size = int(cr.split("/")[-1]) if cr else int(r.headers.get("Content-Length", 0)) + off
            for chunk in r.iter_content(8 << 20):
                out.write(chunk)
                off += len(chunk)
        if size and off >= size:
            break
        raise IOError(f"short read {off}/{size}")
    except Exception as e:  # noqa: BLE001
        tries += 1
        print(f"[resume_stream] retry {tries} at {off}: {e}", file=sys.stderr, flush=True)
        if tries > 200:
            raise
        time.sleep(min(30, 2 * tries))
out.flush()
