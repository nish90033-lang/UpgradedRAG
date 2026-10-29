"""Download and unpack a BEIR dataset (run this on your own machine).

    python scripts/download_beir.py scifact data

Datasets that fit on a laptop: scifact (5k docs), nfcorpus (3.6k), fiqa (57k), arguana (8.7k).
"""

from __future__ import annotations

import os
import shutil
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import certifi

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def _download(url: str, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    ca_bundle = os.environ.get("SSL_CERT_FILE") or certifi.where()
    context = ssl.create_default_context(cafile=ca_bundle)
    try:
        with urllib.request.urlopen(url, context=context) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
    except urllib.error.URLError as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                f"TLS certificate verification failed while downloading {url}. "
                "Update certifi with `python -m pip install --upgrade certifi`, or set "
                "SSL_CERT_FILE to your organization's CA bundle. Certificate verification "
                "remains enabled."
            ) from exc
        raise
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.replace(target)


def main(name: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{name}.zip"
    if not target.exists():
        print(f"downloading {name} ...")
        _download(BASE.format(name=name), target)
    with zipfile.ZipFile(target) as z:
        z.extractall(out_dir)
    print(f"ready: {out_dir / name}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
