"""Corpus download CLI. Starter recipe (~25 GB):

    python -m prism_tse.data.download librispeech --data_root data_root
    python -m prism_tse.data.download musan       --data_root data_root
    python -m prism_tse.data.download rirs_openslr --data_root data_root

Optional extras: `librispeech --also-360`, `dns` (best-effort URLs), `vctk`.
Downloads are resumable (wget -c / curl -C -) and idempotent (marker files).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

OPENSLR = "https://www.openslr.org/resources"

LIBRISPEECH_PARTS = {
    "train-clean-100": f"{OPENSLR}/12/train-clean-100.tar.gz",   # ~6.3 GB
    "dev-clean": f"{OPENSLR}/12/dev-clean.tar.gz",               # ~0.3 GB
}
LIBRISPEECH_360 = {"train-clean-360": f"{OPENSLR}/12/train-clean-360.tar.gz"}  # ~23 GB
MUSAN_URL = f"{OPENSLR}/17/musan.tar.gz"                          # ~11 GB
RIRS_URL = f"{OPENSLR}/28/rirs_noises.zip"                        # ~1.3 GB
VCTK_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

# Best-effort: DNS challenge blob URLs change between challenge editions.
# If these 404, check https://github.com/microsoft/DNS-Challenge download scripts.
DNS_NOISE_URL = (
    "https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband/"
    "noise_fullband/datasets_fullband.noise_fullband.audio_{part:03d}.tar.bz2"
)


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(dest), url]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "-o", str(dest), url]
    else:
        raise RuntimeError("Need wget or curl on PATH")
    print(f"[download] {url}")
    subprocess.run(cmd, check=True)


def extract(archive: Path, out_dir: Path) -> None:
    print(f"[extract] {archive.name} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(out_dir)
    elif archive.suffixes[-2:] == [".tar", ".bz2"]:
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(out_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


def _run(name: str, raw: Path, steps) -> None:
    marker = raw / f".done_{name}"
    if marker.exists():
        print(f"[skip] {name} already downloaded ({marker})")
        return
    steps()
    marker.touch()
    print(f"[done] {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", choices=["librispeech", "musan", "rirs_openslr", "dns", "vctk"])
    ap.add_argument("--data_root", default="data_root")
    ap.add_argument("--also-360", action="store_true",
                    help="librispeech: also fetch train-clean-360 (~23 GB)")
    ap.add_argument("--max-parts", type=int, default=10,
                    help="dns: number of noise archive parts to fetch")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    raw = Path(args.data_root) / "raw"
    dl = raw / "_archives"

    if args.corpus == "librispeech":
        parts = dict(LIBRISPEECH_PARTS)
        if args.also_360:
            parts.update(LIBRISPEECH_360)

        def steps():
            for name, url in parts.items():
                if (raw / "LibriSpeech" / name).exists():
                    print(f"[skip] {name} already extracted")
                    continue
                arc = dl / Path(url).name
                fetch(url, arc)
                extract(arc, raw)  # archives contain LibriSpeech/<part>/...
                if not args.keep_archives:
                    arc.unlink(missing_ok=True)

        _run(f"librispeech{'_360' if args.also_360 else ''}", raw, steps)

    elif args.corpus == "musan":
        def steps():
            arc = dl / "musan.tar.gz"
            fetch(MUSAN_URL, arc)
            extract(arc, raw)  # -> raw/musan/{noise,music,speech}
            if not args.keep_archives:
                arc.unlink(missing_ok=True)

        _run("musan", raw, steps)

    elif args.corpus == "rirs_openslr":
        def steps():
            arc = dl / "rirs_noises.zip"
            fetch(RIRS_URL, arc)
            extract(arc, raw)  # -> raw/RIRS_NOISES/...
            if not args.keep_archives:
                arc.unlink(missing_ok=True)

        _run("rirs_openslr", raw, steps)

    elif args.corpus == "dns":
        def steps():
            ok = 0
            for part in range(args.max_parts):
                url = DNS_NOISE_URL.format(part=part)
                arc = dl / Path(url).name
                try:
                    fetch(url, arc)
                    extract(arc, raw / "dns_noise")
                    if not args.keep_archives:
                        arc.unlink(missing_ok=True)
                    ok += 1
                except (subprocess.CalledProcessError, tarfile.TarError) as e:
                    print(f"[dns] part {part} failed ({e}); URL scheme may have "
                          "changed — see https://github.com/microsoft/DNS-Challenge",
                          file=sys.stderr)
                    break
            if ok == 0:
                raise RuntimeError("No DNS parts downloaded. MUSAN covers the "
                                   "starter recipe; DNS is optional.")
            print(f"[dns] downloaded {ok} parts (48 kHz — build_manifests will resample)")

        _run("dns", raw, steps)

    elif args.corpus == "vctk":
        def steps():
            arc = dl / "VCTK-Corpus-0.92.zip"
            fetch(VCTK_URL, arc)
            extract(arc, raw / "vctk")
            if not args.keep_archives:
                arc.unlink(missing_ok=True)

        _run("vctk", raw, steps)


if __name__ == "__main__":
    main()
