#!/usr/bin/env python3
"""Fetch the MIRD archives from the public folder linked by the author.

No account cookies or user cache are read/written. SHA256 values are local
provenance checks, not independently published publisher checksums.
"""

import argparse
import hashlib
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FOLDER = "https://drive.google.com/drive/folders/1dbm35gjQjG3wAiFiLWAvaguoIx21LIAd"
AUTHOR = "https://sharongannot.group/resources/downloads/"
PATTERN = re.compile(
    r"Impulse_response_Acoustic_Lab_Bar-Ilan_University_"
    r"\(Reverberation_(0\.160|0\.360|0\.610)s\)_"
    r"(3-3-3-8-3-3-3|4-4-4-8-4-4-4|8-8-8-8-8-8-8)\.zip"
)


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(inventory, output, tools_only=False, workers=2):
    rows = json.loads(Path(inventory).read_text())
    selected = []
    for file_id, name, mime in rows:
        if PATTERN.fullmatch(name) or name == "SP_Package_V2.zip":
            if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id) or Path(name).name != name:
                raise ValueError("Unsafe download inventory")
            if not tools_only or name == "SP_Package_V2.zip":
                selected.append((file_id, name))
    expected = 1 if tools_only else 10
    if len(selected) != expected or len({name for _, name in selected}) != expected:
        raise ValueError("Expected all nine RIR archives and the official SP package")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        file_id, name = item
        destination = output / name
        if not destination.exists():
            import gdown

            print(f"Downloading {name}", flush=True)
            result = gdown.download(id=file_id, output=str(destination),
                                    quiet=True, use_cookies=False, resume=True)
            if result is None:
                raise RuntimeError(f"Download failed: {name}")
        with zipfile.ZipFile(destination) as archive:
            damaged = archive.testzip()
            if damaged is not None:
                raise ValueError(f"Archive CRC mismatch: {name}: {damaged}")
            members = len(archive.infolist())
        record = dict(name=name, id=file_id, path=str(destination),
                      bytes=destination.stat().st_size, sha256=checksum(destination),
                      zip_members=members)
        print(f"ZIP integrity checked: {name}: {record['bytes']} bytes", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=workers) as executor:
        files = list(executor.map(fetch, selected))
    report = dict(author_page=AUTHOR, official_folder=FOLDER, files=files,
                  status="transport_integrity_only",
                  checksum_scope="locally computed; publisher checksum unavailable")
    filename = "tools_download.json" if tools_only else "download_transport.json"
    (output / filename).write_text(json.dumps(report, indent=2) + "\n")
    if not tools_only:
        from mird_rir import audit_dataset

        audit = audit_dataset(output)
        (output / "rir_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        if not audit["all_conditions_valid"]:
            raise ValueError(
                "Native MIRD conditions are incomplete; see rir_audit.json"
            )
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tools-only", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    download(args.inventory, args.output, args.tools_only, args.workers)


if __name__ == "__main__":
    main()
