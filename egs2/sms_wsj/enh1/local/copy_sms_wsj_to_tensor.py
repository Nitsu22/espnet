"""Copy only referenced SMS-WSJ audio to tensor's local NVMe, then verify sizes."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import time


def main():
    if socket.gethostname().split(".")[0] != "tensor":
        raise RuntimeError("This copy is configured only for tensor")
    recipe = Path(__file__).resolve().parent.parent
    source = Path("/net/fractal/work2/roland/research/sms_wsj_dump/dump_oracle3")
    metadata = recipe / "dump_roland_sms_wsj_1ch"
    destination = Path("/var/tmp/nitsu/tfgridnet_sms_wsj_1ch")
    report = recipe / "exp_tfgridnet_sms_wsj_1ch_paper/local_copy"
    report.mkdir(parents=True, exist_ok=True)

    def status(message):
        text = time.strftime("%Y-%m-%dT%H:%M:%S%z ") + message
        print(text, flush=True)
        (report / "copy.status").write_text(text + "\n")

    mount_type = subprocess.check_output(
        ["findmnt", "-n", "-o", "FSTYPE", "-T", "/var/tmp"], text=True
    ).strip()
    if mount_type not in {"xfs", "ext4", "btrfs"}:
        raise RuntimeError(f"Destination is not a verified disk filesystem: {mount_type}")
    for path in (destination.parent, destination):
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlink destination: {path}")
        if path.exists() and path.stat().st_uid != os.getuid():
            raise RuntimeError(f"Destination not owned by current user: {path}")
    status("checking referenced file sizes and local free space")
    metadata_bytes = {}
    paths = set()
    splits = ("train_si284_directpath", "cv_dev93_directpath", "test_eval92_directpath")
    for split in splits:
        for item in (metadata / "raw" / split).iterdir():
            if item.is_file() and not item.is_symlink():
                metadata_bytes[item.relative_to(metadata)] = item.read_bytes()
        expected_ids = None
        for name in ("wav.scp", "spk1.scp", "spk2.scp"):
            lines = metadata_bytes[Path("raw") / split / name].decode().splitlines()
            ids = []
            for line in lines:
                uid, path = line.split(maxsplit=1)
                relative = Path(path).relative_to(source)
                if ".." in relative.parts or "\n" in str(relative):
                    raise RuntimeError("Invalid relative audio path")
                paths.add(relative)
                ids.append(uid)
            if expected_ids is not None and ids != expected_ids:
                raise RuntimeError(f"Mixture/reference IDs differ in {split}")
            expected_ids = ids

    sizes = {}
    for index, relative in enumerate(sorted(paths), 1):
        path = source / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"Source audio must be a regular file: {path}")
        sizes[str(relative)] = info.st_size
        if index % 10000 == 0:
            status(f"sized {index}/{len(paths)} source files")
    required = sum(sizes.values())
    free = shutil.disk_usage("/var/tmp").free
    reserve = max(50 * 1024**3, required // 10)
    capacity = {"source_bytes": required, "file_count": len(sizes),
                "free_bytes_before": free, "reserve_bytes": reserve,
                "destination": str(destination), "filesystem": mount_type}
    (report / "capacity.json").write_text(json.dumps(capacity, indent=2) + "\n")
    if free < required + reserve:
        raise RuntimeError(f"Insufficient free space: {capacity}")
    status(f"capacity OK: copy {required / 1024**3:.2f} GiB, "
           f"free {free / 1024**3:.2f} GiB, {len(sizes)} files")
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    audio_root = destination / "audio"
    audio_root.mkdir(exist_ok=True)
    manifest = report / "audio_files.txt"
    manifest.write_text("".join(name + "\n" for name in sorted(sizes)))
    (report / "source_sizes.json").write_text(json.dumps(sizes) + "\n")
    status("copying audio with rsync")
    subprocess.run([
        "rsync", "-rlt", "--partial", "--info=progress2", "--stats",
        "--human-readable", "--outbuf=L", f"--files-from={manifest}",
        str(source) + "/", str(audio_root) + "/",
    ], check=True)
    status("verifying every destination file size")
    for name, size in sizes.items():
        target = audio_root / name
        if not target.is_file() or target.is_symlink() or target.stat().st_size != size:
            raise RuntimeError(f"Copy verification failed: {target}")
    dump = destination / "dump"
    for relative, original in metadata_bytes.items():
        if (metadata / relative).read_bytes() != original:
            raise RuntimeError(f"Input metadata changed during copy: {relative}")
        content = original
        if relative.name in {"wav.scp", "spk1.scp", "spk2.scp"}:
            output = []
            for line in original.decode().splitlines():
                uid, path = line.split(maxsplit=1)
                output.append(f"{uid} {audio_root / Path(path).relative_to(source)}\n")
            content = "".join(output).encode()
        target = dump / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    capacity["free_bytes_after"] = shutil.disk_usage(destination).free
    capacity["metadata_sha256"] = {
        str(k): hashlib.sha256(v).hexdigest() for k, v in metadata_bytes.items()
    }
    capacity["verified"] = "rsync transfer verification and all destination sizes"
    (destination / "COPY_VERIFIED.json").write_text(json.dumps(capacity, indent=2) + "\n")
    (report / "result.json").write_text(json.dumps(capacity, indent=2) + "\n")
    status(f"COPY COMPLETE: {len(sizes)} files; local dump {dump}")


if __name__ == "__main__":
    main()
