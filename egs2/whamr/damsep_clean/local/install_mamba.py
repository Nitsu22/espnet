"""Install matching official wheels into an isolated DAMSEP dependency folder.

Uses prebuilt extensions, never builds CUDA code or upgrades the active PyTorch
environment. Run from the recipe with the existing tf-locoformer Python.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

WHEELS = [
    (
        "https://github.com/state-spaces/mamba/releases/download/v1.2.0.post1/",
        "mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    ),
    (
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.2.0.post2/",
        "causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-"
        "cp310-cp310-linux_x86_64.whl",
    ),
]
PYTHON_PACKAGES = [
    "transformers==4.39.3",
    "tokenizers==0.15.2",
    "safetensors==0.4.5",
    "huggingface-hub==0.23.5",
]
CHECK_IMPORTS = """
import torch, mamba_ssm, causal_conv1d, transformers
from mamba_ssm.ops.triton.layernorm import RMSNorm
from espnet2.enh.damsep.vendor.models.SPMamba import SPMamba
assert mamba_ssm.__version__ == '1.2.0.post1'
assert causal_conv1d.__version__ == '1.2.0.post2'
print('Imported Mamba/causal-conv1d/DAMSEP using the isolated dependency folder')
"""


def verify(target, root):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(target), env.get("PYTHONPATH", "")]
    )
    env.setdefault("NUMBA_CACHE_DIR", str(target.parent / "numba-cache"))
    subprocess.run([sys.executable, "-c", CHECK_IMPORTS], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path(".deps/mamba-cu118-torch21-py310")
    )
    args = parser.parse_args()
    if (
        sys.version_info[:2] != (3, 10)
        or not torch.__version__.startswith("2.1.")
        or torch.version.cuda != "11.8"
        or torch._C._GLIBCXX_USE_CXX11_ABI
    ):
        raise RuntimeError(
            "These wheels require Python 3.10, PyTorch 2.1, CUDA 11.8, ABI=False"
        )
    target = args.target.absolute()
    root = Path(__file__).resolve().parents[4]
    if target.exists():
        if not (target / "installation.json").is_file():
            raise FileExistsError(
                f"Unverified dependency directory already exists: {target}"
            )
        verify(target, root)
        print("Existing isolated installation verified:", target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".install_mamba_", dir=target.parent))
    wheels_dir = target.parent / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    hashes = {}
    try:
        files = []
        for base, name in WHEELS:
            path = wheels_dir / name
            url = base + name.replace("+", "%2B")
            if not path.exists():
                temporary = path.with_suffix(".part")
                print("Downloading official wheel:", name, flush=True)
                urllib.request.urlretrieve(url, temporary)
                temporary.rename(path)
            hashes[name] = dict(
                url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            files.append(str(path))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(staging),
            ]
            + files
            + PYTHON_PACKAGES,
            check=True,
        )
        verify(staging, root)
        (staging / "installation.json").write_text(
            json.dumps(
                dict(
                    python=sys.version.split()[0],
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    cxx11_abi=False,
                    wheels=hashes,
                    python_packages=PYTHON_PACKAGES,
                ),
                indent=2,
            )
            + "\n"
        )
        staging.rename(target)
        print("Installed without modifying the active environment:", target)
    except BaseException:
        shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
