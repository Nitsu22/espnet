#!/usr/bin/env python3
import warnings

warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)

from espnet2.tasks.enh_se_condition_mc_conformer import EnhancementTask as _EnhancementTask

import espnet2.bin.enh_inference_se_condition as _base

# Patch the base module's task to use the MC Conformer variant.
_base.EnhancementTask = _EnhancementTask

get_parser = _base.get_parser
main = _base.main
SeparateSpeech = _base.SeparateSpeech


if __name__ == "__main__":
    _base.main()
