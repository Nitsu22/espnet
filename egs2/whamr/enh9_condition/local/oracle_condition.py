"""Keep noisy WHAMR and NF-WHAMR artifacts separate."""
from pathlib import Path


def settings(condition="nf_whamr", variant="concat"):
    if condition not in ("nf_whamr", "whamr") or variant not in ("concat", "film", "concat_glue"):
        raise ValueError("Unknown Oracle condition/variant")
    if condition == "whamr" and variant not in ("concat", "concat_glue"):
        raise ValueError("Only temporal concatenation is configured for WHAMR")
    if condition == "nf_whamr" and variant == "concat_glue":
        raise ValueError("The Glue experiment is configured for noisy WHAMR only")
    tag = "whamr8k" if condition == "whamr" else "nf8k"
    return dict(
        dump=Path(f"dump_oracle_ctf_{tag}"),
        suffix="mix_both_reverb_min_8k" if condition == "whamr" else "mix_clean_reverb_min_8k",
        config=Path(f"conf/tuning/train_enh_tflocoformer_s_{tag}_oracle_ctf_{variant}.yaml"),
        exp=Path(f"exp/enh_train_tflocoformer_s_{tag}_oracle_ctf_{variant}_seed0"),
        marker=("oracle_ctf_whamr_glue_cuda_check.json" if variant == "concat_glue"
                else "oracle_ctf_whamr_cuda_check.json") if condition == "whamr" else (
            "oracle_ctf_cuda_check.json" if variant == "concat" else "oracle_ctf_film_cuda_check.json"),
    )


def validate_dump(dump, condition="nf_whamr", require_full=True):
    if condition == "whamr":
        from prepare_whamr_oracle_dump import validate
        return validate(dump, require_full=require_full)
    if condition != "nf_whamr":
        raise ValueError("Unknown Oracle condition")
    from prepare_oracle_ctf_dump import validate_prepared_dump
    return validate_prepared_dump(dump, require_full=require_full)
