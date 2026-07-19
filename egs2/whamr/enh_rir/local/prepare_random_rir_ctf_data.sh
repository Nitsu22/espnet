#!/usr/bin/env bash

set -e
set -u
set -o pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
recipe_dir=$(cd "${script_dir}/.." && pwd)
cd "${recipe_dir}"

sets="tr_mix_clean_reverb_min_8k cv_mix_clean_reverb_min_8k tt_mix_clean_reverb_min_8k"
src_dumpdir=dump_rir_clean
dst_dumpdir=dump_rir_clean_random_ctf_seed0
seed=0
python=python3

. utils/parse_options.sh
. ./path.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 [--sets \"<set1 set2 ...>\"] [--src_dumpdir <dir>]" >&2
    echo "          [--dst_dumpdir <dir>] [--seed <int>] [--python <command>]" >&2
    exit 2
fi

if ! [[ "${seed}" =~ ^-?[0-9]+$ ]]; then
    echo "--seed must be an integer, but got: ${seed}" >&2
    exit 2
fi

src_abs=$(
    "${python}" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "${src_dumpdir}"
)
dst_abs=$(
    "${python}" -c \
        'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
        "${dst_dumpdir}"
)
if [ "${src_abs}" = "${dst_abs}" ]; then
    echo "Source and destination dump directories must differ: ${src_abs}" >&2
    exit 2
fi

split_index=0
for dset in ${sets}; do
    src_dir="${src_dumpdir}/raw/${dset}"
    dst_dir="${dst_dumpdir}/raw/${dset}"

    for filename in \
        wav.scp rir_ctf.scp spk1.scp spk2.scp \
        utt2spk spk2utt utt2num_samples feats_type; do
        if [ ! -f "${src_dir}/${filename}" ]; then
            echo "Missing required source metadata: ${src_dir}/${filename}" >&2
            exit 1
        fi
    done

    echo "Preparing random CTF metadata: ${src_dir} -> ${dst_dir}"
    utils/copy_data_dir.sh "${src_dir}" "${dst_dir}"

    # Copy only top-level metadata. Audio and CTF array subdirectories are not copied.
    while IFS= read -r -d '' metadata_file; do
        filename=${metadata_file##*/}
        if [ "${filename}" != rir_ctf.scp ]; then
            cp "${metadata_file}" "${dst_dir}/${filename}"
        fi
    done < <(find "${src_dir}" -maxdepth 1 -type f -print0)

    split_seed=$((seed + split_index))
    "${python}" local/randomize_rir_ctf_scp.py \
        --wav-scp "${src_dir}/wav.scp" \
        --ctf-scp "${src_dir}/rir_ctf.scp" \
        --output-scp "${dst_dir}/rir_ctf.scp" \
        --mapping-tsv "${dst_dir}/rir_ctf_random_mapping.tsv" \
        --manifest-json "${dst_dir}/rir_ctf_random_manifest.json" \
        --seed "${split_seed}"

    split_index=$((split_index + 1))
done

echo "Prepared random CTF dump: ${dst_dumpdir} (base seed=${seed})"
