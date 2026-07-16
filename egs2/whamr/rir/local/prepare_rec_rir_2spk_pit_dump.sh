#!/usr/bin/env bash

set -e
set -u
set -o pipefail

dumpdir=dump_rir_clean
enh_rir_data_dir=../enh_rir/data
sets="tr_mix_clean_reverb_min_8k cv_mix_clean_reverb_min_8k tt_mix_clean_reverb_min_8k"

. utils/parse_options.sh

tmpdir=$(mktemp -d)
trap 'rm -rf "${tmpdir}"' EXIT

for dset in ${sets}; do
  raw_dir="${dumpdir}/raw/${dset}"
  src_dir="${enh_rir_data_dir}/${dset}"

  for path in \
    "${raw_dir}/wav.scp" \
    "${raw_dir}/spk1.scp" \
    "${raw_dir}/spk2.scp" \
    "${src_dir}/spk1_reverb.scp" \
    "${src_dir}/spk2_reverb.scp" \
    "${src_dir}/rir1.scp" \
    "${src_dir}/rir2.scp"; do
    if [ ! -f "${path}" ]; then
      echo "Missing required file: ${path}" >&2
      exit 1
    fi
  done

  mix_keys="${tmpdir}/${dset}.mix.keys"
  awk '{print $1}' "${raw_dir}/wav.scp" | sort > "${mix_keys}"
  for scp in \
    "${raw_dir}/spk1.scp" \
    "${raw_dir}/spk2.scp" \
    "${src_dir}/spk1_reverb.scp" \
    "${src_dir}/spk2_reverb.scp" \
    "${src_dir}/rir1.scp" \
    "${src_dir}/rir2.scp"; do
    check_keys="${tmpdir}/${dset}.$(basename "${scp}").keys"
    awk '{print $1}' "${scp}" | sort > "${check_keys}"
    if ! cmp -s "${mix_keys}" "${check_keys}"; then
      echo "Key mismatch: ${raw_dir}/wav.scp vs ${scp}" >&2
      exit 1
    fi
  done

  cp "${raw_dir}/wav.scp" "${raw_dir}/speech_mix_pit.scp"
  cp "${raw_dir}/spk1.scp" "${raw_dir}/speech_direct1.scp"
  cp "${raw_dir}/spk2.scp" "${raw_dir}/speech_direct2.scp"
  cp "${src_dir}/spk1_reverb.scp" "${raw_dir}/speech_reverb1.scp"
  cp "${src_dir}/spk2_reverb.scp" "${raw_dir}/speech_reverb2.scp"
done
