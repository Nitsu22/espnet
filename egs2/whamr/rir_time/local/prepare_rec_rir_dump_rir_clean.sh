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
    "${raw_dir}/spk1.scp" \
    "${src_dir}/spk1_reverb.scp" \
    "${src_dir}/rir1.scp"; do
    if [ ! -f "${path}" ]; then
      echo "Missing required file: ${path}" >&2
      exit 1
    fi
  done

  raw_keys="${tmpdir}/${dset}.raw.keys"
  reverb_keys="${tmpdir}/${dset}.reverb.keys"
  rir_keys="${tmpdir}/${dset}.rir.keys"
  awk '{print $1}' "${raw_dir}/spk1.scp" | sort > "${raw_keys}"
  awk '{print $1}' "${src_dir}/spk1_reverb.scp" | sort > "${reverb_keys}"
  awk '{print $1}' "${src_dir}/rir1.scp" | sort > "${rir_keys}"

  if ! cmp -s "${raw_keys}" "${reverb_keys}"; then
    echo "Key mismatch: ${raw_dir}/spk1.scp vs ${src_dir}/spk1_reverb.scp" >&2
    exit 1
  fi
  if ! cmp -s "${raw_keys}" "${rir_keys}"; then
    echo "Key mismatch: ${raw_dir}/spk1.scp vs ${src_dir}/rir1.scp" >&2
    exit 1
  fi

  cp "${raw_dir}/spk1.scp" "${raw_dir}/speech_direct.scp"
  cp "${src_dir}/spk1_reverb.scp" "${raw_dir}/speech_reverb.scp"
  cp "${src_dir}/spk1_reverb.scp" "${raw_dir}/speech_mix.scp"
  cp "${src_dir}/rir1.scp" "${raw_dir}/rir_ref.scp"
done
