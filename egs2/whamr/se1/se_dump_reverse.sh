#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# General configuration
stage=3                 # Processes starts from the specified stage.
stop_stage=4           # Processes is stopped at the specified stage.
skip_data_prep=false   # Skip data preparation stages
nj=32                  # The number of parallel jobs.
dumpdir=dump_reverse   # Directory to dump features.

# Feature extraction related
feats_type=raw         # Feature type (raw or fbank_pitch).
audio_format=flac      # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k                 # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

# Training data related
extra_wav_list=        # Extra list of scp files for wav formatting

# [Task dependent] Set the datadir name created by local/data.sh
train_set=             # Name of training set.
valid_set=             # Name of development set.
test_sets=             # Names of evaluation sets. Multiple items can be specified.

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage              # Processes starts from the specified stage (default="${stage}").
    --stop_stage         # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep     # Skip data preparation stages (default="${skip_data_prep}").
    --nj                 # The number of parallel jobs (default="${nj}").
    --dumpdir            # Directory to dump features (default="${dumpdir}").

    # Feature extraction related
    --feats_type         # Feature type (only support raw currently).
    --audio_format       # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs                 # Sampling rate (default="${fs}").
    --min_wav_duration   # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration   # Maximum duration in second (default="${max_wav_duration}").

    --extra_wav_list     # Extra list of scp files for wav formatting (default="${extra_wav_list}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set          # Name of training set (required).
    --valid_set          # Name of development set (required).
    --test_sets          # Names of evaluation sets (required).
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

# Extra files for enhancement process
utt_extra_files="utt2category"

data_feats=${dumpdir}/raw

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

        log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

        # ====== Recreating "wav.scp" ======
        # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
        # shouldn't be used in training process.
        # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
        # and also it can also change the audio-format and sampling rate.
        # If nothing is need, then format_wav_scp.sh does nothing:
        # i.e. the input file format and rate is same as the output.

        for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                _suf="/org"
            else
                _suf=""
            fi
            utils/copy_data_dir.sh data/"${dset}" "${data_feats}${_suf}/${dset}"
            rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel}
            _opts=
            if [ -e data/"${dset}"/segments ]; then
                # "segments" is used for splitting wav files which are written in "wav".scp
                # into utterances. The file format of segments:
                #   <segment_id> <record_id> <start_time> <end_time>
                #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                # Where the time is written in seconds.
                _opts+="--segments data/${dset}/segments "
            fi

            # For dump_reverse: only process wav
            for spk in "wav"; do
                # shellcheck disable=SC2086
                scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                    --out-filename "${spk}.scp" \
                    --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                    "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}" \
                    "${data_feats}${_suf}/${dset}/logs/${spk}" "${data_feats}${_suf}/${dset}/data/${spk}"
            done

            for f in $extra_wav_list; do
                if [ -e "data/${dset}/$f" ]; then
                    # shellcheck disable=SC2086
                    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                        --out-filename "$f" \
                        --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                        "data/${dset}/$f" "${data_feats}/${dset}" \
                        "${data_feats}/${dset}/logs/${f%.*}" "${data_feats}/${dset}/data/${f%.*}"
                fi
            done

            echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

            for f in ${utt_extra_files}; do
                [ -f data/${dset}/${f} ] && cp data/${dset}/${f} ${data_feats}${_suf}/${dset}/${f}
            done

        done
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove short data: ${data_feats}/org -> ${data_feats}"

        for dset in "${train_set}" "${valid_set}"; do
        # NOTE: Not applying to test_sets to keep original data

            # For dump_reverse: only process wav
            _spk_list=""
            _scp_list=""

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            for f in ${utt_extra_files}; do
                if [ -f "${data_feats}/org/${dset}/${f}" ]; then
                    cp "${data_feats}/org/${dset}/${f}" "${data_feats}/${dset}/${f}"
                fi
            done

            _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
            _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
            _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

            # utt2num_samples is created by format_wav_scp.sh
            <"${data_feats}/org/${dset}/utt2num_samples" \
                awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                    '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                    >"${data_feats}/${dset}/utt2num_samples"
            for spk in "wav"; do
                <"${data_feats}/org/${dset}/${spk}.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/${spk}.scp"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${_scp_list} ${utt_extra_files}" "${data_feats}/${dset}"
        done
    fi
else
    log "Skip the data preparation stages"
fi
