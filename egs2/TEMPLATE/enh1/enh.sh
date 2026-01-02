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
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1                 # Processes starts from the specified stage.
stop_stage=10000        # Processes is stopped at the specified stage.
skip_data_prep=false    # Skip data preparation stages
nj=32                   # The number of parallel jobs.
dumpdir=dump            # Directory to dump features.
python=python3          # Specify python to execute espnet commands

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k            # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

# Enhancement model related (needed for data preparation stages)
ref_num=2   # Number of references for training.
            # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
noise_type_num=1    # Number of noise types in the input audio
dereverb_ref_num=1  # Number of reference signals for deverberation
is_tse_task=false   # Whether perform the target speaker extraction task or normal speech enhancement/separation tasks

# Training data related
use_dereverb_ref=false
use_noise_ref=false
use_reverse_mix=true # Whether to add reverse speaker position data (reverbse and anechoic_reverse) to train_set, valid_set, and test_sets
variable_num_refs=false # Whether to use variable numbers of references in spk1.scp, dereverb1.scp, enroll_spk1.scp, etc.
extra_wav_list= # Extra list of scp files for wav formatting

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage              # Processes starts from the specified stage (default="${stage}").
    --stop_stage         # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep     # Skip data preparation stages (default="${skip_data_prep}").
    --nj                 # The number of parallel jobs (default="${nj}").
    --dumpdir            # Directory to dump features (default="${dumpdir}").
    --python             # Specify python to execute espnet commands (default="${python}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type   # Feature type (only support raw currently).
    --audio_format # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").


    # Enhancement model related (needed for data preparation)
    --ref_num    # Number of references for training (default="${ref_num}").
                 # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
    --noise_type_num   # Number of noise types in the input audio (default="${noise_type_num}")
    --dereverb_ref_num # Number of references for dereverberation (default="${dereverb_ref_num}")
    --is_tse_task     # Whether perform the target speaker extraction task or normal speech enhancement/separation tasks (default="${is_tse_task}")

    # Training data related
    --use_dereverb_ref  # Whether or not to use dereverberated signal as an additional reference
                          for training a dereverberation model (default="${use_dereverb_ref}")
    --use_noise_ref     # Whether or not to use noise signal as an additional reference
                          for training a denoising model (default="${use_noise_ref}")
    --use_reverse_mix   # Whether to add reverse speaker position data (reverbse and anechoic_reverse) 
                          to train_set, valid_set, and test_sets (default="${use_reverse_mix}")
    --variable_num_refs # Whether or not to use variable numbers of references in spk1.scp, dereverb1.scp, enroll_spk1.scp, etc. If True, --ref_num and --dereverb_ref_num must be 1. (default="${variable_num_refs}")
    --extra_wav_list    # Extra list of scp files for wav formatting (default="${extra_wav_list}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
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

# Add reverse speaker position data if use_reverse_mix is enabled
# Assumes existing /data and /dump already exist, only adds reverse data
if $use_reverse_mix; then
    log "Adding reverse speaker position data (reverbse and anechoic_reverse) to datasets"
    # Generate reverse dataset names from existing dataset names
    # Pattern: replace _reverb_{min,max}_{8k,16k} with _reverb_reverbse_{min,max}_{8k,16k} and _reverb_anechoic_reverse_{min,max}_{8k,16k}
    _reverse_train_sets=""
    _reverse_valid_sets=""
    _reverse_test_sets=""
    
    # Function to generate reverse dataset names
    _add_reverse_datasets() {
        local _dset=$1
        local _result=""
        # Check if dataset name contains _reverb_ and ends with _{min,max}_{8k,16k}
        if [[ "${_dset}" =~ _reverb_(min|max)_(8k|16k)$ ]]; then
            # Extract the suffix (min_8k, max_8k, min_16k, max_16k)
            local _suffix=$(echo "${_dset}" | sed 's/.*_reverb_//')
            # Generate reverse dataset names
            local _reverbse_dset=$(echo "${_dset}" | sed "s/_reverb_${_suffix}$/_reverb_reverbse_${_suffix}/")
            local _anechoic_reverse_dset=$(echo "${_dset}" | sed "s/_reverb_${_suffix}$/_reverb_anechoic_reverse_${_suffix}/")
            _result="${_reverbse_dset} ${_anechoic_reverse_dset}"
        fi
        echo "${_result}"
    }
    
    for dset in ${train_set}; do
        _reverse_dsets=$(_add_reverse_datasets "${dset}")
        if [ -n "${_reverse_dsets}" ]; then
            _reverse_train_sets+="${_reverse_dsets} "
        fi
    done
    
    for dset in ${valid_set}; do
        _reverse_dsets=$(_add_reverse_datasets "${dset}")
        if [ -n "${_reverse_dsets}" ]; then
            _reverse_valid_sets+="${_reverse_dsets} "
        fi
    done
    
    for dset in ${test_sets}; do
        _reverse_dsets=$(_add_reverse_datasets "${dset}")
        if [ -n "${_reverse_dsets}" ]; then
            _reverse_test_sets+="${_reverse_dsets} "
        fi
    done
    
    # Append reverse datasets to original datasets
    if [ -n "${_reverse_train_sets}" ]; then
        train_set="${train_set} ${_reverse_train_sets}"
        log "Added reverse train sets: ${_reverse_train_sets}"
    fi
    if [ -n "${_reverse_valid_sets}" ]; then
        valid_set="${valid_set} ${_reverse_valid_sets}"
        log "Added reverse valid sets: ${_reverse_valid_sets}"
    fi
    if [ -n "${_reverse_test_sets}" ]; then
        test_sets="${test_sets} ${_reverse_test_sets}"
        log "Added reverse test sets: ${_reverse_test_sets}"
    fi
fi

# Extra files for enhancement process
utt_extra_files="utt2category"

data_feats=${dumpdir}/raw

if $is_tse_task; then
    if $use_noise_ref; then
        log "--use_noise_ref must be false for the target speaker extraction (TSE) task"
        exit 1
    fi
    if $use_dereverb_ref; then
        log "--use_dereverb_ref must be false for the target speaker extraction (TSE) task"
        exit 1
    fi
fi
# No need for model directory setup since we only prepare data

if ${variable_num_refs}; then
    # load variable numbers of speakers in spk1.scp, dereverb1.scp, enroll_spk1.scp, etc.
    if [ "${ref_num}" -ne 1 ]; then
        log "[ERROR] --ref_num must be 1 if --variable_num_refs is true, but got ${ref_num}"
        exit 1
    fi
    if [ "${dereverb_ref_num}" -ne 1 ]; then
        log "[ERROR] --dereverb_ref_num must be 1 if --variable_num_refs is true, but got ${dereverb_ref_num}"
        exit 1
    fi
    if [ ! -e "data/${train_set}/utt2category" ] || [ ! -e "data/${valid_set}/utt2category" ]; then
        log "[ERROR] utt2category must be prepared in data/${train_set} and data/${valid_set} if --variable_num_refs is true."
        exit 1
    else
        log "[WARNING] Variable speaker number is enabled. Please ensure the utt2category file assigns the same category ID to samples with the same number of speakers."
    fi
    if [[ "${audio_format}" == *ark* ]]; then
        log "[WARNING] Since audio_format=*ark* and variable_num_refs=true is applied,\nplease ensure that the first dimension of each array defined in the ark data\nfor 'spk1.scp', 'dereverb1.scp', 'enroll_spk1.scp' and so on corresponds the\nnumber of references (speakers)."
    fi
    log "[INFO] Variable speaker number is enabled. Please make sure the argument 'flexible_numspk' is True in the preprocessor in the model config."
fi


# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for reverse speaker position data"
        # Assumes normal data already exists from data.sh execution
        # Only prepares reverse speaker position data (reverbse and anechoic_reverse)
        local/data_reverse.sh ${local_data_opts}
    fi

    # Stage 2: Speed perturbation is skipped
    # Assumes existing /data and /dump already have speed perturbation applied if needed
    # Only reverse data is added, so speed perturbation is not needed here
    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        log "Skip stage 2: Speed perturbation (assumes existing data already has speed perturbation if needed)"
    fi

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


            _spk_list=" "
            for i in $(seq ${ref_num}); do
                _spk_list+="spk${i} "
                if $is_tse_task; then
                    _spk_list+="enroll_spk${i} "
                fi
            done
            if $use_noise_ref && [ -n "${_suf}" ]; then
                # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
            fi
            if $use_dereverb_ref && [ -n "${_suf}" ]; then
                # references for dereverberation
                _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
            fi

            for spk in "wav" ${_spk_list}; do
                if ${is_tse_task} && [[ "${spk}" == enroll_spk* ]]; then
                    audio_path=$(head -n 1 "data/${dset}/${spk}.scp" | awk '{print $2}')
                    if [[ ("${dset}" == "${train_set}" && "${audio_path:0:1}" == "*") || "${audio_path: -4}" == ".npy" ]]; then
                        # In case of
                        # 1. a special format in `enroll_spk?.scp`:
                        # MIXTURE_UID *UID SPEAKER_ID
                        # 2. speaker embeddings instead of enrollment audios in `enroll_spk?.scp`
                        utils/filter_scp.pl "${data_feats}${_suf}/${dset}/wav.scp" "data/${dset}/${spk}.scp" > "${data_feats}${_suf}/${dset}/${spk}.scp"
                        continue
                    fi
                fi
                if ${variable_num_refs}; then
                    if [[ "${spk}" == spk* ]] || [[ "${spk}" == dereverb* ]] || [[ "${spk}" == enroll_spk* ]]; then
                        # skip formatting for multi-audio-column scp files
                        utils/filter_scp.pl "${data_feats}${_suf}/${dset}/wav.scp" "data/${dset}/${spk}.scp" > "${data_feats}${_suf}/${dset}/${spk}.scp"
                        continue
                    fi
                fi
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

            _spk_list=" "
            _scp_list=" "
            for i in $(seq ${ref_num}); do
                _spk_list+="spk${i} "
                _scp_list+="spk${i}.scp "
                if $is_tse_task; then
                    _spk_list+="enroll_spk${i} "
                    _scp_list+="enroll_spk${i}.scp "
                fi
            done
            if $use_noise_ref; then
                # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
                _scp_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n.scp "; done)
            fi
            if $use_dereverb_ref; then
                # references for dereverberation
                _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
                _scp_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n.scp "; done)
            fi

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            for spk in ${_spk_list};do
                cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
            done
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
            for spk in ${_spk_list} "wav"; do
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



log "Successfully finished. [elapsed=${SECONDS}s]"
