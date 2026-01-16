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
skip_train=false        # Skip training stages
skip_eval=false         # Skip inference and evaluation stages
skip_packing=true       # Skip the packing stage.
skip_upload_hf=true     # Skip uploading to huggingface stage.
ngpu=1                  # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1             # The number of nodes
nj=32                   # The number of parallel jobs.
dumpdir=dump_dif_position            # Directory to dump features.
inference_nj=32         # The number of parallel jobs in inference.
gpu_inference=false     # Whether to perform gpu inference.
expdir=exp              # Directory to save experiments.
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

# Enhancement model related
enh_exp=    # Specify the directory path for enhancement experiment. If this option is specified, enh_tag is ignored.
enh_tag=    # Suffix to the result dir for enhancement model training.
enh_config= # Config for enhancement model training.
enh_args=   # Arguments for enhancement model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in enhancement config.
ref_num=2   # Number of references for training.
            # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
inf_num=    # Number of inferences output by the model
            # Note that if it is not specified, it will be the same as ref_num. Otherwise, it will be overwritten.
            # In MixIT, number of outputs is larger than that of references.
noise_type_num=1    # Number of noise types in the input audio
dereverb_ref_num=1  # Number of reference signals for deverberation
is_tse_task=false   # Whether perform the target speaker extraction task or normal speech enhancement/separation tasks

# Training data related
use_dereverb_ref=false
use_noise_ref=false
variable_num_refs=false # Whether to use variable numbers of references in spk1.scp, dereverb1.scp, enroll_spk1.scp, etc.
extra_wav_list= # Extra list of scp files for wav formatting

# Pretrained model related
# The number of --init_param must be same.
init_param=

# Enhancement related
inference_args="--normalize_output_wav true"
inference_model=valid.loss.ave.pth
download_model=

# Evaluation related
scoring_protocol="STOI SDR SAR SIR SI_SNR"
scoring_opts=""
ref_channel=0
inference_tag=  # Prefix to the result dir for ENH inference.
inference_enh_config= # Config for enhancement.
score_with_asr=false
asr_exp=""       # asr model for scoring WER
lm_exp=""       # lm model for scoring WER
inference_asr_model=valid.acc.best.pth # ASR model path for decoding.
inference_lm=valid.loss.best.pth       # Language model path for decoding.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
inference_asr_tag=    # Suffix to the result dir for decoding.
inference_asr_config= # Config for decoding.
inference_asr_args=   # Arguments for ASR decoding, e.g., "--lm_weight 0.1".

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
enh_speech_fold_length=800 # fold_length for speech data during enhancement training
lang=noinfo      # The language type of corpus

# Upload model related
hf_repo=

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage              # Processes starts from the specified stage (default="${stage}").
    --stop_stage         # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep     # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train         # Skip training stages (default="${skip_train}").
    --skip_eval          # Skip inference and evaluation stages (default="${skip_eval}").
    --skip_packing       # Skip the packing stage (default="${skip_packing}").
    --skip_upload_hf     # Skip uploading to huggingface stage (default="${skip_upload_hf}").
    --ngpu               # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes          # The number of nodes
    --nj                 # The number of parallel jobs (default="${nj}").
    --inference_nj       # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference      # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir            # Directory to dump features (default="${dumpdir}").
    --expdir             # Directory to save experiments (default="${expdir}").
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


    # Enhancemnt model related
    --enh_tag    # Suffix to the result dir for enhancement model training (default="${enh_tag}").
    --enh_config # Config for enhancement model training (default="${enh_config}").
    --enh_args   # Arguments for enhancement model training, e.g., "--max_epoch 10" (default="${enh_args}").
                 # Note that it will overwrite args in enhancement config.
    --ref_num    # Number of references for training (default="${ref_num}").
                 # In supervised learning based speech enhancement / separation, it is equivalent to number of speakers.
    --inf_num    # Number of inference audio generated by the model (default="${ref_num}")
                 # Note that if it is not specified, it will be the same as ref_num. Otherwise, it will be overwritten.
                 # In MixIT, number of outputs is larger than that of references.
    --noise_type_num   # Number of noise types in the input audio (default="${noise_type_num}")
    --dereverb_ref_num # Number of references for dereverberation (default="${dereverb_ref_num}")
    --is_tse_task     # Whether perform the target speaker extraction task or normal speech enhancement/separation tasks (default="${is_tse_task}")

    # Training data related
    --use_dereverb_ref  # Whether or not to use dereverberated signal as an additional reference
                          for training a dereverberation model (default="${use_dereverb_ref}")
    --use_noise_ref     # Whether or not to use noise signal as an additional reference
                          for training a denoising model (default="${use_noise_ref}")
    --variable_num_refs # Whether or not to use variable numbers of references in spk1.scp, dereverb1.scp, enroll_spk1.scp, etc. If True, --ref_num and --dereverb_ref_num must be 1. (default="${variable_num_refs}")
    --extra_wav_list    # Extra list of scp files for wav formatting (default="${extra_wav_list}")

    # Pretrained model related
    --init_param    # pretrained model path and module name (default="${init_param}")

    # Enhancement related
    --inference_args       # Arguments for enhancement in the inference stage (default="${inference_args}")
    --inference_model      # Enhancement model path for inference (default="${inference_model}").
    --inference_enh_config # Configuration file for overwriting some model attributes during SE inference. (default="${inference_enh_config}")

    # Evaluation related
    --scoring_protocol    # Metrics to be used for scoring (default="${scoring_protocol}")
    --scoring_opts        # Additional arguments for scoring (default="${scoring_opts}")
    --ref_channel         # Reference channel of the reference speech will be used if the model
                            output is single-channel and reference speech is multi-channel
                            (default="${ref_channel}")

    # ASR evaluation related
    --score_with_asr       # Enable ASR evaluation (default="${score_with_asr}")
    --asr_exp              # asr model for scoring WER  (default="${asr_exp}")
    --lm_exp               # lm model for scoring WER (default="${lm_exp}")
    --nlsyms_txt           # Non-linguistic symbol list if existing.  (default="${nlsyms_txt}")
    --inference_asr_model  # ASR model path for decoding. (default="${inference_asr_model}")
    --inference_lm         # Language model path for decoding. (default="${inference_lm}")
    --nlsyms_txt           # Non-linguistic symbol list if existing. (default="${nlsyms_txt}")
    --inference_asr_tag    # Suffix to the result dir for decoding. (default="${inference_asr_tag}")
    --inference_asr_config # Config for ASR decoding.  (default="${inference_asr_config}")
    --inference_asr_args   # Arguments for ASR decoding, e.g., "--lm_weight 0.1". (default="${inference_asr_args}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
    --enh_speech_fold_length # fold_length for speech data during enhancement training  (default="${enh_speech_fold_length}").
    --lang         # The language type of corpus (default="${lang}")
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
utt_extra_files=""

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
    if [ -n "$inf_num" ] && [ "$inf_num" != "$ref_num" ]; then
        log "The value of '--inf_num' must be equal to that of '--ref_num' for the target speaker extraction (TSE) task"
        exit 1
    fi
fi
inf_num=${inf_num:=${ref_num}}

# Set tag for naming of model directory
if [ -z "${enh_tag}" ]; then
    if [ -n "${enh_config}" ]; then
        enh_tag="$(basename "${enh_config}" .yaml)_${feats_type}"
    else
        enh_tag="train_${feats_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${enh_args}" ]; then
        enh_tag+="$(echo "${enh_args}" | sed -e "s/--\|\//\_/g" -e "s/[ |=]//g")"
    fi
fi

if [ -z "${inference_asr_tag}" ]; then
    if [ -n "${inference_asr_config}" ]; then
        inference_asr_tag="$(basename "${inference_asr_config}" .yaml)"
    else
        inference_asr_tag=asr_inference
    fi
    # Add overwritten arg's info
    if [ -n "${inference_asr_args}" ]; then
        inference_asr_tag+="$(echo "${inference_asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if [ -n "${lm_exp}" ]; then
        inference_asr_tag+="_lm_$(basename "${lm_exp}")_$(echo "${inference_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    inference_asr_tag+="_asr_model_$(echo "${inference_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi



# The directory used for collect-stats mode
enh_stats_dir="${expdir}/enh_stats_${fs}"
# The directory used for training commands
if [ -z "${enh_exp}" ]; then
enh_exp="${expdir}/enh_${enh_tag}"
fi

if [ -n "${speed_perturb_factors}" ]; then
  enh_stats_dir="${enh_stats_dir}_sp"
  enh_exp="${enh_exp}_sp"
fi

if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_enh_config}" ]; then
        inference_tag="$(basename "${inference_enh_config}" .yaml)"
    else
        inference_tag=enhanced
    fi
fi

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
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data_dif_position.sh ${local_data_opts}
    fi

    if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        if ! $use_dereverb_ref && [ -n "${speed_perturb_factors}" ]; then
           log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

            _scp_list="wav.scp "

           for factor in ${speed_perturb_factors}; do
               if python3 -c "assert ${factor} != 1.0" 2>/dev/null; then
                   scripts/utils/perturb_enh_data_dir_speed.sh --utt_extra_files "${utt_extra_files}" "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}" "${_scp_list}"
                   _dirs+="data/${train_set}_sp${factor} "
               else
                   # If speed factor is 1, same as the original
                   _dirs+="data/${train_set} "
               fi
           done
           utils/combine_data.sh --extra-files "${_scp_list}" "data/${train_set}_sp" ${_dirs}
        else
           log "Skip stage 2: Speed perturbation"
        fi
    fi

    if [ -n "${speed_perturb_factors}" ]; then
        train_set="${train_set}_sp"
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

            for spk in "wav" "wav_dif_position"; do
                # Skip if the source scp file doesn't exist
                if [ ! -f "data/${dset}/${spk}.scp" ]; then
                    continue
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

            # Copy data dir
            utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
            # Copy wav_dif_position.scp explicitly (same as wav.scp is copied by copy_data_dir.sh)
            if [ -f "${data_feats}/org/${dset}/wav_dif_position.scp" ]; then
                cp "${data_feats}/org/${dset}/wav_dif_position.scp" "${data_feats}/${dset}/wav_dif_position.scp"
            fi
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
            for spk in "wav" "wav_dif_position"; do
                <"${data_feats}/org/${dset}/${spk}.scp" \
                    utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                    >"${data_feats}/${dset}/${spk}.scp"
            done

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"
        done
    fi
else
    log "Skip the data preparation stages"
fi
