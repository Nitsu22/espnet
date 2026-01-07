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
dumpdir=dump            # Directory to dump features.
dumpdir_reverse=dump_reverse  # Directory to dump reverse features.
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

# Spatial Encoder model related
enh_exp=    # Specify the directory path for spatial encoder experiment. If this option is specified, enh_tag is ignored.
enh_tag=    # Suffix to the result dir for spatial encoder model training.
enh_config= # Config for spatial encoder model training.
enh_args=   # Arguments for spatial encoder model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in spatial encoder config.
extra_wav_list= # Extra list of scp files for wav formatting

# Pretrained model related
# The number of --init_param must be same.
init_param=

# Spatial Encoder related
inference_args=""
inference_model=valid.loss.ave.pth
download_model=
inference_tag=  # Prefix to the result dir for SE inference.
inference_enh_config= # Config for spatial encoder inference.

# ASR evaluation related (not used in training, but needed for variable initialization)
inference_asr_tag=    # Suffix to the result dir for decoding.
inference_asr_config= # Config for ASR decoding.
inference_asr_args=   # Arguments for ASR decoding.
inference_asr_model=  # ASR model path for decoding.
lm_exp=              # Language model experiment path.
inference_lm=         # Language model path for decoding.

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of development set.
test_sets=       # Names of evaluation sets. Multiple items can be specified.
train_set_reverse=   # Name of training set for reverse data.
valid_set_reverse=   # Name of development set for reverse data.
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
    --dumpdir_reverse    # Directory to dump reverse features (default="${dumpdir_reverse}").
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


    # Spatial Encoder model related
    --enh_tag    # Suffix to the result dir for spatial encoder model training (default="${enh_tag}").
    --enh_config # Config for spatial encoder model training (default="${enh_config}").
    --enh_args   # Arguments for spatial encoder model training, e.g., "--max_epoch 10" (default="${enh_args}").
                 # Note that it will overwrite args in spatial encoder config.
    --extra_wav_list    # Extra list of scp files for wav formatting (default="${extra_wav_list}")

    # Pretrained model related
    --init_param    # pretrained model path and module name (default="${init_param}")

    # Spatial Encoder related
    --inference_args       # Arguments for spatial encoder in the inference stage (default="${inference_args}")
    --inference_model      # Spatial encoder model path for inference (default="${inference_model}").
    --inference_enh_config # Configuration file for overwriting some model attributes during SE inference. (default="${inference_enh_config}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
    --train_set_reverse     # Name of training set for reverse data (required).
    --valid_set_reverse     # Name of development set for reverse data (required).
    --enh_speech_fold_length # fold_length for speech data during spatial encoder training  (default="${enh_speech_fold_length}").
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
[ -z "${train_set_reverse}" ] && { log "${help_message}"; log "Error: --train_set_reverse is required"; exit 2; };
[ -z "${valid_set_reverse}" ] && { log "${help_message}"; log "Error: --valid_set_reverse is required"; exit 2; };

# Extra files for spatial encoder process
utt_extra_files="utt2category"

data_feats=${dumpdir}/raw
data_feats_reverse=${dumpdir_reverse}/raw

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
        inference_tag=embedding
    fi
fi


if ! "${skip_train}"; then
    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        _enh_train_dir="${data_feats}/${train_set}"
        _enh_valid_dir="${data_feats}/${valid_set}"
        log "Stage 5: Spatial Encoder collect stats: train_set=${_enh_train_dir}, valid_set=${_enh_valid_dir}"

        _opts=
        if [ -n "${enh_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${enh_config} "
        fi

        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi

        # 1. Split the key file
        _logdir="${enh_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_enh_train_dir}/${_scp} wc -l)" "$(<${_enh_valid_dir}/${_scp} wc -l)")

        key_file="${_enh_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_enh_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${enh_stats_dir}/run.sh'. You can resume the process from stage 5 using this script"
        mkdir -p "${enh_stats_dir}"; echo "${run_args} --stage 5 \"\$@\"; exit \$?" > "${enh_stats_dir}/run.sh"; chmod +x "${enh_stats_dir}/run.sh"

        # 3. Submit jobs
        log "Spatial Encoder collect-stats started... log: '${_logdir}/stats.*.log'"

        # prepare train and valid data parameters
        # For Spatial Encoder: speech_mix (SC), speech_mix_mc (MC mix), speech_mix_reverse_mc (MC reverse)
        # speech_mix and speech_mix_mc come from dumpdir, speech_mix_reverse_mc comes from dumpdir_reverse
        _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/wav.scp,speech_mix,${_type} "
        _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/speech_mix_mc.scp,speech_mix_mc,${_type} "
        _train_data_param+="--train_data_path_and_name_and_type ${data_feats_reverse}/${train_set_reverse}/wav.scp,speech_mix_reverse_mc,${_type} "
        _valid_data_param="--valid_data_path_and_name_and_type ${_enh_valid_dir}/wav.scp,speech_mix,${_type} "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/speech_mix_mc.scp,speech_mix_mc,${_type} "
        _valid_data_param+="--valid_data_path_and_name_and_type ${data_feats_reverse}/${valid_set_reverse}/wav.scp,speech_mix_reverse_mc,${_type} "

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        train_module=espnet2.bin.enh_se_train
        # shellcheck disable=SC2046,SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m ${train_module} \
                --collect_stats true \
                ${_train_data_param} \
                ${_valid_data_param} \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${enh_args} || { cat $(grep -l -i error "${_logdir}"/stats.*.log) ; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --skip_sum_stats --output_dir "${enh_stats_dir}"

    fi


    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        _enh_train_dir="${data_feats}/${train_set}"
        _enh_valid_dir="${data_feats}/${valid_set}"
        log "Stage 6: Spatial Encoder Training: train_set=${_enh_train_dir}, valid_set=${_enh_valid_dir}"

        _opts=
        if [ -n "${enh_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
            _opts+="--config ${enh_config} "
        fi

        _scp="wav.scp"
        # "sound" supports "wav", "flac", etc.
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        else
            # "sound" supports "wav", "flac", etc.
            _type=sound
        fi
        _fold_length="$((enh_speech_fold_length * 100))"

        # prepare train and valid data parameters
        # For Spatial Encoder: speech_mix (SC), speech_mix_mc (MC mix), speech_mix_reverse_mc (MC reverse)
        # speech_mix and speech_mix_mc come from dumpdir, speech_mix_reverse_mc comes from dumpdir_reverse
        _train_data_param="--train_data_path_and_name_and_type ${_enh_train_dir}/${_scp},speech_mix,${_type} "
        _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/speech_mix_mc.scp,speech_mix_mc,${_type} "
        _train_data_param+="--train_data_path_and_name_and_type ${data_feats_reverse}/${train_set_reverse}/wav.scp,speech_mix_reverse_mc,${_type} "
        _train_shape_param="--train_shape_file ${enh_stats_dir}/train/speech_mix_shape "
        _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/speech_mix_mc_shape "
        _train_shape_param+="--train_shape_file ${enh_stats_dir}/train/speech_mix_reverse_mc_shape "
        _fold_length_param="--fold_length ${_fold_length} "
        _fold_length_param+="--fold_length ${_fold_length} "
        _fold_length_param+="--fold_length ${_fold_length} "
        _valid_data_param="--valid_data_path_and_name_and_type ${_enh_valid_dir}/wav.scp,speech_mix,${_type} "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/speech_mix_mc.scp,speech_mix_mc,${_type} "
        _valid_data_param+="--valid_data_path_and_name_and_type ${data_feats_reverse}/${valid_set_reverse}/wav.scp,speech_mix_reverse_mc,${_type} "
        _valid_shape_param="--valid_shape_file ${enh_stats_dir}/valid/speech_mix_shape "
        _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/speech_mix_mc_shape "
        _valid_shape_param+="--valid_shape_file ${enh_stats_dir}/valid/speech_mix_reverse_mc_shape "

        # Add the category information at the end of the data path list
        if [ -e "${_enh_train_dir}/utt2category" ] && [ -e "${_enh_valid_dir}/utt2category" ]; then
            log "[INFO] Adding the category information for training"
            log "[WARNING] Please make sure the category information is explicitly processed by the preprocessor defined in '${enh_config}' so that it is converted to an integer"

            _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/utt2category,category,text "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/utt2category,category,text "
        fi

        # Add the fs information at the end of the data path list
        if [ -e "${_enh_train_dir}/utt2fs" ] && [ -e "${_enh_valid_dir}/utt2fs" ]; then
            log "[INFO] Adding the sampling frequency information (fs) for training"

            _train_data_param+="--train_data_path_and_name_and_type ${_enh_train_dir}/utt2fs,fs,text_int "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_enh_valid_dir}/utt2fs,fs,text_int "
        fi

        log "Generate '${enh_exp}/run.sh'. You can resume the process from stage 6 using this script"
        mkdir -p "${enh_exp}"; echo "${run_args} --stage 6 \"\$@\"; exit \$?" > "${enh_exp}/run.sh"; chmod +x "${enh_exp}/run.sh"

        log "Spatial Encoder training started... log: '${enh_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${enh_exp})"
        else
            jobname="${enh_exp}/train.log"
        fi
        train_module=espnet2.bin.enh_se_train
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${enh_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${enh_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m ${train_module} \
                ${_train_data_param} \
                ${_valid_data_param} \
                ${_train_shape_param} \
                ${_valid_shape_param} \
                ${_fold_length_param} \
                --resume true \
                --output_dir "${enh_exp}" \
                ${init_param:+--init_param $init_param} \
                ${_opts} ${enh_args}

    fi
else
    log "Skip the training stages"
fi