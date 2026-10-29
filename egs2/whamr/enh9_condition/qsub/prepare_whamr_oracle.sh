#!/usr/bin/env bash
# CPU verification; no CTF fitting or audio copying is necessary.
#$ -cwd
#$ -l cpu_4=1
#$ -l h_rt=00:30:00
#$ -p -5
#$ -N oracle_whamr_inputs
#$ -o qsub_logs
#$ -e qsub_logs
source qsub/oracle_environment.sh
source_root=/gs/bs/tga-shinoda/nitsu/research/tf-locoformer
baseline=${source_root}/espnet-baseline-pipeline/egs2/whamr/enh7_baseline
python local/prepare_whamr_oracle_dump.py \
    --source-dump "${source_root}/espnet-oracle-ctf-nf8k/egs2/whamr/enh9_condition/dump_oracle_ctf_nf8k" \
    --baseline-recipe "${baseline}" \
    --baseline-config "${baseline}/exp/enh_train_tflocoformer_s_whamr_8k_1gpu_batch4/config.yaml" \
    --baseline-stats "${source_root}/espnet/egs2/whamr/enh1/exp/enh_stats_8k" \
    --config conf/tuning/train_enh_tflocoformer_s_whamr8k_oracle_ctf_concat.yaml \
    --output dump_oracle_ctf_whamr8k --workers 4
python local/check_oracle_ctf_inputs.py --condition whamr --dump dump_oracle_ctf_whamr8k
