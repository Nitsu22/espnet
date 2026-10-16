#!/usr/bin/env bash
# Resource/time selection and dependencies are recorded by the controller.
#$ -cwd
#$ -o qsub_logs
#$ -e qsub_logs
#$ -p -5
source qsub/oracle_environment.sh
export PATH="/apps/t4/rhel9/uge/latest/bin/lx-amd64:${PATH}"
command -v qsub qacct >/dev/null
case " $* " in
    *" --phase train "*|*" --phase infer "*) module load cuda/11.8.0 ;;
esac
python local/oracle_ctf_pipeline.py "$@"
