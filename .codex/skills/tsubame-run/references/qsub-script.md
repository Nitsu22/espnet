# qsub Script Pattern

Use this fallback only when no suitable existing qsub script is available. Verify resource and environment settings before use.

Keep qsub scripts close to the existing examples. The stable structure is:

```bash
#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1        # or node_q=1, following the closest working example
#$ -l h_rt=00:03:00   # test; use the requested longer time for production
#$ -N short_job_name
#$ -m abe
#$ -M daichi2ni2two@icloud.com
#$ -o /dev/null
#$ -e /dev/null
#$ -p -5

__conda_setup="$('/gs/bs/tga-shinoda/nitsu/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
elif [ -f "/gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/gs/bs/tga-shinoda/nitsu/anaconda3/bin:$PATH"
fi
unset __conda_setup

module load cuda/11.8.0
conda activate tf-locoformer

CUDA_VISIBLE_DEVICES=0,1,2,3
./run_target.sh --ngpu 4 --stage 6 --stop_stage 8
```

Treat this as a pattern, not a template to apply blindly. Preserve known working details from the closest existing qsub script unless the current run requires a specific change.
