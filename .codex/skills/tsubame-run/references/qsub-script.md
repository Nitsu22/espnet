# qsubの参考例

適切な既存qsubがない場合の参考例。使用前に資源・環境を確認し、同じレシピで成功した設定を優先する。

グループ課金の4 GPU例。`HH:MM:SS` は所要時間に適度な余裕を加えた値に置き換える。ジョブ名・通知先・実行コマンドも依頼に合わせる。4 GPUの確認だからといって3分を既定値にしない。

ログディレクトリはジョブ開始前に必要なため、TSUBAMEのレシピディレクトリで作成してから投入する。

```bash
mkdir -p logs/qsub
qsub -g tga-shinoda qsub/<script>.sh
```

```bash
#!/usr/bin/env bash
#$ -cwd
#$ -l node_f=1
#$ -l h_rt=HH:MM:SS
#$ -N short_job_name
#$ -m abe
#$ -M daichi2ni2two@icloud.com
#$ -o logs/qsub
#$ -e logs/qsub
#$ -p -5

# 環境準備に失敗したら停止する。conda初期化中は未定義変数を許容する。
set -eo pipefail
if __conda_setup="$('/gs/bs/tga-shinoda/nitsu/anaconda3/bin/conda' 'shell.bash' 'hook')"; then
    eval "$__conda_setup"
elif [ -f "/gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/gs/bs/tga-shinoda/nitsu/anaconda3/etc/profile.d/conda.sh"
else
    printf '%s\n' 'condaの初期化に失敗しました' >&2
    exit 1
fi
unset __conda_setup

module load cuda/11.8.0
conda activate tf-locoformer
set -u

# 既存のGPU割当を保持し、未設定の場合はこの4 GPU例の値を使う。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
./run_target.sh --ngpu 4 --stage 6 --stop_stage 8
```

標準出力・エラーは `logs/qsub/` にジョブごとに保存される。GPU数やステージを変更するときは、資源指定と実行引数も合わせる。
