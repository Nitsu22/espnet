# WHAMR / NF-WHAMR 8 kHz: Oracle CTF temporal conditioning

DAI-8 の TF-Locoformer-S ベースラインと比較するための実験。
入力・無響教師は enh4_clean/dump_clean の元 WAV をそのまま使い、
2話者の Oracle Direct→Reverb CTF を時間方向の条件トークンとして前置する。

## モデルと比較条件

音声は既存の入力 Conv、各 CTF の実部・虚部は共通の 1×1 投影で
96 次元へ変換する。4 blocks の前に [CTF1:60][CTF2:60][speech:T] を連結し、
出力 Conv の前に CTF 区間を除く。追加パラメータは288個。
学習では発話ごとに2話者の CTF 順をランダム交換し、音声の PIT を維持する。
可変長バッチは有効区間を発話ごとに処理し、padding を条件情報へ混入させない。

DAI-8 と同じ FP32、seed0、実 batch4、accum_grad1、valid batch1、
4秒 crop、STFT256/hop64、4 blocks/embedding96/FFN128、AdamW と
warmup/plateau、最大150 epoch・patience10を使う。入力と教師を再生成しない。

## Oracle CTF と専用 dump

既存の物理 direct/reverb RIR ペアを、その direct peak で共通にスケールし、
既存 Rec-RIR の8.192秒 Sweep に畳み込む。遅延を揃え直したり末尾を切らず、
同じ STFT256/hop128/sqrt-Hann と full convolution で、60タップの
正則化付き複素最小二乗を周波数ごとに解く。
正則化は各周波数の励起エネルギーに対して1e-6。
係数は因果順 (lag0が現在フレーム)。CTF の16ms/tapは音声の8ms/frameと
同じ時間目盛りではなく、条件トークンの索引として扱う。

RIR 側の再生成音声とベースラインの全28,000例を、同じID・左チャネル・
サンプルレート・長さで照合する。差は既知の FLOAT/PCM16 量子化と
クリッピングで説明できる場合だけ許容し、自由なgain/delay fittingはしない。
materialな差があれば CTF 生成を中止する。

`dump_oracle_ctf_nf8k` は独立した SCP・manifest・CTF を持つ。
各 raw split の audio シンボリックリンクは元 dump の WAV を共有する。
保存形式は float32 `[60,2,129,2]` (tap,speaker,frequency,real/imag)。
manifest は計算条件、元 RIR/音声/索引の SHA256、対応検証、再構成 MSE と
RIMag、出力 SHA256 を記録する。limited/smoke dump は本学習に使えない。
計算と輸出は専用の一時ディレクトリから完了時に公開し、既存 dump を上書きしない。

これは有限長 CTF の Oracle 参照で、分離 SI-SNR の保証された上限ではない。
推定器の RIMag 損失と LS 目的関数も同一ではない。
将来の推定 CTF 比較では同じ native grid、因果順、投影・連結方式を使う。

## ローカルでの準備

このレシピのディレクトリで、tf-locoformer 環境を有効にして実行する。
CPU のみで計算し、共有音声・既存統計は書き換えない。

```bash
bash run.sh --stage 1 --stop_stage 1 --ngpu 0 --workers 8
python local/check_oracle_ctf_inputs.py --dump dump_oracle_ctf_nf8k
```

## TSUBAME への反映・検証

実行中ジョブがある既存 checkout は更新せず、専用 checkout を使用する。
管理コードは Git の同じ commit を同期する。
ローカルの完成 dump を `dump_oracle_ctf_nf8k_imported` として転送し、
CPU ジョブで元 DAI-8 音声との全 SHA256 を検証して参照先を再構築する。
TSUBAME 側には元 RIR の再転送を要しない。元 manifest は provenance に残す。

```bash
mkdir -p qsub_logs
qsub -g tga-shinoda qsub/prepare_imported_oracle_ctf.sh dump_oracle_ctf_nf8k_imported
# 保存済み dump の再検証だけを行う場合:
qsub -g tga-shinoda qsub/check_oracle_ctf_inputs_cpu.sh
# CPU準備の正常終了後に投入する。hold_jidだけでは成功を保証しない。
qsub -g tga-shinoda qsub/check_oracle_ctf_concat_gpu.sh
```

CPU準備は cpu_8×1・1時間・priority-5、上限見積もり0.024ポイント。
保存済み dump の再検証は cpu_4×1・30分・priority-5、上限見積もり0.006ポイント。
CUDA検証は gpu_1×1・15分・priority-5、上限見積もり0.04ポイント。
CUDA検証は実batch4の有限loss/gradient/optimizer step、最長validation入力、
checkpoint経由の推論を確認し、probeの重みを破棄する。
準備成功とCUDA検証成功を確認してから本学習を投入する。

```bash
qsub -g tga-shinoda qsub/train_oracle_ctf_concat_1gpu.sh
# 時間切れ後の同条件継続:
qsub -g tga-shinoda qsub/train_oracle_ctf_concat_1gpu.sh true
```

本学習は gpu_1×1・24時間・priority-5、上限見積もり3.84ポイント。
これらは確定課金ではない。本学習の投入は準備・検証とは別に行う。
学習前にCUDA検証時のモデル・実行コード、config/dump hashの一致を確認する。
自動進行やドキュメントだけの変更では、同一の検証済みモデルを再利用できる。
出力は `exp/enh_train_tflocoformer_s_nf8k_oracle_ctf_concat_seed0`。

```bash
bash run.sh --stage 7 --stop_stage 7 --ngpu 1
bash run.sh --stage 8 --stop_stage 8 --ngpu 0
```

valid.loss.best.pth を採用し、cv/tt の2話者無響音声に対する SI-SNR/SDR 等を
既存 ESPnet の採点で評価する。元 DAI-8 のモデル・出力は変更しない。

### 出力保存条件を揃えた再評価

Stage 7はBaselineと同じ `--normalize_output_wav true` を必須にする。
PCM16保存前に各話者の波形ピークを0.9へ揃え、クリッピングを防ぐ。
学習済みモデルの重み・音声入力・教師・採点器を変更しない。

既存出力を保持して再評価する場合は、次の2ジョブを投入する。
前回の実測は推論約24分、CPU採点約15分。GPUは前回の最大キャッシュが
約54 GBのため `gpu_1`、採点は前回と同じ8並列で `cpu_8` を使う。

```bash
qsub -g tga-shinoda -l gpu_1=1 -l h_rt=01:00:00 -N oracle_norm_infer \
    qsub/reevaluate_normalized.sh infer
qsub -g tga-shinoda -l cpu_8=1 -l h_rt=01:00:00 -hold_jid <infer_job_id> \
    -N oracle_norm_score qsub/reevaluate_normalized.sh score
```

出力は元実験配下の `evaluation_normalized_20261016`。元のconfig、学習完了
checkpoint、120 epochのbest重みを参照する。重みのSHA256を前回評価の
記録と照合し、保存済みconfigと重みのSHA256を再評価前後で確認する。
新しい `.pipeline` で採点するため、以前の
score shardを再利用しない。先行失敗時は完了markerなしで採点を拒否する。
旧 `RESULTS.md`、音声、ジョブ履歴は元の場所に保持する。

## Stage 8までの自動進行

### Oracle CTF Cross-Attention＋全層FiLMとの比較

`--variant film` は、同じOracle dump、音声・教師、batch4、valid batch1、
FP32、seed0、4秒crop、optimizer/schedulerで条件付けだけを変更する。
CTFの2話者の実部・虚部を既存の3×3 Conv＋gLNへ渡し、各周波数で
音声をQuery、CTFをKey/Valueとして各ブロック直前に参照する。
FiLMはγ=1・β=0で初期化し、ESPnetの全体初期化後にも恒等性を復元する。
CTF順序のランダム交換と可変長入力の処理は時間連結版と共通。
追加パラメータは227,232個。CTFを音声列へ追加しないため、Decoderは
元の音声フレーム数を受け取る。追加のtap位置符号は導入しない。

専用checkoutで元の完成dumpを読み取り用symlinkとして共有する。
時間連結版のcheckpointや出力は共有・上書きせず、FiLM版は新規学習する。
GPU検証は実batch4の2更新、CTF encoderへの勾配、最長validationと
checkpoint推論を確認する。probeの重みを本学習に流用しない。

```bash
qsub -g tga-shinoda qsub/check_oracle_ctf_film_gpu.sh
```

検証成功時だけ学習を投入し、24時間枠終了後のcheckpoint継続、学習終了、
正規化ありのStage 7、Stage 8まで自動進行する。出力は
`exp/enh_train_tflocoformer_s_nf8k_oracle_ctf_film_seed0`。
CUDA検証の記録・pipelineのpin・job journalはFiLM版専用に保存する。
検証はgpu_1×1・20分・priority−5、上限0.0533ポイント。
本学習と後続の資源・上限は以下の時間連結版と同じで、区間ごとの実投入を
FiLM実験の `.pipeline/jobs.json` に記録する。確定課金ではない。

### 時間連結版

専用checkoutで次を1回投入する。初期制御ジョブが学習と終了待ちの
制御ジョブを登録する。学習完了後に推論、推論全8,000件の完了確認後に
8並列のCPU採点へ進み、`RESULTS.md` と `.pipeline/score.done.json` を作る。

```bash
qsub -g tga-shinoda -l cpu_4=1 -l h_rt=00:10:00 -p -5 \
    -N oracle_control qsub/oracle_ctf_pipeline.sh --phase control
```

学習の時間切れはSGE accountingとcheckpointのepoch進捗を確認して再開する。
正常終了、max_epoch150またはpatience10の到達、best modelの整合を確認して
Stage 7へ進む。時間切れ以外の異常、進捗のない再開、欠落した出力は停止する。
自動再投入は学習12区間・推論2回・採点3回を上限とする。
採点の再実行では、全IDと全指標が揃った区間を再利用する。

各ジョブはpriority-5。学習gpu_1/24hは上限3.84ポイント/区間、
推論gpu_1/4hは0.64、採点cpu_8/6hは0.144、
制御cpu_4/10分は0.002/回。実行時間と再開回数は未確定で、
全再試行上限を使った場合の上限見積もりは47.828ポイント。
投入ごとのID・資源・上限見積もりは `.pipeline/jobs.json` に記録する。
確定課金とは異なる。制御・実行中はこのcheckoutを更新しない。

## 雑音ありWHAMR!・Oracle時間連結 S

DAI-7の1 GPU・実batch4・FP32・seed0 Baselineと比較する。
Baselineの出力先は
`/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet-baseline-pipeline/egs2/whamr/enh7_baseline/exp/enh_train_tflocoformer_s_whamr_8k_1gpu_batch4`。
既存の最良単一checkpointによるtest絶対SI-SNRは11.46 dB、SDRは12.47 dB。
論文のSI-SNRi/SDRiや5best平均とは区別する。

入力はDAI-7の `mix_both_reverb_min_8k`、教師は同じ元の無響2話者音声。
モデル・optimizer・scheduler・batch/crop/STFT・停止条件を保存済みconfigと照合し、
元のStage 5の3種類のshapeと全train/validサンプル数・長さも照合する。
separatorを時間連結版へ変更し、元の5,125,252パラメータへ288個だけ追加する。
Baselineの重みを流用せず、スクラッチ学習する。

専用 `dump_oracle_ctf_whamr8k` の音声はDAI-7のWAVへのsymlink、CTF配列は
検証済みNF Oracle dumpの `features` へのsymlink。全28,000件でID、
2話者教師のSHA256、音声長を照合し、同じ話者・RIRのCTFだけを再利用する。
train/validは保存済みの独立した雑音トラックとも照合する。
NF混合音の非飽和サンプルで `WHAMR = NF + noise` をPCM16量子化3 LSB以内で
確認し、除外したNF飽和サンプル数を記録する。testの元dumpには雑音トラックが
保存されていないため、testはID・教師・長さと非ゼロの混合差分までを検証し、
独立した雑音との一致を検証したとは扱わない。
元NF dumpの特徴・音声の全hash検証と、新WHAMR dumpの入力hash検証を行う。
manifestへこの対応検証と索引hashを残し、音声・CTF・既存結果を再生成しない。

専用checkoutでコードをcommit/push/syncした後、このレシピから投入する。

```bash
mkdir -p qsub_logs exp
qsub -g tga-shinoda qsub/prepare_whamr_oracle.sh
# CPU準備のexit0とmanifestを確認後:
qsub -g tga-shinoda qsub/check_whamr_oracle_gpu.sh
```

GPU検証は実batch4の更新・CTF投影への勾配・最長valid・保存復元推論を確認し、
検証成功時だけ `--condition whamr --phase control` を呼ぶ。
その後は24時間枠の学習再開、最良単一checkpointのStage 7、Stage 8まで自動進行。
Stage 7の `normalize_output_wav true` とStage 8の採点器はBaselineと同じ。
NFのdump・モデル・結果とは出力を分離する。新しい出力先は
`exp/enh_train_tflocoformer_s_whamr8k_oracle_ctf_concat_seed0`。

priorityはすべて-5。CPU準備はcpu_4/30分（上限0.006ポイント）、
GPU検証はgpu_1/20分（0.0533）、学習はgpu_1/24時間（3.84/区間）。
同じ構造・長さのNF版での推論約24分・採点約15分を参考に、
WHAMR推論はgpu_1/1時間（上限0.16）、採点はcpu_8/1時間（0.024）とする。
後続制御はcpu_4/10分（0.002/回）。実投入は `.pipeline/jobs.json` に残す。
すべて確定課金ではなく上限見積もり。時間切れ以外の異常時は停止する。
