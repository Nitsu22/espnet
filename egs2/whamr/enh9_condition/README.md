# NF-WHAMR 8 kHz: Oracle CTF temporal conditioning

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
# CPU準備の正常終了後に投入する。hold_jidだけでは成功を保証しない。
qsub -g tga-shinoda qsub/check_oracle_ctf_concat_gpu.sh
```

CPU準備は cpu_8×1・1時間・priority-5、上限見積もり0.024ポイント。
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
学習前にCUDA検証のcommit/config/dump hash一致を確認する。
出力は `exp/enh_train_tflocoformer_s_nf8k_oracle_ctf_concat_seed0`。

```bash
bash run.sh --stage 7 --stop_stage 7 --ngpu 1
bash run.sh --stage 8 --stop_stage 8 --ngpu 0
```

valid.loss.best.pth を採用し、cv/tt の2話者無響音声に対する SI-SNR/SDR 等を
既存 ESPnet の採点で評価する。元 DAI-8 のモデル・出力は変更しない。
