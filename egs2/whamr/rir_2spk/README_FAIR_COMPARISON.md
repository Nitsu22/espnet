# NF-WHAMR 8 kHz：Rec-RIR Baseline / 遅延関係NB / 遅延関係Full

## 比較条件

3条件を新規学習し、CTF Headの構造を比較する。Baselineは元の
`BiSpatialNetPIT`（α・β融合、時間方向NB、重み付き時間Pooling、全CTF出力）を使う。
NB/Fullは `[q; k; q×k] → MLP → 有効時間の平均` から60遅延分の特徴を作る。
モデル容量・計算量は構造とともに変わるため、同じパラメータ数の比較とはしない。

| 項目 | 全3条件で共通 |
| --- | --- |
| データ | 同じNF-WHAMR tr/cv/tt、2話者、min、左ch、8 kHz |
| 波形 | TSUBAME既存16 kHz portable dumpをresample_poly(1,2)、FLOATで保存 |
| 音声側 | 96次元、noise 2層、speech 6層、同じ構造と初期重み |
| 初期化 | seed 0の未学習Baselineから音声側だけを共通化。CTF側は各構造で新規初期化 |
| CTF | 60 taps、CTFブロック6層、同じタップ順とPIM |
| 教師 | direct1/2・reverb1/2の同じ索引。教師directと推定CTFで再構成 |
| 損失 | RIMag、clean/reverb/reconstruction各1、3ロス共通PIT |
| 長さ処理 | 実長ごとにSTFT前から処理。発話ごとのロスをバッチ平均 |
| 入力 | 同じseed・データ順・4秒random crop。短い発話は同じ実長処理 |
| バッチ | folded、batch 4、入力shapeのみ、fold_length 80000、valid batch 1 |
| 学習 | AMP、AdamW lr .001、同じscheduler、最大100 epoch、patience 10 |
| モデル選択 | valid.loss最小、keep_nbest 5 |
| GPU | 各gpu_1=1、優先度-5、GPU割当はschedulerに従う |

各epochのデータ順とworkerのcrop乱数はESPnetの`seed + epoch`で決まる。
同じseedでもモデル構築順によりDecoder初期値は変わるため、共通のランダム音声重みを
`exp/lag_relation_comparison/profile/shared_speech_init.pth`に保存し、全条件へ読み込む。
学習済み重みの流用ではない。共通Speech側も全条件で学習する。

Baselineの長さ処理だけを揃えるため、`rec_rir_pit`の`model_conf.length_aware: true`
を追加した。設定しない既存Baselineは従来のwrapperを使う。パラメータ名とネットワークは
変わらず、旧checkpointをstrictに読み込める。提案版も同じ長さ対応wrapperを継承する。

## スクリプトと成果物

- 設定：`conf/tuning/train_rec_rir_nf8k_fair_{baseline,nb,full}.yaml`
- 共通データ：`dump_nf8k_fair/`。`preparation.json`に元索引SHA、変換条件、件数、
  長さ範囲、noise-free残差を保存。既存データは変更しない。
- 共通shape：`exp/rir_stats_nf8k_fair/`
- CUDA確認：`local/profile_lag_relation_comparison.py`
- 確認結果：`exp/lag_relation_comparison/profile/{baseline,nb,full}.json`と`ready.json`
- 学習先：`exp/rir_train_rec_rir_nf8k_fair_{baseline,nb,full}_seed0/`
- ジョブログ・終了コード：`exp/lag_relation_comparison/jobs/`

CUDA確認では、実データbatch 4のAMP更新、全勾配の有限性、出力形状、padding不変性、
発話平均、教師交換PIT不変性、推論の順序復元、短入力のエラー、RIR出力、最長検証入力、
Checkpointingの出力・勾配一致、CTF側パラメータの独立性を検査する。
Baselineのstate_dict互換性と、音声側初期重みの完全一致も確認する。
成功時だけ`ready.json`を書き、各学習ジョブがcommit・設定・データ・初期重みのSHAを確認する。
AMPの初期loss scaleで勾配がoverflowした場合は、本学習のTrainerと同じく更新をskipして
scaleを下げる。最大32試行以内に4回の有限な更新が必要で、調整履歴も確認結果へ保存する。
どの条件もこの確認用更新を学習に引き継がず、共通の未学習重みから再開する。

TSUBAME専用checkoutへGit同期後、レシピのディレクトリから投入する。
準備CPUジョブ → CUDA確認GPUジョブ → 学習3ジョブの順でhold_jidを指定する。
依存先の終了だけでは成功を保証しないため、後続では必要な完了ファイルも検査する。
CUDA確認の第5引数に`true`を渡すと、成功後にだけ学習3ジョブを自動投入する。
既存の共通データ・shapeが完成済みの場合、準備ジョブを再実行する必要はない。
各学習ジョブIDは`profile/submitted_{baseline,nb,full}.job`へ直ちに保存する。

```bash
qsub -g tga-shinoda -l cpu_8=1 -l h_rt=01:00:00 -p -5 \
  -N recrir_fair_prepare qsub/lag_relation_comparison.sh prepare all 3600
qsub -g tga-shinoda -l gpu_1=1 -l h_rt=00:20:00 -p -5 \
  -hold_jid PREPARE_JOB -N recrir_fair_profile \
  qsub/lag_relation_comparison.sh profile all 1200
qsub -g tga-shinoda -l gpu_1=1 -l h_rt=24:00:00 -p -5 \
  -hold_jid PROFILE_JOB -N recrir_fair_nb \
  qsub/lag_relation_comparison.sh train nb 86400 false
```

確認後に自動投入する場合は、学習を別途投入せず、2つ目のコマンドを
`qsub/lag_relation_comparison.sh profile all 1200 false true`で実行する。

`baseline`と`full`も最後と同じ形式で投入する。全条件を同じ最大epochとearly stopping
で学習するため、24時間内に終わらない場合だけ、保存済みcheckpointから同じ条件で
自動継続投入する。通常の学習エラーでは継続しない。継続ジョブIDは各ジョブログ内へ保存する。
学習が正常終了しただけで性能改善とは判断せず、同じcheckpoint選択・PIM・テスト参照・
評価指標による比較が必要。
