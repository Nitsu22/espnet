# 学習済みRec-RIR Baselineに合わせる遅延関係Headの比較

## 比較対象

既存の二話者・3ロスBaselineを再利用し、提案NBとFullだけを新規学習する。
学習済みBaselineの正本は以下。

```text
/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir/exp/rir_train_rec_rir_2spk_pit_clean_8192
```

100 epoch完了、`valid.loss.best.pth`は97epochを参照する。5best平均も存在するが、
比較時は同じ選択基準を使う。学習済み重みを提案版の初期値として読み込むことはしない。

| 項目 | 元Baselineから引き継ぐ内容 |
| --- | --- |
| データ | 同じNF-WHAMR native 8 kHz。Train 20,000、Valid 5,000 |
| 波形 | 元のWAV bytesを保持。Mix/DirectのPCM16・Stereoも変換しない |
| 前処理 | 元のRecRIRPITPreprocessor、先頭ch、Trainだけ同じ位置の32,000 crop |
| 短い発話 | バッチ最大長までのpaddingを含めてSTFT・特徴・損失を計算 |
| バッチ | folded、batch4、valid_batch_size=nullで検証も最大4 |
| shape | 混合・Direct1/2・Reverb1/2の5本すべて、ファイルbytes・行順も保持 |
| fold_length | 5本すべて32,000（Train 10,124 batch/epoch） |
| 損失 | RIMag、Clean/Reverb/Reconstruction各1、共通PIT、教師Directによる再構成 |
| 初期化 | seed0、元と同じ標準初期化。共通初期重みファイルの作成・読込なし |
| 学習 | AMP、float32、AdamW .001、同じscheduler、clip1、100epoch、patience10 |
| モデル選択 | valid.loss最小、keep_nbest_models=5 |

`train_rec_rir_original_baseline_{nb,full}.yaml`は、学習済みの保存configから作成した。
CTF固有の項目、移送先パス、出力先・resume等の実行管理項目を除き、保存値と照合する。
ネットワーク容量・計算量の一致を意味するものではない。

## 長さ処理

提案版の`model_conf.length_aware: false`を指定する。このモードの学習とCTF推論は
元の`ESPnetRecRIRPITModel`を直接使う。個々の実長でグループ分けする経路を通らない。
Headは渡されたバッチ全体のTを使い、遅延範囲外だけを除くT-dフレームを平均するため、
音声のpadding由来の特徴も元Baselineと同じ条件で処理される。

`length_aware: true`の既存動作は維持するが、今回の比較では使わない。
CTF Head自体は`[q; k; q×k] → MLP → Average → NB/Full 6層 → 4出力`。

## データと検証

`local/prepare_original_baseline_comparison.py`は元の保存configと実際のtrain.log先頭の
投入コマンドを照合し、元WAVの同一bytes、SCPキー・行順、全5shapeを移送用にまとめる。
再サンプリング・再量子化・音量変更をしない。完了manifestは最後に書く。

元configのSHA256：

```text
d4e3461ca974c7c41ae643b4ba4aa9479007cea7902b77c9002884bbff34137f
```

TSUBAMEのレシピ内`exp/original_baseline_inputs`に転送後、CPUジョブで全WAVのSHA256と
サイズを検証する。全非Head設定と全5shapeも照合する。元データや既存checkpointは変更しない。
今回作り直したFLOATの`dump_nf8k_fair`や`shared_speech_init.pth`は使わない。

GPU確認ではNB/Fullの実データ4更新、AMP初期scale調整、元wrapperとの学習・推論一致、
共通PIT、形状、Checkpointingの値・勾配一致を確認する。元のfolded samplerで最大の
padding付き検証バッチを選び、損失の有限性とメモリも確認する。Baselineの再学習はない。

## TSUBAME投入

```bash
# 事前にGit同期・完全なデータ転送・qsub_logs作成を済ませ、rir_2spkから投入する。
qsub -g tga-shinoda -terse -p -5 -l cpu_8=1 -l h_rt=00:45:00 \
  -N recrir_orig_verify qsub/original_baseline_lag_relation.sh verify all 2700
```

このCPU検証が成功するとGPU確認を投入し、それも両条件で成功した場合だけNB/Fullの
2学習を投入する。各GPUは`gpu_1=1`、優先度-5。学習は24時間上限で、時間切れかつ
checkpointがある場合のみ同じ条件で継続する。通常エラーでは自動継続しない。

- 入力照合：`exp/original_baseline_comparison/input_audit.json`
- GPU確認：`exp/original_baseline_comparison/profile/ready.json`
- 学習：`exp/rir_train_rec_rir_original_baseline_{nb,full}_seed0`
- ジョブID：`exp/original_baseline_comparison/submitted_jobs.tsv`
- 開始・終了・終了コード：`exp/original_baseline_comparison/jobs/`

最初のCPU45分・GPU確認20分・学習24時間×2の上限ポイントは、通常優先度で
`0.03×0.8×0.75 + 0.2×0.8×(1/3) + 2×0.2×0.8×24 = 7.7513`。
実行時間は未計測であり確定課金ではない。時間切れ後の継続分は別途。
