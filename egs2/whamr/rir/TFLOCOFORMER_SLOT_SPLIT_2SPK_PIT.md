# TF-Locoformer Slot-Split 2話者PIT

更新日: 2026-08-24

## 目的

既存TF-Locoformer CTF-only 2話者PITのlate-split構造を変更し、4層の
中間で匿名speaker slotを作る。既存baselineの学習・推論・採点経路は
変更せず、slot分岐の効果を同じ条件で比較する。

## 既存実験

既存TF-Locoformerは、4個のTF-Locoformer blockをすべて共有した後、
1個のtime scorerと1個のjoint CTF headから2話者分を同時に出力する。

~~~text
mixture STFT
  -> Conv encoder
  -> shared block x 4
  -> shared time pooling
  -> joint head: C -> 2 speakers x real/imag x 60 taps
~~~

既存runの学習条件は8 kHz、4秒segment、batch size 4、seedの実効値0、
AdamW、RIMag reconstruction PIT、60-tap CTFである。100 epochの総学習
時間は38時間1分40秒、1 epochはtrain約19分47秒、valid約2分56秒だった。

既存の掲載scoreは次のとおり。ただし valid.loss.best の推論は学習途中の
epoch 58時点のsnapshotであり、学習完了後の最終bestは未採点である。

| checkpoint | RMSE | RMSE50 | corr | RT60 MAE | DRR MAE | C50 MAE |
|---|---:|---:|---:|---:|---:|---:|
| epoch 58時点 valid.loss.best | 0.010345 | 0.037553 | 0.815532 | 0.134263 | 2.401678 | 5.240298 |
| 学習完了後 ave-5-best | 0.010285 | 0.037341 | 0.817324 | 0.124420 | 2.300156 | 5.110687 |

## 新方式

前半2層は1本のstreamで共有する。2層目の出力へ学習可能なslot embeddingを
加算し、2本のspeaker-slot activationを作る。

~~~text
x_shared = SharedBlock2(SharedBlock1(x))

x_slot1 = x_shared + slot_embedding1
x_slot2 = x_shared + slot_embedding2
~~~

slot embeddingはshape [2, C]、標準偏差0.02の正規分布で初期化し、時間・
周波数方向へbroadcastする。slotは固定話者IDではなく、PITで対応を決める
匿名output slotである。

後半2層はslot軸をbatch軸へ畳み、同じblock parameterを適用する。

~~~text
[B, 2, C, T, F]
  -> [B*2, C, T, F]
  -> weight-shared Block3
  -> weight-shared Block4
  -> [B, 2, C, T, F]
~~~

したがってTF-Locoformer blockは既存と同じ4 parameter setだけを持つ。
activationと計算は後半のみ2 streamとなり、block実行量は6 block相当になる。

parameter数の実測は以下のとおり。

| model | parameters | baselineとの差 |
|---|---:|---:|
| 既存TF-Locoformer | 5,163,793 | - |
| Slot-Split | 5,182,706 | +18,913 (+0.37%) |

追加parameterはslot embeddingと、2個目のtime scorerおよび独立headの
中間Linearである。4個のTF-Locoformer block parameterは増えていない。

最後のtime scorerとCTF headはslotごとに独立させる。

~~~text
slot 1 -> scorer 1 -> Softmax(T) -> head 1: C -> C -> 2 x 60
slot 2 -> scorer 2 -> Softmax(T) -> head 2: C -> C -> 2 x 60
stack -> complex CTF [B, 2, F, 60]
~~~

## 公平比較条件

新configは既存configのコピーであり、差は以下の3項目だけである。

~~~yaml
slot_split: true
num_shared_layers: 2
slot_embedding_std: 0.02
~~~

以下は既存TF-Locoformerと共通である。

- train/valid/test dataとshape file
- 1 GPU、batch size 4、folded batch
- 8 kHz、4秒segment、STFT 256/128
- 4 TF-Locoformer layers、embedding 96
- 60-tap CTF、RIMag reconstruction PIT
- AdamW、学習率、scheduler、AMP、grad clip、patience
- CTFからRIRへのPIM
- peak alignment、peak scaling、RMSE50による評価PIT

stats directoryも既存の exp/rir_stats_8k_rec_rir_pit を再利用する。
出力expだけを分離する。

## 実行

学習:

~~~bash
cd /net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir
./run_rec_rir_tflocoformer_slot_split_2spk_pit_clean_8192.sh
~~~

学習完了後のbest checkpoint推論と標準採点:

~~~bash
./run_eval_rec_rir_tflocoformer_slot_split_2spk_pit_clean_8192_current.sh
~~~

比較時は既存baselineと新方式の両方について、学習完了後の
valid.loss.best.pth同士、またはave-5-best同士を採点する。既存のepoch 58
snapshotを最終比較には使用しない。

## 変更範囲

- espnet2/rir/rec_rir/tflocoformer_ctf_pit.py
- test/espnet2/rir/rec_rir/test_tflocoformer_ctf_pit_slot_split.py
- conf/tuning/train_rec_rir_tflocoformer_slot_split_2spk_pit_clean_8192.yaml
- run_rec_rir_tflocoformer_slot_split_2spk_pit_clean_8192.sh
- run_eval_rec_rir_tflocoformer_slot_split_2spk_pit_clean_8192_current.sh

既存baseline class、defaultのslot_split=false、既存checkpointのstate-dict名、
loss、PIT、estimate_ctf、estimate_rirは維持する。

## 実装確認

以下を確認済み。

- Python syntax compile、shell syntax check、git diff check
- 新旧configの差がslot分岐の3項目だけであること
- CTF出力shapeが [B, 2, F, 60] であること
- 後半block入力が [B*2, C, T, F] となり、4 block parameterだけを使うこと
- slot embedding、共有前半/後半block、2個のscorer/headへfinite gradientが流れること
- seedを揃えたとき、既存版と新方式の4 block初期値が完全一致すること
- 実TF-Locoformer、AMP、forward/backward、AdamW step
- batch size 4、32000 samplesで1 step成功
- estimate_ctfが [2, 129, 60]、estimate_rirが [2, 8192]を返すこと
- 既存TF-Locoformer checkpointが従来predictorへstrict loadできること
- 新configがSlot-Split predictorを選択すること

turingのQuadro RTX 5000で行ったランダム1 batchのsmoke testでは、peak
allocated memoryが10.18 GB、peak reserved memoryが12.80 GBだった。既存
学習ログとはGPU/runtimeが異なるため、既存値との速度・memory比較には使わない。

tf-locoformer環境にはpytestがないためpytest一括実行は未実施。追加テストの
主要4項目は同環境のPythonから個別に実行済み。
