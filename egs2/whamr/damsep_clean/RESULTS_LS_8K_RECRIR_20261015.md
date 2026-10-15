# Native 8 kHz LS評価: Rec-RIR・DAMSEP・BiMamba

2026-10-15。NF-WHAMRの全3,000混合・6,000音源を同じ条件で評価した。
Rec-RIRがRMSE・相関・RT60/C50誤差で最良。DRR誤差はRec-RIRとDAMSEPでほぼ同じ。

| 指標 | Rec-RIR / best97 | DAMSEP 3損失 / best29 | BiMamba / best45 |
|---|---:|---:|---:|
| RMSE ↓ | 0.001870 | 0.002951 | 0.002447 |
| 相関 ↑ | 0.980354 | 0.955394 | 0.967939 |
| RT60 MAE [s] ↓ | 0.059014 | 1.882301 | 0.079019 |
| DRR MAE [dB] ↓ | 1.107059 | 1.105765 | 1.504365 |
| C50 MAE [dB] ↓ | 1.019290 | 2.514572 | 1.598799 |

Rec-RIRのRMSEはBiMambaより23.6%、DAMSEPより36.6%小さい。

## 評価条件

- ユーザーが用意したnative 8 kHz LS direct-to-reverb参照を再利用。参照を作り直していない。
- 2秒 / 16,000 samples、ordinary sweep PIM、mono FLOAT WAV。
- 3モデルとも同じformatted PCM16 NF混合音声を全発話・左chで推論した。Rec-RIRの入力indexは前回BiMambaとバイト単位で一致。
- peak alignment、独立peak normalization、RMSEで2話者PIT。極性反転なし。
- 正解direct RIRをPIM励起に与えない。絶対遅延・絶対ゲインは採点対象外。
- LS参照は相対伝達フィルタであり、RT60/DRR/C50はその波形の診断値。物理的な部屋の指標とは区別する。
- 前回のDAMSEP/BiMamba予測を再利用して同じ採点器で再集計し、両モデルの全summary値が前回と完全一致することを確認。

## Rec-RIR

`rir_train_rec_rir_2spk_pit_clean_8192`の`valid.loss.best.pth`は`97epoch.pth`。
8 kHz / 2話者PIT / clean・残響・再構成の3損失（重みは各1.0）。
FFT256 / window256 / hop128 / sqrt-Hann。推論はFP32。
元の学習設定とcheckpointをコピーして固定し、既存のRec-RIR推論コードを使用した。
学習・モデル実装・重みは変更していない。以前の8,192 samplesの出力をゼロ埋めせず、16,000 samplesを再推論した。

チェックポイントの選択条件・学習損失は3モデル間で異なるため、今回の数値は各学習済みモデルの比較である。
旧Rec-RIR評価は物理RIR参照・8,192 samples・rmse_50msでのPITであり、旧スコアとの数値差を性能改善として扱わない。

## 検証・終了確認

- Python構文、シェル構文、Git差分の検査に成功。
- 実音声3件のGPU推論に成功してから全件を実行。
- 同3件の入力は以前のRec-RIR推論時と完全一致。先頭8,192 samplesの6音源出力は旧PCM16出力と1 LSB以内で一致（legacy_prefix_validation.json）。
- 3モデルとも3,000混合・6,000音源を採点。RMSE・相関・RT60・DRR・C50の全6,000値が有限。
- exit 0、complete.json、comparison/evaluation_complete.json、final_checks.jsonを確認。
- 実行用tmuxセッションと自分の専用TMPDIRは終了後に消去。

## 実行・来歴

- ホスト: pasmo / NVIDIA GeForce RTX 4090 × 1（GPU番号0）。採点はCPU。
- 開始: 2026-10-15 20:48:37、終了: 2026-10-15 20:56:43。
- 実行コード: `d436c5bd72ebf6aa323b2274956218b70e1e244f`（codex/damsep-ls8k-eval）。
- 新規スクリプト: `egs2/whamr/rir_2spk/local/evaluate_rec_rir_ls8k.py`、`local/run_rec_rir_ls8k.sh`。
- 出力: `/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/exp/rec_rir_ls8k_20261015`
- checkpoint SHA256: `5fe27c7996e70d290d8646fb39f9cb6fe6a3f2972138e5f16b42730c8f4cafc8`
- config SHA256: `d4e3461ca974c7c41ae643b4ba4aa9479007cea7902b77c9002884bbff34137f`
- LS参照: `/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k`
- LS参照preparation SHA256: `8d5d16dd191bc87a7f211c004f3def28a70d5069fce1062c8fca7df762813103`
- 比較表: `comparison/comparison.csv`、`comparison/comparison.json`。
- 音源別結果: `comparison/models/00/per_source.csv`（DAMSEP）、`01/per_source.csv`（BiMamba）、`02/per_source.csv`（Rec-RIR）。
- 入力・checkpoint・旧比較のハッシュは`metadata.json`に保存。ログ・終了コード・実行時刻は出力ディレクトリと同名のsidecarに保存。
