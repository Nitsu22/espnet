# Rec-RIR / DAMSEP 音声分離評価 — NF-WHAMR 8 kHz

2026-10-16。全3,000混合・各出力6,000音源を、同じ入力・参照・ESPnet採点条件で比較した。
Rec-RIRはLS評価でも使ったbest97、DAMSEPは既存3損失版best29。

## クリーン出力：分離＋残響除去（direct-path参照）

| モデル | SI-SNR [dB] | SDR [dB] | SIR [dB] | SAR [dB] | STOI [%] |
|---|---:|---:|---:|---:|---:|
| RecRIR_best97 | 14.16 | 15.23 | 26.70 | 15.62 | 96.53 |
| DAMSEP_best29 | 13.50 | 14.76 | 26.70 | 15.11 | 96.05 |

Rec-RIR − DAMSEP: SI-SNR +0.66 dB、SDR +0.47 dB、STOI +0.49ポイント。

## 残響出力：残響を残した分離

| モデル | SI-SNR [dB] | SDR [dB] | SIR [dB] | SAR [dB] | STOI [%] |
|---|---:|---:|---:|---:|---:|
| RecRIR_best97 | 10.71 | 11.31 | 18.07 | 12.68 | 90.50 |
| DAMSEP_best29 | -6.61 | 2.46 | 15.81 | 2.83 | 75.06 |

Rec-RIR − DAMSEP: SI-SNR +17.32 dB、SDR +8.84 dB、STOI +15.44ポイント。

## 解釈

クリーン出力ではRec-RIRのSI-SNR・SDR・STOIが上回る。SIRは両者とも約26.70 dBでほぼ同じ。
残響出力でもRec-RIRが上回り、特にDAMSEPの低いSI-SNRが目立つ。
同じ評価条件にそろえた、既存の学習済みモデル同士の比較である。
学習損失、訓練期間、停止条件、モデル構造は異なるため、差の原因をアーキテクチャだけに帰属しない。
クリーン参照と残響参照は別の目標なので、2つの表のSI-SNR同士を直接比較してモデルの良し悪しを判断しない。
数値は入力に対する改善量ではなく、出力音声の絶対スコア。

## 統一した評価条件

- NF-WHAMR / 8 kHz / min / 全3,000混合 / 左ch / 全発話。
- DAMSEPで使用したdump_cleanのwav.scp・spk1/2.scpと、dump_reverbのspk1/2_reverb.scpを使用。
- 各モデルのclean出力は同じdirect-path参照、reverb出力は同じ残響参照で採点。
- 推論出力は各話者・各枝で0.9ピーク正規化しPCM16保存。DAMSEPの既存処理と一致。
- 入力と同じサンプル長を保つ。参照を使った時間位置の補正は加えない。
- ESPnet enh_scoring.pyをそのまま使用。mir_eval bss_eval_sourcesによる既存のSIR基準のPIT割り当てを、全指標で共通使用。
- SI-SNR、SDR、SIR、SAR、STOIを6,000音源で平均。STOIは%表示。
- DAMSEPの全件スコアは2026-10-13にTSUBAMEで完了した結果を再利用し、未丸めの音源別スコアから同じ方法で再集計。

## 実装と検証

Rec-RIRの学習済みspeech headsを使用。入力の混合音声を学習設定に従って正規化し、clean・reverb推定スペクトルを逆STFTして元のゲインを復元する。
逆STFTには入力サンプル長を明示し、最後の部分フレームを保持する。モデル本体・学習・重みは変更していない。

- 音量復元・話者/枝の取り出し・端数長を検証するCPUテスト4件が成功。
- Python/シェル構文・Git差分の検査に成功。
- DAMSEPの保存済み出力3混合・両枝をラボ環境で再採点。5指標×2話者が以前のスコアと一致。
- numpy/scipy/torch/soundfile/mir_eval/pystoiの版はDAMSEP評価環境とすべて同じ。
- 確認用のclean/reverb参照音声はTSUBAME側とローカル側の左chが完全一致。
- Rec-RIRの実音声3件でGPU推論成功後、全件を推論。
- 両モデル・両枝・各5指標の6,000値がすべて有限。全ID集合と集計値を独立に再確認。
- exit 0、complete.json、validation/final_checks.jsonを確認。専用TMPDIRとtmuxセッションを終了後に清掃。

新規ファイル:

- espnet2/rir/rec_rir/speech_inference.py
- espnet2/bin/rec_rir_speech_inference.py
- egs2/whamr/rir_2spk/local/evaluate_rec_rir_speech.py
- egs2/whamr/rir_2spk/local/run_rec_rir_speech.sh
- test/espnet2/rir/test_rec_rir_speech_inference.py

## 実行・来歴

- 実行ホスト: pasmo / RTX 4090 × 1（GPU番号0）。採点CPUワーカー8、各1スレッド。
- 開始: 2026-10-16 09:55:20、終了: 2026-10-16 10:07:10。
- 実行コード: `e34e5e1e5addd3b32d127f875e3f18fd86b7bc6a`（codex/damsep-ls8k-eval）。
- 出力: `/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/exp/rec_rir_damsep_speech_20261016`
- Rec-RIR checkpoint SHA256: `5fe27c7996e70d290d8646fb39f9cb6fe6a3f2972138e5f16b42730c8f4cafc8`
- 採点コード SHA256: `5925a81bf0aad878b8a9268ca8eae30df23503338900ea7bf6561133c9350cbb`
- DAMSEPスコア再現の最大絶対差: clean 1.07e-14、reverb 1.42e-14。
- 全件集計: `comparison.csv` / `comparison.json`。
- Rec-RIR音源別スコア: `rec_rir/inference/clean/scoring/` と `rec_rir/inference/reverb/scoring/`。
- DAMSEP保存済みスコア: `input/damsep/clean/` と `input/damsep/reverb/`。
- 設定・参照index・版・hash: `metadata.json` / `rec_rir/`。検証記録: `validation/`。
