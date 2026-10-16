# Rec-RIR 3損失版・再構成損失のみ版の8 kHz LS評価

2026-10-16。NF-WHAMRのノイズなし・8 kHz・2話者、全3,000混合／6,000音源で比較した。
主比較は両版の `valid.loss.ave_5best.pth`（各版の検証損失上位5モデルの平均重み）。
3損失版best97は既存予測を再採点した参考値であり、平均重み同士の比較とは区別する。

| 指標 | 3損失・平均5 | 再構成のみ・平均5 | 参考：3損失・best97 |
|---|---:|---:|---:|
| RMSE ↓ | 0.002106 | 0.002216 | 0.001870 |
| 波形相関 ↑ | 0.977178 | 0.973570 | 0.980354 |
| RT60 MAE [s] ↓ | 0.072618 | 0.081597 | 0.059014 |
| DRR MAE [dB] ↓ | 1.891975 | 1.341042 | 1.107059 |
| C50 MAE [dB] ↓ | 1.390322 | 1.341336 | 1.019290 |

平均重み同士では3損失版のRMSEが4.96%小さく、波形相関・RT60 MAEも良い。
再構成のみ版はDRR MAE・C50 MAEが小さい。全指標で一方が優位という結果ではない。
参考の3損失best97は、ここに示した両平均モデルより5指標すべてで良い。
この表は再構成のみ版best64との比較ではなく、単一best同士の優劣は判断していない。

## 比較条件

- 既存のnative 8 kHz LS direct-to-reverb参照を再利用。参照の作り直しは行っていない。
- 既存のRec-RIR 8 kHz LS評価と同じformatted PCM16入力一覧をコピーし、全発話・左chをFP32で推論。
- ordinary sweep PIM、16,000 samples（2秒）、mono FLOAT WAV。以前の8,192 samples出力のゼロ埋めではない。
- ピーク位置合わせ、独立ピーク正規化、全体RMSEによる2話者PIT、極性反転なし。
- LS参照はdirect-to-reverb相対伝達を表す。RT60・DRR・C50はその波形の診断値で、物理RIRの部屋指標とは区別する。絶対ゲイン・絶対遅延は採点対象外。
- 保存された学習設定は、設定ファイル名・出力先を除くとロス重みだけが異なる。モデル構造・データ・seed・最適化設定は同じ。
- 3損失の重みはdirect音声・残響音声・再構成が各1。再構成のみは0・0・1。両方とも音声出力枝を保持する。
- 各版固有の検証損失で重みを選択。3損失版は100 epoch終了、再構成のみ版は75 epochで早期終了。seed 0の既存学習結果の比較である。

## 検証

- Python構文、シェル構文、Git差分検査に成功。
- 各平均モデルを実音声3混合で動作確認してから、各3,000混合を全件推論。
- 参照・予測のID、8 kHz・16,000 samples・mono FLOAT形式を検証。各モデルの全6,000音源について5指標が有限。
- 3損失best97の再採点summaryは、既存の8 kHz LS評価と全項目で完全一致。
- 音源別CSVから5指標を独立再集計し、summaryと一致（許容差1e-12）。
- 終了コード0、complete.json、final_checks.jsonを確認。専用tmux・一時ディレクトリの消去とGPU解放を確認。

## 実行・成果物

- 実行ホスト：pasmo／NVIDIA GeForce RTX 4090 × 1（ホストGPU番号0）。
- 開始：2026-10-16 22:10:13、終了：2026-10-16 22:26:34。
- 実行コード：`1c3549dca8274adc26cb9267b01110dc59c990d9`（`codex/rec-rir-loss-ls8k-20261016`）。
- 出力：`/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/exp/rec_rir_losses_ls8k_20261016`。
- 集計：`comparison/comparison.csv`、`comparison/comparison.json`。
- 音源別結果：`comparison/models/00/per_source.csv`（3損失平均5）、`01/per_source.csv`（再構成のみ平均5）、`02/per_source.csv`（best97参考）。
- 入力・参照・設定・重みのハッシュ：`metadata.json`と各モデルの`metadata.json`。使用重みと設定もコピーして固定。
- 検証記録：`final_checks.json`、`independent_checks.json`。実行ログ・時刻・終了状態・GPU情報は出力ディレクトリと同名のsidecarに保存。
- 実行入口：`local/run_rec_rir_losses_ls8k.sh`、`local/evaluate_rec_rir_losses_ls8k.py`。既存のLS採点器・参照・モデル実装は変更していない。
- 参照preparation SHA256：`8d5d16dd191bc87a7f211c004f3def28a70d5069fce1062c8fca7df762813103`。
- 入力index SHA256：`ecc778e816628e73d3f7b0754309bbc1eb06a3810d5c5bd6cc879bb53bc2fa1d`。
