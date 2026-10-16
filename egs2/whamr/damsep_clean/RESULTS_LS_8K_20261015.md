# DAMSEP / native 8 kHz BiMamba: LS direct-to-reverb evaluation

2026-10-15。NF-WHAMRの全テスト3,000混合・6,000音源を評価した。
波形RMSE・相関・RT60/C50誤差はBiMamba、DRR誤差はDAMSEPが良い。

| 指標 | DAMSEP 3損失 / best29 | BiMamba 8 kHz / best45 | 良い方 |
|---|---:|---:|---|
| RMSE ↓ | 0.002951 | 0.002447 | BiMamba |
| 相関 ↑ | 0.955394 | 0.967939 | BiMamba |
| RT60 MAE [s] ↓ | 1.882301 | 0.079019 | BiMamba |
| DRR MAE [dB] ↓ | 1.105765 | 1.504365 | DAMSEP |
| C50 MAE [dB] ↓ | 2.514572 | 1.598799 | BiMamba |

## 解釈

DAMSEPの波形相関は0.9554だが、LS参照に対するRT60の過大推定が大きい。
推定RT60の平均はDAMSEP 2.2095 s、BiMamba 0.2536 s、LS参照 0.3272 s。
これらは相対伝達フィルタをPIM測定した波形の診断であり、物理的な部屋のRT60/DRR/C50ではない。
また、ピーク位置・ピーク振幅を合わせる評価なので絶対遅延・絶対ゲインは採点していない。
この比較だけではRT60誤差の原因を特定できない。

## 共通の評価条件

- ユーザーが用意したnative 8 kHz LS dumpを再生成せず使用。16 kHz版のリサンプルではない。
- LS: causal FIR 16,000 taps / 2 s、ridge alpha=1e-6、全応答でフィット。
- ordinary sweep PIM: 8.192 s、62.5–4,000 Hz、元の逆フィルタ、ピークの2.5 ms前から16,000 samples。
- 出力はmono FLOAT WAV / 8,000 Hz / 16,000 samples。入力は全発話の左ch。
- 参照と予測のpeak alignment、独立peak normalization、RMSEでPIT。極性反転は許可しない。
- 両者とも同じformatted PCM16 NF混合音声を使用した。BiMambaの通常のFLOAT dumpとの差を全件検証し、PCM16の飽和と1 LSB以下の量子化差で説明できることを確認。
- 飽和を含む混合は134件、飽和サンプルは両ch合計1,247。クリップ後の最大差は3.051735e-5。入力の発話対応とファイルSHA256はinput_correspondence.jsonに保存。
- モデル選択条件は異なる。DAMSEPはclean SI-SDR項＋0.1×残響RI+Mag＋0.5×CTF再構成RI+Magのvalidation合計で選択されたbest29、BiMambaはCTF/sweep validation lossのbest45。アーキテクチャ単独の優劣を結論づける比較ではない。

## CTFの扱い

DAMSEPは学習済み3損失版の生の複素CTF [2,257,60] を再利用した。
cleanのみ版・clean+reverbのみ版はCTF枝を学習しないため、このCTF比較の対象には含めていない。
DAMSEPの復号は学習時のCTF再構成ロスと同じFFT512 / window256 / hop128 / sqrt-Hann。
ネットワーク入力のFFT512 / window512 / Hannとは区別した。
タップ0が現在フレームとなる因果的な全線形畳み込みを使用。
既存RecRIR PIMのunfold/matmulには逆順の係数が必要だが、追加したFFT畳み込みはDAMSEPの順序を直接扱う。
BiMambaは既存の推論実装をそのまま使い、FFT256 / window256 / hop128 / sqrt-Hann、129 bins・60 taps。
両者の最終出力を同じLS参照と採点器に渡した。PIM励起には正解direct RIRを与えていない。

## 検証

3件の変換テストが成功。独立した複素線形畳み込みとの一致、逆順RecRIR演算との一致、正のタップが正の遅延になること、恒等/遅延恒等CTFのordinary-sweep測定、非有限値の拒否を確認。
シェル構文・Git差分の検査も成功。
両モデルの3,000 ID・6,000音源を確認し、RMSE/corr/RT60/DRR/C50の全6,000値が有限。
ジョブはexit 0、complete.jsonとevaluation_complete.jsonを保存。
専用の一時ディレクトリとtmuxセッションは終了後に削除された。

## 実装したファイル

- espnet2/rir/damsep_pim.py: 学習時のタップ順序・STFTでCTFをordinary-PIM RIRへ変換。
- egs2/whamr/rir_2spk/local/evaluate_damsep_bimamba_ls8k.py: 発話対応・入力照合、BiMamba推論、CTF変換、共通LS採点、来歴と完了記録。
- egs2/whamr/rir_2spk/local/run_damsep_bimamba_ls8k.sh: 専用TMPDIR、スレッド/GPU指定、ログ・終了状態・後片付け。
- test/espnet2/rir/test_damsep_pim.py: 上記の変換検証。

## 実行と成果物

- ホスト: pasmo / NVIDIA GeForce RTX 4090 × 1（GPU番号0）。CTF変換・採点はCPU。
- 開始: 2026-10-15 18:27:38、終了: 2026-10-15 18:38:40。
- 実行コード: 175349076993122244f01a39279692422f6e1dcc（codex/damsep-ls8k-eval）。
- 出力: `/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/exp/damsep_bimamba_ls8k_20261015`
- 参照: `/net/midgar/work2/nitsu/learning/tf-locoformer/espnet/egs2/whamr/rir_2spk/dump_ls_direct_reverb/raw/tt_rir_2spk_nf_min_8k`
- 参照preparation SHA256: `8d5d16dd191bc87a7f211c004f3def28a70d5069fce1062c8fca7df762813103`
- BiMamba checkpoint SHA256: `728afa1b8d27d526ba87cb73ca48193ee08f91ee48023cc1f26d9f44bd8a990e`
- DAMSEP元出力: `/gs/bs/tga-shinoda/nitsu/research/tf-locoformer/espnet/egs2/whamr/damsep_clean/exp/damsep_nf_8k/enhanced_tt/ctf.scp`
- 生の比較: `comparison/comparison.csv`, `comparison/comparison.json`
- 音源別: `comparison/models/00/per_source.csv`（DAMSEP）、`comparison/models/01/per_source.csv`（BiMamba）
- 検証: `validation/final_checks.json`、入力対応: `input_correspondence.json`。
