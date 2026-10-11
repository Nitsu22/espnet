# enh9_condition

NF-WHAMR 8 kHz の条件付け実験用ひな形。現在は enh4_clean の
TF-Locoformer small 設定をコピーした状態で、条件付け設定は未追加。

- run.sh、cmd.sh、conf/tuning の設定、local/path.sh は編集用コピー。
- 共通 ESPnet 処理は enh_tmp と同じ TEMPLATE への相対リンク。
- local/ のデータ準備処理は enh4_clean/local への相対リンク。
- dump_clean、data、exp/enh_stats_8k は enh4_clean の既存データ・統計への相対リンク。
- 学習・評価出力は専用の exp/enh_train_tflocoformer_small_nf_4gpu。

入力は雑音なし・残響ありの2話者混合、教師は無響音声。
train/valid/test は tr/cv/tt_mix_clean_reverb_min_8k。

run.sh は既存 dump・統計を使う Stage 6–8、4 GPU の設定を引き継ぐ。
実行前に実験条件と GPU 数を調整すること。作成時には学習は実行していない。
共有データ・統計の上書きを避けるため Stage 1–5 は実行しない。
再生成する場合は共有リンクを外し、専用保存先を用意する。
リンク先が存在する同じリポジトリ配置で使用する。
