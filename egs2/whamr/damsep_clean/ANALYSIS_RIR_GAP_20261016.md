# Rec-RIR / DAMSEP のCTF・RIR精度差の考察（2026-10-16）

原因は未確定。主な仮説は、DAMSEPの分離後音声を経由するCTF推定と、学習目標・最適化条件の違い。
既存の学習済みモデルの比較であり、モデルサイズや構造単独の因果効果を測った実験ではない。

## 確認した事実

- 共通LS参照・全3,000混合/6,000音源で、Rec-RIR / DAMSEPのRMSEは0.001870 / 0.002951、相関は0.9804 / 0.9554。
- DRR誤差は1.107 / 1.106 dBとほぼ同じ。RT60誤差は0.059 / 1.882 s、C50誤差は1.019 / 2.515 dB。
- DAMSEPはSPMambaの分離後波形x_sepを再度STFTし、話者ごとに共有後段へ渡してCTFを推定する。
- Rec-RIRは混合音声から2話者のclean/reverb/CTFを推定する。DAMSEPと同じ直列の波形分離段を経由しない。
- 残響音声出力のSI-SNRはRec-RIR 10.71 dB、DAMSEP -6.61 dB。DAMSEP後段の入力が残響音声として良好ではないことを示す。ただし、その表現がCTF推定に不適切であるという因果関係は未検証。
- 後段のhidden幅はDAMSEP 64、Rec-RIR 96。総パラメータ数の大小はCTF枝の容量の大小と同じではない。
- 両モデルとも、正解direct音声に推定CTFを畳み込んで正解残響音声を再構成する間接的なCTF教師信号を使う。今回のRec-RIRにも正解RIR/LSフィルタを直接当てる学習損失はない。

## 学習・選択条件の違いと訂正

学習に使用したDAMSEPコードcommit 72526c93eacを確認した。
DAMSEPのvalidation `loss`はclean SI-SDR項だけでなく、0.1×残響RI+Magと0.5×CTF再構成RI+Magを含む合計。
会話中の「cleanの指標だけで選択」という説明は不正確だったため訂正する。旧LS結果記録の曖昧な記述も訂正した。

- DAMSEP訓練: clean負SNR＋0.1×残響RI+Mag＋0.5×CTF再構成RI+Mag。
- Rec-RIR訓練・validation: clean/reverb/CTF再構成のRI+Mag、重み各1。
- DAMSEPのPIT対応はclean損失で決め、補助損失へ共有。Rec-RIRは3損失の合計で対応を決める。
- DAMSEPはbatch1・best29、Rec-RIRはbatch4・best97。学習停止やLRスケジュールも異なる。epoch数だけでは更新回数を比較できず、DAMSEPの学習不足とは断定しない。
- 損失の単位と正規化が異なるため、係数0.5と1だけからCTFへの勾配の強さを比較しない。

## 復号の追加診断（既存出力のみ、再学習なし）

既存の全6,000波形について、各自のピークで正規化し、ピークから1秒後以降のRMSを算出した。

| 波形 | RMS中央値 | RMS平均 |
|---|---:|---:|
| LS参照 | 1.01563e-5 | 1.07789e-5 |
| Rec-RIR | 2.24018e-5 | 2.45536e-5 |
| DAMSEP | 1.83164e-5 | 1.96293e-5 |

DAMSEPの1秒後以降の残留量はRec-RIRより大きくない。したがって「1秒以降の長い尾が多い」だけではRT60の大差を説明できない。
RT60は減衰曲線の回帰から外挿した指標であり、今回の2秒窓から得た数秒の推定値をそのまま物理的な長い残響の存在と解釈しない。

復号窓について、参照index順の `[::100]` で選んだ30混合・60音源を診断した。
同じCTF・LS参照・RMSE-PIT・peak alignment/scaleを使い、DAMSEPのPIM復号側の窓のみ変更。

| FFT512 / hop128 の復号窓 | RMSE | 相関 | RT60 MAE [s] | DRR MAE [dB] |
|---|---:|---:|---:|---:|
| window256 / sqrt-Hann（既存主評価・再構成損失の基底） | 0.00285596 | 0.958892 | 1.935982 | 1.149770 |
| window512 / Hann（ネットワーク入出力の基底） | 0.00286426 | 0.958798 | 1.881957 | 0.735941 |
| window512 / sqrt-Hann | 0.00287336 | 0.958588 | 1.873716 | 0.721446 |

この少数例では窓変更だけで大きなRT60誤差は解消しない。一方DRRは動くため、復号の影響が全指標で無視できるとも結論づけない。
診断は限定集合であり、既存の全件主評価の数値・復号条件は変更していない。

## 現時点の解釈

分離段で音声復元に都合のよい表現ができても、物理的な残響特性を同定する情報が保持されるとは限らない。
今回のDAMSEPでは中間残響波形の精度が低いことから、この直列構造が有力な原因候補。
損失・PIT・後段容量・バッチ/LR/停止条件も同時に異なるため、構造だけが原因とは断定できない。
原因の確定には、これらを分けた対照実験や再構成誤差・減衰曲線の詳細分析が必要。

## 根拠

- 既存全件比較: `egs2/whamr/rir_2spk/exp/rec_rir_ls8k_20261015/comparison/`。
- 音声評価: `egs2/whamr/rir_2spk/exp/rec_rir_damsep_speech_20261016/comparison.json`。
- 訓練時DAMSEP: `git show 72526c93eac:espnet2/enh/damsep/espnet_model.py`。
- DAMSEPの直列構造: `espnet2/enh/damsep/vendor/models/SPMamba.py:606`。
- Rec-RIRの学習: `espnet2/rir/rec_rir/espnet_model_pit.py`。
- RT60計算: `egs2/whamr/rir_2spk/local/score_rir.py:219`。
