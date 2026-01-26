#!/usr/bin/env python3
"""
音声ファイルの類似度を計算するスクリプト
複数の指標（相関係数、MSE、スペクトログラム、MFCC）で比較します
"""

import numpy as np
import wave
import struct
import sys
from pathlib import Path


def load_audio_wave(filepath):
    """waveモジュールを使って音声ファイルを読み込む"""
    try:
        with wave.open(filepath, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            
            # バイトデータを読み込む
            frames = wf.readframes(n_frames)
            
            # サンプル幅に応じてデータを解釈
            if sample_width == 1:
                # 8-bit unsigned
                dtype = np.uint8
                data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                data = (data - 128) / 128.0  # -1.0 to 1.0に正規化
            elif sample_width == 2:
                # 16-bit signed
                dtype = np.int16
                data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                data = data / 32768.0  # -1.0 to 1.0に正規化
            elif sample_width == 4:
                # 32-bit signed
                dtype = np.int32
                data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                data = data / 2147483648.0  # -1.0 to 1.0に正規化
            else:
                raise ValueError(f"サポートされていないサンプル幅: {sample_width}")
            
            # ステレオの場合はモノラルに変換
            if n_channels == 2:
                data = data.reshape(-1, 2)
                data = np.mean(data, axis=1)
            elif n_channels > 2:
                data = data.reshape(-1, n_channels)
                data = np.mean(data, axis=1)
            
            return data, sr
    except Exception as e:
        print(f"エラー: {filepath} の読み込みに失敗しました (wave): {e}", file=sys.stderr)
        return None, None


def load_audio(filepath):
    """音声ファイルを読み込む（複数の方法を試す）"""
    # まずwaveモジュールで試す
    data, sr = load_audio_wave(filepath)
    if data is not None:
        return data, sr
    
    # waveモジュールで読み込めなかった場合はNoneを返す
    return None, None


def normalize_audio(data):
    """音声データを正規化"""
    if len(data) == 0:
        return data
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val
    return data


def align_length(data1, data2):
    """2つの音声データの長さを揃える（短い方に合わせる）"""
    min_len = min(len(data1), len(data2))
    return data1[:min_len], data2[:min_len]


def calculate_correlation(data1, data2):
    """相関係数を計算"""
    data1, data2 = align_length(data1, data2)
    if len(data1) == 0:
        return 0.0
    corr = np.corrcoef(data1, data2)[0, 1]
    return corr if not np.isnan(corr) else 0.0


def calculate_mse(data1, data2):
    """平均二乗誤差を計算"""
    data1, data2 = align_length(data1, data2)
    if len(data1) == 0:
        return float('inf')
    mse = np.mean((data1 - data2) ** 2)
    return mse


def calculate_spectral_similarity(data1, data2, sr):
    """スペクトログラムの類似度を計算（NumPyのみで実装）"""
    data1, data2 = align_length(data1, data2)
    if len(data1) == 0:
        return 0.0
    
    try:
        # 短時間フーリエ変換（STFT）を簡易実装
        n_fft = min(2048, len(data1))
        hop_length = n_fft // 4
        window = np.hanning(n_fft)
        
        # スペクトログラムを計算
        stft1 = []
        stft2 = []
        
        for i in range(0, len(data1) - n_fft + 1, hop_length):
            frame1 = data1[i:i+n_fft] * window
            frame2 = data2[i:i+n_fft] * window
            fft1 = np.fft.rfft(frame1, n=n_fft)
            fft2 = np.fft.rfft(frame2, n=n_fft)
            stft1.append(np.abs(fft1))
            stft2.append(np.abs(fft2))
        
        if len(stft1) == 0:
            return 0.0
        
        mag1 = np.array(stft1)
        mag2 = np.array(stft2)
        
        # コサイン類似度を計算
        mag1_flat = mag1.flatten()
        mag2_flat = mag2.flatten()
        
        # 正規化
        norm1 = np.linalg.norm(mag1_flat)
        norm2 = np.linalg.norm(mag2_flat)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(mag1_flat, mag2_flat) / (norm1 * norm2)
        return cosine_sim
    except Exception as e:
        print(f"  警告: スペクトログラム計算でエラー: {e}", file=sys.stderr)
        return 0.0


def calculate_mfcc_similarity(data1, data2, sr):
    """簡易的な周波数領域特徴量の類似度を計算（MFCCの代替）"""
    data1, data2 = align_length(data1, data2)
    if len(data1) == 0:
        return 0.0
    
    try:
        # FFTで周波数特性を取得
        n_fft = min(2048, len(data1))
        fft1 = np.abs(np.fft.rfft(data1, n=n_fft))
        fft2 = np.abs(np.fft.rfft(data2, n=n_fft))
        
        # メルスケールに近い対数変換
        fft1_log = np.log1p(fft1)
        fft2_log = np.log1p(fft2)
        
        # コサイン類似度を計算
        norm1 = np.linalg.norm(fft1_log)
        norm2 = np.linalg.norm(fft2_log)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(fft1_log, fft2_log) / (norm1 * norm2)
        return cosine_sim
    except Exception as e:
        print(f"  警告: 周波数特徴量計算でエラー: {e}", file=sys.stderr)
        return 0.0


def compare_audio_files(file1, file2):
    """2つの音声ファイルを比較"""
    print(f"\n{'='*80}")
    print(f"比較: {Path(file1).name} vs {Path(file2).name}")
    print(f"{'='*80}")
    
    # 音声ファイルを読み込む
    data1, sr1 = load_audio(file1)
    data2, sr2 = load_audio(file2)
    
    if data1 is None or data2 is None:
        print("エラー: ファイルの読み込みに失敗しました")
        return None
    
    # サンプリングレートを統一
    if sr1 != sr2:
        print(f"警告: サンプリングレートが異なります ({sr1} vs {sr2})")
        # NumPyの補間でリサンプリング
        if sr1 > sr2:
            num_samples = int(len(data1) * sr2 / sr1)
            indices = np.linspace(0, len(data1) - 1, num_samples)
            data1 = np.interp(indices, np.arange(len(data1)), data1)
            sr = sr2
        else:
            num_samples = int(len(data2) * sr1 / sr2)
            indices = np.linspace(0, len(data2) - 1, num_samples)
            data2 = np.interp(indices, np.arange(len(data2)), data2)
            sr = sr1
    else:
        sr = sr1
    
    # 正規化
    data1 = normalize_audio(data1)
    data2 = normalize_audio(data2)
    
    # 基本情報
    print(f"\n基本情報:")
    print(f"  ファイル1: {len(data1)} サンプル, {len(data1)/sr:.2f} 秒, SR={sr}")
    print(f"  ファイル2: {len(data2)} サンプル, {len(data2)/sr:.2f} 秒, SR={sr}")
    
    # 各種類似度を計算
    print(f"\n類似度指標:")
    
    # 1. 相関係数
    corr = calculate_correlation(data1, data2)
    print(f"  相関係数 (Correlation): {corr:.6f} (1.0に近いほど類似)")
    
    # 2. MSE
    mse = calculate_mse(data1, data2)
    print(f"  平均二乗誤差 (MSE): {mse:.6f} (0に近いほど類似)")
    
    # 3. スペクトログラム類似度
    spec_sim = calculate_spectral_similarity(data1, data2, sr)
    print(f"  スペクトログラム類似度 (Cosine): {spec_sim:.6f} (1.0に近いほど類似)")
    
    # 4. MFCC類似度
    mfcc_sim = calculate_mfcc_similarity(data1, data2, sr)
    print(f"  MFCC類似度 (Cosine): {mfcc_sim:.6f} (1.0に近いほど類似)")
    
    # 判定
    print(f"\n判定:")
    if corr > 0.99 and mse < 0.001 and spec_sim > 0.99:
        print(f"  → ほぼ同一の音声ファイルです")
    elif corr > 0.95 and mse < 0.01 and spec_sim > 0.95:
        print(f"  → 非常に類似しています")
    elif corr > 0.8 and spec_sim > 0.8:
        print(f"  → 類似しています")
    elif corr > 0.5:
        print(f"  → ある程度類似しています")
    else:
        print(f"  → 異なる音声ファイルです")
    
    return {
        'correlation': corr,
        'mse': mse,
        'spectral_similarity': spec_sim,
        'mfcc_similarity': mfcc_sim
    }


def main():
    if len(sys.argv) < 3:
        print("使用方法: python compare_audio_similarity.py <file1> <file2> [file3 ...]")
        sys.exit(1)
    
    files = sys.argv[1:]
    
    print(f"\n音声ファイル類似度比較ツール")
    print(f"比較対象: {len(files)} ファイル")
    
    # 全ファイルのペアを比較
    results = {}
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            key = (files[i], files[j])
            results[key] = compare_audio_files(files[i], files[j])
    
    # サマリー
    print(f"\n{'='*80}")
    print(f"サマリー")
    print(f"{'='*80}")
    for (file1, file2), result in results.items():
        if result:
            print(f"\n{Path(file1).name} vs {Path(file2).name}:")
            print(f"  相関係数: {result['correlation']:.6f}")
            print(f"  スペクトログラム類似度: {result['spectral_similarity']:.6f}")


if __name__ == "__main__":
    main()
