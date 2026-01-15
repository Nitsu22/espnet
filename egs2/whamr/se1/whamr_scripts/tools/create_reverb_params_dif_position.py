#!/usr/bin/env python3
"""
話者位置を少しだけ変えたreverb_params_*_dif_position.csvファイルを作成するスクリプト

元のreverb_params_*.csvファイルを読み込み、話者位置（s1_x, s1_y, s2_x, s2_y）を
z座標を変えずに0.1-0.5m程度ランダムに移動させ、壁を越えないように制約を適用する。
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def move_speaker_position(
    current_x, current_y, room_x, room_y, min_distance_from_wall=0.1, 
    min_move_distance=0.1, max_move_distance=0.5
):
    """
    話者位置をランダムに移動させる
    
    Args:
        current_x: 現在のx座標
        current_y: 現在のy座標
        room_x: 部屋のx方向のサイズ
        room_y: 部屋のy方向のサイズ
        min_distance_from_wall: 壁からの最小距離（デフォルト: 0.1m）
        min_move_distance: 最小移動距離（デフォルト: 0.1m）
        max_move_distance: 最大移動距離（デフォルト: 0.5m）
    
    Returns:
        new_x, new_y: 新しいx, y座標
    """
    # 壁からの最小距離を考慮した有効範囲
    min_x = min_distance_from_wall
    max_x = room_x - min_distance_from_wall
    min_y = min_distance_from_wall
    max_y = room_y - min_distance_from_wall
    
    # 最大試行回数（無限ループを防ぐ）
    max_attempts = 100
    
    for attempt in range(max_attempts):
        # ランダムな移動方向（0から2πの範囲）
        angle = np.random.uniform(0, 2 * np.pi)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # その方向に移動できる最大距離を計算
        # x方向の制約
        if abs(cos_angle) < 1e-10:
            # x方向に移動しない（垂直方向）
            max_dist_x = float('inf')
        elif cos_angle > 0:
            # 正のx方向に移動
            max_dist_x = (max_x - current_x) / cos_angle
        else:
            # 負のx方向に移動
            max_dist_x = (min_x - current_x) / cos_angle
        
        # y方向の制約
        if abs(sin_angle) < 1e-10:
            # y方向に移動しない（水平方向）
            max_dist_y = float('inf')
        elif sin_angle > 0:
            # 正のy方向に移動
            max_dist_y = (max_y - current_y) / sin_angle
        else:
            # 負のy方向に移動
            max_dist_y = (min_y - current_y) / sin_angle
        
        # 両方向の制約を満たす最大距離（負の値は無視）
        max_dist_x = max(0, max_dist_x) if max_dist_x != float('inf') else float('inf')
        max_dist_y = max(0, max_dist_y) if max_dist_y != float('inf') else float('inf')
        max_possible_distance = min(max_dist_x, max_dist_y)
        
        # 最大距離が最小移動距離より小さい場合は、別の方向を試す
        if max_possible_distance < min_move_distance:
            continue
        
        # 移動可能な距離の範囲でランダムに選ぶ
        # ただし、要求される移動距離の範囲と、実際に移動できる距離の範囲の交差を取る
        actual_max_distance = min(max_move_distance, max_possible_distance)
        actual_min_distance = min_move_distance
        
        if actual_max_distance < actual_min_distance:
            # 移動できない場合は、最大距離で移動
            move_distance = max_possible_distance
        else:
            # 移動可能な範囲でランダムに選ぶ
            move_distance = np.random.uniform(actual_min_distance, actual_max_distance)
        
        # 移動ベクトル
        dx = move_distance * np.cos(angle)
        dy = move_distance * np.sin(angle)
        
        # 新しい位置
        new_x = current_x + dx
        new_y = current_y + dy
        
        # 念のため範囲チェック（浮動小数点誤差を考慮）
        new_x = np.clip(new_x, min_x, max_x)
        new_y = np.clip(new_y, min_y, max_y)
        
        return new_x, new_y
    
    # 最大試行回数に達した場合は、現在の位置から可能な限り移動
    # これは非常に稀なケース（部屋が非常に小さい場合など）
    print(f"Warning: Could not find valid move direction after {max_attempts} attempts. "
          f"Using current position with slight adjustment.")
    # 現在の位置を少し調整（可能な範囲内で）
    new_x = np.clip(current_x, min_x, max_x)
    new_y = np.clip(current_y, min_y, max_y)
    return new_x, new_y


def process_reverb_params_file(input_path, output_path, seed=None, 
                                min_distance_from_wall=0.1,
                                min_move_distance=0.1, max_move_distance=0.5):
    """
    reverb_params CSVファイルを処理して、話者位置を変更した新しいファイルを作成
    
    Args:
        input_path: 入力CSVファイルのパス
        output_path: 出力CSVファイルのパス
        seed: 乱数のシード（再現性のため）
        min_distance_from_wall: 壁からの最小距離
        min_move_distance: 最小移動距離
        max_move_distance: 最大移動距離
    """
    if seed is not None:
        np.random.seed(seed)
    
    # CSVファイルを読み込む
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Processing {len(df)} rows...")
    
    # 各行について話者位置を変更
    for idx, row in df.iterrows():
        # 話者1の位置を変更
        new_s1_x, new_s1_y = move_speaker_position(
            row['s1_x'], row['s1_y'],
            row['room_x'], row['room_y'],
            min_distance_from_wall=min_distance_from_wall,
            min_move_distance=min_move_distance,
            max_move_distance=max_move_distance
        )
        
        # 話者2の位置を変更
        new_s2_x, new_s2_y = move_speaker_position(
            row['s2_x'], row['s2_y'],
            row['room_x'], row['room_y'],
            min_distance_from_wall=min_distance_from_wall,
            min_move_distance=min_move_distance,
            max_move_distance=max_move_distance
        )
        
        # データフレームを更新
        df.at[idx, 's1_x'] = new_s1_x
        df.at[idx, 's1_y'] = new_s1_y
        df.at[idx, 's2_x'] = new_s2_x
        df.at[idx, 's2_y'] = new_s2_y
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} rows...")
    
    # 新しいCSVファイルに書き出す
    print(f"Writing to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Done! Created {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='話者位置を少しだけ変えたreverb_params_*_dif_position.csvファイルを作成'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data',
        help='入力CSVファイルがあるディレクトリ（デフォルト: data）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='出力CSVファイルを保存するディレクトリ（デフォルト: 入力ディレクトリと同じ）'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['tr', 'cv', 'tt'],
        help='処理するスプリット（デフォルト: tr cv tt）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='乱数のシード（再現性のため）'
    )
    parser.add_argument(
        '--min-distance-from-wall',
        type=float,
        default=0.1,
        help='壁からの最小距離（メートル、デフォルト: 0.1）'
    )
    parser.add_argument(
        '--min-move-distance',
        type=float,
        default=0.1,
        help='最小移動距離（メートル、デフォルト: 0.1）'
    )
    parser.add_argument(
        '--max-move-distance',
        type=float,
        default=0.5,
        help='最大移動距離（メートル、デフォルト: 0.5）'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各スプリットについて処理
    for split in args.splits:
        input_file = input_dir / f'reverb_params_{split}.csv'
        output_file = output_dir / f'reverb_params_{split}_dif_position.csv'
        
        if not input_file.exists():
            print(f"Warning: {input_file} does not exist. Skipping...")
            continue
        
        print(f"\nProcessing {split} split...")
        process_reverb_params_file(
            input_file,
            output_file,
            seed=args.seed,
            min_distance_from_wall=args.min_distance_from_wall,
            min_move_distance=args.min_move_distance,
            max_move_distance=args.max_move_distance
        )
    
    print("\nAll done!")


if __name__ == '__main__':
    main()
