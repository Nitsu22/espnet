#!/usr/bin/env python3
"""
生成されたreverb_params_*_dif_position.csvファイルが要件を満たしているか検証するスクリプト

検証項目:
1. 部屋の中に収まっているか（壁からの最小距離0.1mを確保）
2. 移動距離が0.1-0.5mの範囲になっているか
"""

import pandas as pd
import numpy as np
from pathlib import Path


def verify_file(original_path, new_path, min_distance_from_wall=0.1, 
                min_move_distance=0.1, max_move_distance=0.5):
    """
    ファイルを検証する
    
    Returns:
        (is_valid, issues): 検証結果と問題点のリスト
    """
    issues = []
    
    # CSVファイルを読み込む
    df_original = pd.read_csv(original_path)
    df_new = pd.read_csv(new_path)
    
    if len(df_original) != len(df_new):
        issues.append(f"行数が一致しません: 元={len(df_original)}, 新={len(df_new)}")
        return False, issues
    
    # 各項目を検証
    wall_violations = 0
    move_distance_too_small = 0
    move_distance_too_large = 0
    z_coordinate_changed = 0
    
    for idx in range(len(df_original)):
        row_orig = df_original.iloc[idx]
        row_new = df_new.iloc[idx]
        
        room_x = row_orig['room_x']
        room_y = row_orig['room_y']
        min_x = min_distance_from_wall
        max_x = room_x - min_distance_from_wall
        min_y = min_distance_from_wall
        max_y = room_y - min_distance_from_wall
        
        # 1. 部屋の中に収まっているかチェック（話者1）
        s1_x_new = row_new['s1_x']
        s1_y_new = row_new['s1_y']
        if s1_x_new < min_x or s1_x_new > max_x or s1_y_new < min_y or s1_y_new > max_y:
            wall_violations += 1
            if wall_violations <= 5:  # 最初の5件のみ記録
                issues.append(
                    f"行{idx+2}: 話者1が部屋の範囲外 "
                    f"(s1_x={s1_x_new:.6f}, s1_y={s1_y_new:.6f}, "
                    f"範囲: x=[{min_x:.6f}, {max_x:.6f}], y=[{min_y:.6f}, {max_y:.6f}])"
                )
        
        # 1. 部屋の中に収まっているかチェック（話者2）
        s2_x_new = row_new['s2_x']
        s2_y_new = row_new['s2_y']
        if s2_x_new < min_x or s2_x_new > max_x or s2_y_new < min_y or s2_y_new > max_y:
            wall_violations += 1
            if wall_violations <= 5:  # 最初の5件のみ記録
                issues.append(
                    f"行{idx+2}: 話者2が部屋の範囲外 "
                    f"(s2_x={s2_x_new:.6f}, s2_y={s2_y_new:.6f}, "
                    f"範囲: x=[{min_x:.6f}, {max_x:.6f}], y=[{min_y:.6f}, {max_y:.6f}])"
                )
        
        # 2. 移動距離をチェック（話者1）
        s1_x_orig = row_orig['s1_x']
        s1_y_orig = row_orig['s1_y']
        s1_move_distance = np.sqrt((s1_x_new - s1_x_orig)**2 + (s1_y_new - s1_y_orig)**2)
        
        if s1_move_distance < min_move_distance:
            move_distance_too_small += 1
            if move_distance_too_small <= 5:
                issues.append(
                    f"行{idx+2}: 話者1の移動距離が小さすぎます "
                    f"({s1_move_distance:.6f}m < {min_move_distance}m)"
                )
        elif s1_move_distance > max_move_distance:
            move_distance_too_large += 1
            if move_distance_too_large <= 5:
                issues.append(
                    f"行{idx+2}: 話者1の移動距離が大きすぎます "
                    f"({s1_move_distance:.6f}m > {max_move_distance}m)"
                )
        
        # 2. 移動距離をチェック（話者2）
        s2_x_orig = row_orig['s2_x']
        s2_y_orig = row_orig['s2_y']
        s2_move_distance = np.sqrt((s2_x_new - s2_x_orig)**2 + (s2_y_new - s2_y_orig)**2)
        
        if s2_move_distance < min_move_distance:
            move_distance_too_small += 1
            if move_distance_too_small <= 5:
                issues.append(
                    f"行{idx+2}: 話者2の移動距離が小さすぎます "
                    f"({s2_move_distance:.6f}m < {min_move_distance}m)"
                )
        elif s2_move_distance > max_move_distance:
            move_distance_too_large += 1
            if move_distance_too_large <= 5:
                issues.append(
                    f"行{idx+2}: 話者2の移動距離が大きすぎます "
                    f"({s2_move_distance:.6f}m > {max_move_distance}m)"
                )
        
        # z座標が変更されていないかチェック
        if abs(row_new['s1_z'] - row_orig['s1_z']) > 1e-10:
            z_coordinate_changed += 1
            if z_coordinate_changed <= 5:
                issues.append(
                    f"行{idx+2}: 話者1のz座標が変更されています "
                    f"(元={row_orig['s1_z']:.6f}, 新={row_new['s1_z']:.6f})"
                )
        
        if abs(row_new['s2_z'] - row_orig['s2_z']) > 1e-10:
            z_coordinate_changed += 1
            if z_coordinate_changed <= 5:
                issues.append(
                    f"行{idx+2}: 話者2のz座標が変更されています "
                    f"(元={row_orig['s2_z']:.6f}, 新={row_new['s2_z']:.6f})"
                )
    
    # 統計情報を追加
    if wall_violations > 0:
        issues.append(f"\n【統計】部屋の範囲外: {wall_violations}件")
    if move_distance_too_small > 0:
        issues.append(f"【統計】移動距離が小さすぎる: {move_distance_too_small}件")
    if move_distance_too_large > 0:
        issues.append(f"【統計】移動距離が大きすぎる: {move_distance_too_large}件")
    if z_coordinate_changed > 0:
        issues.append(f"【統計】z座標が変更された: {z_coordinate_changed}件")
    
    is_valid = (wall_violations == 0 and move_distance_too_small == 0 and 
                move_distance_too_large == 0 and z_coordinate_changed == 0)
    
    return is_valid, issues


def main():
    data_dir = Path('data')
    splits = ['tr', 'cv', 'tt']
    min_distance_from_wall = 0.1
    min_move_distance = 0.1
    max_move_distance = 0.5
    
    print("=" * 80)
    print("reverb_params_*_dif_position.csvファイルの検証")
    print("=" * 80)
    print(f"検証項目:")
    print(f"  1. 部屋の中に収まっているか（壁からの最小距離: {min_distance_from_wall}m）")
    print(f"  2. 移動距離が{min_move_distance}-{max_move_distance}mの範囲内か")
    print(f"  3. z座標が変更されていないか")
    print("=" * 80)
    print()
    
    all_valid = True
    
    for split in splits:
        original_file = data_dir / f'reverb_params_{split}.csv'
        new_file = data_dir / f'reverb_params_{split}_dif_position.csv'
        
        if not original_file.exists():
            print(f"⚠️  {split}: 元ファイルが見つかりません: {original_file}")
            continue
        
        if not new_file.exists():
            print(f"⚠️  {split}: 新ファイルが見つかりません: {new_file}")
            continue
        
        print(f"検証中: {split} split...")
        is_valid, issues = verify_file(
            original_file, new_file,
            min_distance_from_wall=min_distance_from_wall,
            min_move_distance=min_move_distance,
            max_move_distance=max_move_distance
        )
        
        if is_valid:
            print(f"✅ {split}: すべての要件を満たしています")
        else:
            print(f"❌ {split}: 問題が見つかりました")
            for issue in issues:
                print(f"  - {issue}")
            all_valid = False
        
        print()
    
    print("=" * 80)
    if all_valid:
        print("✅ すべてのファイルが要件を満たしています！")
    else:
        print("❌ 一部のファイルに問題があります。上記を確認してください。")
    print("=" * 80)


if __name__ == '__main__':
    main()
