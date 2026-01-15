#!/usr/bin/env python3
"""
移動距離の統計情報を表示するスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_move_distances(original_path, new_path):
    """移動距離を計算"""
    df_original = pd.read_csv(original_path)
    df_new = pd.read_csv(new_path)
    
    s1_distances = []
    s2_distances = []
    
    for idx in range(len(df_original)):
        row_orig = df_original.iloc[idx]
        row_new = df_new.iloc[idx]
        
        # 話者1の移動距離
        s1_dist = np.sqrt(
            (row_new['s1_x'] - row_orig['s1_x'])**2 + 
            (row_new['s1_y'] - row_orig['s1_y'])**2
        )
        s1_distances.append(s1_dist)
        
        # 話者2の移動距離
        s2_dist = np.sqrt(
            (row_new['s2_x'] - row_orig['s2_x'])**2 + 
            (row_new['s2_y'] - row_orig['s2_y'])**2
        )
        s2_distances.append(s2_dist)
    
    return s1_distances, s2_distances


def main():
    data_dir = Path('data')
    splits = ['tr', 'cv', 'tt']
    
    print("=" * 80)
    print("移動距離の統計情報")
    print("=" * 80)
    print()
    
    for split in splits:
        original_file = data_dir / f'reverb_params_{split}.csv'
        new_file = data_dir / f'reverb_params_{split}_dif_position.csv'
        
        if not original_file.exists() or not new_file.exists():
            continue
        
        s1_distances, s2_distances = calculate_move_distances(original_file, new_file)
        all_distances = s1_distances + s2_distances
        
        print(f"{split} split:")
        print(f"  話者1: 最小={np.min(s1_distances):.6f}m, "
              f"最大={np.max(s1_distances):.6f}m, "
              f"平均={np.mean(s1_distances):.6f}m, "
              f"中央値={np.median(s1_distances):.6f}m")
        print(f"  話者2: 最小={np.min(s2_distances):.6f}m, "
              f"最大={np.max(s2_distances):.6f}m, "
              f"平均={np.mean(s2_distances):.6f}m, "
              f"中央値={np.median(s2_distances):.6f}m")
        print(f"  全体: 最小={np.min(all_distances):.6f}m, "
              f"最大={np.max(all_distances):.6f}m, "
              f"平均={np.mean(all_distances):.6f}m, "
              f"中央値={np.median(all_distances):.6f}m")
        
        # 範囲内の割合
        in_range = sum(1 for d in all_distances if 0.1 <= d <= 0.5)
        print(f"  0.1-0.5mの範囲内: {in_range}/{len(all_distances)} ({100*in_range/len(all_distances):.2f}%)")
        print()


if __name__ == '__main__':
    main()
