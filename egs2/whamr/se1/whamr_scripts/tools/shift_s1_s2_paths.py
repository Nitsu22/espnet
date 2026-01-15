#!/usr/bin/env python3
"""
CSVファイルのs1_pathとs2_pathをずらすスクリプト

各行のs1_pathとs2_pathを、別の行から選んだペアに置き換えます。
元の組み合わせと同じにはなりません。
"""

import csv
import random
import sys
from pathlib import Path


def shift_s1_s2_paths(input_csv_path, output_csv_path, max_attempts=1000):
    """
    CSVファイルのs1_pathとs2_pathをずらす
    
    Args:
        input_csv_path: 入力CSVファイルのパス
        output_csv_path: 出力CSVファイルのパス
        max_attempts: 同じ組み合わせを避けるための最大試行回数
    """
    # CSVを読み込む
    rows = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    if len(rows) < 2:
        print("エラー: 行数が少なすぎます（最低2行必要）", file=sys.stderr)
        return
    
    # 各行について、別の行からs1_pathとs2_pathのペアを選ぶ
    random.seed()  # ランダムシードを初期化
    shifted_rows = []
    
    for i, row in enumerate(rows):
        original_s1 = row['s1_path']
        original_s2 = row['s2_path']
        
        # 別の行をランダムに選ぶ（元の組み合わせと同じでないことを確認）
        attempts = 0
        while attempts < max_attempts:
            # 自分自身以外の行をランダムに選ぶ
            other_idx = random.randint(0, len(rows) - 1)
            while other_idx == i:
                other_idx = random.randint(0, len(rows) - 1)
            
            new_s1 = rows[other_idx]['s1_path']
            new_s2 = rows[other_idx]['s2_path']
            
            # 元の組み合わせと同じでないことを確認
            if not (new_s1 == original_s1 and new_s2 == original_s2):
                break
            
            attempts += 1
        
        if attempts >= max_attempts:
            print(f"警告: 行 {i+1} で適切な組み合わせが見つかりませんでした。元の値を保持します。", file=sys.stderr)
            new_s1 = original_s1
            new_s2 = original_s2
        
        # 新しい行を作成（output_filenameは固定）
        shifted_row = {
            'output_filename': row['output_filename'],
            's1_path': new_s1,
            's2_path': new_s2
        }
        shifted_rows.append(shifted_row)
    
    # 新しいCSVファイルに書き込む
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['output_filename', 's1_path', 's2_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(shifted_rows)
    
    print(f"処理完了: {len(shifted_rows)} 行を処理しました")
    print(f"出力ファイル: {output_csv_path}")


def main():
    """メイン関数"""
    # デフォルトのパス
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    input_csv = data_dir / 'mix_2_spk_filenames_cv.csv'
    output_csv = data_dir / 'mix_2_spk_filenames_cv_shifted.csv'
    
    # コマンドライン引数で上書き可能
    if len(sys.argv) > 1:
        input_csv = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_csv = Path(sys.argv[2])
    
    if not input_csv.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_csv}", file=sys.stderr)
        sys.exit(1)
    
    shift_s1_s2_paths(input_csv, output_csv)


if __name__ == '__main__':
    main()
