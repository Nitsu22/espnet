#!/usr/bin/env python3
"""
scpファイルを先頭から5000エントリに削減するスクリプト

元のディレクトリ: ./dump/raw/tr_mix_both_reverb_min_8k
出力ディレクトリ: ./dump/raw/tr_mix_both_reverb_min_8k_5000
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


def read_scp_file(filepath, max_lines=None):
    """
    scpファイルを読み込む
    
    Args:
        filepath: scpファイルのパス
        max_lines: 読み込む最大行数（Noneの場合は全て）
    
    Returns:
        list: 各行の内容（文字列のリスト）
    """
    lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            lines.append(line.rstrip('\n'))
    return lines


def write_scp_file(filepath, lines):
    """
    scpファイルに書き込む
    
    Args:
        filepath: 出力ファイルのパス（Pathオブジェクトまたは文字列）
        lines: 書き込む行のリスト
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def extract_utterance_id(line):
    """
    scpファイルの行からutterance_idを抽出
    
    Args:
        line: scpファイルの1行
    
    Returns:
        str: utterance_id（最初の空白またはタブまでの部分）
    """
    return line.split()[0] if line.strip() else ""


def generate_spk2utt(utt2spk_lines):
    """
    utt2spkからspk2uttを生成
    
    Args:
        utt2spk_lines: utt2spkファイルの行のリスト
    
    Returns:
        list: spk2uttファイルの行のリスト
    """
    spk2utt_dict = defaultdict(list)
    
    for line in utt2spk_lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            utt_id = parts[0]
            spk_id = parts[1]
            spk2utt_dict[spk_id].append(utt_id)
    
    # 話者IDでソートして、各話者のutterance_idもソート
    spk2utt_lines = []
    for spk_id in sorted(spk2utt_dict.keys()):
        utt_ids = sorted(spk2utt_dict[spk_id])
        line = f"{spk_id} {' '.join(utt_ids)}"
        spk2utt_lines.append(line)
    
    return spk2utt_lines


def main():
    # パス設定（相対パス）
    base_dir = Path("./dump/raw")
    source_dir = base_dir / "tr_mix_both_reverb_min_8k"
    target_dir = base_dir / "tr_mix_both_reverb_min_8k_5000"
    
    # エントリ数
    num_entries = 5000
    
    print(f"元のディレクトリ: {source_dir}")
    print(f"出力ディレクトリ: {target_dir}")
    print(f"エントリ数: {num_entries}")
    print()
    
    # ソースディレクトリの存在確認
    if not source_dir.exists():
        raise FileNotFoundError(f"ソースディレクトリが見つかりません: {source_dir}")
    
    # ターゲットディレクトリが既に存在する場合は警告
    if target_dir.exists():
        print(f"警告: ターゲットディレクトリは既に存在します: {target_dir}")
        response = input("続行しますか？ (y/n): ")
        if response.lower() != 'y':
            print("処理を中止しました。")
            return
    
    # ターゲットディレクトリを作成
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 処理するファイルのリスト（ルート直下のみ）
    scp_files = [
        'wav.scp',
        'spk1.scp',
        'spk2.scp',
        'noise1.scp',
        'utt2spk',
        'utt2num_samples',
    ]
    
    # 各scpファイルを処理
    utt2spk_lines = None
    for filename in scp_files:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        if not source_file.exists():
            print(f"警告: ファイルが見つかりません: {source_file}")
            continue
        
        print(f"処理中: {filename}")
        
        # 先頭からnum_entries行を読み込む
        lines = read_scp_file(source_file, max_lines=num_entries)
        
        # utt2spkは後でspk2utt生成に使用するため保存
        if filename == 'utt2spk':
            utt2spk_lines = lines
        
        # ファイルに書き込む
        write_scp_file(target_file, lines)
        print(f"  -> {len(lines)}行を書き込みました")
    
    # spk2uttを生成
    if utt2spk_lines is not None:
        print("処理中: spk2utt (生成)")
        spk2utt_lines = generate_spk2utt(utt2spk_lines)
        target_spk2utt = target_dir / 'spk2utt'
        write_scp_file(target_spk2utt, spk2utt_lines)
        print(f"  -> {len(spk2utt_lines)}行を書き込みました")
    
    # feats_typeファイルがあればコピー
    feats_type_source = source_dir / 'feats_type'
    if feats_type_source.exists():
        feats_type_target = target_dir / 'feats_type'
        shutil.copy2(feats_type_source, feats_type_target)
        print(f"コピー: feats_type")
    
    print()
    print("処理が完了しました。")
    print(f"出力ディレクトリ: {target_dir}")


if __name__ == "__main__":
    main()
