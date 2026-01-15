#!/usr/bin/env bash

# Copyright 2024
# Apache 2.0

# 話者位置を少しだけ変えたreverb_params_*_dif_position.csvファイルを作成するスクリプト

set -e
set -u
set -o pipefail

# デフォルト値
input_dir=data
output_dir=
splits="tr cv tt"
seed=
min_distance_from_wall=0.1
min_move_distance=0.1
max_move_distance=0.5

# スクリプトのディレクトリを取得
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python_script="${script_dir}/tools/create_reverb_params_dif_position.py"

# 引数の解析
usage() {
    cat << EOF
Usage: $0 [options]

Options:
    --input-dir <dir>             入力CSVファイルがあるディレクトリ（デフォルト: data）
    --output-dir <dir>             出力CSVファイルを保存するディレクトリ（デフォルト: 入力ディレクトリと同じ）
    --splits <split1> [split2] ... 処理するスプリット（デフォルト: tr cv tt）
    --seed <seed>                  乱数のシード（再現性のため、デフォルト: なし）
    --min-distance-from-wall <m>  壁からの最小距離（メートル、デフォルト: 0.1）
    --min-move-distance <m>       最小移動距離（メートル、デフォルト: 0.1）
    --max-move-distance <m>       最大移動距離（メートル、デフォルト: 0.5）
    -h, --help                     このヘルプメッセージを表示

Example:
    $0 --input-dir data --seed 42
    $0 --input-dir data --output-dir data --splits tr cv --seed 123
EOF
    exit 1
}

# 引数の解析
while [ $# -gt 0 ]; do
    case $1 in
        --input-dir)
            input_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --splits)
            splits="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --min-distance-from-wall)
            min_distance_from_wall="$2"
            shift 2
            ;;
        --min-move-distance)
            min_move_distance="$2"
            shift 2
            ;;
        --max-move-distance)
            max_move_distance="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Pythonスクリプトの存在確認
if [ ! -f "${python_script}" ]; then
    echo "Error: Python script not found: ${python_script}"
    exit 1
fi

# 入力ディレクトリの存在確認
if [ ! -d "${input_dir}" ]; then
    echo "Error: Input directory not found: ${input_dir}"
    exit 1
fi

# 引数の構築
args=(
    --input-dir "${input_dir}"
    --min-distance-from-wall "${min_distance_from_wall}"
    --min-move-distance "${min_move_distance}"
    --max-move-distance "${max_move_distance}"
)

if [ -n "${output_dir}" ]; then
    args+=(--output-dir "${output_dir}")
fi

if [ -n "${splits}" ]; then
    args+=(--splits ${splits})
fi

if [ -n "${seed}" ]; then
    args+=(--seed "${seed}")
fi

# Pythonスクリプトの実行
echo "Running: python ${python_script} ${args[*]}"
cd "${script_dir}"
python "${python_script}" "${args[@]}"

echo "Done!"
