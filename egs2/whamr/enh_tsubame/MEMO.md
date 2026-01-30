conda create -n tf-locoformer python=3.10.8 -y
./setup_python.sh "$(which python)"
make TH_VERSION=2.1.0 CUDA_VERSION=11.8
pip install rotary-embedding-torch==0.6.1
cd ../../  
git clone https://github.com/merlresearch/tf-locoformer.git
cd tf-locoformer
./copy_files_to_espnet.sh /net/midgar/work/nitsu/learning/tf-locoformer/espnet
pip install pyroomacoustics==0.2.0 --no-build-isolation