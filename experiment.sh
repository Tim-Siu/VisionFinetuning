nvidia-smi

cd /temp

mkdir condap

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /temp/condap/miniconda.sh

bash /temp/condap/miniconda.sh -b -u -p /temp/condap

/temp/condap/bin/conda init

source ~/.bashrc

rm -rf /temp/condap/miniconda.sh

conda create -n finetune python=3.11 -y

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install matplotlib -y

wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -P /temp/data

cd temp/data && unzip PennFudanPed.zip
