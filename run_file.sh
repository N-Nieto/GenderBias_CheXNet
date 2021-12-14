#!/bin/bash
echo What python file plus args would you like to run ex: training.py 1?
read filename

MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
MINICONDA_PREFIX=/usr/local
wget https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
chmod +x $MINICONDA_INSTALLER_SCRIPT
./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX
conda install --channel defaults conda python=3.7 --yes
conda update --channel defaults --all --yes

yes | conda create --name gender_bias --file requirements.txt

source activate gender_bias
pip install pillow==4.2.0
pip install opencv-python==4.1.0.25
pip install imgaug==0.2.9
pip install numpy==1.13.3
yes | conda install -c anaconda cudatoolkit

pip list

python $filename

mv -f ./Cross_validation_splits/0%_female_images/Fold_0/output_female_finetune_100 /content/drive/MyDrive/output_finetune_100
mv -f ./Cross_validation_splits/0%_female_images/Fold_0/output_female_finetune_500 /content/drive/MyDrive/output_finetune_500
mv -f ./Cross_validation_splits/0%_female_images/Fold_0/output_female_finetune_1000 /content/drive/MyDrive/output_finetune_1000
mv -f ./Cross_validation_splits/0%_female_images/Fold_0/output_female_finetune_2500 /content/drive/MyDrive/output_finetune_2500
mv -f ./Cross_validation_splits/0%_female_images/Fold_0/output_female_finetune_5000 /content/drive/MyDrive/output_finetune_5000
mv -f ./Cross_validation_splits/0%_female_images/Fold_0/output_female_finetune_10000 /content/drive/MyDrive/output_finetune_10000
mv -f ./Cross_validation_splits/0%_female_images/Fold_0/output_female_finetune_20000 /content/drive/MyDrive/output_finetune_20000