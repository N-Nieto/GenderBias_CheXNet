#!/bin/bash
echo What python file plus args would you like to run ex: training.py 0 -g female?
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

gsutil -m cp -r /content/colab_directory/Cross_validation_splits/100%_female_images gs://gender-bias-data/Cross_validation_splits/100%_female_images
gsutil -m cp -r /content/colab_directory/Cross_validation_splits/0%_female_images gs://gender-bias-data/Cross_validation_splits/0%_female_images