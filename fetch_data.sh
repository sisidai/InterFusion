#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

mkdir -p data
mkdir -p PoseGen/ICON/data/smpl_related/models

# username and password input
echo -e "\nYou need to register at https://icon.is.tue.mpg.de/, according to Installation Instruction."
read -p "Username (ICON):" username
read -p "Password (ICON):" password
username=$(urle $username)
password=$(urle $password)

# SMPL (Male, Female)
echo -e "\nDownloading SMPL..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip&resume=1' -O './PoseGen/ICON/data/smpl_related/models/SMPL_python_v.1.0.0.zip' --no-check-certificate --continue
unzip PoseGen/ICON/data/smpl_related/models/SMPL_python_v.1.0.0.zip -d PoseGen/ICON/data/smpl_related/models
mv PoseGen/ICON/data/smpl_related/models/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl PoseGen/ICON/data/smpl_related/models/smpl/SMPL_FEMALE.pkl
mv PoseGen/ICON/data/smpl_related/models/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl PoseGen/ICON/data/smpl_related/models/smpl/SMPL_MALE.pkl
cd PoseGen/ICON/data/smpl_related/models
rm -rf *.zip __MACOSX smpl/models smpl/smpl_webuser
cd ../../../../..

# SMPL (Neutral, from SMPLIFY)
echo -e "\nDownloading SMPLify..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' -O './PoseGen/ICON/data/smpl_related/models/mpips_smplify_public_v2.zip' --no-check-certificate --continue
unzip PoseGen/ICON/data/smpl_related/models/mpips_smplify_public_v2.zip -d PoseGen/ICON/data/smpl_related/models
mv PoseGen/ICON/data/smpl_related/models/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl PoseGen/ICON/data/smpl_related/models/smpl/SMPL_NEUTRAL.pkl
cd PoseGen/ICON/data/smpl_related/models
rm -rf *.zip smplify_public 
cd ../../../../..

# SMPL-X 
echo -e "\nDownloading SMPL-X..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1' -O './PoseGen/ICON/data/smpl_related/models/models_smplx_v1_1.zip' --no-check-certificate --continue
unzip PoseGen/ICON/data/smpl_related/models/models_smplx_v1_1.zip -d PoseGen/ICON/data/smpl_related/models
unzip PoseGen/ICON/data/smpl_related/models/models_smplx_v1_1.zip -d data/smplx_model
rm -rf PoseGen/ICON/data/smpl_related/models/models_smplx_v1_1.zip

# ICON
echo -e "\nDownloading ICON..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=icon&sfile=icon_data.zip&resume=1' -O './PoseGen/ICON/data/icon_data.zip' --no-check-certificate --continue
cd PoseGen/ICON/data && unzip icon_data.zip
mv smpl_data smpl_related/
rm -rf icon_data.zip
cd ../../..

# PIXIE
mkdir -p PoseGen/ICON/data/HPS/pixie_data
echo -e "\nDownloading PIXIE..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './PoseGen/ICON/data/HPS/pixie_PoseGen/ICON/data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue
# PIXIE pretrained model and utilities
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=pixie_model.tar&resume=1' -O './PoseGen/ICON/data/HPS/pixie_PoseGen/ICON/data/pixie_model.tar' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=pixie&sfile=utilities.zip&resume=1' -O './PoseGen/ICON/data/HPS/pixie_PoseGen/ICON/data/utilities.zip' --no-check-certificate --continue
cd PoseGen/ICON/data/HPS/pixie_data
unzip utilities.zip
rm -rf utilities.zip
cd ../../../../..

# VPoser
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=V02_05.zip&resume=1' -O './data/V02_05.zip' --no-check-certificate --continue
cd data && unzip V02_05.zip
mv V02_05 vposer
rm -rf V02_05.zip

# Our Data: Human Body Texture & Codebook
gdown https://drive.google.com/uc?id=1n3b8QaBqhViVdNCMTXJttpKEotGlDkf6
unzip data.zip
rm -rf data.zip
cd ..