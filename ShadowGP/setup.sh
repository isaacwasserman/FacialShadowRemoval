conda env create -f environment.yml
conda activate shadowgp

gdown --folder https://drive.google.com/drive/folders/1Rg5He8XIY8qP4JYPFRRGUIvfZUcqm8zt
mv ShadowGP-Checkpoints/ checkpoint/
tar -zxvf checkpoint/weights.tar.gz

gdown --folder https://drive.google.com/drive/folders/1pLX9BIsG51XH5rnftXrO66dw1a2SKAnj
apt-get install update
apt-get install unzip
unzip evaluation/input.zip -d evaluation
unzip evaluation/gt.zip -d evaluation

rm imgs/*
mv evaluation/input/* imgs
mkdir gts
mv evaluation/gt/* gts
rm -rf evaluation