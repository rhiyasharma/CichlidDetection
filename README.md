# CichlidDetection

Command for downloading this repo (execute in order)
-------------------------------------------
mkdir -p ~/data
cd data
git clone https://github.com/ptmcgrat/CichlidDetection.git

Commands for configuring anaconda environment (execute in order)
-------------------------------------------
conda create -n CichlidDetection python=3.7
conda activate CichlidDetection
conda install -yc anaconda numpy pandas opencv matplotlib tqdm pillow seaborn scikit-learn
conda install -yc conda-forge scikit-image tensorboard shapely rclone
conda install -yc pytorch pytorch torchvision cudatoolkit=10.1

