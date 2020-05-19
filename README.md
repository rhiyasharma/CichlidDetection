# CichlidDetection

Commands for configuring anaconda environment (execute in order)
-------------------------------------------
conda create -n CichlidDetection python=3.7
conda activate CichlidDetection
conda install -yc anaconda numpy pandas opencv matplotlib tqdm pillow seaborn scikit-learn
conda install -yc conda-forge scikit-image tensorboard
conda install -yc pytorch pytorch torchvision
