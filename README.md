# CichlidDetection

Command for downloading this repo (execute in order)
-------------------------------------------
mkdir -p ~/data <br/>
cd data <br/>
git clone https://github.com/ptmcgrat/CichlidDetection.git <br/>

Commands for configuring anaconda environment (execute in order)
-------------------------------------------
conda create -n CichlidDetection python=3.7 <br/>
conda activate CichlidDetection <br/>
conda install -yc anaconda numpy pandas opencv matplotlib tqdm pillow seaborn scikit-learn <br/>
conda install -yc conda-forge scikit-image tensorboard shapely rclone <br/>
conda install -yc pytorch pytorch torchvision cudatoolkit=10.1 <br/>

Other Requirements
-------------------------------------------
imagemagick (https://imagemagick.org/script/download.php) run 'convert -version' to see if it's already installed

Files
-------------------------------------------
##### VideoDetection.py 
A pipeline that allows you to do the following:
- download images or videos from different projects
- trim the video files into smaller manageable chunks
- run the images/videos through the machine learning model to produce predictions
- produce the final output videos where the cichlids are being tracked 
- upload final output files to the cloud 
