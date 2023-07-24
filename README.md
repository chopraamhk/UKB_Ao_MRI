# UKB_Aortic_Image_Analysis
*Requirements and installation till now* 

conda create env --work2 python=3.8

conda install tensorflow-gpu cudatoolkit

conda install -c nvidia cudnn 

conda update -n base -c defaults conda

pip install nvidia-cudnn-cu11

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

pip install tensorflow 

# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

As i was getting an error for cudnn==8.0.1, i tried installing cudnn==7.4 using, 
conda install -c anaconda cudnn=7.4

pip3 install numpy matplotlib seaborn pandas python-dateutil pydicom SimpleITK nibabel scikit-image opencv-python vtk

