# UKBB Aortic Magnetic Resonance Imaging Analysis

Code has been taken from ```https://github.com/baiwenjia/ukbb_cardiac``` and debugged later for compatibility with TensorFlow's latest version (with cuda + cudnn).

The below text is from the reference given below:
To train the network end-to-end, we require the ground truth label map sequence across the time frames. However, the typical manual annotation is temporally sparse. For example, our dataset only has manual annotations at two time frames, end-diastole (ED) and end-systole (ES). In order to obtain the annotations at other time frames, we perform label propagation. Non-rigid image registration is performed to estimate the motion between each pair of successive time frames. Based on the motion estimate, the label map at each time frame is propagated from either ED or ES annotations, whichever is closer. For segmentation accuracy, we evaluate the Dice overlap metric and the mean contour distance between automated segmentation and
manual annotation at ED and ES time frames.
Reference : https://arxiv.org/abs/1808.00273

# environment with tensorflow 1.15.0 (not recommended as a lot od functions have been deprecated and keras is included in the version tf2) 
#conda will take care of the other cuda and cudnn requirements here.
```
srun --pty --preserve-env -p gpu /bin/bash
conda create -n tf -c conda-forge tensorflow-gpu=1.15
conda activate tf 
pip install numpy scipy matplotlib seaborn pandas python-dateutil pydicom SimpleITK nibabel scikit-image opencv-python vtk
```

# Step 1: Go to GPU (it's an interactive gpu session - can avoid it by adding the environment directly to the bash script while running)
```
srun --pty --preserve-env -p gpu /bin/bash
```

# Step 2: Create an environment (the codes in the directory are compatible with tensorflow = 2.12.1)
*Requirements and installation* 
```
module load Anaconda3/2023.03-1
conda create -n mycode python=3.9
conda activate mycode
module load cuda/11.8.0
module load cudnn_for_cuda11/8.6.0
pip install tensorflow==2.12.1
conda install -c "conda-forge/label/cf201901" unzip #unzip required for downloading script
conda install -c conda-forge screen #for screen sessions or use tmux
python --version #change in the python version can lead to incompatibility of tensorflow with GPUs.
pip3 install numpy scipy matplotlib seaborn pandas python-dateutil pydicom SimpleITK nibabel scikit-image opencv-python vtk
#python3 demo_pipeline.py
```

# Verify installation of tensorflow with GPU's:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

OR

```
#!/usr/bin/env python3

import tensorflow as tf

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(x)
print(x.shape)
print(x.dtype)

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")
```

# Step 3: Get the .enc file, key file directory location and output with the field id and download the data. 
Make sure that the paths is download_data_ukbb_general.py are correct. 
edit the download.py file by below path::
data_root #where the data will be downloaded
util_dir #ukbb utilities directory
ukbkey #authentication file -> application id + password
csv_dir_path #lists the anonymised ids of the subject

extract.py code is to extract the eid of participants that will be downloaded
```
python3 download_data_ukbb_general.py

#the code will download all the images and convert them from DICOM to NIFTI files. Keep biobank_utils.py as it contains the functions that will help download.
#extract.py contains the eid of participants that will be downloaded.
#it looks like:
eid
1000095
10000**
so on..
```
# Step 4: Download the models and run the code to generate the segmentation of MRI's
```
#oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`

export TF_ENABLE_ONEDNN_OPTS=0

python3 deploy_network_ao.py --model UNet-LSTM --model_path /home/mchopra/wbai/ukbb_cardiac/models/UNet-LSTM_ao --data_dir /home/mchopra/wbai/ukbb_cardiac/images/validation/
```

# Step 5: Generate the distensibility by running the code:
```
python3 aorta_pass_quality_control.py ## contains quality measures

python3 eval_aortic_area.py --data_dir <path of input data> --pressure_csv <path_of csv_file> --output_csv <path_of_output_file>
```

# RULE: NA's ARE NOT ZEROS. 
make sure the file is not adding zeros at the place of missing values
```
pressure_csv looks like :
,"Central pulse pressure during PWA","Central pulse pressure during PWA"
,"12678-2.0","12678-2.1","12678-2.3","12678-2.4",
1,3,4,2,,
2,2,3,,
```

# sources might be helpful for later on:
1. https://www.tensorflow.org/install/pip
2. https://www.tensorflow.org/install/source#gpu   ##versions**
3. https://docs.anaconda.com/free/anaconda/applications/tensorflow/#python-2
4. https://transang.me/cuda-cudnn-driver-gcc-tensorflow-python-version-compatibility-charts/
5. https://gist.github.com/Madhusakth/66fa3daaffee8b7e11b83df5e2eb1c4e
6. https://gist.github.com/johndpope/f8feb553c6959f0000318f730f3d181f
7. https://medium.com/@poom.wettayakorn/nvidia-cuda-10-cudnn-7-4-and-tensorflow-gpu-1-13-on-ubuntu-16-04-lts-5227854136a1
