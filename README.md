Code has been taken from ```https://github.com/baiwenjia/ukbb_cardiac``` and debugged later for compatibility with TensorFlow's latest version (with cuda + cudnn).

# UKBB Aortic Magnetic Resonance Imaging Analysis
*Requirements and installation* 
```
module load Anaconda3/2023.03-1
conda create -n mycode
conda activate mycode
module load cuda/11.8.0
module load cudnn_for_cuda11/8.6.0
pip install tensorflow==2.12.1
python --version
python3 demo_pipeline.py
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

```
python3 deploy_network_ao.py --model UNet-LSTM --model_path /home/mchopra/wbai/ukbb_cardiac/models/UNet-LSTM_ao --data_dir /home/mchopra/wbai/ukbb_cardiac/images/validation/
```

# sources might be helpful for later on:
1. https://www.tensorflow.org/install/pip
2. https://www.tensorflow.org/install/source#gpu   ##versions**
3. https://docs.anaconda.com/free/anaconda/applications/tensorflow/#python-2
4. https://transang.me/cuda-cudnn-driver-gcc-tensorflow-python-version-compatibility-charts/
5. https://gist.github.com/Madhusakth/66fa3daaffee8b7e11b83df5e2eb1c4e
6. https://gist.github.com/johndpope/f8feb553c6959f0000318f730f3d181f
7. https://medium.com/@poom.wettayakorn/nvidia-cuda-10-cudnn-7-4-and-tensorflow-gpu-1-13-on-ubuntu-16-04-lts-5227854136a1
