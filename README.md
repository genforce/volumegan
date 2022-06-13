# VolumeGAN - 3D-aware Image Synthesis via Learning Structural and Textural Representations

![image](./docs/assets/framework.png)
**Figure:** *Framework of VolumeGAN.*

> **3D-aware Image Synthesis via Learning Structural and Textural Representations** <br>
> Yinghao Xu, Sida Peng, Ceyuan Yang, Yujun Shen, Bolei Zhou <br>
> *Computer Vision and Pattern Recognition (CVPR), 2022*

[[Paper](https://arxiv.org/pdf/2112.10759.pdf)]
[[Project Page](https://genforce.github.io/volumegan/)]
[[Demo](https://www.youtube.com/watch?v=p85TVGJBMFc)]

This paper aims at achieving high-fidelity 3D-aware images synthesis. We propose a novel framework, termed as VolumeGAN, for synthesizing images under different camera views, through explicitly learning a structural representation and a textural representation. We first learn a feature volume to represent the underlying structure, which is then converted to a feature field using a NeRF-like model. The feature field is further accumulated into a 2D feature map as the textural representation, followed by a neural renderer for appearance synthesis. Such a design enables independent control of the shape and the appearance. Extensive experiments on a wide range of datasets show that our approach achieves sufficiently higher image quality and better 3D control than the previous methods.


## Usage

### Installation

Make sure your Python >= 3.7, CUDA version >= 10.2, and CUDNN version >= 7.6.5.

1. Install package requirements.

   Option 1 (Recommend): Create a virtual environment via `conda`.

   ```shell
   conda create -n hammer python=3.7  # create virtual environment with python 3.7
   conda activate hammer
   conda install --yes --file requirements.txt
   ```

   Option 2: Install via `pip3` (or `pip`).

   ```shell
   pip3 install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html
   ```

2. To use video visualizer (optional), please also install `ffmpeg`.

   - Ubuntu: `sudo apt-get install ffmpeg`.
   - MacOS: `brew install ffmpeg`.

3. To reduce memory footprint (optional), you can switch to either `jemalloc` (recommended) or `tcmalloc` rather than your default memory allocator.

   - jemalloc (recommended):
     - Ubuntu: `sudo apt-get install libjemalloc`
   - tcmalloc:
     - Ubuntu: `sudo apt-get install google-perftools`

4. (optional) To speed up data loading on NVIDIA GPUs, you can install [DALI](https://github.com/NVIDIA/DALI), together with [CuPy](https://cupy.dev/) for customized operations if needed:

    ```shell
    pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-<CUDA_VERSION>
    pip3 install cupy
    ```

    For example, on CUDA 10.2, DALI can be installed via:

    ```shell
    pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
    pip3 install cupy
    ```

### Test Demo
```bash
python render.py --checkpoint ${MODEL_PATH} --num ${NUM} --work_dir ${WORK_DIR} --seed ${SEED} --render_mode ${RENDER_MODE} --generate_html ${SAVE_HTML} volumegan-ffhq
```

where 

- `<MODEL_PATH>` refers to the path of the pretrained model.

- `<NUM>` refers the number of the samples to synthesize.

- `<WORK_DIR>` refers the path to save the results.

- `<SEED>` refers the random seed.

- `<RENDER_MODE>` refers the type of the rendered results. We provide two choices: one is `video` and the other is `shape`. 

- `<SAVE_HTML>` refers whether to save images into a html file when rendering videos. The default value is False. 

We provide the following pretrained models for inference.

| Pretrained Models | 
| :--- | 
|[FFHQ_256x256](https://www.dropbox.com/s/ygwhufzwi2vb2t8/volumegan_ffhq256.pth?dl=0)|


### Train Demo

#### Train VolumeGAN on FFHQ in Resolution of 256x256

In your Terminal, run:

```bash
PORT=<PORT> ./scripts/training_demos/volumegan_ffhq256.sh <NUM_GPUS> <PATH_TO_DATA> [OPTIONS]
```

where

- `<PORT>` refers to the communication port for distributed training.

- `<NUM_GPUS>` refers to the number of GPUs. Setting `<NUM_GPUS>` as 1 helps launch a training job on single-GPU platforms.

- `<PATH_TO_DATA>` refers to the path of FFHQ dataset (in resolution of 256x256) with `zip` format. If running on local machines, a soft link of the data will be created under the `data` folder of the working directory to save disk space.

- `[OPTIONS]` refers to any additional option to pass. Detailed instructions on available options can be shown via `./scripts/training_demos/volumegan_ffhq256.sh <NUM_GPUS> <PATH_TO_DATA> --help`.

This demo script uses `volumegan_ffhq256` as the default value of `job_name`, which is particularly used to identify experiments. Concretely, a directory with name `job_name` will be created under the root working directory (with is set as `work_dirs/` by default). To prevent overwriting previous experiments, an exception will be raised to interrupt the training if the `job_name` directory has already existed. To change the job name, please use `--job_name=<NEW_JOB_NAME>` option.

### Prepare Datasets

See [dataset preparation](./docs/dataset_preparation.md) for details.

## BibTeX

```bibtex
@inproceedings{xu2021volumegan,
  title     = {3D-aware Image Synthesis via Learning Structural and Textural Representations},
  author    = {Xu, Yinghao and Peng, Sida and Yang, Ceyuan and Shen, Yujun and Zhou, Bolei},
  booktitle = {CVPR},
  year      = {2022}
}
```
