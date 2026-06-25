# DiffPhysCam-NovelViewSynthesis

This repository contains the official implementation code of the following paper:

📄 [**DiffPhysCam: Differentiable Physics-Based Camera Simulation for Inverse Rendering and Embodied AI**](https://arxiv.org/abs/2508.08831) (arXiv preprint)

The other GitHub repository for this paper is here: [DiffPhysCam-CamCaliExp](https://github.com/DanielYamChen/DiffPhysCam-CamCaliExp)

---

## Relationship to NVIDIA NVDiffRecMC

This codebase is **derived from and extends** the official implementation of:

> [*Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising*](https://nvlabs.github.io/nvdiffrecmc/) (NeurIPS 2022)  
> [https://github.com/NVlabs/nvdiffrecmc](https://github.com/NVlabs/nvdiffrecmc)

All original components from *NVDiffRecMC* remain subject to NVIDIA's original license terms.

All modifications and newly added components related to the **DiffPhysCam camera model** are original contributions by the authors of this repository.

---

## ⚖️ License

This repository contains code under two licenses:

* Original code from *NVDiffRecMC* is licensed under the **NVIDIA Source Code License**, see [LICENSE-NVIDIA.txt](LICENSE-NVIDIA.txt).

* New code and modifications authored in this repository are released under the **BSD 3-Clause License**, see [LICENSE](LICENSE).

Please review both licenses carefully before use.

---

## Disclaimer

This project is an independent academic research effort and is not affiliated with or endorsed by NVIDIA.

---

## 📌 Scope

This repository aims to reproduces the results in *Sec. IV: REAL-TO-SIM VIRTUAL ENVIRONMENT CREATION STUDY* in the paper.

---

## 🗃️ Dataset

Please download the folder named `DiffPhysCam_Data` of the dataset and the output results from [HuggingFace](https://huggingface.co/datasets/DanielYamChen/DiffPhysCam_Data) (we will also have a backup in Zenodo) and arrange the folder structure on your machine as follows:
```
├─ DiffPhysCam-CamCaliExp/
│  └─ Src/
│  └─ ...
│
├─ DiffPhysCam-NovelViewSynthesis/
│  ├─ Src/
│  ├─ train.py
│  └─ test.py
│  └─ ...
│
└─ DiffPhysCam_Data/
   ├─ CameraCalibrateExp_Data/
   ├─ CameraCalibrateExp_Output/
   ├─ NovelViewSynthesis_Data/
   └─ NovelViewSynthesis_Output/
```

---

## 📦 Required Packages

* Firstly, you need to clone the other repo of this paper ([DiffPhysCam-CamCaliExp](https://github.com/DanielYamChen/DiffPhysCam-CamCaliExp)) as shown above, which contains the calibrated camera model parameters defined in `DiffPhysCam-CamCaliExp/phys_cam_model_params_defocus_{gaussian,uniform}.json`.

This project was tested using Anaconda3. Crucial packages and their versions to install are listed below.

* A system-wide CUDA 12.3 installed in `/usr/local/cuda-12.3`
* Python 3.10
* pytorch 2.10.0 (compiled with CUDA 12.8)
* torchvision 0.25.0
* pytorch-cuda 12.1
* **nvdiffrast** 0.4.0
* **tinycudann** 2.0
* **numba** 0.64.0 (Notice: this is NOT `numba-cuda` from NVIDIA)

Please refer the GitHub repo of [NVDiffRecMC](https://github.com/nvlabs/nvdiffrecmc) to see how to install the packages above.

---

## ▶️ Run

### (Optional) Generate synthetic photos of Sim Scenes 1, 2, and 3

```bash
python get_calibtated_cam_output.py --cond {SimScene01, SimScene02, SimScene03} --defocus_type gaussian
```
The synthetic photos of Sim Scenes 1, 2, and 3 will be saved in ``

---

Given camera configuration data `trnsfrms_and_configs_{train,test}_wo_ground.json` (which contains corresponding calibrated camera poses, camera setting parameters, defocus matrix name, and environment map name), training photos, depth maps, and environment maps, one can follow the steps below for 3D scene reconstruction and novel view synthesis.


### Generate and save defocus matrices from depth maps for real-world scenes
Go to `Src/` folder and run
```bash
python get_defocus_matrix_from_depth_map.py --scene {RealScene01, RealScene02} --defocus_type {gaussian, uniform}
```
The output defocus matrices will be saved in ``.


### Optimize the mesh and material textures
```bash
# full camera condtion
python train.py --scene SimScene01 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene01_full --add_phys_cam --defocus_type gaussian --cond full --learning_rate 0.0250 0.0030

# w/o defocus-blur condition
python train.py --scene SimScene01 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene01_wo_defocus --add_phys_cam --cond wo_defocus --learning_rate 0.0250 0.0030

# w/o exposure-related condition
python train.py --scene SimScene01 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene01_wo_expsr --add_phys_cam --cond wo_expsr --defocus_type gaussian --learning_rate 0.0250 0.0030

# NVDiffRecMC-only (w/o camera) condition
python train.py --scene SimScene01 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene01_wo_camera --learning_rate 0.0250 0.0030
```
For SimScene01, the optimization results will be saved in `DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene01_{full,wo_defocus,wo_expsr,wo_camera}`.


```bash
# full camera condtion
python train.py --scene SimScene02 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene02_full --add_phys_cam --defocus_type gaussian --cond full --learning_rate 0.0250 0.0030

# w/o defocus-blur condition
python train.py --scene SimScene02 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene02_wo_defocus --add_phys_cam --cond wo_defocus --learning_rate 0.0250 0.0030

# w/o exposure-related condition
python train.py --scene SimScene02 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene02_wo_expsr --add_phys_cam --cond wo_expsr --defocus_type gaussian --learning_rate 0.0250 0.0030

# NVDiffRecMC-only (w/o camera) condition
python train.py --scene SimScene02 --mesh_scale 1.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene02_wo_camera --learning_rate 0.0250 0.0030
```
For SimScene02, the optimization results will be saved in `DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene02_{full,wo_defocus,wo_expsr,wo_camera}`.


```bash
# full camera condtion
python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_full --add_phys_cam --cond full --defocus_type gaussian --learning_rate 0.0250 0.0030

# w/o defocus-blur condition
python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_wo_defocus --add_phys_cam --cond wo_defocus --learning_rate 0.0250 0.0030

# w/o exposure-related condition
python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_wo_expsr --add_phys_cam --cond wo_expsr --defocus_type gaussian --learning_rate 0.0250 0.0030

# NVDiffRecMC-only (w/o camera) condition
python train.py --scene SimScene03 --mesh_scale 1.6 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_wo_camera --learning_rate 0.0250 0.0030
```
For SimScene03, the optimization results will be saved in `DiffPhysCam_Data/NovelViewSynthesis_Output/SimScene03_{full,wo_defocus,wo_expsr,wo_camera}`.


```bash
# full camera with Gaussian-defocused condtion
python train.py --scene RealScene01 --mesh_scale 1.4 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_full_defocus_gaussian --add_phys_cam --defocus_type gaussian --cond full --learning_rate 0.0250 0.0030

# full camera with Uniform-defocused condtion
python train.py --scene RealScene01 --mesh_scale 1.4 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_full_defocus_uniform --add_phys_cam --defocus_type uniform --cond full --learning_rate 0.0250 0.0030

# w/o defocus-blur condition
python train.py --scene RealScene01 --mesh_scale 1.4 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_wo_defocus --add_phys_cam --defocus_type gaussian --cond wo_defocus --learning_rate 0.0250 0.0030

# w/o exposure-related with Gaussian-defocused condition
python train.py --scene RealScene01 --mesh_scale 1.4 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_wo_expsr_defocus_gaussian --add_phys_cam --defocus_type gaussian --cond wo_expsr --learning_rate 0.0250 0.0030

# w/o exposure-related with Uniform-defocused condition
python train.py --scene RealScene01 --mesh_scale 1.4 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_wo_expsr_defocus_uniform --add_phys_cam --defocus_type uniform --cond wo_expsr --learning_rate 0.0250 0.0030

# NVDiffRecMC-only (w/o camera) condition
python train.py --scene RealScene01 --mesh_scale 1.4 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_wo_camera --learning_rate 0.0250 0.0030
```
For RealScene01, the optimization results will be saved in `DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene01_{full_defocus_gaussian,full_defocus_uniform,wo_defocus,wo_expsr_defocus_gaussian,wo_expsr_defocus_uniform,wo_camera}`.


```bash
# full camera with Gaussian-defocused condtion
python train.py --scene RealScene02 --mesh_scale 3.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene02_full_defocus_gaussian --add_phys_cam --cond full --defocus_type gaussian --learning_rate 0.0150 0.0018

# full camera with Uniform-defocused condtion
python train.py --scene RealScene02 --mesh_scale 3.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene02_full_defocus_uniform --add_phys_cam --cond full --defocus_type uniform --learning_rate 0.0150 0.0018

# w/o defocus-blur condition
python train.py --scene RealScene02 --mesh_scale 3.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene02_wo_defocus --add_phys_cam --cond wo_defocus --learning_rate 0.0150 0.0018

# w/o exposure-related with Gaussian-defocused condition
python train.py --scene RealScene02 --mesh_scale 3.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene02_wo_expsr_defocus_gaussian --add_phys_cam --cond wo_expsr --defocus_type gaussian --learning_rate 0.0150 0.0018

# w/o exposure-related with Uniform-defocused condition
python train.py --scene RealScene02 --mesh_scale 3.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene02_wo_expsr_defocus_uniform --add_phys_cam --cond wo_expsr --defocus_type uniform --learning_rate 0.0150 0.0018

# NVDiffRecMC-only (w/o camera) condition
python train.py --scene RealScene02 --mesh_scale 3.8 --out_dir ../DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene02_wo_camera --learning_rate 0.0150 0.0018
```
For RealScene02, the optimization results will be saved in `DiffPhysCam_Data/NovelViewSynthesis_Output/RealScene02_{full_defocus_gaussian,full_defocus_uniform,wo_defocus,wo_expsr_defocus_gaussian,wo_expsr_defocus_uniform,wo_camera}`.

---

## 💬 Questions?

Please raise issues here if you have any questions.