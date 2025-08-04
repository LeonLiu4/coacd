````markdown
# CoACD Parameter Optimization with RL

This repository uses reinforcement learning to find optimal parameters for CoACD convex decomposition on a given mesh.

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/LeonLiu4/coacd.git --recursive
   cd coacd
````

2. **Create and activate the environment**

   ```bash
   conda env create -f environment.yml
   conda activate coacd_env
   ```

3. **Install PyTorch3D**

   * **Linux**

     ```bash
     pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
     ```

   * **macOS**

     ```bash
     export KMP_DUPLICATE_LIB_OK=TRUE
     pip install --no-build-isolation --no-deps \
       git+https://github.com/facebookresearch/pytorch3d.git@stable
     unset KMP_DUPLICATE_LIB_OK
     ```

4. **Run training**

   ```bash
   # CPU only
   CUDA_VISIBLE_DEVICES= python -m src.models.coacd_ppo_train --device cpu

   # GPU
   python -m src.models.coacd_ppo_train
   ```
