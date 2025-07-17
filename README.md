# COACD parameter optimization using RL

- **`GOAL: For given input mesh, find the best parameters for COACD to do convex decomposition on the input mesh.`**
- To solve this problem, we will use `RL` to train a network that processes the input mesh, and determine the parameters for COACD.

SETUP (for linux): 

Paste this into environment.yml:

# environment.yml  (cleaned)
name: coacd_env
channels:
 - pytorch        # keep pytorch channel first so conda sees its builds
 - conda-forge
 - defaults


dependencies:
 # ---- core runtime (conda) ----
 - python=3.10
 - numpy                     # conda picks compatible build
 - gymnasium
 - rtree
 - pytorch                   # CPU build; add `cpuonly` if conda chooses CUDA by default
 - pip


 # ---- pip packages ----
 - pip:
     # project / repo
     # (remove editable line because extern/coacd isn't a Python package)
     # - -e ./extern/coacd


     # libraries
     - absl-py==2.3.1
     - cmake==3.27.9
     - contourpy==1.3.2
     - cycler==0.12.1
     - filelock==3.18.0
     - fonttools==4.59.0
     - fsspec                  # let pip resolve
     - fvcore==0.1.5.post20221221
     - grpcio==1.73.1
     - gymnasium==1.1.1        # duplicate safe
     - iopath==0.1.10
     - Jinja2==3.1.6
     - kiwisolver==1.4.8
     - Markdown==3.8.2
     - MarkupSafe==3.0.2
     - matplotlib==3.10.3
     - mpmath==1.3.0
     - networkx==3.4.2
     - packaging==25.0
     - pandas==2.3.1
     - pillow==11.3.0
     - portalocker==3.2.0
     - protobuf==6.31.1
     - pyparsing==3.2.3
     - python-dateutil==2.9.0.post0
     # REMOVE pytorch3d here; will install after env creation
     - pytz==2025.2
     - PyYAML==6.0.2
     - rl==3.2
     - six==1.17.0
     - stable-baselines3==2.6.0
     - sympy==1.14.0
     - tabulate==0.9.0
     - tensorboard==2.19.0
     - tensorboard-data-server==0.7.2
     - termcolor==3.1.0
     - tqdm==4.67.1
     - trimesh==4.7.0
     - typing_extensions        # keep only one spelling
     - tzdata==2025.2
     - Werkzeug==3.1.3
     - yacs==0.1.8

After cloning git create conda environment:
conda env create -f environment.yml 
conda activate coacd_env

Install pytorch3d:

pip install ninja cmake
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

Build binary:
cd ~/coacd/extern/CoACD
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target main -j"$(nproc)"

Compatibility wrapper:
cat > "$CONDA_PREFIX/bin/coacd" <<'EOF'

#!/usr/bin/env bash
# CoACD compatibility wrapper
if [[ -n "${COACD_MAIN:-}" && -x "$COACD_MAIN" ]]; then
  : # use as-is
else
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  _cands=(
    "$COACD_MAIN"                                   # env override (may be empty)
    "$script_dir/../extern/CoACD/build/main"        # relative to wrapper
    "${CONDA_PREFIX:-}/extern/CoACD/build/main"     # conda env
    "$HOME/coacd/extern/CoACD/build/main"           # legacy fallback
  )

  for c in "${_cands[@]}"; do
    [[ -n "$c" && -x "$c" ]] || continue
    COACD_MAIN="$c"
    break
  done
fi

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)       args+=(-i "$2"); shift 2 ;;
    --output)      args+=(-o "$2"); shift 2 ;;
    --threshold)   args+=(-t "$2"); shift 2 ;;
    --merge)       shift ;;                 # merge is default
    --no-merge)    args+=(-nm); shift ;;
    --max_hull)    args+=(-c "$2"); shift 2 ;;
    *)             args+=("$1"); shift ;;
  esac
done

exec "$COACD_MAIN" "${args[@]}"
EOF

chmod +x "$CONDA_PREFIX/bin/coacd"

To run the training:
cd ~/coacd
CUDA_VISIBLE_DEVICES=  python -m src.models.coacd_ppo_train --device cpu

SETUP (for mac):

conda env create -f environment.yml 
conda activate coacd_env

pytorch3d:

export KMP_DUPLICATE_LIB_OK=TRUE

pip install --no-build-isolation --no-deps \
  git+https://github.com/facebookresearch/pytorch3d.git@2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7

unset KMP_DUPLICATE_LIB_OK

export COACD_MAIN="/Users/leonliu/my_project/extern/CoACD/build/main"

wrapper:

cat > "$CONDA_PREFIX/bin/coacd" <<'EOF'
#!/usr/bin/env bash
# Minimal CoACD wrapper: translate long flags -> CoACD short flags.
# Requires COACD_MAIN to point to the actual binary.

if [[ -z "${COACD_MAIN:-}" || ! -x "$COACD_MAIN" ]]; then
  echo "coacd compat: set \$COACD_MAIN to CoACD binary (extern/CoACD/build/main)." >&2
  exit 127
fi

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)       args+=(-i "$2"); shift 2 ;;
    --output)      args+=(-o "$2"); shift 2 ;;
    --threshold)   args+=(-t "$2"); shift 2 ;;
    --merge)       shift ;;                 # default is merge
    --no-merge)    args+=(-nm); shift ;;
    --max_hull)    args+=(-c "$2"); shift 2 ;;
    -h|--help)     exec "$COACD_MAIN" -h ;;
    --)            shift; args+=("$@"); break ;;
    *)             args+=("$1"); shift ;;
  esac
done

exec "$COACD_MAIN" "${args[@]}"
EOF
chmod +x "$CONDA_PREFIX/bin/coacd"

To run:
python -m src.models.coacd_ppo_train
