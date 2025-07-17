# COACD parameter optimization using RL

* **`GOAL: For given input mesh, find the best parameters for COACD to do convex decomposition on the input mesh.`**
* To solve this problem, we will use `RL` to train a network that processes the input mesh, and determine the parameters for COACD.

## SETUP:

### LINUX:

```bash
conda env create -f environment.yml
conda activate coacd_env
```

Install pytorch3d:

```bash
pip install ninja cmake
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

Build binary:

```bash
cd ~/coacd/extern/CoACD
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target main -j"$(nproc)"
```

Compatibility wrapper:

```bash
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
```

```bash
chmod +x "$CONDA_PREFIX/bin/coacd"
```

To run the training:

```bash
cd ~/coacd
CUDA_VISIBLE_DEVICES=  python -m src.models.coacd_ppo_train --device cpu
```

### MAC:

```bash
conda env create -f environment.yml
conda activate coacd_env
```

pytorch3d:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE

pip install --no-build-isolation --no-deps \
  git+https://github.com/facebookresearch/pytorch3d.git@2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7

unset KMP_DUPLICATE_LIB_OK
```

```bash
export COACD_MAIN="/Users/leonliu/my_project/extern/CoACD/build/main"
```

wrapper:

```bash
cat > "$CONDA_PREFIX/bin/coacd" <<'EOF'
#!/usr/bin/env bash
# Minimal CoACD wrapper: translate long flags -> CoACD short flags.
# Requires COACD_MAIN to point to the actual binary.

if [[ -z "${COACD_MAIN:-}" || ! -x "$COACD_MAIN" ]]; then
  echo "coacd compat: set $COACD_MAIN to CoACD binary (extern/CoACD/build/main)." >&2
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
```

```bash
chmod +x "$CONDA_PREFIX/bin/coacd"
```

To run:

```bash
python -m src.models.coacd_ppo_train
```
