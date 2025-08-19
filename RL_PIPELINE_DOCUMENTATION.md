# CoACD Reinforcement Learning Pipeline Documentation

## Overview

This document describes the Reinforcement Learning (RL) pipeline for optimizing CoACD (Convex Approximate Convex Decomposition) parameters. The system uses Proximal Policy Optimization (PPO) to learn optimal parameters for mesh decomposition.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Mesh    │───▶│   CoACD Agent   │───▶│  Decomposed     │
│   (Bunny.obj)   │    │   (PPO)         │    │  Parts          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Environment   │
                       │  (CoACDEnv)     │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Reward        │
                       │  Calculation    │
                       └─────────────────┘
```

## Detailed Pipeline Flow

### 1. Environment Setup (CoACDEnv)

```
┌─────────────────────────────────────────────────────────────┐
│                    CoACDEnv Environment                     │
├─────────────────────────────────────────────────────────────┤
│ Input:                                                     │
│  - Mesh: assets/bunny_simplified.obj                       │
│  - Baseline metrics (Hausdorff, runtime, parts, vertices)  │
│  - Point sampling parameters (4096 points, 25 angles)      │
├─────────────────────────────────────────────────────────────┤
│ State Space:                                               │
│  - Current CoACD parameters (threshold, mcts_nodes, etc.)  │
│  - Previous performance metrics                            │
├─────────────────────────────────────────────────────────────┤
│ Action Space:                                              │
│  - Parameter adjustments (continuous values)               │
│  - Bounded by reasonable ranges for each parameter         │
└─────────────────────────────────────────────────────────────┘
```

### 2. Point Cloud Generation Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Decomposed    │───▶│   Depth Map     │───▶│   Point Cloud   │
│   Mesh Parts    │    │   Rendering     │    │   Generation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   MeshLab       │
                       │  Simplification │
                       └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Final Point   │
                       │   Cloud (4096)  │
                       └─────────────────┘
```

**Detailed Point Sampling Process:**
```
1. Render depth maps from 25 camera directions at 512x512 resolution
2. Backproject depth maps to 3D points (~747k points)
3. Deduplicate points (tolerance 1e-2) → ~35k unique points
4. MeshLab grid sampling → 966 uniform points
5. Add 3,130 random points → 4,096 total points
```

### 3. Reward Function Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Reward Calculation                       │
├─────────────────────────────────────────────────────────────┤
│ Components:                                                 │
│                                                             │
│ 1. Hausdorff Distance (H):                                 │
│    - Compare reconstructed vs original point clouds        │
│    - Target: H < baseline_H * 1.5 (relaxed)               │
│                                                             │
│ 2. Runtime Performance:                                     │
│    - Measure CoACD execution time                          │
│    - Target: runtime < baseline_runtime * 1.5              │
│                                                             │
│ 3. Part Count:                                             │
│    - Number of convex hulls                                │
│    - Target: parts < baseline_parts * 1.5                  │
│                                                             │
│ 4. Vertex Count:                                           │
│    - Total vertices across all parts                       │
│    - Target: vertices < baseline_vertices * 1.5            │
├─────────────────────────────────────────────────────────────┤
│ Success Condition:                                         │
│ H < baseline_H * 0.75 AND runtime < baseline_runtime * 1.2 │
└─────────────────────────────────────────────────────────────┘
```

### 4. Training Loop

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Initialize    │───▶│   Agent Step    │───▶│   Environment   │
│   Environment   │    │   (PPO)         │    │   Step          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Collect       │    │   Run CoACD     │
                       │   Experience    │    │   with Params   │
                       └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Update        │    │   Calculate     │
                       │   Policy        │    │   Reward        │
                       └─────────────────┘    └─────────────────┘
                              │                        │
                              └──────────┬─────────────┘
                                         ▼
                              ┌─────────────────┐
                              │   Continue      │
                              │   Training?     │
                              └─────────────────┘
                                         │
                                         ▼
                              ┌─────────────────┐
                              │   Save Best     │
                              │   Parameters    │
                              └─────────────────┘
```

## Key Components

### 1. CoACD Parameters (Action Space)

```
┌─────────────────────────────────────────────────────────────┐
│                    CoACD Parameters                         │
├─────────────────────────────────────────────────────────────┤
│ Parameter          │ Baseline │ Range      │ Description    │
├─────────────────────────────────────────────────────────────┤
│ threshold          │ 0.1      │ [0.01, 0.5]│ Decomposition  │
│                    │          │            │ threshold      │
├─────────────────────────────────────────────────────────────┤
│ mcts_nodes         │ 20       │ [5, 50]    │ MCTS tree nodes│
├─────────────────────────────────────────────────────────────┤
│ mcts_iterations    │ 100      │ [20, 200]  │ MCTS iterations│
├─────────────────────────────────────────────────────────────┤
│ mcts_max_depth     │ 3        │ [2, 5]     │ MCTS max depth │
├─────────────────────────────────────────────────────────────┤
│ preprocess_res     │ 30       │ [10, 100]  │ Preprocess res │
├─────────────────────────────────────────────────────────────┤
│ resolution         │ 1000     │ [500, 2000]│ Final resolution│
└─────────────────────────────────────────────────────────────┘
```

### 2. Baseline Metrics

```
┌─────────────────────────────────────────────────────────────┐
│                    Baseline Performance                     │
├─────────────────────────────────────────────────────────────┤
│ Metric              │ Value    │ Description                │
├─────────────────────────────────────────────────────────────┤
│ Hausdorff Distance  │ 0.003680 │ Point cloud similarity     │
├─────────────────────────────────────────────────────────────┤
│ Runtime             │ 5.735s   │ CoACD execution time       │
├─────────────────────────────────────────────────────────────┤
│ Parts               │ 14       │ Number of convex hulls     │
├─────────────────────────────────────────────────────────────┤
│ Total Vertices      │ 1592     │ Vertices across all parts  │
└─────────────────────────────────────────────────────────────┘
```

### 3. Point Sampling Configuration

```
┌─────────────────────────────────────────────────────────────┐
│                    Point Sampling Config                    │
├─────────────────────────────────────────────────────────────┤
│ Parameter           │ Value    │ Description                │
├─────────────────────────────────────────────────────────────┤
│ Target Points       │ 4096     │ Final point cloud size     │
├─────────────────────────────────────────────────────────────┤
│ Camera Angles       │ 25       │ Number of viewing dirs     │
├─────────────────────────────────────────────────────────────┤
│ Resolution          │ 512x512  │ Depth map resolution       │
├─────────────────────────────────────────────────────────────┤
│ Tolerance           │ 1e-2     │ Deduplication tolerance    │
├─────────────────────────────────────────────────────────────┤
│ MeshLab Filter      │ Grid     │ Uniform sampling method    │
└─────────────────────────────────────────────────────────────┘
```

## Training Process

### Phase 1: Environment Interaction
```
Agent → Environment → CoACD → Point Sampling → Reward Calculation
```

### Phase 2: Experience Collection
```
- Collect trajectories of (state, action, reward, next_state)
- Store in PPO buffer for policy updates
- Use baseline metrics for reward normalization
```

### Phase 3: Policy Update
```
- Update PPO policy using collected experience
- Optimize for better CoACD parameters
- Balance exploration vs exploitation
```

## Performance Optimization

### 1. Fast Sampling for Training
```
- 25 camera angles (reduced from higher counts)
- 512x512 resolution (balanced quality/speed)
- MeshLab grid sampling for uniformity
- 4096 target points (efficient for training)
```

### 2. Reward Relaxation
```
- 1.5x relaxation factor for all metrics
- Makes training targets more achievable
- Prevents overly strict constraints
- Encourages exploration of parameter space
```

## Expected Outcomes

### Training Goals
```
1. Hausdorff Distance: < 0.00276 (baseline * 0.75)
2. Runtime: < 6.882s (baseline * 1.2)
3. Parts: < 21 (baseline * 1.5)
4. Vertices: < 2388 (baseline * 1.5)
```

### Success Criteria
```
- Hausdorff distance below relaxed threshold
- Runtime within acceptable bounds
- Reasonable number of parts
- Efficient vertex count
```

## File Structure

```
my_project/
├── src/
│   ├── envs/
│   │   └── coacd_env.py          # RL environment
│   ├── models/
│   │   └── pointnet_param_net.py # Neural network
│   └── utils/
│       ├── geometry.py           # Point sampling
│       └── visualization.py      # Visualization tools
├── baseline_coacd.py             # Baseline calculation
├── coacd_ppo_train.py           # Training script
├── baseline_metrics.json         # Baseline performance
└── assets/
    └── bunny_simplified.obj      # Input mesh
```

## Usage

### Running Training
```bash
conda activate coacd_clean
python coacd_ppo_train.py
```

### Creating Baseline
```bash
python baseline_coacd.py
```

### Generating Point Cloud
```bash
python create_baseline_pointcloud.py
```

## Key Innovations

1. **Depth-based Point Sampling**: Uses depth rendering instead of raycasting
2. **MeshLab Integration**: Ensures uniform point distribution
3. **Relaxed Reward Function**: 1.5x relaxation for better training
4. **Fast Training Sampling**: Optimized parameters for speed
5. **Comprehensive Metrics**: Hausdorff, runtime, parts, vertices

This RL pipeline represents a novel approach to optimizing mesh decomposition parameters through reinforcement learning, with careful attention to both performance and training efficiency.
