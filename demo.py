import torch, os
from src.envs.coacd_env import CoACDEnv
from src.models.pointnet_param_net import PointNetFeatureExtractor   # ← path fixed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mesh_path = "assets/bunny.obj"
if not os.path.isfile(mesh_path):
    raise FileNotFoundError(mesh_path)

env  = CoACDEnv(mesh_path, npts=1024)            # optional: fewer points
net  = PointNetFeatureExtractor(env.observation_space).to(device)

obs, _ = env.reset()
with torch.no_grad():
    action = (
        net(torch.tensor(obs).unsqueeze(0).to(device))
        .squeeze(0)
        .cpu()
        .numpy()
    )

_, reward, _, _, info = env.step(action)
print(f"reward = {reward:.4f}")
print("params =", info['params'])
