import torch, os
from envs.coacd_env import CoACDEnv
from models.pointnet_param_net import PointNetParamNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mesh_path = "assets/bunny.obj"      # change to your mesh
if not os.path.isfile(mesh_path):
    raise FileNotFoundError(mesh_path)

env   = CoACDEnv(mesh_path)
net   = PointNetParamNet().to(device)

obs, _ = env.reset()
action = (
    net(torch.tensor(obs).unsqueeze(0).to(device))   # forward pass
      .squeeze(0)                                    # (3,)
      .detach()                                      # ← stop autograd
      .cpu()
      .numpy()
)

_, reward, _, _, info = env.step(action)
print(f"reward = {reward:.4f}")
print("params =", info["params"])
