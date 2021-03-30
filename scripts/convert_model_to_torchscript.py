import torch

from models import MPNet, MPNetJIT

weight = torch.load("./data/pytorch_model/mpnet_weight.pt")

mpnet = MPNetJIT()
mpnet.load_state_dict(weight)

mpnet.encoder.save("./data/pytorch_model/enet_script_gpu.pt")
mpnet.pnet.save("./data/pytorch_model/pnet_script_gpu.pt")
mpnet.save("./data/pytorch_model/mpnet_script_gpu.pt")
