# import torch
# obj = torch.load("/home/fanfengzhao/project/prompt_pair/LLaMA-Factory/prompts/bur-en_step14965.pt", map_location="cpu")

# print(type(obj))
# if isinstance(obj, dict):
#     print("keys:", list(obj.keys())[:20])



import torch
p = "/home/fanfengzhao/project/prompt_pair/LLaMA-Factory/prompts/bur-en_step29930.pt"   # 改成你的文件路径
obj = torch.load(p, map_location="cpu")
print(type(obj))
print("keys:", list(obj.keys()))
print("\nCONFIG:")
from pprint import pprint
pprint(obj.get("config"))
print("\nMETADATA:")
pprint(obj.get("metadata"))

sd = obj.get("encoder_state_dict") or obj.get("state_dict")
print("\nstate_dict sample keys (first 30):")
for k in list(sd.keys())[:30]:
    print(" ", k)



# save as download_comet.py 或 直接在 python 交互运行
# from comet import load_from_checkpoint
# model = load_from_checkpoint("/home/fanfengzhao/project/prompt_pair/wmt22-comet-da/checkpoints/model.ckpt")
# print("loaded:", type(model))

