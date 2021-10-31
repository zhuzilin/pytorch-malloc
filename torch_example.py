import torch
  
for i in range(3):
    print(f"start allocate {i}")
    a = torch.zeros((4, 4), device=torch.device("cuda:0"))
    print(f"end allocate {i}")
