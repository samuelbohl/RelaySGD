import torch
import bagua.torch_api as bagua
import time


torch.cuda.set_device(bagua.get_local_rank())
bagua.init_process_group()

t = torch.rand(1000000).cuda()
nb = list(range(bagua.get_world_size()))

avg_bagua = 0
avg_pyt = 0
N = 12
for its in range(N):
    start = time.time()
    for n in nb:
        if n == bagua.get_local_rank():
            continue
        if n < bagua.get_local_rank():
            bagua.send(t, n)
            bagua.recv(t, n)
        else:
            bagua.recv(t, n)
            bagua.send(t, n)
    end = time.time()
    if its > 3: avg_bagua += end-start

    start = time.time()
    for n in nb:
        if n == bagua.get_local_rank():
            continue
        if n < bagua.get_local_rank():
            torch.distributed.send(t, n)
            torch.distributed.recv(t, n)
        else:
            torch.distributed.recv(t, n)
            torch.distributed.send(t, n)
    torch.cuda.synchronize()
    end = time.time()
    if its > 3: avg_pyt += end-start

if bagua.get_local_rank() == 1: print("BAGUA_AVG: {}".format(avg_bagua/(N-1)))
if bagua.get_local_rank() == 1: print("PYT_AVG: {}".format(avg_pyt/(N-1)))