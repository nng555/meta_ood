from pynvml import *
import torch

def print_mem():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total / 1000**3}GB')
    print(f'free     : {info.free / 1000**3}GB')
    print(f'used     : {info.used / 1000**3}GB')
    mem_out = torch.cuda.mem_get_info()
    print(f"Using {(mem_out[1] - mem_out[0]) / 1000**3}GB / {mem_out[1] / 1000**3}GB")
