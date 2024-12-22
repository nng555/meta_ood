from pynvml import *
import torch

OOD_ICL_PROMPT = """You are a sentiment classification system. Below, you will be provided with labeled text examples from a source domain, as well as unlabeled text examples from a target domain. Given these examples, classify the user provided text from the target domain as either \"positive\" or \"negative.\"

"""

def print_mem():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total / 1000**3}GB')
    print(f'free     : {info.free / 1000**3}GB')
    print(f'used     : {info.used / 1000**3}GB')
    mem_out = torch.cuda.mem_get_info()
    print(f"Using {(mem_out[1] - mem_out[0]) / 1000**3}GB / {mem_out[1] / 1000**3}GB")

def get_label_map(label):
    if label in ['1', '0']:
        return {
            '1': 1,
            '0': 0,
        }
    elif label in [1, 0]:
        return {
            1: 1,
            0: 0,
        }
    elif label in ['positive', 'negative']:
        return {
            'positive': 1,
            'negative': 0,
        }
    else:
        raise Exception

