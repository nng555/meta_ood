import os
import sys
import traceback

def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, show_locals=False)

sys.excepthook = custom_excepthook

from utils import *
# os.environ['HF_HOME'] = "/scratch/nhn234/cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
NN_TOKEN = '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'
QC_TOKEN = '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'

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, DataCollatorWithPadding,
)
from datasets import load_dataset

import typer
app = typer.Typer()

@app.command()
def eval_icl(
    model_name: str='meta-llama/Llama-3.2-1B',
    id_dataset: str='stanfordnlp/imdb',
    id_config: str=None,
    id_split: str='train',
    ood_dataset: str='stanfordnlp/sst2',
    ood_config: str = None,
    ood_split: str='train',
    max_id_icl: int=8,
    max_ood_icl: int=8,
    n_ood_test: int=100,
    nsamples: int=10,
    seed: int=1,
    bsize: int=4,
):

    out_name = id_dataset.split('/')[-1] + '_' + ood_dataset.split('/')[-1]

    id_prompt = "Review: {}\nSentiment: {}\n\n"
    ood_prompt = "Review: {}\n\nSentiment:\n\n"
    test_prompt = "Review: {}\nSentiment:"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,
            token=QC_TOKEN).eval()
    #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=QC_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    print_mem()

    #tok_choices = tokenizer.convert_tokens_to_ids([" Positive", " Negative"])
    #tok_choices = [48314, 40695] # negative, positive
    tok_choices = [51957, 45003]

    id_data = load_dataset(id_dataset, id_config, split=id_split, trust_remote_code=True).shuffle(seed=seed)
    ood_data = load_dataset(ood_dataset, ood_config, trust_remote_code=True).shuffle(seed=seed)

    if len(ood_data.keys()) == 1:
        ood_data = ood_data['train'].train_test_split(test_size=0.2)

    max_log_id = int(np.ceil(np.log2(max_id_icl)))
    max_log_ood = int(np.ceil(np.log2(max_ood_icl)))

    res = np.zeros((max_log_id + 2, max_log_ood + 2, nsamples))
    np.random.seed(seed)

    id_exemplars = np.random.choice(len(id_data), size=(nsamples, max_id_icl), replace=False)
    ood_exemplars = np.random.choice(len(ood_data[ood_split]), size=(nsamples, max_ood_icl), replace=False)

    for n_log_id in list(range(max_log_id + 2))[::-1]: # need at least one ID ICL
        for n_log_ood in range(max_log_ood + 2):
            if n_log_id == 0:
                n_id = 0
            else:
                n_id = 2**(n_log_id - 1)

            if n_log_ood == 0:
                n_ood = 0
            else:
                n_ood = 2**(n_log_ood - 1)

            for i in tqdm(range(nsamples), desc=f"n_id: {n_id}, n_ood: {n_ood}"):
                id_idxs = id_exemplars[i, :n_id]
                ood_idxs = ood_exemplars[i, :n_ood]

                id_exs = []
                ood_exs = []

                for id_idx in id_idxs:
                    id_idx = int(id_idx)
                    data = id_data[id_idx]
                    label = "Positive" if data['label'] else "Negative"
                    # id_exs.append(id_prompt.format(data['sentence'], label))
                    id_exs.append(id_prompt.format(data['text'], label))

                for ood_idx in ood_idxs:
                    ood_idx = int(ood_idx)
                    ood_exs.append(ood_prompt.format(ood_data[ood_split][ood_idx]['sentence']))

                full_prompt = ''.join(id_exs) + ''.join(ood_exs)

                # start evaluation
                bid = 0
                ncorrect = 0

                while bid < n_ood_test:
                    batch = []
                    labels = []

                    while len(batch) < bsize and bid < n_ood_test:
                        batch.append(full_prompt + test_prompt.format(ood_data['test'][bid]['sentence']))
                        labels.append(ood_data['test'][bid]['label'])
                        bid += 1

                    batch = tokenizer(batch)
                    ex_lens = [len(bid) - 1 for bid in batch['input_ids']]
                    batch = DataCollatorWithPadding(tokenizer)(batch)
                    batch = {k: v.cuda() for k, v in batch.items()}
                    out = model(**batch)

                    preds = out.logits[torch.arange(out.logits.shape[0]), ex_lens][:, tok_choices].argmax(-1).cpu().numpy()
                    ncorrect += (preds == labels).sum()
                    del out

                res[n_log_id, n_log_ood, i] = ncorrect / n_ood_test

            print(res.mean(-1))
            print(res.var(-1))
            np.save(out_name + '.npy', res, allow_pickle=True)

if __name__ == "__main__":
    app()
