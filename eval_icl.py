import os
import sys
import traceback

def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, show_locals=False)

sys.excepthook = custom_excepthook

from utils import *

# set HF cache on NYU cluster
if os.environ["USER"] == 'nhn234':
    os.environ['HF_HOME'] = "/scratch/nhn234/cache"
    HF_TOKEN = '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'
else:
    HF_TOKEN = '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'

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    model_name: str='meta-llama/Llama-3.2-1B-Instruct',
    src_dataset: str='stanfordnlp/sst2',
    src_config: str=None,
    src_split: str='train',
    src_field: str='sentence',
    tgt_dataset: str='takala/financial_phrasebank',
    tgt_config: str = 'sentences_50agree',
    tgt_split: str='train',
    tgt_test_split: str='test',
    tgt_field: str='sentence',
    max_src_icl: int=16,
    max_tgt_icl: int=16,
    n_tgt_test: int=600,
    nsamples: int=10,
    seed: int=1,
    bsize: int=4,
):

    out_name = src_dataset.split('/')[-1] + '_' + tgt_dataset.split('/')[-1]

    src_prompt = "Text:\n{}\n\nSentiment:\n{}\n\n"
    tgt_prompt = "Text:\n{}\n\n"
    test_prompt = "Text:\n{}"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16,
            token=HF_TOKEN).eval()
    #model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    print_mem()

    tok_choices = tokenizer.convert_tokens_to_ids(["negative", "positive"])
    label_map = get_label_map

    src_data = load_dataset(src_dataset, src_config, split=src_split, trust_remote_code=True).shuffle(seed=seed)
    tgt_data = load_dataset(tgt_dataset, tgt_config, trust_remote_code=True).shuffle(seed=seed)
    tgt_data = tgt_data.filter(lambda x: x['label'] not in ['2', 2, 'neutral'])

    src_label_map = get_label_map(src_data[0]['label'])
    tgt_label_map = get_label_map(tgt_data[tgt_split][0]['label'])

    if len(tgt_data.keys()) == 1:
        tgt_data = tgt_data[tgt_split].train_test_split(test_size=0.2)

    max_log_id = int(np.ceil(np.log2(max_src_icl)))
    max_log_ood = int(np.ceil(np.log2(max_tgt_icl)))

    res = np.zeros((max_log_id + 2, max_log_ood + 2, nsamples))
    np.random.seed(seed)

    src_exemplars = np.random.choice(len(src_data), size=(nsamples, max_src_icl), replace=False)
    tgt_exemplars = np.random.choice(len(tgt_data[tgt_split]), size=(nsamples, max_tgt_icl), replace=False)

    for n_log_id in list(range(max_log_id + 2)):
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
                src_idxs = src_exemplars[i, :n_id]
                tgt_idxs = tgt_exemplars[i, :n_ood]

                src_exs = []
                tgt_exs = []

                for src_idx in src_idxs:
                    src_idx = int(src_idx)
                    data = src_data[src_idx]
                    label = src_label_map[data['label']]
                    src_exs.append(src_prompt.format(data[src_field], label))

                for tgt_idx in tgt_idxs:
                    tgt_idx = int(tgt_idx)
                    tgt_exs.append(tgt_prompt.format(tgt_data[tgt_split][tgt_idx][tgt_field]))

                full_sys_prompt = OOD_ICL_PROMPT +"### Source Examples\n\n" + ''.join(src_exs) + "### Target Examples\n\n" + ''.join(tgt_exs)

                messages = [
                    {'role': 'system', 'content': full_sys_prompt},
                ]

                # start evaluation
                bid = 0
                ncorrect = 0

                while bid < n_tgt_test:
                    batch = []
                    labels = []

                    while len(batch) < bsize and bid < n_tgt_test:
                        ex = tgt_data[tgt_test_split][bid]
                        inputs = tokenizer.apply_chat_template(
                            messages + [
                                {'role': 'user', 'content': test_prompt.format(ex[tgt_field])},
                                {'role': 'system', 'content': "Sentiment:\n"}
                            ],
                            add_generation_prompt=False,
                            return_tensors='pt',
                        ).cuda()
                        batch.append({'input_ids': inputs[0][:-1]})
                        labels.append(tgt_label_map[ex['label']])
                        bid += 1

                    ex_lens = [len(bid['input_ids']) - 1 for bid in batch]
                    batch = DataCollatorWithPadding(tokenizer)(batch)
                    batch = {k: v.cuda() for k, v in batch.items()}
                    out = model(**batch)

                    preds = out.logits[torch.arange(out.logits.shape[0]), ex_lens][:, tok_choices].argmax(-1).cpu().numpy()
                    ncorrect += (preds == labels).sum()
                    del out

                res[n_log_id, n_log_ood, i] = ncorrect / n_tgt_test

            print(res.mean(-1))
            print(res.var(-1))
            np.save(out_name + '.npy', res, allow_pickle=True)

if __name__ == "__main__":
    app()
