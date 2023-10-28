import pandas as pd
import logging
import os
from os import truncate
from typing import List, Optional, Tuple, Union

import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import json
from dataclasses import dataclass, asdict
from multiprocessing import Pool
import multiprocessing
import math
from random import sample
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList
)
import random


logger = logging.getLogger(__name__)

def read_data(file_name):
    f = open(file_name, 'r', encoding='utf-8').readlines()
    data = [json.loads(d) for d in f]
    inputs = []
    targets = []
    task_type = []
    choice = []
    for index, d in enumerate(data):
        if isinstance(d['target'], list):
            if len(d['target']) < 1:
                continue
        else:
            if pd.isnull(d['target']) or pd.isna(d['target']):
                continue
        inputs.append(d['input'])
        targets.append(d['target'])
        if 'task_type' in d:
            task_type.append(d['task_type'])
        else:
            task_type.append('')
        if 'choice' in d:
            choice.append(d['choice'])
        else:
            choice.append('')
    negative = {}
    for i, c in zip(inputs, choice):
        negative[i] = c
    dict_ = {'input': inputs, 'output': targets, 'task_type': task_type, "choice": choice}
    df_data = pd.DataFrame(dict_)
    df_data.dropna(axis=0, how='any')

    return df_data, negative


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False
    
def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


class Seq2SeqDataset(Dataset):
    def __init__(self, data):
        inputs = list(data["input"])
        outputs = list(data['output'])
        # choices = list(data['choice'])
        self.examples = [[i, o] for i, o in zip(inputs, outputs)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

class Seq2SeqCollator(object):
    def __init__(self, args, tokenizer, negative, mode="train"):
        self.tokenizer = tokenizer
        self.args = args    
        self.mode = mode
        self.negative = negative

    def __call__(self, batch):
        if self.args.training_mode in ["choice", "match"]:
            inputs, targets, choices = NegativeSampling(batch, self.negative, self.args)
        else:
            inputs = [MultiChoicePrompting(self.args, d[0], "") for d in batch]
            targets = [d[1] for d in batch]
            choices = [None] * len(inputs)
        if self.mode == "dev":
            max_length = self.args.max_length if self.args.model_type == "decoder" else self.args.max_seq_length
            tokenized_inputs = self.tokenizer(inputs, max_length=max_length, 
                                  truncation=True, padding=True, return_tensors='pt')
            return tokenized_inputs, targets, choices
        
        inputs = preprocess_data_batch([inputs, targets], self.tokenizer, self.args)

        return inputs
    
def MultiChoicePrompting(args, input, choice):
    return args.prompt.format(question=input, choice=choice)


def preprocess_data_batch(data, tokenizer, args):
    inputs, targets = data
    new_targets = []
    for t in targets:
        t = t.split("\t")
        if len(t) > 5:
            t_ = random.sample(t, 5)
            t = "\t".join(t_)
            new_targets.append(t)
        else:
            new_targets.append("\t".join(t))
    targets = new_targets

    if args.model_type == "decoder":
        if args.mode == "pretrain":
            inputs = tokenizer(
                inputs,
                max_length=args.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            labels = inputs['input_ids'].clone().contiguous()
            labels[labels[:, :] == tokenizer.pad_token_id] = -100
            type_token_ids = inputs['attention_mask'].long()
            inputs['labels'] = labels
            inputs["type_token_ids"] = type_token_ids
            return inputs
            
        # decoder-only model
        inputs = tokenizer(
            inputs
        )
        targets = tokenizer(
            targets,
            add_special_tokens=False,
        )
        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        concat_input = [input_ids[i] + target_ids[i] for i in range(len(input_ids))]
        if not args.open_ended:
            concat_input = [c_ids + [tokenizer.eos_token_id] for c_ids in concat_input]
        concat_input = [c_[: args.max_length] for c_ in concat_input]

        type_token_ids = [[0] * min(len(concat_input[i]), len(input_ids[i])) + [1] * (len(concat_input[i]) - len(input_ids[i])) for i in range(len(input_ids))]
        attention_mask = [[1] * len(concat_input[i]) for i in range(len(input_ids))]
        
        max_batch_length = 0
        for i in range(len(input_ids)):
            max_batch_length = max(max_batch_length, len(type_token_ids[i]))
        type_token_ids = [[0] * (max_batch_length - len(ids)) + ids for ids in type_token_ids]
        attention_mask = [[0] * (max_batch_length - len(ids)) + ids for ids in attention_mask]
        concat_input = [[tokenizer.pad_token_id] * (max_batch_length - len(ids)) + ids for ids in concat_input]
        type_token_ids = torch.Tensor(type_token_ids).long()
        attention_mask = torch.Tensor(attention_mask).long()
        concat_input = torch.Tensor(concat_input).long()
        labels = concat_input.clone().contiguous()
        labels[type_token_ids[:, :] == 0] = -100
        if "chatglm" in args.model_name_or_path and not "chatglm2" in args.model_name_or_path:
            attention_mask = attention_mask.bool()
        return {
            "input_ids": concat_input,
            "attention_mask": attention_mask,
            "type_token_ids": type_token_ids,
            "labels": labels
        }
    else:
        ## encoder-decoder model
        inputs = tokenizer(
            inputs,
            max_length=args.max_seq_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        targets = tokenizer(
            targets,
            max_length=args.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        target_ids = targets['input_ids']
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids[:, :] == tokenizer.pad_token_id] = 0
        type_token_ids = torch.ones_like(target_ids)
        type_token_ids[target_ids[:, :] == tokenizer.pad_token_id] = 0
        labels = target_ids.clone().contiguous()
        labels[target_ids[:, :] == tokenizer.pad_token_id] = -100
        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
            "attention_mask": torch.LongTensor(attention_mask),
            "type_token_ids": torch.LongTensor(type_token_ids)
        }


def NegativeSampling(batch, ori_negative, args):
    '''
    inputs: list(questions)
    target
    negative: {q: a} dict, a = [[alias + answer]]
    '''
    negative = ori_negative.copy()
    inputs = [d[0] for d in batch]
    targets = [getCWQAnswer(d[1]) for d in batch]
    self_answer = [negative.pop(inputs[i]) if inputs[i] in negative else targets[i].split(", ") for i in range(len(inputs))]
    negative_choices = list(negative.values())
    choosed_samples = [random.sample(negative_choices, 2) for i in range(len(inputs))]
    choosed_answers = [[getCWQAnswer(a[0]), getCWQAnswer(a[1])] for a in choosed_samples]
    if args.training_mode == 'match':
        answers = [getCWQAnswer(t) for t in self_answer]
        inputs_prompted = [args.prompt.format(question=i, choice=c) for i, c in zip(inputs, answers)]
        inputs_negative = [args.prompt.format(question=i, choice=c[0]) for i, c in zip(inputs, choosed_answers)]
        final_inputs = inputs_prompted + inputs_negative
        final_targets = [f'''Yes, the answer is {t}''' for t in answers] +[f'''No, the answer should be {t}''' for t in answers]
    elif args.training_mode == "choice":
        one_correct_answers = [[a] + random.sample(c, 1) for a, c in zip(targets, choosed_answers)]
        for shuffle in one_correct_answers:
            random.shuffle(shuffle)
        one_correct_answers = [' or '.join(a) for a in one_correct_answers]
        choosed_answers = [' or '.join(a) for a in choosed_answers]
        inputs_prompted = [args.prompt.format(question=i, choice=c) for i, c in zip(inputs, one_correct_answers)]
        inputs_negative = [args.prompt.format(question=i, choice=c) for i, c in zip(inputs, choosed_answers)]
        final_inputs = inputs_prompted + inputs_negative
        final_targets = targets + [f'''Neither, the correct answer should be {a}''' for a in targets]
    return final_inputs, final_targets, self_answer + self_answer

def getCWQAnswer(list_answers):
    if isinstance(list_answers, str):
        list_answers = list_answers.split("\t")
        list_answers = [[d] for d in list_answers]
    this_answer = []
    for alias_answer in list_answers:
        this_answer.append(random.choice(alias_answer))
    if len(this_answer) > 5:
        this_answer = random.sample(this_answer, 5)
    this_answer = list(set(this_answer))
    return ", ".join(this_answer)



@dataclass
class ModelArgs:
    model_type: str = "decoder"
    model_name_or_path: str = None
    checkpoint_dir: str = None
    output_dir: str = None
    data_dir: str = None
    deepspeed_config = None
    do_train: bool = True
    do_eval: bool = False
    num_train_epochs = 10
    warmup_ratio: float = 0.1
    warmup_steps: int = None
    save_steps: int = 500
    weight_decay: float = 0.0
    max_seq_length: int = 96
    max_length: int = 32
    num_beams: int = 1
    do_sample: bool = False
    top_k: int = None
    top_p: float = None
    learning_rate: float = 3e-5
    preprocess_inputs: bool = True
    clip_norm: float = 1.0
    open_ended: bool = False
    batch_size: int = 32
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    lora: bool = True
    lora_dim: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_module_name: str = 'q_proj,k_proj,v_proj,query_key_value'
    seed: int = 42
    offload_optimizer: bool = False
    deepspeed_config: str = None
    zero_shot: bool = False
    mode: str = "sft"
    gradient_checkpointing: bool = False
    prompt: str = "Which is the correct answer for the question {question}\n{choice}\nAssistant:"
    training_mode: str = None
    num_return_sequences: int = 1

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            f.write(json.dumps(asdict(self), indent=5))

    def update(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))