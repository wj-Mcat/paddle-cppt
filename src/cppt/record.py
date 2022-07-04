from __future__ import annotations
import os
from collections import defaultdict
from typing import Union, List, Dict
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
import pickle
import torch
import paddle
import numpy as np
from tqdm import tqdm

from tabulate import tabulate


@dataclass_json
@dataclass
class Record:
    run_id: str
    name: str
    key: str
    sum_value: str
    logits: str
    last_logits: str


class TensorRecorder:
    def __init__(self, name: str, run_id: Optional[str] = None, record_file: Optional[str] = None):
        self.name = name

        self.run_id = run_id or os.environ.get("run_id", None) or '1'
        self.record_file = record_file or os.environ.get("record_file", None) or './records.json'

    def _save_record(self, record: Record):
        with open(self.record_file, 'a+', encoding='utf-8') as f:
            f.write(
               record.to_json() + '\n'
            )

    def add_str(self, key: str, data: str):
        record = Record(
            run_id=self.run_id, name=self.name,
            key=key,
            logits=data,
            last_logits=data,
            sum_value=data
        ) 
        self._save_record(record)
        
    def add(self, key: str, data):
        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()
        elif paddle.is_tensor(data):
            data = data.numpy()
        else:
            data = np.array(data)

        # add to the record file
        data: np.ndarray = data
        logits = data.reshape(-1)

        indice = min(10, len(logits))

        record = Record(
            run_id=self.run_id, name=self.name,
            key=key,
            sum_value=str(np.sum(data)),
            logits=str(logits[:indice]),
            last_logits=str(logits[-indice:])
        )
        self._save_record(record)    
    
class TensorComparer:
    def __init__(self, first_recorder: TensorRecorder, second_recorder: TensorRecorder):
        self.first_recorder = first_recorder
        self.second_recorder = second_recorder

        self.first_records: Dict[str, Record] = {} 
        
        with open(first_recorder.record_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                record: Record = Record.from_json(line)
                if record.name == first_recorder.name and record.run_id == first_recorder.run_id:
                    self.first_records[record.key] = record

        self.second_records: Dict[str, Record] = {}
        with open(second_recorder.record_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                record: Record = Record.from_json(line)
                if record.name == second_recorder.name and record.run_id == second_recorder.run_id:
                    self.second_records[record.key] = record

    def all_record_keys(self):
        keys = set()
        for key in self.first_records.keys():
            keys.add(key)

        for key in self.second_records.keys():
            keys.add(key)

        keys = list(keys)
        keys.sort() 

        return keys

    def compare(self):
        keys = self.all_record_keys()

        tables = []
        for key in keys:
            row_values = [key]

            # first logits
            logits, sum_value = [], .0
            if key in self.first_records:
                logits = self.first_records[key].logits
                sum_value = self.first_records[key].sum_value
            row_values.append(logits)
            row_values.append(sum_value)

            # second logits
            logits, sum_value = [], .0
            if key in self.second_records:
                logits = self.second_records[key].logits
                sum_value = self.second_records[key].sum_value
            row_values.append(logits)
            row_values.append(sum_value)

            tables.append(row_values)
             
            
        heads = [
            self.first_recorder.name + '-logit', self.first_recorder.name, '-sum-value',
            self.second_recorder.name + '-logit', self.second_recorder.name, '-sum-value',
        ]

        print(tabulate(
            tabular_data=tables,
            headers=heads,
            tablefmt='grid'
        ))