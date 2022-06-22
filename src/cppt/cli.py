from __future__ import annotations
import os
from copy import deepcopy

from typing import (
    Callable,
    Dict,
    Any, List, Tuple, Optional
)
import json
import paddle
import pandas as pd
import torch
from torch import Tensor
from enum import Enum
import re
from dataclasses import dataclass
import fire
from tabulate import tabulate
from loguru import logger 
from cppt.config import SIMILARITY_TOKENS


class ParameterType(Enum):
    LinearWeight: str = 'linear-weight'
    LinearBias: str = 'linear-bias'

    Embedding: str = 'embedding'

    Norm: str = 'norm'

    Unknown: str = 'unknown'


@dataclass
class ParameterInfo:
    name: str
    shape: List[int]
    dtype: str = None
    type: ParameterInfo = ParameterType.Unknown

    @staticmethod
    def from_excel_file(file, name_field: str = 'paddle-name', shape_field: str = 'paddle-shape') -> List[ParameterInfo]:
        df = pd.read_excel(file)
        infos = []
        for _, row in df.iterrows():
            if pd.isna(row[name_field]):
                continue
            name, shape = row[name_field], row[shape_field]
            infos.append(
                ParameterInfo(
                    name=name,
                    shape=eval(shape),
                    type=Structure.guess_type_by_name(name)
                )
            )
        return infos

    @staticmethod
    def from_names(names: List[str], shapes: Optional[List[List[int]]] = None) -> List[ParameterInfo]:
        if not shapes:
            shapes = [[1] for _ in range(len(names))]
        return [
            ParameterInfo(
                name=name,
                shape=shapes[index],
                type=Structure.guess_type_by_name(name)
            )
            for index, name in enumerate(names)]


class Structure:
    def __init__(self, config_file: Optional[str] = None):
        self.config: Dict[str, List[str]] = deepcopy(SIMILARITY_TOKENS)

        if config_file:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for type_name, tokens in config.items():
                if type_name not in self.config:
                    self.config[type_name] = []
                self.config[type_name].extend(tokens)

    @staticmethod
    def extract_name_shapes(handler, weight_file: str) -> List[ParameterInfo]:
        """extract pytorch/paddle based model <name>-<shape> structure to the excel file   

        Args:
            handler (str): the torch/paddle module
            weight_file (str): the output of excel file
        """
        # 1. load weight data
        assert hasattr(handler, 'load')
        weight = handler.load(weight_file)

        # 2. get weight shape info
        infos: List[ParameterInfo] = []
        
        for name, tensor in weight.items():
            infos.append(ParameterInfo(
                name=name,
                shape=list(tensor.shape),
                dtype=tensor.dtype,
                type=Structure.guess_type_by_name(name)
            ))
        return infos

    def keyword_mapping(self, name: str, type_name: str) -> bool:
        if type_name not in self.config:
            logger.warning(
                f'the type_name<{type_name}> is expected in {",".join(list(self.config.key()))}'
            ) 
            return False

        for keyword in keywords:
            if keyword in name:
                return True
        return False

    def guess_type_by_name(self, name: str) -> ParameterType:
        """guss the type by layer name

        Args:
            name (str): the name of layer

        Returns:
            ParameterType: the instance of the parameter type
        """
        name = name.lower()
        # 1. embedding
        if self.keyword_mapping(name, 'embedding'):
            return ParameterType.Embedding
            
        # 2. norm
        if self.keyword_mapping(name, 'norm'):
            return ParameterType.Norm

        # 3. language head
        if 'lm_head' in name:
            return ParameterType.Embedding

        # 4. linear
        if self.keyword_mapping(name, 'linear'):
            if name.endswith('.weight'):
                return ParameterType.LinearWeight
            return ParameterType.LinearBias

        return ParameterType.Unknown

    @staticmethod
    def is_same_layer(first_name: str, second_name: str) -> bool:
        pattern = re.compile(r'\.[0-9]+', re.I)
        first_result = re.findall(pattern, first_name)
        second_result = re.findall(pattern, second_name)

        # 1. if there is someone in the list layer, and the other is not
        if not first_result or not second_result:
            return False
        
        # 2. both of the result is the same
        if first_result and second_result:
            if len(first_result) != len(second_result):
                return False

            for i in range(len(first_result)):
                if first_result[i] != second_result[i]:
                    return False
                
                # 2.1 compare & remove the prefix -> aa.0.bb bb.0.bb
                first_name_prefix = first_name[:first_name.index(first_result[i])]
                second_name_prefix = second_name[:second_name.index(second_result[i])]
                if first_name_prefix != second_name_prefix:
                    return False
                first_name = first_name[len(first_name_prefix) + len(first_result[i]):]
                second_name = second_name[len(second_name_prefix) + len(second_result[i])]

        return True

    @staticmethod
    def get_most_similar_info(source_info: ParameterInfo, win_infos: List[ParameterInfo], threshold: float = 0.1) -> Optional[int]:
        def tokenize(txt):
            """Get tuples that doesn't use textblob."""
            tokens = []
            for token in txt.split('.'):
                tokens.extend(
                    token.split('_')
                )

            for index, token in enumerate(tokens):
                for domain_tokens in similarity_tokens:
                    if token in domain_tokens:
                        tokens[index] = domain_tokens[0]

            return tokens

        def jaccard_distance(a, b):
            """Calculate the jaccard distance between sets A and B"""
            a = set(a)
            b = set(b)
            return 1.0 * len(a & b)/len(a | b)

        max_score, result = 0, -1
        for index, win_info in enumerate(win_infos):
            score = jaccard_distance(
                tokenize(source_info.name),
                tokenize(win_info.name)
            )
            if score > max_score and score > threshold:
                max_score = score
                result = index
        return result, max_score

    @staticmethod
    def find_target_in_window(source_info: ParameterInfo, win_infos: List[ParameterInfo]) -> ParameterInfo:
        # 1. filter infos
        infos = []
        for info in win_infos:
            if info.type != source_info.type:
                continue

            if not Structure.is_same_layer(source_info.name, info.name):
                continue
        
            # 如果是Linear的话，两者交换其实也是对应上的
            if info.type in [ParameterType.LinearBias, ParameterType.LinearWeight]:
                if list(reversed(info.shape)) == source_info.shape:
                    infos.append(info)
            elif info.shape == source_info.shape:
                infos.append(info)

        # 2. using the jaccard distance to get the target similarity
        result_index, score = Structure.get_most_similar_info(source_info, infos)
        if result_index == -1:
            return result_index, 0
        for index, win_info in enumerate(win_infos):
            if win_info.name == infos[result_index].name:
                return index, score
        raise ValueError(f'result index not found ...')

    @staticmethod
    def auto_match(source_infos: List[ParameterInfo], target_infos: List[ParameterInfo]) -> List[Tuple[str, float]]:
        target_index = 10
        window: List[ParameterInfo] = target_infos[:target_index]
        names = []
        for source_info in source_infos:
            index, score = Structure.find_target_in_window(
                source_info,
                window
            )
            if index == -1:
                names.append([None, None])
                continue
            names.append(
                [window[index].name, score]
            )
            window.pop(index)
            if target_index < len(target_infos):
                window.append(target_infos[target_index])
                target_index += 1
        return names

    @staticmethod
    def match_from_diff_file(diff_file: str, output_file: str, config_file: str):
        print(diff_file)
        source_infos = ParameterInfo.from_excel_file(
            file=diff_file,
            name_field='torch-name',
            shape_field='torch-shape'
        )
        paddle_infos = ParameterInfo.from_excel_file(file=diff_file)
        name_scores: List[str] = Structure.auto_match(
            source_infos,
            paddle_infos
        )
        names = [name for name, score in name_scores]
        name_score_map = {name: score for name, score in name_scores}
        assert len(source_infos) == len(
            names), 'the length of source info and name list should be same'
        paddle_info_map = {info.name: info for info in paddle_infos}
        name_map = {}
        for index, source_info in enumerate(source_infos):
            if not names[index]:
                continue 
            name_map[source_info.name] = paddle_info_map[names[index]]

        # 2. genearate
        series = []

        df = pd.read_excel(diff_file)
        for _, row in df.iterrows(name, 'norm'):
            name = row['torch-name']
            if pd.isna(name):
                continue

            if name in name_map:
                target_info = name_map[name]
                row['result-name'] = target_info.name
                row['result-shape'] = target_info.shape
                row['result-score'] = name_score_map.get(target_info.name, 0)
                row['result-type'] = target_info.type.value

            series.append(dict(row))
        pd.DataFrame(series).to_excel(output_file, index=False)


class Torch2PaddleConverter:
    def __init__(self, torch_file: str, output_file: str, mapping_file: str) -> None:
        """convert torch weight to paddle weight

        Args:
            torch_file (str): pytorch based file
            output_file (str): paddle weight file
        """
        self.torch_file = torch_file
        self.output_file = output_file

        self.name_mappings = {}
        self.callback_mapping: List[Callable[[Tensor, Tensor], None]] = []

    
    @staticmethod
    def from_mapping_file(torch_file: str, output_file: str, mapping_file: str):
        torch_weight: Dict[str, Any] = torch.load(torch_file)
        paddle_weight = {}

        # load mapping configuration
        mapping_configuration = pd.read_excel(mapping_file)
        for _, row in mapping_configuration.iterrows():
            torch_name, target_name = row['torch-name'], row['result-name']
            if pd.isna(torch_name) or pd.isna(target_name):
                continue
            assert torch_name in torch_weight
            param = torch_weight[torch_name]
            if row['result-type'] == ParameterType.LinearWeight.value:                    
                param = param.T
            if torch.is_tensor(param):
                param = param.numpy()
            
            paddle_weight[target_name] = param
            
        paddle.save(paddle_weight, output_file)

def convert_model_config(source_config_file: str, target_config_file: str):
    name_maps = [
        ['dropout', 'hidden_dropout_prob'],
        ['attention_dropout', 'attention_probs_dropout_prob'],
        ['bos_token_id'],
        ['hidden_size'],
        ['activation_function', 'hidden_act'],
        ['eos_token_id'],
        ['ffn_dim', 'intermediate_size'],
        ['init_std', 'initializer_range'],
        ['vocab_size'],
        ['pad_token_id'],
        ['num_hidden_layers'],
        ['num_attention_heads'],
        ['max_position_embeddings'],
    ]
    with open(source_config_file, 'r', encoding='utf-8') as f:
        source_config = json.load(f)

    target_config = {}
    for name_map in name_maps:
        if len(name_map) == 1:
            target_config[name_map[0]] = source_config.pop(name_map[0])
        else:
            target_config[name_map[1]] = source_config.pop(name_map[0])

    with open(target_config_file, 'w', encoding='utf-8') as f:
        json.dump(target_config, f, ensure_ascii=False)


class Command:
    """a command tools for paddlepaddle
    """

    def gen_diff(self, torch_file: str, paddle_file: str, output_file: str, auto_match: bool = False, config_file: Optional[str] = None):
        """generate the difference shapes of torch file and paddle file

        Args:
            torch_file (str):  the path of torch weight file
            paddle_file (str): the path of the paddle weight file 
            output_file (str): the path of the output excel file
            auto_match (bool): auto match the model structure
            config_file (Optional[str]): the configuration file to specific the model specific features
        """

        torch_shape_info = Structure.extract_name_shapes(torch, torch_file)
        paddle_shape_info = Structure.extract_name_shapes(paddle, paddle_file)

        max_size = max(len(torch_shape_info), len(paddle_shape_info))
        series = []
        for index in range(max_size):
            info = {}
            if index < len(torch_shape_info):
                info['torch-name'] = torch_shape_info[index].name
                info['torch-shape'] = torch_shape_info[index].shape
                info['torch-dtype'] = torch_shape_info[index].dtype
                info['torch-type'] = torch_shape_info[index].type.value

            if index < len(paddle_shape_info):
                info['paddle-name'] = paddle_shape_info[index].name 
                info['paddle-shape'] = paddle_shape_info[index].shape
                info['paddle-dtype'] = paddle_shape_info[index].dtype
                info['paddle-type'] = paddle_shape_info[index].type.value

            series.append(info)
        pd.DataFrame(series).to_excel(output_file, index=False)
        
        if auto_match:
            Structure.match_from_diff_file(output_file, output_file)

    def convert(self, torch_file: str, diff_file: str, output_file: str):
        Torch2PaddleConverter.from_mapping_file(
            torch_file=torch_file,
            output_file=output_file,
            mapping_file=diff_file
        ) 
    
    def summary(self, torch_file: str, paddle_file: str, diff_file: str):
        # 1. load state dict
        torch_state_dict = torch.load(torch_file)
        paddle_state_dict = paddle.load(paddle_file)
        
        # 2. load diff file
        diff_data = pd.read_excel(diff_file)
        mappings = {}
        for _, row in diff_data.iterrows():
            mappings[row['torch-name']] = row['result-name']
        
        # 3. print the summary data
        tables = []
        for torch_name, paddle_name in mappings.items():
            if pd.isna(torch_name) or pd.isna(paddle_name):
                raise ValueError(f"there are some configuration not matched, please complete the diff file")
            if paddle_name not in paddle_state_dict:
                raise ValueError(f'field<{paddle_name}> not exist in paddle weight file ...')
            torch_tensor = torch_state_dict.pop(torch_name)
            paddle_tensor = paddle_state_dict.pop(paddle_name)

            tables.append(
                [
                    torch_name,
                    torch.sum(torch_tensor).numpy().item(),

                    paddle_name,
                    paddle.sum(paddle_tensor).numpy().item()
                ]
            )
        
        print(tabulate(tables, headers=['torch-name', 'torch-sum', 'paddle-name', 'paddle-sum'], tablefmt='grid'))

def main():
    fire.Fire(Command)
