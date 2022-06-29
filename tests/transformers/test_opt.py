from __future__ import annotations
import os
from typing import Union, Tuple, Type
import json
import inspect
import paddle
from paddle import nn
from paddle.nn import Layer
import torch
from torch.nn import Module
import numpy as np

import pytest


class Config(dict):
    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, value)
        
        self._source_dict = data

    def to_dict(self):
        return self._source_dict

    @staticmethod
    def from_file(file: str) -> Config:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Config(data)
     

def load_state_dict(model: Union[Layer, Module], state_dict_file: str, prefix_name: str, handler):
    # 1. load the state dict
    state_dict = handler.load(state_dict_file)
    if prefix_name:
        state_dict = {name.replace(prefix_name, ''): param for name, param in state_dict.items() if name.startswith(prefix_name)}
    
    # 2. get the model name
    model_state_dict = {}
    # assert len(state_dict) == len(list(model.named_parameters()))
    for name, _ in model.named_parameters():
        assert name in state_dict
        model_state_dict[name] = state_dict[name]

    if isinstance(model, Layer):
        model.set_state_dict(model_state_dict)
    else:
        from transformers.models.opt.modeling_opt import OPTForCausalLM 
        if isinstance(model, OPTForCausalLM):
            model_state_dict['lm_head.weight'] = state_dict['lm_head.weight']

        model.load_state_dict(model_state_dict)

    return model

def init_class(class_type, kwargs: dict):
    parameters_dict = inspect.signature(class_type.__init__).parameters
    final_kwargs = {}
    for k, v in kwargs.items():
        if k in parameters_dict:
            final_kwargs[k] = v
    return class_type(**final_kwargs) 

def equal(first_np, second_np) -> bool:
    return np.allclose(first_np, second_np, atol=0.001)

@pytest.fixture
def torch_dir():
    return '/root/paddle/pretrained-models/opt-350m'

@pytest.fixture
def paddle_dir():
    return "/root/paddle/PaddleNLP/pretrained/paddle/opt-350m"

@pytest.fixture
def torch_model(torch_dir):
    from transformers.models.opt.modeling_opt import OPTForCausalLM
    torch_model = OPTForCausalLM.from_pretrained(torch_dir)
    torch_model.eval()
    return torch_model

@pytest.fixture
def paddle_model(paddle_dir):
    from paddlenlp.transformers.opt.modeling import OPTForCausalLM
    paddle_model = OPTForCausalLM.from_pretrained(paddle_dir)
    paddle_model.eval()
    return paddle_model

@pytest.fixture()
def tokenzier(torch_dir):
    from transformers.models.gpt2 import GPT2Tokenizer 
    return GPT2Tokenizer.from_pretrained(torch_dir)

@pytest.fixture
def torch_hidden_state():
    input_embedding = np.load('input.npy')
    torch_input_embedding = torch.tensor(input_embedding).float()
    return torch_input_embedding

@pytest.fixture
def paddle_hidden_state():
    input_embedding = np.load('input.npy')
    paddle_input_embedding = paddle.to_tensor(input_embedding, dtype=paddle.float32)
    return paddle_input_embedding

def test_embedding(torch_model, paddle_model, tokenzier):
    sentence = 'I love China & Paddle, and'
    features = tokenzier(sentence, max_length=15, return_attention_mask=True, return_tensors='np', return_position_ids=True)
    
    # get pytorch model embeddign
    word_embeddings = torch_model.model.decoder.embed_tokens(torch.tensor(features['input_ids']))
    word_embeddings = torch_model.model.decoder.project_in(word_embeddings)
    position_embeddings = torch_model.model.decoder.embed_positions(torch.tensor(features['attention_mask']))
    torch_embeddigns = (word_embeddings + position_embeddings).detach().numpy()

    embeddings = paddle_model.opt.embeddings(paddle.to_tensor(features['input_ids'])).numpy()
    assert equal(torch_embeddigns, embeddings)


@pytest.mark.parametrize("self_index", [
    0,1,2,3,4,5,6,7,8,9,10,11
])
def test_self_attention(self_index: int, paddle_model, torch_model) -> Tuple[Layer, Module]:
    """load paddle & torch self-attention layer

    Returns:
        Tuple[Layer, Module]: paddle layer and torch module 
    """
    # 1. load paddle word embedding 
    input_embedding = np.load('input.npy')
    torch_input_embedding = torch.tensor(input_embedding).float()
    paddle_input_embedding = paddle.to_tensor(input_embedding, dtype=paddle.float32)
    
    torch_result = torch_model.model.decoder.layers[self_index].self_attn(torch_input_embedding)[0].detach().numpy()
    paddle_result = paddle_model.opt.decoder.layers[self_index].self_attn(
        paddle_input_embedding, 
        paddle_input_embedding, 
        paddle_input_embedding
    ).numpy()

    if isinstance(paddle_result, (list, tuple)):
        paddle_result = paddle_result[0]

    assert equal(torch_result, paddle_result)


@pytest.mark.parametrize("self_index", [
    0,1,2,3,4,5,6,7,8,9,10,11
])
def test_decoder_layer(self_index: int, torch_model, paddle_model, torch_hidden_state, paddle_hidden_state):
    # input_embedding = np.load('input.npy')
    torch_result = torch_model.model.decoder.layers[self_index](torch_hidden_state)[0].detach().numpy()
    paddle_result = paddle_model.opt.decoder.layers[self_index](
        paddle_hidden_state,
        memory=None
    ).numpy()

    if isinstance(paddle_result, (list, tuple)):
        paddle_result = paddle_result[0]

    assert equal(torch_result, paddle_result)

def test_opt_lm_head_model(torch_model, paddle_model, torch_hidden_state, paddle_hidden_state):

    torch_hidden_state= torch_hidden_state.reshape([-1, 768]) 
    paddle_hidden_state = paddle_hidden_state.reshape([-1, 768]) 

    torch_result = torch_model.lm_head(torch_hidden_state).detach().numpy()
    paddle_result = paddle_model.lm_head(paddle_hidden_state).numpy()

    if isinstance(paddle_result, (list, tuple)):
        paddle_result = paddle_result[0]

    assert equal(torch_result, paddle_result)
