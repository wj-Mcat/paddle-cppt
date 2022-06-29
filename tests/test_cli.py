from __future__ import annotations
from typing import List
import pytest
from copy import deepcopy
from cppt.cli import Structure, ParameterType, ParameterInfo


@pytest.fixture
def structure() -> Structure:
    return Structure()


def test_is_same_layer(structure: Structure):
    first_name, second_name = 'aa.0.bb', 'aa.0.cc'
    assert structure.is_same_layer(first_name, second_name)

    first_name, second_name = 'aa.0.bb', 'bb.0.cc'
    assert not structure.is_same_layer(first_name, second_name)

    first_name, second_name = 'aa.0.bb', 'aa.0.1.cc'
    assert not structure.is_same_layer(first_name, second_name)

    first_name, second_name = 'model.decoder.embed_tokens.weight', 'opt.embeddings.word_embeddings.weight'
    assert structure.is_same_layer(first_name, second_name)


def create_parama_infos_by_name(names: str) -> List[ParameterInfo]:
    infos = []
    for name in names.split():
        infos.append(
            ParameterInfo(name=name, type=ParameterType.Embedding, shape=[1,1])
        )
    return infos
        

def test_guess_name(structure: Structure):
    param_type: ParameterType = structure.guess_type_by_name('model.decoder.embed_positions.weight')
    assert param_type == ParameterType.Embedding

    param_type: ParameterType = structure.guess_type_by_name('model.decoder.embed_tokens.weight')
    assert param_type == ParameterType.Embedding


def test_similarity(structure: Structure):
    info = ParameterInfo(name='a.b.c', shape=[1,1], dtype='float32', type=ParameterType.Embedding)
    
    win_infos: List[ParameterInfo] = [
        ParameterInfo(name='b.c.d', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='c.d.e', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
    ]
    index, _ = structure.get_similarity(info, win_infos) 
    assert index == 0


def test_similarity_embedding_token(structure: Structure):
    info = ParameterInfo(name='gpt.word_embedding.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding)
    
    win_infos: List[ParameterInfo] = [
        ParameterInfo(name='opt.embed_embedding.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.embed_position.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
    ]
    index, _ = structure.get_similarity(info, win_infos) 
    assert index == 0
 

def test_find_target_in_window(structure: Structure):
    info = ParameterInfo(name='gpt.word_embedding.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding)
    
    win_infos: List[ParameterInfo] = [
        ParameterInfo(name='opt.embed_embedding.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.embed_position.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
    ]
    index, _ = structure.find_target_in_window(
        info,
        win_infos
    )
    assert index == 0

def test_find_target_in_window_qkv(structure: Structure):
    info = ParameterInfo(name='gpt.layers.q_proj.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding)
    
    win_infos: List[ParameterInfo] = [
        ParameterInfo(name='opt.layers.k_fc.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.layers.v_fc.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.layers.q_fc.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.layers.proj.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
    ]
    index, _ = structure.find_target_in_window(
        info,
        win_infos
    )
    assert index == 2


def test_find_target_in_window_boundary(structure: Structure):
    structure = deepcopy(structure)
    structure.prefix_words = ['model.', 'opt.']
    structure.threshold = 0.0

    weight_info = ParameterInfo(name='model.decoder.layers.0.self_attn.k_proj.weight', shape=[1,1], type=ParameterType.Embedding)
    bias_info = ParameterInfo(name='model.decoder.layers.0.self_attn.k_proj.bias', shape=[1,1], type=ParameterType.Embedding)

    win_infos: List[ParameterInfo] = [
        ParameterInfo(name='opt.decoder.layers.0.self_attn.q_proj.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.self_attn.q_proj.bias', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.self_attn.k_proj.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.self_attn.k_proj.bias', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.self_attn.v_proj.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.self_attn.v_proj.bias', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.self_attn.out_proj.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.self_attn.out_proj.bias', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        
        ParameterInfo(name='opt.decoder.layers.0.linear1.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.linear1.bias', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.linear2.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.linear2.bias', shape=[1,1], dtype='float32', type=ParameterType.Embedding),

        ParameterInfo(name='opt.decoder.layers.0.norm1.weight', shape=[1,1], dtype='float32', type=ParameterType.Embedding),
        ParameterInfo(name='opt.decoder.layers.0.norm2.bias', shape=[1,1], dtype='float32', type=ParameterType.Embedding),

    ]
    index, _ = structure.find_target_in_window(weight_info, win_infos)
    assert index == 2
    index, _ = structure.find_target_in_window(bias_info, win_infos)
    assert index == 3


def test_auto_match_simple(structure: Structure):
    source_infos = create_parama_infos_by_name("""model.decoder.final_layer_norm.weight""")

    win_infos = create_parama_infos_by_name("""opt.decoder.layers.0.self_attn.q_proj.weight
opt.decoder.layers.0.self_attn.q_proj.bias
opt.decoder.layers.0.self_attn.k_proj.weight
opt.decoder.layers.0.self_attn.k_proj.bias
opt.decoder.layers.0.self_attn.v_proj.weight
opt.decoder.layers.0.self_attn.v_proj.bias
opt.decoder.layers.0.self_attn.out_proj.weight
opt.decoder.layers.0.self_attn.out_proj.bias
opt.decoder.layers.0.linear1.weight
opt.decoder.layers.0.linear1.bias
opt.decoder.layers.0.linear2.weight
opt.decoder.layers.0.linear2.bias
opt.decoder.layers.0.norm1.weight
opt.decoder.layers.0.norm1.bias
opt.decoder.layers.0.norm2.weight
opt.decoder.layers.0.norm2.bias""")
    
    name_maps = structure.auto_match(
        source_infos,
        win_infos
    )
    assert len(name_maps) == 1
    assert name_maps[0] is not None

def test_auto_match(structure: Structure):
    source_infos = create_parama_infos_by_name("""model.decoder.layers.0.self_attn.k_proj.weight
model.decoder.layers.0.self_attn.k_proj.bias
model.decoder.layers.0.self_attn.v_proj.weight
model.decoder.layers.0.self_attn.v_proj.bias
model.decoder.layers.0.self_attn.q_proj.weight
model.decoder.layers.0.self_attn.q_proj.bias
model.decoder.layers.0.self_attn.out_proj.weight
model.decoder.layers.0.self_attn.out_proj.bias
model.decoder.layers.0.self_attn_layer_norm.weight
model.decoder.layers.0.self_attn_layer_norm.bias
model.decoder.layers.0.fc1.weight
model.decoder.layers.0.fc1.bias
model.decoder.layers.0.fc2.weight
model.decoder.layers.0.fc2.bias""")

    win_infos = create_parama_infos_by_name("""opt.decoder.layers.0.self_attn.q_proj.weight
opt.decoder.layers.0.self_attn.q_proj.bias
opt.decoder.layers.0.self_attn.k_proj.weight
opt.decoder.layers.0.self_attn.k_proj.bias
opt.decoder.layers.0.self_attn.v_proj.weight
opt.decoder.layers.0.self_attn.v_proj.bias
opt.decoder.layers.0.self_attn.out_proj.weight
opt.decoder.layers.0.self_attn.out_proj.bias
opt.decoder.layers.0.linear1.weight
opt.decoder.layers.0.linear1.bias
opt.decoder.layers.0.linear2.weight
opt.decoder.layers.0.linear2.bias
opt.decoder.layers.0.norm1.weight
opt.decoder.layers.0.norm1.bias
opt.decoder.layers.0.norm2.weight
opt.decoder.layers.0.norm2.bias""")
    
    name_maps = structure.auto_match(
        source_infos,
        win_infos
    )
    for name, score in name_maps:
        assert name is not None