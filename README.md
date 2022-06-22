# paddle-cppt

Convert Pytorch to Paddle Toolkit

This Repo contains the toolkit that help you transform the pytorch model to paddle model. eg: Weight file Comparer, Weight Converter, Weight Summary, and so on ...

## Features

* generate the diff between paddle weight and torch weight file ...
* convert the torch weight file to paddle weight file ...
* summary the info between paddle weight and torch weight file ...

## Getting Started

### Installation

```shell
pip install paddle-cppt
```

## gen diff

```shell
cppt gen_diff \
    --torch_file=/path/to/pytorch_model.bin \
    --paddle_file=/path/to/model_state.pdparams \
    --output_file=/path/to/diff.xlsx --auto_match
```
with this command, you will get the `diff.xlsx` file which contains the layer name of paddle/torch weight file, eg: 

|                     torch-name                    |  torch-shape |  torch-dtype  |   torch-type  |                 paddle-name                | paddle-shape |  paddle-dtype  |  paddle-type  |
|:-------------------------------------------------:|:------------:|:-------------:|:-------------:|:------------------------------------------:|:------------:|:--------------:|:-------------:|
| embeddings.position_ids                           | [1, 512]     | torch.int64   | embedding     | embeddings.word_embeddings.weight          | [30522, 768] | paddle.float32 | embedding     |
| embeddings.word_embeddings.weight                 | [30522, 768] | torch.float32 | embedding     | embeddings.position_embeddings.weight      | [512, 768]   | paddle.float32 | embedding     |
| embeddings.position_embeddings.weight             | [512, 768]   | torch.float32 | embedding     | embeddings.token_type_embeddings.weight    | [2, 768]     | paddle.float32 | embedding     |
| embeddings.token_type_embeddings.weight           | [2, 768]     | torch.float32 | embedding     | embeddings.layer_norm.weight               | [768]        | paddle.float32 | embedding     |
| embeddings.LayerNorm.weight                       | [768]        | torch.float32 | embedding     | embeddings.layer_norm.bias                 | [768]        | paddle.float32 | embedding     |
| embeddings.LayerNorm.bias                         | [768]        | torch.float32 | embedding     | encoder.layers.0.self_attn.q_proj.weight   | [768, 768]   | paddle.float32 | linear-weight |
| encoder.layer.0.attention.self.query.weight       | [768, 768]   | torch.float32 | linear-weight | encoder.layers.0.self_attn.q_proj.bias     | [768]        | paddle.float32 | linear-bias   |
| encoder.layer.0.attention.self.query.bias         | [768]        | torch.float32 | linear-bias   | encoder.layers.0.self_attn.k_proj.weight   | [768, 768]   | paddle.float32 | linear-weight |
| encoder.layer.0.attention.self.key.weight         | [768, 768]   | torch.float32 | linear-weight | encoder.layers.0.self_attn.k_proj.bias     | [768]        | paddle.float32 | linear-bias   |
| encoder.layer.0.attention.self.key.bias           | [768]        | torch.float32 | linear-bias   | encoder.layers.0.self_attn.v_proj.weight   | [768, 768]   | paddle.float32 | linear-weight |
| encoder.layer.0.attention.self.value.weight       | [768, 768]   | torch.float32 | linear-weight | encoder.layers.0.self_attn.v_proj.bias     | [768]        | paddle.float32 | linear-bias   |
| encoder.layer.0.attention.self.value.bias         | [768]        | torch.float32 | linear-bias   | encoder.layers.0.self_attn.out_proj.weight | [768, 768]   | paddle.float32 | linear-weight |
| encoder.layer.0.attention.output.dense.weight     | [768, 768]   | torch.float32 | linear-weight | encoder.layers.0.self_attn.out_proj.bias   | [768]        | paddle.float32 | linear-bias   |
| encoder.layer.0.attention.output.dense.bias       | [768]        | torch.float32 | linear-bias   | encoder.layers.0.linear1.weight            | [768, 3072]  | paddle.float32 | linear-weight |
| encoder.layer.0.attention.output.LayerNorm.weight | [768]        | torch.float32 | norm          | encoder.layers.0.linear1.bias              | [3072]       | paddle.float32 | linear-bias   |
| encoder.layer.0.attention.output.LayerNorm.bias   | [768]        | torch.float32 | norm          | encoder.layers.0.linear2.weight            | [3072, 768]  | paddle.float32 | linear-weight |
| encoder.layer.0.intermediate.dense.weight         | [3072, 768]  | torch.float32 | linear-weight | encoder.layers.0.linear2.bias              | [768]        | paddle.float32 | linear-bias   |
| encoder.layer.0.intermediate.dense.bias           | [3072]       | torch.float32 | linear-bias   | encoder.layers.0.norm1.weight              | [768]        | paddle.float32 | norm          |
| encoder.layer.0.output.dense.weight               | [768, 3072]  | torch.float32 | linear-weight | encoder.layers.0.norm1.bias                | [768]        | paddle.float32 | norm          |
| encoder.layer.0.output.dense.bias                 | [768]        | torch.float32 | linear-bias   | encoder.layers.0.norm2.weight              | [768]        | paddle.float32 | norm          |
| encoder.layer.0.output.LayerNorm.weight           | [768]        | torch.float32 | norm          | encoder.layers.0.norm2.bias                | [768]        | paddle.float32 | norm          |
| encoder.layer.0.output.LayerNorm.bias             | [768]        | torch.float32 | norm          | encoder.layers.1.self_attn.q_proj.weight   | [768, 768]   | paddle.float32 | linear-weight |



## convert torch model to paddle weight

```shell
cppt convert \
    --torch_file=/path/to/pytorch_model.bin \
    --output_file=/path/to/model_state.pdparams \
    --diff_file=/path/to/diff.xlsx
```

## print the summary info between torch and paddle model

```shell
cppt summary \
    --torch_file=/path/to/pytorch_model.bin \
    --output_file=/path/to/model_state.pdparams \
    --diff_file=/path/to/diff.xlsx
```

## Creators

- [@wj-Mcat](https://github.com/wj-Mcat) - Jingjing WU (吴京京)

## Copyright & License

- Code & Docs © 2022 wj-Mcat <wjmcater@gmail.com>
- Code released under the Apache-2.0 License
- Docs released under Creative Commons
