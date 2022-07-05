# paddle-cppt

<div align="center" style="font-size: 30px">
Convert Pytorch to Paddle Toolkit
</div>

<div align="center">

[![wj-Mcat - paddle-cppt](https://img.shields.io/static/v1?label=wj-Mcat&message=paddle-cppt&color=blue&logo=github)](https://github.com/wj-Mcat/paddle-cppt "Go to GitHub repo")
[![stars - paddle-cppt](https://img.shields.io/github/stars/wj-Mcat/paddle-cppt?style=social)](https://github.com/wj-Mcat/paddle-cppt)
[![forks - paddle-cppt](https://img.shields.io/github/forks/wj-Mcat/paddle-cppt?style=social)](https://github.com/wj-Mcat/paddle-cppt)
[![PyPI](https://github.com/wj-Mcat/paddle-cppt/actions/workflows/python-publish.yml/badge.svg)](https://github.com/wj-Mcat/paddle-cppt/actions/workflows/python-publish.yml)
[![GitHub release](https://img.shields.io/github/release/wj-Mcat/paddle-cppt?include_prereleases=&sort=semver&color=blue)](https://github.com/wj-Mcat/paddle-cppt/releases/)
[![License](https://img.shields.io/badge/License-Apache_License_2.0-blue)](#license)
[![issues - paddle-cppt](https://img.shields.io/github/issues/wj-Mcat/paddle-cppt)](https://github.com/wj-Mcat/paddle-cppt/issues)


</div>


This Repo contains the toolkit that help you transform the pytorch model to paddle model. eg: Weight file Comparer, Weight Converter, Weight Summary, and so on ...

## Features

* `cppt gen_diff`: generate the diff between paddle weight and torch weight file, for more details you check out ...
* `cppt auto_match`: auto match the names of paddle and torch occording to the name semantic, eg: torch.model.embeddings.embed_token.weight -> paddle.opt.model.word_embeddings.weight ...
* `cppt convert`: convert the torch weight file to paddle weight file according to the diff file ...
* `cppt summary`: summary the tensor meta info according to the weight files and diff file ...

## Intro

In order to convert pytorch weight file to paddle weight file and make sure that the logits of paddle model is absolute align with pytorch, there are some steps you should follow.

* first, you should get the layer names of you paddle model. In this abastract, you can init the paddle model with the same configuration as pytorch model, and save the state_dict.
* second, in order to convert pytorch weight file to paddle weight file, you should find the name mapping between weight files, so you can load the state dict of paddle weight and pytorch weight file and find the diffs. In this abstract, you can use the command `cppt gen_diff` to find the diffs between layer names. 
* third, mapping the names between pytorch and paddle models is a boring work, so let's make it more intelligent. You can use `cppt auto_match` command to auto match the names with similarity algo. you can edit the final diff file, and make it correct. 
* fourth, you get the correct name mapping with third step, you can use the `cppt convert` command to convet the pytorch weight file to paddle weight file. In this abstract, the script will automaticly transpose the linear-weight tensor.
* finaly, in order to checking the tensor data of paddle weight file, you can use the command `cppt summary` the generate the meta info between paddle weight file.

So, it's cool right ? these codes help me `converting` work more soft.

But, there are also some great works can be done:

- [ ] compare the computing gragh between pytorch and paddle code. 
- [ ] compare the outputs of different layers, eg: embedding layer, transformer layer, lm_head layer and so on. 
- [ ] convert the pytorch code to paddle code using the ast. Objecoive： We can't convert it and make it run at onece, but you can convert it and edit it with simple works to make it run. 

If you have more ideas about it, you can post [issue](https://github.com/wj-Mcat/paddle-cppt/issues/new) to discuss with us. We look forward to discussing it with you. 

## Quick Start

### Installation

```shell
pip install paddle-cppt
```

or install from the source code:

```shell
git clone https://github.com/wj-Mcat/paddle-cppt
cd paddle-cpp
python setup.py install
```

### save paddle state dict

If you complete the code, you can init the paddle weight file with the following code:

```python
from paddlenlp.transformers.{your-model}.modeling import {YourModel}

# this code will be different according to different model, but anyway the final result is to save the state dict of model which contains the layer names of your model code.
model = {YourModel}(
    model={YourModel}(
        **{YourModel}.pretrained_init_configuration['{name-of-configuration}']
    )
)
model.save_pretrained('/path/to/local/dir')
```

### cppt gen_diff

this command will generate the name mapping 

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


## cppt auto_match

this command will generate the final name mapping into the excel file.

```shell
cppt auto_match \
    --diff_file=/path/to/diff.xlsx \
    --output_file=/path/to/diff-result.xlsx
```

## cppt convert

convert torch model to paddle weight according the final diff file.

```shell
cppt convert \
    --torch_file=/path/to/pytorch_model.bin \
    --output_file=/path/to/model_state.pdparams \
    --diff_file=/path/to/diff-result.xlsx
```

## cppt summary

print the summary metadata info between torch and paddle model

```shell
cppt summary \
    --torch_file=/path/to/pytorch_model.bin \
    --output_file=/path/to/model_state.pdparams \
    --diff_file=/path/to/diff.xlsx
```

## Creators


[![wj-Mcat - paddle-cppt](https://img.shields.io/static/v1?label=wj-Mcat&message=paddle-cppt&color=blue&logo=github)](https://github.com/wj-Mcat/paddle-cppt "Go to GitHub repo")


## License

Released under [Apache License 2.0](/LICENSE) by [@wj-Mcat](https://github.com/wj-Mcat).

## Copyright & License

- Code & Docs © 2022 wj-Mcat <wjmcater@gmail.com>
- Code released under the Apache-2.0 License
- Docs released under Creative Commons
