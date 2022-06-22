from __future__ import annotations

from fire import Fire
import logging

logger = logging.getLogger(__name__)


class Commands:
    def gen_weight(self):
        # 1. gen paddle random bert weight file
        logger.info('init & save paddle weight file ...')
        from paddlenlp.transformers.bert.modeling import BertForPretraining, BertModel
        model_name = 'bert-base-uncased'

        model = BertModel(**BertModel.pretrained_init_configuration[model_name]) 
        model.save_pretrained(f'./paddle/{model_name}')
        
        # 2. gen torch pretrained model weight file

        from transformers.models.bert.modeling_bert import BertModel, BertConfig
        bert_config = BertConfig.from_pretrained(model_name)
        torch_model = BertModel(bert_config)
        torch_model.save_pretrained(f'./torch/{model_name}')

if __name__ == "__main__":
    Fire(Commands)
