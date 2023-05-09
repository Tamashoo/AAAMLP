import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        config = transformers.BertConfig.from_json_file("Chapter-10/input/bert_base_uncased/bert_config.json")
        self.bert = transformers.BertForPreTraining(config)
        #self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH, from_tf=True)
        #self.bert = transformers.BertModel.from_pretrained("bert-base-uncased")
        self.bert_drop = nn.Dropout(0.3)
        #self.out = nn.Linear(768, 1)
        self.out = nn.Linear(30522, 1)
    
    def forward(self, ids, mask, token_type_ids):
        lhs, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        #print(f"lhs.shape: {lhs.shape}") lhs.shape: torch.Size([4, 512, 30522])
        #print(f"o2.shape: {o2.shape}") o2.shape: torch.Size([4, 2])
        o2 = lhs[:, 0, :]
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
    
"""    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        #print(f"o2.shape: {o2.shape}")
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output.squeeze(1) """