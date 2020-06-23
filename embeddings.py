from transformers import BertModel, BertTokenizer
import torch
from torch import nn

def create_embedding(sentence):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states

tokenizer = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased-ner')
model = BertModel.from_pretrained('KB/bert-base-swedish-cased-ner')

emb1 = create_embedding('Detta är ett test som jag hoppas fungerar bra')
emb2 = create_embedding('Detta är ett test som jag hoppas fungerar jättebra')

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
sim = cos(emb1, emb2)
print('Resulting similarity:', sim.mean().item())