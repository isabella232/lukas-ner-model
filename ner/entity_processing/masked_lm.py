import torch
from transformers import pipeline, BertTokenizer, BertForMaskedLM

text = "Johannes Lindén är en slags handledare och ganska ofta frågar Lukas Johannes obegripliga frågor."

model_nam = "KB/bert-base-swedish-cased-ner"
model_name = "KB/bert-base-swedish-cased"
nlp = pipeline("ner", model=model_nam, tokenizer=model_nam)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()

entities = nlp(text)

[print(entity) for entity in entities]

masked_indexes = []
for i in range(len(entities)):
    if entities[i - 1]["index"] == entities[i]["index"] - 1:
        masked_indexes += [entities[i]["index"]]

splitted_text = ["[CLS]"] + text.split() + ["[SEP]"]

for i in masked_indexes:
    splitted_text[i] = "[MASK]"

masked_text = " ".join(splitted_text)

# masked_text = "[CLS] Den här maten var fantastisk, bland det [MASK] jag har ätit. [SEP]"

tokenized_text = tokenizer.tokenize(masked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
masked_indexes = [i for i, token in enumerate(tokenized_text) if token == "[MASK]"]
print(tokenized_text)
print(masked_indexes)

segments_ids = [0] * len(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

for masked_index in masked_indexes:
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)
