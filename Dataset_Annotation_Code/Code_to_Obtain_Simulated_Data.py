#Code to obtain the simulated data given an existing real-time data
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import json


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

#List used to dump the data from AspectBasedSentimentClassification_Dataset.json
data = []

#Read AspectBasedSentimentClassification_Dataset.json dataset and store it in a list 
with open('AspectBasedSentimentClassification_Dataset.json' , 'r') as f:
    for l in f.readlines():
        if not l.strip():
            continue
        jd = json.loads(l)
        data.append(jd)

f.close()

#Read the sentences from the list one by one and generate simulated data a specified number (num_return_sequences) of times.  
for i in range(0, len(data)):
  sent=data[i]['sentence']
  text =  "paraphrase: " + sent + " </s>"
  
  max_len = 256
  encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
  
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
  
  beam_outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True,
    max_length=256,
    top_k=120,
    top_p=0.98,
    early_stopping=True,
    num_return_sequences=5 #Number of sentences to return
)


  print("Paraphrase: ")
  
  #List used to store simulated data
  para_sents1 = []

  for i,line in enumerate(beam_outputs):
      paraphrase = tokenizer.decode(line,skip_special_tokens=True,clean_up_tokenization_spaces=True)
      para_sents1.append(paraphrase)
      print(f"{i+1}. {paraphrase}")