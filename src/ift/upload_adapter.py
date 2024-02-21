import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained('tmp')
tokenizer = AutoTokenizer.from_pretrained('22h/cabrita7b', devide_map='auto')

model = PeftModel.from_pretrained(model, 'adapter_3epochs')

hf_repository = '22h/cabrita7B-IFT-3epochs'
model.push_to_hub(hf_repository, repo_type='model', create_pr=True)

