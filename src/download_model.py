from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'google/gemma-7b'
access_token = "your-api-key-here"
mod = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
tok = AutoTokenizer.from_pretrained(model_name, token=access_token)

mod.half().save_pretrained('gemma7bhf', safe_serialization=False)
tok.save_pretrained('gemma7bhf')