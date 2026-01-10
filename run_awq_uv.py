import os
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/home/ubuntu/OpenCharacterTraining/models/elias_vance_merged'
quant_path = '/home/ubuntu/OpenCharacterTraining/models/elias_vance_merged-awq'

print(f'Loading model from {model_path}...')
try:
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print('Quantizing model (w_bit=4, q_group_size=128)...')
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
    model.quantize(tokenizer, quant_config=quant_config)

    print(f'Saving quantized model to {quant_path}...')
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    print("AWQ Quantization Success!")

except Exception as e:
    print(f"FAILED: {e}")
    # Print more debug info if possible
    import traceback
    traceback.print_exc()
