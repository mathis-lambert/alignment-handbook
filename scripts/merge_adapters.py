from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, set_seed
import torch
import transformers

print("merging")
peft_config = PeftConfig.from_pretrained("./data/zephyr-7b-dpo-lora", device_map="auto")

model_kwargs = dict(
    use_flash_attention_2=True,
    torch_dtype="auto",
    use_cache=False,
)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    **model_kwargs,
     device_map="auto"
)
model = PeftModel.from_pretrained(
    base_model, "./data/zephyr-7b-dpo-lora",
     device_map="auto"
)
model.to("cuda")
model.eval()
model = model.merge_and_unload()
model_kwargs = None

model.save_pretrained("./data/zephyr_7b_lora_merged")
print("merged and saved")