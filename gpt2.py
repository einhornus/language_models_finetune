from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
import os
os.environ["WANDB_DISABLED"] = "true"

model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="F://hf_cache")

"""
from transformers import pipeline
chef = pipeline('text-generation', model="models//bible_gpt", tokenizer='gpt2')
result = chef('I love to be your little slut')[0]['generated_text']
print(result)
exit(0)
"""

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data//bible_small_train.txt",
    block_size=128)

test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data//bible_small_test.txt",
    block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="F://hf_cache")
training_args = TrainingArguments(
    output_dir="models//bible_gpt",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model()
