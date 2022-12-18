from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
import os
import torch
import time
import HuggingFaceLanguageModel
from transformers import pipeline

os.environ["WANDB_DISABLED"] = "true"
torch.zeros(1).cuda()

#model_name = "facebook/opt-350m"
model_name = "sberbank-ai/rugpt3large_based_on_gpt2"


#lm = HuggingFaceLanguageModel.HuggingFaceLanguageModel("models//bible_gpt", model_name, use_cuda=True)
lm = HuggingFaceLanguageModel.HuggingFaceLanguageModel(model_name, model_name, use_cuda=True)

for i in range(1):
    t1 = time.time()
    result = lm.predict("Александр Сергеевич Пушкин", output_length=100, temperature=0.01)
    t2 = time.time()
    print(t2-t1, result)
exit(0)


tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data//bible_train.txt",
    block_size=32)

test_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="data//bible_test.txt",
    block_size=32)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

model = AutoModelForCausalLM.from_pretrained(model_name)
training_args = TrainingArguments(
    output_dir="models//bible_gpt",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    log_level="debug",
    logging_steps=1,
    no_cuda=False,
    fp16=True,
    save_strategy='no',
    evaluation_strategy='steps',
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
