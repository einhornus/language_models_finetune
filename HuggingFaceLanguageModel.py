import LanguageModel as LM
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceLanguageModel(LM.LanguageModel):
    def __init__(self, name, tokenizer, use_cuda=False):
        self.use_cuda = use_cuda
        super().__init__(name)
        self.model = AutoModelForCausalLM.from_pretrained(name)
        if use_cuda:
            self.model.cuda()
        #self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)

        #self.model = AutoModelForCausalLM.from_pretrained(name)


    def predict(self, prompt, temperature = 0.1, output_length = 1, output_count = 1, suffix = ''):
        if self.use_cuda:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        generated_ids = self.model.generate(input_ids, do_sample=True, num_return_sequences=output_count,
                                       max_length=input_ids.shape[1] + output_length,
                                       temperature=temperature)
        res = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i in range(len(res)):
            res[i] = res[i][len(prompt):]
        return res