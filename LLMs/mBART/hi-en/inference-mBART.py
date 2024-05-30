import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BitsAndBytesConfig
import numpy as np

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


from peft import PeftModel


print("loading model")

model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBartTokenizer.from_pretrained(model_name , use_fast = True)

loaded_model = MBartForConditionalGeneration.from_pretrained(model_name)

ft_model = PeftModel.from_pretrained(loaded_model, "./mBART-fine-tuned-hi-en",torch_dtype=torch.float16,is_trainable=False)

print("model loaded")


from transformers import pipeline

translator = pipeline("translation", model=ft_model,
                    src_lang = "hi_IN" , tgt_lang= "en_US",
                    tokenizer=tokenizer)



text = "आज बहुत गरम दिन है |"

print(translator(text)[0]['translation_text'])