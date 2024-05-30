import os

os.environ['HF_HOME'] = "./hf/"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BitsAndBytesConfig
import numpy as np

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


import argparse



argparser = argparse.ArgumentParser()


argparser.add_argument("--epochs" , "-e" , type = int , default = 2 , help = "Number of Epochs")

argparser.add_argument("--train_batch_size" , "-tbs" , type = int , default = 50 , help = "Train Batch Size")

argparser.add_argument("--validation_batch_size" , "-vbs" , type = int , default = 50 , help = "Validation Batch Size")

argparser.add_argument("--eval_steps" , "-es" , type = int , default = 500 , help = "Evaluation Steps")

argparser.add_argument("--early_stopping_patience" , "-esp" , type = int , default = 5 , help = "Early Stopping Patience")

argparser.add_argument("--lora_rank" , "-lr" , type = int , default = 16 , help = "LoRA rank parameter")

argparser.add_argument("--output_dir" , "-out" , type = str , default = "output" , help = "Output Directory")

args = argparser.parse_args()








config = LoraConfig(
    r=args.lora_rank, #Rank
    lora_alpha=32,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense'
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)


compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )


print("Getting Models")

model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBartTokenizer.from_pretrained(model_name , use_fast = True)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Preparing the Model for QLoRA
model = prepare_model_for_kbit_training(model)

# peft model
model = get_peft_model(model, config)

from datasets import load_dataset

dataset = load_dataset("cfilt/iitb-english-hindi")

from datasets import Dataset

def generate_dataset(dataset , split):
    filtered_dataset = dataset[split]['translation']
    english_dataset = [data['en'] for data in filtered_dataset]
    hindi_dataset = [data['hi'] for data in filtered_dataset]

    print("Total Dataset length : " , len(english_dataset))
    
    data_dictionary = {
        "english" : english_dataset,
        "hindi" : hindi_dataset
    }
    return Dataset.from_dict(data_dictionary)


train_dataset = generate_dataset(dataset, "train")

test_dataset = generate_dataset(dataset , "test")

validation_dataset = generate_dataset(dataset , "validation")

def tokenize_dataset_new(example):
    model_inputs = tokenizer(example["english"], max_length=512, truncation=True)
    labels = tokenizer(example["hindi"], max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



train_tokenised_dataset = train_dataset.map(tokenize_dataset_new , batched=True , num_proc=5)
train_tokenised_dataset = train_tokenised_dataset.remove_columns(['english' , 'hindi'])


test_tokenised_dataset = test_dataset.map(tokenize_dataset_new , batched=True , num_proc=5)
test_tokenised_dataset = test_tokenised_dataset.remove_columns(['english' , 'hindi' ])


validation_tokenised_dataset = validation_dataset.map(tokenize_dataset_new , batched=True , num_proc=5)
validation_tokenised_dataset = validation_tokenised_dataset.remove_columns(['english' , 'hindi'])


from transformers import DataCollatorForSeq2Seq
data_collector = DataCollatorForSeq2Seq(tokenizer=tokenizer)



train_batch_size = args.train_batch_size
validation_batch_size = args.validation_batch_size



data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


training_args = Seq2SeqTrainingArguments(
    # training
    num_train_epochs=args.epochs,
    per_device_train_batch_size=train_batch_size,
    # per device training batch size is used to train the model on the given batch size

    # evaluation
    per_device_eval_batch_size=validation_batch_size,
    # per device evaluation batch size is used to evaluate the model on the given batch size
    # gradient_accumulation_steps=8,
    # gradient accumulation steps is used to accumulate the gradients over the given number of steps
    # this helps in reducing the memory usage during training
    # eval_accumulation_steps=10,
    # eval accumulation steps is used to accumulate the evaluation results over the given number of steps
    # this helps in reducing the memory usage during evaluation
    evaluation_strategy="steps",
    # if the evaluation strategy is steps, then the evaluation will be done every eval_steps
    # else if it is epoch, then the evaluation will be done every epoch and eval accumulation steps will be ignored
    eval_steps=args.eval_steps,


    # checkpointing


    # logging
    logging_dir="./logs",
    logging_steps=10,

    # misc
    warmup_steps=500,
    # warmup steps is used to warmup the learning rate over the given number of steps
    # this helps in reducing the impact of the randomness in the initial learning rate
    # this is very useful when the learning rate is very high
    # this is also useful when the model is very large
    output_dir="./output",
    save_steps=args.eval_steps,
    # save steps is used to save the model over the given number of steps
    # this is useful when the model is very large
    save_strategy="steps",
    # save strategy is used to save the model every epoch
    # if the save strategy is steps, then the model will be saved every save_steps
    # else if it is epoch, then the model will be saved every epoch
    # and save_steps will be ignored
    save_total_limit=5,

    # save the best model
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better = False,
    # generate tensorboard logs
    report_to=None,

)


from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)



trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenised_dataset,
    eval_dataset=validation_tokenised_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks = [early_stopping]
    # compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./mBART-fine-tuned-en-hi")


from transformers import pipeline


translator = pipeline("translation", model=model,
                    src_lang = "en_XX" , tgt_lang= "hi_IN",
                    tokenizer=tokenizer)



text = "hi how are you"
translator(text)