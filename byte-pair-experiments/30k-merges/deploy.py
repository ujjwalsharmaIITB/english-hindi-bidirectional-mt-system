import os

import streamlit as sl
import ctranslate2
from subword_nmt.apply_bpe import BPE
import codecs




# set the title
sl.set_page_config(page_title="Hi<->En MT System")
sl.title("NMT-Based Bi-directional Hindi-English Machine Translation")


print("getting BPE codes ...")
# bpe codes
codes_english = codecs.open("bpe/learn/english-bpe.codes" , encoding='utf-8')
bpe_english = BPE(codes_english)

codes_hindi = codecs.open("bpe/learn/hindi-bpe.codes" , encoding='utf-8')
bpe_hindi = BPE(codes_hindi)



model_options = (
    "English to Hindi",
    "Hindi to English"
)



@sl.cache_resource
def load_mt_models():
    
    print("Loading mt models ...")
    models = {
        "en-hi": ctranslate2.Translator('models/en-hi-model'),
        "hi-en": ctranslate2.Translator('models/hi-en-model'),
    }
    print("Models loaded ...")

    return models

mt_models =  load_mt_models()

def get_model(model):
    match model:
        case "English to Hindi":
            return mt_models['en-hi']
        case "Hindi to English":
            return mt_models['hi-en']
        case _:
            return None


def get_bpe(model):
    match model:
        case "English to Hindi":
            return bpe_english
        case "Hindi to English":
            return bpe_hindi
        case _:
            return None


def predict(model , sentence):
    tokenizer = get_bpe(model)
    bpe_tokens = tokenizer.process_line(sentence).split(" ")
    translator = get_model(model)
    if translator is None:
        return None
    translation = translator.translate_batch([bpe_tokens] , beam_size = 10 , max_batch_size=1)
    return " ".join(translation[0].hypotheses[0]).replace("@@ " , "")
    

text = sl.text_input("Enter Text based on language to translate")

model = sl.selectbox("Select Models" , model_options )

translate = sl.button("Translate")

if translate:
    if text == "":
        sl.warning("Enter sentence:")
    else:
        translation = predict(model , text)
        if translation is None:
            sl.error("Model not found")
        else:
            sl.info(translation)