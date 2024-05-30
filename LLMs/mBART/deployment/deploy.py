import streamlit as sl
import os

# getting llms

from getModels import *


# set the title
sl.set_page_config(page_title="Hi<->En MT System")
sl.title("Fine-Tuned Bi-directional Hindi-English Machine Translation")


model_options = (
    "English to Hindi",
    "Hindi to English"
)



@sl.cache_resource
def getMTModels():

    print("Loading Models")

    english_to_hindi = getEnglishToHindiModel()
    hindi_to_english = getHindiToEnglishModel()

    return english_to_hindi, hindi_to_english

english_to_hindi, hindi_to_english = getMTModels()


def get_model(model):
    match model:
        case "English to Hindi":
            return english_to_hindi
        case "Hindi to English":
            return hindi_to_english
        case _:
            return None , None


def predict(model, sentence):
    translator = get_model(model)
    translation = translator(sentence)[0]['translation_text']
    return translation



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