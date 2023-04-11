# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:37:05 2023

@author: PeterJordan
"""
import requests
import openai
import traceback
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast

openai.api_key = "API_KEY_HERE"

def passage_segmenter(passage, interval=8000):
    segment = []
    count = 0
    while count < len(passage):
        segment.append(passage[count:count + interval])
        count += interval
    return segment

MODEL = "gpt-3.5-turbo"

def ask_question(messages, api_key, model):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.2,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        stream=True  # Set stream=True for streaming completions
    )

    output = ""
    for chunk in response:
        if "delta" in chunk["choices"][0]:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                content = delta["content"]
                output += content
                print(content, end="")  # Add end parameter to prevent newline character

    return output

def count_tokens(string, n_positions=8191):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(string)
    return len(tokens) - 1

def create_analysis(passage, api_key):
    messages = [{"role": "system", "content": "Generate a bullet point summary of the text snippet that includes the main topic, any key details, and relevant background information that provides context for the snippet.\n\nSection: " + passage}]
    return ask_question(messages, api_key, "gpt-3.5-turbo")

def reformat_analysis(analysis, api_key):
    messages = [{"role": "system", "content": "Reformat the following bullet pointed notes into a clear, concise, and well formatted set of bullet pointed notes that are about 500 words in length and discuss the main topic, overview of key details, and outline of the text as a whole. Group related information together and use bullet points or numbered lists to enhance readability.\n\nNotes: " + analysis}]
    return ask_question(messages, api_key, "gpt-4")

def takeNotes(inputq, api_key):
    while True:
        segments = passage_segmenter(inputq, 13000)
        count = 1
        unformatted = ""
        list_of_notes = []
        for segment in segments:
            print("Section #" + str(count) + " of " + str(len(segments)) + ": ")
            analysis = create_analysis(segment, api_key)
            print("\n\n")
            unformatted += analysis + "\n"
            list_of_notes.append(analysis)
            count += 1
        notes = reformat_analysis(unformatted, api_key)
        return notes, list_of_notes
        break
