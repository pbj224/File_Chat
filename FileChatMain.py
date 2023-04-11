# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:49:58 2023

@author: PeterJordan
"""
import PyPDF2
import requests
import openai
import traceback
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
from openai import api_key, Embedding, embeddings_utils
import FileChatNotes
import time
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkthemes import ThemedTk
from langchain.document_loaders import WebBaseLoader
from tkinter import font
import sys


global global_api_key

global list_of_text
list_of_text = []

# Determine the script path
script_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))

# Create a data folder if it doesn't exist
data_folder = os.path.join(script_path)
print(data_folder)
os.makedirs(data_folder, exist_ok=True)

def set_openai_api_key():
    import openai
    openai.api_key = global_api_key
    os.environ["OPENAI_API_KEY"] = global_api_key

def count_tokens(string, n_positions=8191):
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = tokenizer.tokenize(string)
    return len(tokens)-1

def extract_text_from_link(link:str):
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    all_p_tags = soup.find_all('p')
    return '\n'.join([p.get_text() for p in all_p_tags])

def show_done_message():
    embed_text.config(state=tk.NORMAL)
    embed_text.delete("1.0", tk.END)
    embed_text.insert(tk.END, "Done!")
    embed_text.config(state=tk.DISABLED)
    
def _extract_pdf_text():
    threading.Thread(target=extract_pdf_text).start()

def extract_pdf_text():
    global pdf_text
    global yes_pdf
    pdf_file_paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
    pdf_text = ""
    for pdf_file_path in pdf_file_paths:
        with open(pdf_file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            for page in range(len(pdf.pages)):
                pdf_text += pdf.pages[page].extract_text()
    chunks = split_text_into_chunks(pdf_text)
    for chunk in chunks:
        list_of_text.append(chunk.strip("\n"))
    yes_pdf = True
    show_done_message()
    
def _extract_text_file():
    threading.Thread(target=extract_text_file).start()
    
def extract_text_file():
    global list_of_text
    global yes_text
    text_file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
    for text_file_path in text_file_paths:
        with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        chunks = split_text_into_chunks(text)
        for chunk in chunks:
            list_of_text.append(chunk.strip("\n"))
    yes_text = True
    show_done_message()
    
def extract_text_from_sublink(link:str):
    loader = WebBaseLoader(link)
    datas = loader.load()
    text_data = ""
    for data in datas:
        text_data += str(data).replace("\\n", "").replace("\\", "")
    if "langchain" in link:
        return text_data.replace("page_content='Welcome to LangChain — 🦜🔗 LangChain 0.0.123Skip to main contentCtrl+K🦜🔗 LangChain 0.0.123Getting StartedQuickstart GuideModulesPrompt TemplatesGetting StartedKey ConceptsHow-To GuidesCreate a custom prompt templateCreate a custom example selectorProvide few shot examples to a promptPrompt SerializationExample SelectorsOutput ParsersReferencePromptTemplatesExample SelectorLLMsGetting StartedKey ConceptsHow-To GuidesGeneric FunctionalityCustom LLMFake LLMLLM CachingLLM SerializationToken Usage TrackingIntegrationsAI21Aleph AlphaAnthropicAzure OpenAI LLM ExampleBananaCerebriumAI LLM ExampleCohereDeepInfra LLM ExampleForefrontAI LLM ExampleGooseAI LLM ExampleHugging Face HubManifestModalOpenAIPetals LLM ExamplePromptLayer OpenAISageMakerEndpointSelf-Hosted Models via RunhouseStochasticAIWriterAsync API for LLMStreaming with LLMsReferenceDocument LoadersKey ConceptsHow To GuidesCoNLL-UAirbyte JSONAZLyricsBlackboardCollege ConfidentialCopy PasteCSV LoaderDirectory LoaderEmailEverNoteFacebook ChatFigmaGCS DirectoryGCS File StorageGitBookGoogle DriveGutenbergHacker NewsHTMLiFixitImagesIMSDbMarkdownNotebookNotionObsidianPDFPowerPointReadTheDocs DocumentationRoams3 Directorys3 FileSubtitle FilesTelegramUnstructured File LoaderURLWeb BaseWord DocumentsYouTubeUtilsKey ConceptsGeneric UtilitiesBashBing SearchGoogle SearchGoogle Serper APIIFTTT WebHooksPython REPLRequestsSearxNG Search APISerpAPIWolfram AlphaZapier Natural Language Actions APIReferencePython REPLSerpAPISearxNG SearchDocstoreText SplitterEmbeddingsVectorStoresIndexesGetting StartedKey ConceptsHow To GuidesEmbeddingsHypothetical Document EmbeddingsText SplitterVectorStoresAtlasDBChromaDeep LakeElasticSearchFAISSMilvusOpenSearchPGVectorPineconeQdrantRedisWeaviateChatGPT Plugin RetrieverVectorStore RetrieverAnalyze DocumentChat IndexGraph QAQuestion Answering with SourcesQuestion AnsweringSummarizationRetrieval Question/AnsweringRetrieval Question Answering with SourcesVector DB Text GenerationChainsGetting StartedHow-To GuidesGeneric ChainsLoading from LangChainHubLLM ChainSequential ChainsSerializationTransformation ChainUtility ChainsAPI ChainsSelf-Critique Chain with Constitutional AIBashChainLLMCheckerChainLLM MathLLMRequestsChainLLMSummarizationCheckerChainModerationPALSQLite exampleAsync API for ChainKey ConceptsReferenceAgentsGetting StartedKey ConceptsHow-To GuidesAgents and VectorstoresAsync API for AgentConversation Agent (for Chat Models)ChatGPT PluginsCustom AgentDefining Custom ToolsHuman as a toolIntermediate StepsLoading from LangChainHubMax IterationsMulti Input ToolsSearch ToolsSerializationAdding SharedMemory to an Agent and its ToolsCSV AgentJSON AgentOpenAPI AgentPandas Dataframe AgentPython AgentSQL Database AgentVectorstore AgentMRKLMRKL ChatReActSelf Ask With SearchReferenceMemoryGetting StartedKey ConceptsHow-To GuidesConversationBufferMemoryConversationBufferWindowMemoryEntity MemoryConversation Knowledge Graph MemoryConversationSummaryMemoryConversationSummaryBufferMemoryConversationTokenBufferMemoryAdding Memory To an LLMChainAdding Memory to a Multi-Input ChainAdding Memory to an AgentChatGPT CloneConversation AgentConversational Memory CustomizationCustom MemoryMultiple MemoryChatGetting StartedKey ConceptsHow-To GuidesAgentChat Vector DBFew Shot ExamplesMemoryPromptLayer ChatOpenAIStreamingRetrieval Question/AnsweringRetrieval Question Answering with SourcesUse CasesAgentsChatbotsGenerate ExamplesData Augmented GenerationQuestion AnsweringSummarizationQuerying Tabular DataExtractionEvaluationAgent Benchmarking: Search + CalculatorAgent VectorDB Question Answering BenchmarkingBenchmarking TemplateData Augmented Question AnsweringUsing Hugging Face DatasetsLLM MathQuestion Answering Benchmarking: Paul Graham EssayQuestion Answering Benchmarking: State of the Union AddressQA GenerationQuestion AnsweringSQL Question Answering Benchmarking: ChinookModel ComparisonReferenceInstallationIntegrationsAPI ReferencesPromptsPromptTemplatesExample SelectorUtilitiesPython REPLSerpAPISearxNG SearchDocstoreText SplitterEmbeddingsVectorStoresChainsAgentsEcosystemLangChain EcosystemAI21 LabsAtlasDBBananaCerebriumAIChromaCohereDeepInfraDeep LakeForefrontAIGoogle Search WrapperGoogle Serper WrapperGooseAIGraphsignalHazy ResearchHeliconeHugging FaceMilvusModalNLPCloudOpenAIOpenSearchPetalsPGVectorPineconePromptLayerQdrantRunhouseSearxNG Search APISerpAPIStochasticAIUnstructuredWeights & BiasesWeaviateWolfram Alpha WrapperWriterAdditional ResourcesLangChainHubGlossaryLangChain GalleryDeploymentsTracingDiscordProduction Support.rst.pdfWelcome to LangChain Contents Getting StartedModulesUse CasesReference DocsLangChain EcosystemAdditional Resources", "")
    else:
        return text_data

def split_text_into_chunks(text, chunk_size=8):
    custom_delimiter = delimiter.get("1.0", tk.END).strip()
    if len(custom_delimiter) > 0:
        sentences = text.split(custom_delimiter)
    else:
        sentences = text.split('.')
    chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
    return ['. '.join(chunk) for chunk in chunks]

def create_log(df, filename):
    # Select only the question column from the DataFrame
    df = df[['question']]
    # Remove any rows with missing values
    df = df.dropna()
    # Initialize the GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Split rows that have too many tokens into two separate rows
    df['n_tokens'] = df.question.apply(lambda x: len(tokenizer.encode(x)))
    df = df[df.n_tokens<4000]
    df = df[df.n_tokens>2]
    # Create a new column called "babbage_search" with the Babbage search embeddings for each question text
    df['babbage_search'] = df.question.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    # Save the DataFrame to a csv file
    df.to_csv(filename + '_with_embeddings_2k.csv')
    return df

def search_questions(df, search_query, n, pprint=True):
    # Get the Babbage search embedding for the search query
    embedding = get_embedding(
      search_query,
      engine="text-embedding-ada-002"
    )
    # Calculate the cosine similarity between the search embedding and the babbage_search embeddings in the DataFrame
    df["similarities"] = df.babbage_search.apply(lambda x: cosine_similarity(x, embedding))
    # Sort the DataFrame by the similarities column in descending order and select the top n rows
    res = (
      df.sort_values("similarities", ascending=False)
      .head(n)
      .question
    )
    result_string = ""
    for r in res:
      result_string += r[:] + "\n"
    return result_string

MODEL = "gpt-4"

def ask_self_question(query, notes, messages):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You will be tasked with generating examples of how to rewrite a bad query for text retrieval using embeddings and cosine similarity to a good one. A good query is more specific and directly addresses the topics that the given sentences cover. It contains relevant keywords and phrases, which makes it easier for the search function to identify the most relevant sentences by calculating the cosine similarity between the query's embeddings and the sentence embeddings. A bad query is considered bad because it is too broad and uses more general terms. These terms might not have strong direct associations with the given sentences. The search function, which relies on cosine similarity, may struggle to find the most relevant sentences as the embeddings of these general terms might not have high similarity scores with the embeddings of the specific sentences in the dataset. A broad query might return results that are not directly related to the question if the question relates to subtopics within a broader context. In summary, a good query is specific and contains relevant keywords that can be effectively matched with the embeddings of the sentences in the dataset. A bad query, on the other hand, uses broad or vague terms that might not have strong direct associations with the sentences in the dataset, making it harder for the search function to find the most relevant results."},
            {"role": "system", "content": "Example: Suppose we have the following 5 sentences in the DataFrame: \"The solar system has eight planets.\" \"Jupiter is the largest planet in our solar system.\" \"The ozone layer protects the Earth from harmful UV radiation.\" \"The Earth's atmosphere is composed mainly of nitrogen and oxygen.\" \"Mars is the fourth planet in our solar system.\" A good query: \"Tell me about planets in the solar system and Earth's atmosphere.\" A bad query: \"Tell me about space and the environment.\""},
            {"role": "system", "content": f"Below is a set of bullet point notes about the text that is being retrieved:\nNotes: {notes}"},
            {"role": "user", "content": f"Given the context of the conversation, {messages},\nRe-word this query so that it is optimized for retrieving relevant text using embeddings and cosine similarity. Print just the new query. Do not add any quotation marks, titles, or labels but just the new query: {query}"}
        ],
        temperature=0.2,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    output = response["choices"][0]["message"]["content"].strip()
    return output

def ask_question(messages, res, notes, notes_res):
    response = openai.ChatCompletion.create(
        model=MODEL,
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
                # Update the tkinter chat frame with the received content
                chat_text.config(state=tk.NORMAL)
                chat_text.insert(tk.END, content)
                chat_text.see(tk.END)
                chat_text.config(state=tk.DISABLED)

    return output

def get_sublinks(link, main_link):
    # Send a GET request to the URL
    response = requests.get(link)
    # Parse the HTML content of the response
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find all the <a> tag on the page
    a_tags = soup.find_all('a')
    # Extract the href attribute of each <a> tag
    sublinks = []
    for a_tag in a_tags:
        href = a_tag.get('href')
        if href is not None:
            if 'https' not in href:
                sublinks.append(main_link +href)
    return sublinks

def read_file_remove_error_characters(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        notes = file.read()
    return notes

def read_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    df = df[['question']]
    text = ""
    for index, row in df.iterrows():
        text += ",".join([str(val) for val in row.values]) + "\n"
    return text

def process_text_sources(user_inputs, base_file_name):
    if new_sources_t:
        if len(user_inputs["sublink"]) > 1:
            link = user_inputs["sublink"]
            main_link = user_inputs["mainlink"]
            token_count = 0
            sublinks = get_sublinks(link, main_link)
            print("Total sublinks: " + str(len(sublinks)))
            counter = 1
            for sublink in sublinks:
                text = extract_text_from_sublink(sublink)
                tokens = count_tokens(text)
                print("Text Tokens for link #" + str(counter) + ": " + str(tokens))
                counter += 1
                token_count += tokens
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    list_of_text.append(chunk.strip("\n"))
            print("Total Tokens: " +str( token_count))
            print("Cost: $"+str((token_count/1000)*0.0008))
        elif len(user_inputs["links"])>1:
            for link in user_inputs["links"].split(", "):
                text = extract_text_from_link(link)
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    list_of_text.append(chunk.strip("\n"))
        df = pd.DataFrame(list_of_text, columns=['question'])
        create_log(df, os.path.join(data_folder, user_inputs["datafile_path"]))
    return list_of_text

def prepare_dataframe_and_notes(list_of_text, user_inputs, base_file_name):
    file_path = os.path.join(data_folder, user_inputs["datafile_path"]).replace("\\", "/")
    df = pd.read_csv(file_path+"_with_embeddings_2k.csv")
    if user_inputs["new_sources"] == True:
        notes_text = "".join(list_of_text)
        notes, for_embed = FileChatNotes.takeNotes(notes_text, global_api_key)
        df2 = pd.DataFrame(for_embed, columns=['question'])
        create_log(df2, file_path + "_notes")
        notes_file_path = file_path + "_notes.txt"
        with open(notes_file_path, 'w') as file:
            file.write(notes)
        df2 = pd.read_csv(file_path+"_notes_with_embeddings_2k.csv")
    else:
        notes_file_path = file_path + "_notes.txt"
        notes_embeddings_file_path = file_path + "_notes_with_embeddings_2k.csv"
        with open(notes_file_path, 'r') as file:
            notes = file.read()
        df2 = pd.read_csv(notes_embeddings_file_path)

    df['babbage_search'] = df.babbage_search.apply(eval).apply(np.array)
    df2['babbage_search'] = df2.babbage_search.apply(eval).apply(np.array)
    isDone = True
    return df, notes, isDone, df2


def main_conversation_loop(df, notes, df2):
    embed_text.config(state=tk.NORMAL)
    embed_text.insert(tk.END, "General Ovewrview:\n" + notes+ "\n\n\n\n")
    embed_text.config(state=tk.DISABLED)
    messages=[
        {"role": "system", "content": "You are a question answering chatbot. You will be provided with a question relating to content from a website and/or PDF document along with helpful notes and relevant snippets of the text from the website. Your job is to answer the question using only the information provided to you. Try to include direct quotes from the text provided to you in your answer when it doesnt reduce the quality of your answer. If the information provided to is not relevant to the question asked, respond only with 'I dont know'. Most importantly, not try to answer the question using your training data."},
        {"role": "system", "content": f"The following is the information provided for this question:\nNotes: {notes}"},
        {"role": "user", "content": "Hello, I would like to ask you questions about some external sources"},
        {"role": "assistant", "content": "Sure! Go right ahead. How can I help you today?"}
    ]
    return messages

def handle_new_query(df, notes, messages, df2):
    threading.Thread(target=lambda: _handle_new_query(df, notes, messages, df2)).start()

def _handle_new_query(df, notes, messages, df2):
    query = input_text.get("1.0", tk.END).strip()
    messages.append({"role": "user", "content": query})
    chat_text.insert(tk.END, "\n\nUser: " + input_text.get("1.0", tk.END).strip() + "\n\nAI: ")
    chat_text.see(tk.END)
    search_query = ask_self_question(query, notes, messages)
    print(search_query)
    try:
        res = search_questions(df, search_query, n=5).strip("\n").strip("\n\n\n\n\n")
        notes_res = search_questions(df2, search_query, n=1).strip("\n")
        messages.append({"role": "system", "content": f"Overview: {notes_res}.\nRelevant sections from the websites text content: {res}"})  # Add this line
        response = ask_question(messages, res, notes, notes_res)
        out_new = {"role": "assistant", "content": response}
        messages.append(out_new)
        
        embed_text.config(state=tk.NORMAL)
        embed_text.insert(tk.END, "Brief section overview: " + notes_res +"\n\n")
        embed_text.insert(tk.END, "Direct relevant source text: " + res)
        embed_text.config(state=tk.DISABLED)
        messages.pop(len(messages)-2)
    except:
        try:
            res = search_questions(df, search_query, n=1).strip("\n").strip("\n\n\n\n\n")
            notes_res = search_questions(df2, search_query, n=1).strip("\n")
            messages.append({"role": "system", "content": f"Overview: {notes_res}.\nRelevant sections from the websites text content: {res}"})  # Add this line
            response = ask_question(messages, res, notes, notes_res)
            out_new = {"role": "assistant", "content": response}
            messages.append(out_new)
            
            embed_text.config(state=tk.NORMAL)
            embed_text.insert(tk.END, "bried section overview: " + notes_res + "\n\n")
            embed_text.insert(tk.END, "Direct source text: " + res)
            embed_text.config(state=tk.DISABLED)
            messages.pop(len(messages)-2)
        except:
            res = search_questions(df, search_query, n=1).strip("\n")
            res = res[:2000]
            notes_res = search_questions(df2, search_query, n=1).strip("\n")
            messages.append({"role": "system", "content": f"Overview: {notes_res}.\nRelevant sections from the websites text content: {res}"})  # Add this line
            response = ask_question(messages, res, notes, notes_res)
            out_new = {"role": "assistant", "content": response}
            messages.append(out_new)
            embed_text.config(state=tk.NORMAL)
            embed_text.insert(tk.END, "bried section overview: " + notes_res + "\n\n")
            embed_text.insert(tk.END, "Direct source text: " + res)
            embed_text.config(state=tk.DISABLED)
            messages.pop(len(messages)-2)
    input_text.delete("1.0", tk.END)
    calculate_price_queries = ""
    calculate_price_response = ""
    for message in messages:
        calculate_price_queries += query
        calculate_price_response += search_query + response
    query_price = float((count_tokens(calculate_price_queries)/1000)*0.03)
    response_price = float((count_tokens(calculate_price_response)/1000)*0.06)
    print("Cycle cost: $" + str(query_price + response_price))

def chat_start(df, notes, df2):
    messages = main_conversation_loop(df, notes, df2)
    return messages

def start_chat_thread():
    global df, notes, messages, df2, global_api_key
    try:
        user_inputs = {
            "new_sources": new_sources_t,
            "links": link_text.get("1.0", tk.END).strip(),
            "sublink": sublink_text.get("1.0", tk.END).strip(),
            "mainlink": mainlink_text.get("1.0", tk.END).strip(),
            "pdf": yes_pdf,
            "datafile_path": os.path.join("data", base_file_name),
            "api_key": api_key_text.get("1.0", tk.END).strip(),
        }
    except:
        user_inputs = {
            "new_sources": new_sources_t,
            "links": link_text.get("1.0", tk.END).strip(),
            "sublink": sublink_text.get("1.0", tk.END).strip(),
            "mainlink": mainlink_text.get("1.0", tk.END).strip(),
            "pdf": False,
            "datafile_path": os.path.join("data", base_file_name),
            "api_key": api_key_text.get("1.0", tk.END).strip(),
        }
    try:
        global_api_key = user_inputs["api_key"]
        list_of_text = process_text_sources(user_inputs, base_file_name)
        df, notes, isDone, df2 = prepare_dataframe_and_notes(list_of_text, user_inputs, base_file_name)
        messages = chat_start(df, notes, df2)
    except Exception as e:
        print(f"Error: {e}")

def start_chat():
    threading.Thread(target=start_chat_thread).start()
    
def on_run_button_click():
    try:
        threading.Thread(target=lambda: _on_run_button_click()).start()
    except Exception as e:
        print(f"Error: {e}")

def _on_run_button_click():
    embed_text.config(state=tk.NORMAL)  # Make the embed_text widget editable
    embed_text.delete("1.0", tk.END)
    chat_text.config(state=tk.NORMAL)  # Make the embed_text widget editable
    handle_new_query(df, notes, messages, df2)

def on_key_press(event):
    input_text.update_idletasks()
    width = input_text["width"]
    lines = input_text.get("1.0", tk.END).split("\n")
    longest_line = max(lines, key=len)
    height = len(lines)
    if len(longest_line) > width:
        width = len(longest_line)
    input_text.config(height=height, width=width)
    
def new_sources_clicked():
    try:
        global new_sources_t
        new_sources_t = True
        global base_file_name
        base_file_name = inputted_file_name.get("1.0", tk.END).strip()
        start_chat()
        embed_text.config(state=tk.NORMAL)
        embed_text.delete("1.0", tk.END)
        embed_text.config(state=tk.DISABLED)
    except Exception as e:
        print(f"Error: {e}")

def old_sources_clicked():
    try:
        global new_sources_t
        new_sources_t = False
        global base_file_name
        base_file_name = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")]).replace(os.path.join(data_folder, "").replace("/", "\\"), "").replace("_with_embeddings_2k.csv", "")
        start_chat()
    except Exception as e:
        print(f"Error: {e}")
        
def update_input_text_height(event=None):
    # Calculate the required height based on the content
    input_text_content = input_text.get("1.0", tk.END)
    num_lines = input_text_content.count("\n")

    # Set the new height
    if num_lines < 7:
        input_text.configure(height=num_lines)
    else:
        input_text.configure(height=7)
def on_enter_key_pressed(event=None):
    on_run_button_click()
        
def clear_temp_message(event):
    if delimiter.get("1.0", tk.END).strip() == "Do not input anything here unless you're not querying documents that contain periods":
        delimiter.delete("1.0", tk.END)
        delimiter.config(foreground='black')

def set_temp_message(event=None):
    if not delimiter.get("1.0", tk.END).strip():
        delimiter.insert("1.0", "Do not input anything here unless you're not querying documents that contain periods")
        delimiter.config(foreground='grey')
        
def clear_temp_message_filename(event):
    if inputted_file_name.get("1.0", tk.END).strip() == "IMPORTANT: Name file before clicking new sources to initiate the process of processing your docs":
        inputted_file_name.delete("1.0", tk.END)
        inputted_file_name.config(foreground='black')

def set_temp_message_filename(event=None):
    if not inputted_file_name.get("1.0", tk.END).strip():
        inputted_file_name.insert("1.0", "IMPORTANT: Name file before clicking new sources to initiate the process of processing your docs")
        inputted_file_name.config(foreground='grey')
        
app = ThemedTk(theme="arc")  # Use ThemedTk instead of tk.Tk and choose a theme
app.title("File Chat")

frame = ttk.Frame(app, padding="3 3 12 12")
frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
app.columnconfigure(0, weight=1)
app.rowconfigure(0, weight=1)
button_width = 20
custom_font = font.Font(family="Sans Serif", size=10)

ttk.Label(frame, text="OpenAI API key:").grid(column=1, row=1, columnspan=1, sticky=tk.W)
api_key_text = tk.Text(frame, height=1, width=60, wrap=tk.WORD, relief="flat", font=custom_font)
api_key_text.grid(column=2, row=1, columnspan=14, sticky=(tk.W, tk.E))
api_key_text.bind("<KeyPress>", on_key_press)
    
new_sources_button = ttk.Button(frame, text="New Sources", command=new_sources_clicked, width=button_width)
new_sources_button.grid(row=6, column=3, sticky=tk.EW)

old_sources_button = ttk.Button(frame, text="Old Sources", command=old_sources_clicked, width=button_width)
old_sources_button.grid(row=6, column=2, sticky=tk.EW)

link_text = tk.Text(frame, height=1, width=60, wrap=tk.WORD, relief="flat", font=custom_font)
link_text.grid(column=2, row=2, columnspan=14, sticky=(tk.W, tk.E))
link_text.bind("<KeyPress>", on_key_press)

sublink_text = tk.Text(frame, height=1, width=60, wrap=tk.WORD, relief="flat", font=custom_font)
sublink_text.grid(column=2, row=3, columnspan=14, sticky=(tk.W, tk.E))
sublink_text.bind("<KeyPress>", on_key_press)

mainlink_text = tk.Text(frame, height=1, width=60, wrap=tk.WORD, relief="flat", font=custom_font)
mainlink_text.grid(column=2, row=4, columnspan=14, sticky=(tk.W, tk.E))
mainlink_text.bind("<KeyPress>", on_key_press)

pdf_upload_button = ttk.Button(frame, text="Upload PDF", command=_extract_pdf_text, width=button_width)
pdf_upload_button.grid(row=5, column=2, sticky=tk.EW)

text_upload_button = ttk.Button(frame, text="Upload .txt", command=_extract_text_file, width=button_width)
text_upload_button.grid(row=5, column=3, sticky=tk.EW)

ttk.Label(frame, text="Individual links:").grid(column=1, row=2, columnspan=1, sticky=tk.W)
ttk.Label(frame, text="Sublink:").grid(column=1, row=3, columnspan=1, sticky=tk.W)
ttk.Label(frame, text="Main links:").grid(column=1, row=4, columnspan=1, sticky=tk.W)

# Use tk.Text widget instead of ttk.Entry
input_text = tk.Text(frame, height=1, width=60, wrap=tk.WORD, relief="flat", font=custom_font)
input_text.grid(column=1, row=12, columnspan=8, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
input_text.bind("<KeyPress>", on_key_press)
input_text.bind("<KeyPress>", update_input_text_height)  # Bind the height update function
input_text.bind("<KeyRelease>", update_input_text_height)  # Bind the height update function
input_text.bind("<Return>", on_enter_key_pressed)  # Bind the new function

ttk.Label(frame, text="Name new sources:").grid(column=4, row=6, columnspan=1, sticky=tk.W)
inputted_file_name = tk.Text(frame, height=1, width=60, wrap=tk.WORD, relief="flat", font=custom_font)
inputted_file_name.grid(column=5, row=6, columnspan=4, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
set_temp_message_filename()
inputted_file_name.bind("<FocusIn>", clear_temp_message_filename)
inputted_file_name.bind("<FocusOut>", set_temp_message_filename)
inputted_file_name.bind("<KeyPress>", on_key_press)

ttk.Label(frame, text="Custom Delimiter:").grid(column=4, row=5, columnspan=1, sticky=tk.W)
delimiter = tk.Text(frame, height=1, width=60, wrap=tk.WORD, relief="flat", font=custom_font)
delimiter.grid(column=5, row=5, columnspan=4, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
set_temp_message()
delimiter.bind("<FocusIn>", clear_temp_message)
delimiter.bind("<FocusOut>", set_temp_message)
delimiter.bind("<KeyPress>", on_key_press)

# Start Chat button
run_button = ttk.Button(frame, text="Enter", command=on_run_button_click)
run_button.grid(column=9, row=12, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

# Chat interface
chat_frame = ttk.Frame(frame, borderwidth=5, relief="flat")
chat_frame.grid(row=9, column=5, columnspan=4, pady=20, sticky=(tk.W, tk.E, tk.N, tk.S))

chat_text = tk.Text(chat_frame, state=tk.DISABLED, height=25, font=custom_font)
chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Embeddings interface
embed_frame = ttk.Frame(frame, borderwidth=5, relief="flat")
embed_frame.grid(row=9, column=1, columnspan=4, pady=20, sticky=(tk.W, tk.E, tk.N, tk.S))

embed_text = tk.Text(embed_frame, state=tk.DISABLED, height=25, font=custom_font)
embed_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


scrollbar = ttk.Scrollbar(chat_frame, orient=tk.VERTICAL, command=chat_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_text['yscrollcommand'] = scrollbar.set

scrollbar_two = ttk.Scrollbar(embed_frame, orient=tk.VERTICAL, command=embed_text.yview)
scrollbar_two.pack(side=tk.RIGHT, fill=tk.Y)
embed_text['yscrollcommand'] = scrollbar_two.set


# Configure the column weights
for col in range(1, 40):
    frame.columnconfigure(col, weight=1)
    
# Configure the column weights
for row in range(1, 15):
    frame.rowconfigure(row, weight=1)

# Add padding to all child widgets
for child in frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

# Set focus to the input_text widget
input_text.focus()

# Run the tkinter application
app.mainloop()
