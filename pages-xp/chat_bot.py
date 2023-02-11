import streamlit as st
from streamlit_chat import message
import openai
import numpy as np
import pandas as pd
import os
import pathlib
from typing import List, Dict
import pickle
import tiktoken

st.set_page_config(
    page_title="Kastam GPT Chat - Demo",
    page_icon=":robot:"
)

openai.api_key = st.secrets['openapi']['api_key']
COMPLETIONS_MODEL =   "text-davinci-003" # "text-curie-001" 
EMBEDDING_MODEL = "text-embedding-ada-002"

parent_path = pathlib.Path(__file__).parent.parent.resolve()
data_path = os.path.join(parent_path, "data")
olectra_transcript_embedding_csv_filename =  "olectra_transcript_embedding.csv"


st.title("Kastam ChatGPT - Demo")
st.markdown("[Github](https://github.com/sahas-t/kastam-gpt)")


# Open a csv file from data folder
def load_data(file_name :str, header :int = 0 ):
    data = pd.read_csv(f'{data_path}/{file_name}', header = header)
    return data

data_load_state = st.text("Loading data...")
df = load_data('olectra_call_transcript.csv')
data_load_state.text("Loading data...done!") 
df.set_index(['title', 'content'])
# st.write(f"{len(df)} rows in the data.")
# st.write(df.sample(5))





def get_embedding(text: str, id: int = 0, model: str=EMBEDDING_MODEL) -> list[float]:
    # st.write(f"Getting embedding for id: {id} and content: {text[:10]}...")
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_transcript_doc_embeddings(df: pd.DataFrame) -> dict[str, list[float]]:

    return {
        "transcript " + str(idx) : get_embedding(r.content, idx) for idx, r in df.iterrows() if len(r.content) > 10
    }


def save_to_csv(transcript_embeddings: dict[str, list[float]], file_name: str):
    df = pd.DataFrame(transcript_embeddings)
    df = df.T
    df.to_csv(f'{data_path}/{file_name}')
    st.write(f"Saved to {file_name}.")

# Uncomment these lines if new embeddings are required

# transcript_embeddings = compute_transcript_doc_embeddings(df)

# save_to_csv(transcript_embeddings, olectra_transcript_embedding_csv_filename)



def load_embeddings(fname: str) -> dict[str, list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df = load_data(fname)
    # st.write(df.columns[0])
    max_dim = max([int(col_name) for col_name in df.columns if col_name.strip() != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

transcript_embeddings = load_embeddings("olectra_transcript_embedding.csv")


# example_entry = list(transcript_embeddings.items())[0]
# st.write(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)")

# 2) Find the most similar document embeddings to the question embedding

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict[str, np.array]) -> list[(float, str)]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


# 3) Add the most relevant document sections to the query prompt

TOKEN_LENGTH = 4  # 1 tokens per 4 characters for gpt2
MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    # st.write(most_relevant_document_sections)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[df.title == section_index].iloc[0]
        
        # st.write(document_section)
        tokens = int(len(document_section.content)/TOKEN_LENGTH)
        # st.write("tokens" + str(tokens))
        chosen_sections_len += tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + str(document_section.content))
        # st.write(chosen_sections[0])
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    # st.write(f"Selected {len(chosen_sections)} document sections:")
    # st.write("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know.".
    The below is a call transcript from the annual company finacial connect. It contains queries asked by clients and the company representatives responds these questions one after another usually but not always separated by a new line.
    You need to understand which sentences are queries & which are responses. And then answer the question asked to you based on this context. \n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


# 4) Answer the user's question based on the context.

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 200,
    "model": COMPLETIONS_MODEL,
}

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print("Prompt" + prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("Ask a query: ","", key="input")
    return input_text 


user_input = get_text()

if user_input:
    output =  answer_query_with_context(user_input, df, transcript_embeddings, show_prompt=True)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')


