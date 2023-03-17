import logging
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

import os
import pandas as pd
import tiktoken

from website_parser import WebsiteParser
from embedding_generator import EmbeddingGenerator

logging.basicConfig(level=logging.INFO)

#
# Model configs
#
MODEL = 'text-davinci-003'
# token encoding corresponds to the engine
EMBEDDINGS_ENGINE = 'text-embedding-ada-002'
TOKEN_ENCODING = '"cl100k_base"'
# configure openai's RPM quota
quota_size = 20
# designate max number of tokens per embedding
max_tokens = 1000

domain = 'gusto.com'

output_path = 'output'
website_path = output_path + '/text/' + domain
processed_path = output_path + '/processed/' + domain

# if path does not exist, scrape a website and dump text files to the path
if not os.path.exists(website_path):
    logging.info('Scraping website: ' + domain)
    websiteParser = WebsiteParser(output_path=output_path, valid_paths=[
                                  'product', 'about', 'services', 'partners'], max_pages=20)
    websiteParser.parse('https://' + domain)

if ((not os.path.exists(processed_path)) or (len(os.listdir(processed_path)) == 0)):
    logging.info('Generating embedding vectors for ' + domain)
    dir_list = os.listdir(website_path)
    file_names = []
    for file in dir_list:
        file_names.append(website_path + '/' + file)

    embedding_generator = EmbeddingGenerator(quota_size=quota_size, engine=EMBEDDINGS_ENGINE,
                                             output_path=output_path,
                                             domain=domain,
                                             tokenizer=tiktoken.get_encoding(
                                                 TOKEN_ENCODING),
                                             max_tokens=max_tokens)
    embedding_generator.generate(file_names)

embeddings_df = pd.read_csv(processed_path + '/embeddings.csv', index_col=0)
embeddings_df['embeddings'] = embeddings_df['embeddings'].apply(
    eval).apply(np.array)


def create_context(question: str, embeddings_df: pd.DataFrame, max_len=1800):
    # Get the embeddings for the question
    question_embeddings = openai.Embedding.create(
        input=question, engine=EMBEDDINGS_ENGINE)['data'][0]['embedding']

    # Get the distances from the embeddings
    embeddings_df['distances'] = distances_from_embeddings(
        question_embeddings, embeddings_df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in embeddings_df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['num_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row['text'])

    # Return the context
    return "\n\n###\n\n".join(returns)


def answer_question(
        embeddings_df: pd.DataFrame, question: str,
        model=MODEL,
        max_len=1800, debug=False, max_tokens=150, stop_sequence=None):

    context = create_context(
        question=question, embeddings_df=embeddings_df, max_len=max_len)

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )

        if debug:
            print(response)

        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""


print(answer_question(embeddings_df,
      question="Is Gusto a Software company?", debug=True))
print(answer_question(embeddings_df,
      question="Does Gusto perform research and development?"))
