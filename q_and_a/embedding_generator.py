import logging
import numpy as np
import openai
import os
import pandas as pd
import tiktoken


class EmbeddingGenerator:

    def __init__(self, quota_size: int, embedding_engine: str, output_path: str, domain: str,
                 tokenizer: tiktoken.Encoding, max_tokens: int):
        self.embedding_engine = embedding_engine
        self.quota_size = quota_size
        self.output_path = output_path
        self.domain = domain
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

    def generate(self, file_names) -> None:
        os.makedirs(self._get_output_path(), exist_ok=True)

        scraped_data_frame = self._create_scraped_file(file_names=file_names)

        # Tokenize the text and save the number of tokens to a new column
        scraped_data_frame['num_tokens'] = scraped_data_frame.text.apply(
            lambda x: len(self.tokenizer.encode(x)))

        # enforce max tokens for scraped data
        shortened_text_list = self._get_shortened_tokens(
            data_frame=scraped_data_frame, max_tokens=self.max_tokens)

        # create data frame from the shortened text
        embeddings_data_frame = pd.DataFrame(
            shortened_text_list, columns=['text'])

        if (len(embeddings_data_frame) > self.quota_size):
            logging.info('%i text chunks exceeds %i quota size. Taking first %i blocks',
                         len(embeddings_data_frame), self.quota_size, self.quota_size)
            embeddings_data_frame = embeddings_data_frame.iloc[0:self.quota_size, 0:]

        embeddings_data_frame['num_tokens'] = embeddings_data_frame.text.apply(
            lambda x: len(self.tokenizer.encode(x)))

        embeddings_data_frame['embeddings'] = embeddings_data_frame.text.apply(lambda x: openai.Embedding.create(
            input=x, engine=self.embedding_engine)['data'][0]['embedding'])

        self.embeddings_data_file = self._get_output_path() + '/embeddings.csv'

        # write embeddings out to disk
        embeddings_data_frame.to_csv(self.embeddings_data_file)

    def _get_output_path(self) -> str:
        return self.output_path + '/processed/' + self.domain

    def _remove_newlines(self, text: str) -> str:
        text = text.str.replace('\n', ' ')
        text = text.str.replace('\\n', ' ')
        text = text.str.replace('  ', ' ')
        text = text.str.replace('  ', ' ')

        return text

    def _create_scraped_file(self, file_names) -> pd.DataFrame:
        texts = []

        # Get all the text files in the text directory
        for file_name in file_names:
            # Open the file and read the text
            with open(file_name, 'r', encoding='UTF-8') as file:
                text = file.read()

                # drop last 4 chars of file_name and remove any directory prefix
                texts.append(
                    (file_name[file_name.rindex('/'):~4].replace('-', ' ').replace('_', ' '), text))

        # Create a dataframe from the list of texts
        data_frame = pd.DataFrame(texts, columns=['title', 'text'])

        # Set the text column to be the raw text with the newlines removed
        data_frame['text'] = data_frame.title + ". " + \
            self._remove_newlines(data_frame.text)

        self.scraped_file_name = self._get_output_path() + '/scraped.csv'

        data_frame.to_csv(self.scraped_file_name)

        return data_frame

    def _split_into_many(self, text: str, max_tokens: int):
        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        num_tokens = [len(self.tokenizer.encode(" " + sentence))
                      for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, num_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append('. '.join(chunk) + '.')
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1
        return chunks

    def _get_shortened_tokens(self, data_frame: pd.DataFrame, max_tokens: int) -> any:
        shortened = []

        # Loop through the dataframe
        for row in data_frame.iterrows():

            # If the text is None, go to the next row
            if row[1]['text'] is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row[1]['num_tokens'] > max_tokens:
                shortened += self._split_into_many(
                    text=row[1]['text'], max_tokens=max_tokens)

            # Otherwise, add the text to the list of shortened texts
            else:
                shortened.append(row[1]['text'])

        return shortened
