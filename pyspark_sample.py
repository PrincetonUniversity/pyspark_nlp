"""
Utility to download Moby Dick from Project Gutenberg,
parse it and create nGrams.
"""
import os
import re
import shutil
import sys

from nltk.tokenize.punkt import PunktSentenceTokenizer
from pyspark.ml.feature import NGram, StopWordsRemover, Tokenizer
from pyspark.sql.functions import explode
from pyspark.sql import SparkSession
import requests



DATA_DIR = 'data'
MOBY_BASE = 'https://www.gutenberg.org/files/2701/'
MOBY_TXT = '2701-0.txt'
MOBY_PATH = os.path.join(DATA_DIR, MOBY_TXT)

def get_moby():
    """Download the text to Moby Dick from Project Gutenberg,
    if it does not already exist."""
    if not os.path.exists(os.path.join(DATA_DIR)):
        sys.stdout.write('Creating download dir...\n')
        os.mkdir(DATA_DIR)

    if not os.path.exists(MOBY_PATH):
        res = requests.get('{}{}'.format(MOBY_BASE, MOBY_TXT))
        if res.status_code != 200:
            raise ValueError('Download did not complete successfully!')
        with open(MOBY_PATH, 'w') as fp:
            fp.write(res.text)
        sys.stdout.write('Downloaded Moby Dick text to 2701-0.txt')
    else:
        sys.stdout.write('Text file exists... skipping.\n')


def get_sentences():
    """Open the downloaded file and use nltk to split it up by sentence,
    stripping Project Gutenberg headers."""
    with open(MOBY_PATH, 'r') as fp:
        # this remove Gutenberg header and footer
        lines = [l.strip('\n') for l in fp.readlines()[848:21964]]
        sentences = PunktSentenceTokenizer().sentences_from_text(' '.join(lines))
        for i, sentence in enumerate(sentences):
            # format as tuples with id and string, expected format of
            # createDataFrame below
            sentences[i] = (i, re.sub(r'[^\w\s]','', sentence))
        return sentences



def main():
    # basic cleaning and getting of files
    get_moby()
    sentences = get_sentences()

    # create spark app, for use in iPython notebook OR as a standalone.
    spark = SparkSession\
        .builder\
        .appName("NGramSample")\
        .getOrCreate()

    # build a distributed dataframe
    sentence_df = spark.createDataFrame(sentences, ['id', 'sentences'])

    # create a tokenizer and write a 'words' column to DF
    tokenizer = Tokenizer(inputCol='sentences', outputCol='words')
    words = tokenizer.transform(sentence_df)

    # create ngram generators for bi, tri, and quad grams
    bigram = NGram(n=2, inputCol='words', outputCol='bigrams')
    trigram = NGram(n=3, inputCol='words', outputCol='trigrams')
    quadgram = NGram(n=4, inputCol='words', outputCol='quadgrams')

    # add each one in turn to the df
    bigrams = bigram.transform(words)
    trigrams = trigram.transform(bigrams)
    final = quadgram.transform(trigrams)

    # write as traversable JSON
    if os.path.exists('ngrams'):
        shutil.rmtree('ngrams')
    final.coalesce(1).write.json('ngrams')

    # as an example, write out quadgrams to CSV
    if os.path.exists('bigrams'):
        shutil.rmtree('bigrams')

    # This tricky bit selects bigrams, explodes it, and regroups by unique
    # bigram, then adds a count, after filtering out extremely uncommon bigrams
    # It finally writes to a CSV
    final.select('bigrams')\
        .withColumn('bigrams', explode('bigrams'))\
        .groupBy('bigrams').count().orderBy('count', ascending=False)\
        .filter('count > 10')\
        .coalesce(1).write.csv('bigrams')


if __name__ == '__main__':
    main()