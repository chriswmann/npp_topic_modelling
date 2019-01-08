#!/usr/bin/env python3

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from pathlib import Path
from spacy.lang.en import English
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from wand.image import Image as WImage
from autocorrect import spell

import gensim
import matplotlib.pyplot as plt
import mimetypes
import numpy as np
import os
import re
import seaborn as sns
import spacy

home = str(Path.home())
cwd = os.getcwd()
nlp = spacy.load('en')

# documents to analyse
target = os.path.join(cwd, './scraped_docs/')
print('Target is: ' + target)

# read in document
def find_files(path):
    fnames = []
    if os.path.isdir(path):
        for dirpath, dirs, files in os.walk(path):
            for f in files:
                if mimetypes.guess_type(f)[0] == 'text/plain':
                    fnames.append(os.path.join(dirpath, f))
        return fnames
    else:
        return [path]

target_files = find_files(target)

if target_files:
    print('Target files are: ')
    for f in target_files:
        print(f)
else:
    path = os.path.abspath(os.path.join(os.getcwd(), target))
    print('No target files found at {}\nQuitting...'.format(path))
    quit()

STOP_WORDS = set('''
\'s \t -PRON- uk protective marking not protectively protectivel marked the a and next is will was be because not
chapter page report low stage use visitor s3000 m n\'t \'s \'m ca 10kV ni fcg3 ci + ° 0.00e+0 0.00e+00 e 0.00e+000 00e 000e
'''.split())

# A custom stoplist
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS) + list(STOP_WORDS))

text = []

print('Reading and cleaning target files...')
'''
The regex below includes upper case letters despite the text being converted to lower case.
This was a lazy work around because Gensim was picking up different cased words as different
topic elements.
'''
for f in target_files:
    with open(f, 'r') as fin:
        raw_text = fin.read().lower()
    # replace any non-alphanumeric character with a space
    raw_text = re.sub(r'[^a-zA-Z\d\s]', ' ', raw_text)
    #replace certain mixtures of alpha and numeric words with spaces
    raw_text = re.sub(r'\b[a-zA-Z]{1,3}[0-9]{1,2}\b|\b[0-9]{1,3}[a-zA-Z]{1,2}\b', ' ', raw_text)
    
    # remove stop words
    for s in STOPLIST:
        try:
            recomp = re.compile(r'\b%s\b' %s, re.IGNORECASE)
            raw_text = recomp.sub(' ', raw_text)
        except:
            pass
    # shrink all whitespace to a single space
    raw_text = re.sub(r'\s+', ' ', raw_text)
    text.append(raw_text)

texts, article = [], []
texts_spell, article_spell = [], []
print('Analysing text with Spacy and preparing texts for corpus...')
# analyse text via spacy nlp pipeline
'''
I'm not sure if this multithreading is implemented correctly.  It works but the documentation being 
analysed is too small to notice a difference in timing and I am not inclined to 
time it computationally.
'''
for doc in nlp.pipe(text, disable=['tagger', 'parser', 'ner', 'textcat'], batch_size=100, n_threads=4):
    for w in doc:
        '''
        double check for punctuation, numbers, stopwords
        '''
        if not w.is_punct and not w.like_num and not w.is_stop and len(w.text) > 3:
                article.append(w.lemma_)
                article_spell.append(spell(w.lemma_))
    if article:
        texts.append(article)
        texts_spell.append(article_spell)
        article = []
        article_spell = []

print('Generating corpus...')
# generate dictionary and corpus
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
dictionary_spell = Dictionary(texts_spell)
corpus_spell = [dictionary_spell.doc2bow(text) for text in texts_spell]

# HDP Hierarchical Dirichlet Process - unsupervised method that determines number of topics itself
print('HDP Hierarchical Dirichlet Process')
hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
with open('./tm_results.txt', 'w') as f:
    f.write('Without Spelling Correction\nHDP\n')
    for topic in hdp_model.show_topics(formatted=True):
        f.write('{}\t{}\n'.format(topic[0], topic[1]))

num_topics = len(hdp_model.show_topics())

# LSI Latent Symantex Indexing
print('LSI Latent Symantex Indexing')
lsi_model = LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
with open('./tm_results.txt', 'a') as f:
    f.write('LSI\n')
    for topic in lsi_model.show_topics(formatted=True):
        f.write('{}\t{}\n'.format(topic[0], topic[1]))

# LDA Latent Dirichlet Allocation
print('LDA Latent Dirichlet Allocation')
lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
with open('./tm_results.txt', 'a') as f:
    f.write('LDA\n')
    for topic in lda_model.show_topics(formatted=True):
        f.write('{}\t{}\n'.format(topic[0], topic[1]))

# HDP Hierarchical Dirichlet Process - unsupervised method that determines number of topics itself
print('HDP Hierarchical Dirichlet Process')
hdp_model = HdpModel(corpus=corpus_spell, id2word=dictionary_spell)
with open('./tm_results.txt', 'a') as f:
    f.write('\nWith Spelling Correction\nHDP\n')
    for topic in hdp_model.show_topics(formatted=True):
        f.write('{}\t{}\n'.format(topic[0], topic[1]))

num_topics = len(hdp_model.show_topics())

# LSI Latent Symantex Indexing
print('LSI Latent Symantex Indexing')
lsi_model = LsiModel(corpus=corpus_spell, num_topics=num_topics, id2word=dictionary_spell)
with open('./tm_results.txt', 'a') as f:
    f.write('LSI\n')
    for topic in lsi_model.show_topics(formatted=True):
        f.write('{}\t{}\n'.format(topic[0], topic[1]))

# LDA Latent Dirichlet Allocation
print('LDA Latent Dirichlet Allocation')
lda_model = LdaModel(corpus=corpus_spell, num_topics=num_topics, id2word=dictionary_spell)
with open('./tm_results.txt', 'a') as f:
    f.write('LDA\n')
    for topic in lda_model.show_topics(formatted=True):
        f.write('{}\t{}\n'.format(topic[0], topic[1]))


def computeCoherence():
    hdp_topics = [[word for word, prob in topic] for topicid, topic in hdp_model.show_topics(formatted=False)]
    lsi_topics = [[word for word, prob in topic] for topicid, topic in lsi_model.show_topics(formatted=False)]
    lda_topics = [[word for word, prob in topic] for topicid, topic in lda_model.show_topics(formatted=False)]
    
    hdp_coherence = CoherenceModel(topics=hdp_topics[:5], texts=texts, dictionary=dictionary, window_size=5).get_coherence()
    lsi_coherence = CoherenceModel(topics=lsi_topics[:5], texts=texts, dictionary=dictionary, window_size=5).get_coherence()
    lda_coherence = CoherenceModel(topics=lda_topics[:5], texts=texts, dictionary=dictionary, window_size=5).get_coherence()
    return lsi_coherence, hdp_coherence, lda_coherence
    
def generate_bar_graph(coherences, indices):
    sns.axes_style('white')
    sns.set_style('white')
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    ax = sns.barplot(x, coherences, palette="GnBu_d")
    ax.set(xlabel='Models', ylabel='Coherence Value')
    ax.set(xticklabels=indices)
    plt.show()



hdp_coherence, lsi_coherence, lda_coherence = computeCoherence()
generate_bar_graph([hdp_coherence, lsi_coherence, lda_coherence], ['HDP', 'LSI', 'LDA'])

hdp_topics = [topic for topic in hdp_model.show_topics(formatted=True)]
lsi_topics = [topic  for topic in lsi_model.show_topics(formatted=True)]
lda_topics = [topic for topic in lda_model.show_topics(formatted=True)]

print('\nHDP Topics: ')
for topic in hdp_topics:
    print(topic)

print('\nLSI Topics: ')
for topic in lsi_topics:
    print(topic)

print('\nLDA Topics: ')
for topic in lda_topics:
    print(topic)
