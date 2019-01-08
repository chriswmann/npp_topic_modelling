#!/usr/bin/env python3

from PIL import Image
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import re
import enchant


cwd = os.getcwd()
results = './tm_results.txt'
mask_path = './SSN-791.png'
default_figure_path = './topic_cloud.pdf'

with open(results, 'r') as f:
    raw_data = f.readlines()
d = enchant.Dict('en-GB')
topics = {}
for l in raw_data:
    if 'Spelling Correction' not in l:
        l = l.strip()
        if re.match(r'^[A-Z]{3}', l):
            model = l
            topics[model] = {}
        else:
            index = l.split('\t')[0]
            words_scores = l.split('\t')[1].strip()
            topics[model][index] = {}
            rank = 0
            max_topic_rank = 0
            topics[model][index]['topic_model'] = []
            topics[model][index]['normalised_topic_model'] = []
            for line in words_scores.split(' + '):
                s = abs(float(line.split('*')[0]))
                w = line.split('*')[1].strip().replace('"','')
                if d.check(w):
                    topics[model][index]['topic_model'].append((w, s))
                    rank += s
                    if rank > max_topic_rank:
                        max_topic_rank = rank
            topics[model][index]['rank'] = rank
            topics[model]['max_topic_rank'] = max_topic_rank
            for topic_model in topics[model][index]['topic_model']:
                normalised_score = topic_model[1] / topics[model]['max_topic_rank']
                topics[model][index]['normalised_topic_model'].append((topic_model[0], normalised_score))

pprint.pprint(topics)

plot_dict = {}
for model in topics.keys():
    plot_dict[model] = {}
    for index in topics[model]:
        if not index == 'max_topic_rank':
            plot_dict[model].update(dict(topics[model][index]['normalised_topic_model']))

def make_image(text, save=False, fname=default_figure_path):
    mask = np.array(Image.open(mask_path))


    wc = WordCloud(background_color="rgb(20,20,20)", max_words=1000, mask=mask)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if save:
        plt.savefig(fname, bbox_inches='tight', dpi=400, facecolor='w', edgecolor='w',
            orientation='landscape', papertype='A4', format=None,
            transparent=False)
    plt.show()

for k in sorted(plot_dict.keys()):
    fname = os.path.join(cwd, 'topic_cloud_' + k)
    make_image(plot_dict[k], save=True, fname=fname)

plot_dict_combined = {}
for model in topics.keys():
    for index in topics[model]:
        if not index == 'max_topic_rank':
            plot_dict_combined.update(dict(topics[model][index]['normalised_topic_model']))

fcombined = os.path.join(cwd, 'combined_topic_cloud')
make_image(plot_dict_combined, save=True, fname=fcombined)
