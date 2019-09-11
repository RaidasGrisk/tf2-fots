import os
import numpy as np
import config
import scipy.io as sio

# read data / path to gt.mat
dataset = sio.loadmat(config.FLAGS['training_data_path'] + 'gt.mat')

# 1. paths to imgs
img_paths = [i[0] for i in dataset['imnames'][0, :]]
with open('synthtext/path_to_imgs.txt', 'w') as f:
    for item in img_paths:
        f.write("%s\n" % item)

# 2. word list
dataset_word_list = []
for img_words in dataset['txt'][0, :]:
    img_word_list = []
    for words in img_words:
        for word in [item for sublist in [i.strip().split(' ') for i in words.split('\n')] for item in sublist]:
            if word != '':
                img_word_list.append(word)
    dataset_word_list.append(img_word_list)

# 3. polys and words
iter = 0
for polys, words, img_path in zip(dataset['wordBB'][0, :], dataset_word_list, img_paths):

    # fix poly
    if len(polys.shape) == 2:
        polys = polys[:, :, np.newaxis]
    polys = polys.transpose([-1, 0, 1])

    # check if all match up
    if polys.shape[0] != len(words):
        print('number of polys and words do not mathc')
        break

    # write file
    with open('synthtext/annotation/{}.txt'.format(img_path.split('/')[1].split('.')[0]), 'w') as f:
        for poly, word in zip(polys, words):
            line = np.around(poly, 1).ravel()
            line = np.concatenate([line, [word]], axis=0)
            for item in line:
                f.write("%s " % item)
            f.write("\n")

    iter += 1
    if iter % 100 == 0:
        print(iter, 'out of', len(img_paths))