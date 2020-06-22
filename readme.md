# NLP Utils

This repository contains code for training and running two types of NLP models:

## 1. Punctuation restorer
This model treats the task of restoring punctuation as a sequence labeling task. By default, we use BERT to obtain token embeddings because it is known to encode structural data (syntax) that can be used to define the positions of specific punctuation tags in a sentence. By default, we only train the model to predict commas, full stops, and the absence of punctuation, because other punctuation marks may be ambiguous.

## 2. Spellchecker
This model is character-level, i. e., it obtains token embeddings via a Seq2Vec encoder from character embeddings. The specific model defined in this repository is a joint model that solves the task of predicting "clean" tokens as well as restoring punctuation. Note, however, that the metrics on the second task will likely be significantly lower compared to the first model, because this one does not use any pretrained contextual embeddings. One way of overcoming that is introducing said pretrained models into the model by using them alongside with token embeddings obtained from character embeddings.
