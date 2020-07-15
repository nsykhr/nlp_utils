# NLP Utils

This repository contains code for training and running two types of NLP models:

## 1. Punctuation restorer
This model treats the task of restoring punctuation as a sequence labeling task. By default, we use **BERT** to obtain token embeddings because it is known to encode structural information (syntax) that can be used to define the positions of specific punctuation tags in a sentence. By default, we only train the model to predict commas, full stops, and the absence of punctuation, because other punctuation marks may be ambiguous.

## 2. Spellchecker
This model expects character-level input and outputs token-level predictions. To do so, it obtains token embeddings from character embeddings via a Seq2Vec encoder. The specific model defined in this repository is a joint model that solves the task of predicting "clean" tokens as well as restoring punctuation (in other words, it optimizes the sum of the two losses during training). Note, however, that the metrics on the second task will likely be lower compared to the first model, because this one does not use any pretrained contextual embeddings. One way to overcome this is to introduce said pretrained models by using their token representations alongside those obtained from character embeddings. Preferably, we should use a model that was trained on a noisy corpus (such as DeepPavlov's **ConvBERT**) since the model expects noisy data as input.
