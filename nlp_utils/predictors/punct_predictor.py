import numpy as np
from nltk import sent_tokenize
from overrides import overrides
from typing import List, Iterator

from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance, Token

punctuation = ('.', ',', '!', '?', ';', ':')

decoding_dict = {
    'COMMA': ',',
    'FULL_STOP': '.',
    '_': ''
}


@Predictor.register('punct_predictor')
class PunctPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader):
        super().__init__(model=model, dataset_reader=dataset_reader)
        self.labels = '.,'

    @overrides
    def _json_to_instance(self, json_dict: dict) -> Instance:
        tokens = json_dict['tokens']
        tokens = [Token(token) if isinstance(token, str) else token
                  for token in tokens if token]
        return self._dataset_reader.text_to_instance(tokens)

    def _predict(self, json_dict: JsonDict) -> List[str]:
        if not json_dict:
            return []

        predictions = self.predict_json(json_dict)
        output = [
            self._model.vocab.get_token_from_index(
                np.argmax(token_logits).item(),
                namespace=self._model.label_namespace)
            for token_logits, mask in zip(predictions['logits'], predictions['mask']) if mask
        ]

        return output

    def __segment(self, tokens: List[str]) -> Iterator[List[str]]:
        for sentence in sent_tokenize(' '.join(tokens), 'russian'):
            yield [token.strip(self.labels) for token in sentence.split() if token not in self.labels]

    def predict(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []

        preds = [x for sent_tokens in self.__segment(tokens) for x in self._predict({'tokens': sent_tokens})]

        output = []
        for i, (token, label) in enumerate(zip(tokens, preds)):
            if not token.endswith(punctuation):
                output.append(token + decoding_dict[label])
                continue

            if output and token in punctuation and preds[i-1] != '_':
                output[-1] = tokens[i-1]
                output.append(token)
                continue

            output.append(token)

        return output
