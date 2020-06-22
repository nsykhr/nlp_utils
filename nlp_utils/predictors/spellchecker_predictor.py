import re
import numpy as np
from itertools import product
from string import punctuation
from overrides import overrides
from nltk import edit_distance, sent_tokenize
from typing import List, Tuple, Set, Iterator, Union, Callable, Optional

from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance

punctuation += '«»—…“”№–'
punctuation_to_keep = re.compile(r'[.,]*([!"#$%&\'()*+\-/:;<=>?@\[\]\\^_`{}|~«»—…“”№–]*)')

NAMED_ENTITIES = {
    'ФизЛицо',
    'НЮЛ',
    'ГосОрган',
    'Адрес'
}


class SpellCheckerPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 use_ner_model: bool = False,
                 ner_model: Optional[Callable] = None,
                 named_entities: Set[str] = None):
        super().__init__(model=model, dataset_reader=dataset_reader)
        self.use_ner_model = use_ner_model
        self.ner_model = ner_model
        self.named_entities = named_entities or NAMED_ENTITIES
        self.labels = '.,'
        self.decoding_dict = {
            'COMMA': ',',
            'FULL_STOP': '.',
            '_': ''
        }
        self.char2digit = {
            'о': '0',
            'г': '1',
            '!': '1',
            '|': '1',
            '?': '2',
            'з': '3',
            'э': '3',
            'б': '6',
            'ь': '6',
            '/': '7',
            'в': '8',
            '$': '8'
        }
        self.all_digit2chars = None

        self.oov_token = self._model.vocab._oov_token
        self.character_vocab = set(self._model.vocab.get_token_to_index_vocabulary('token_characters').keys())
        self.token_vocab_lowercase = set(map(str.lower, self._model.vocab.get_token_to_index_vocabulary(
            'target_tokens').keys()))

    @overrides
    def _json_to_instance(self, json_dict: dict) -> Instance:
        tokens = list(map(lambda x: x.rstrip('.,'), json_dict['tokens']))
        return self._dataset_reader.text_to_instance(tokens, add_noise=False, add_target=False)

    @staticmethod
    def __segment(boxes: List[str]) -> Iterator[List[str]]:
        return (sentence.split() for sentence in sent_tokenize(' '.join(boxes), 'russian'))

    def _predict(self, json_dict: JsonDict) -> List[str]:
        if not json_dict:
            return []

        predictions = self.predict_json(json_dict)
        predictions = [
            (self._model.vocab.get_token_from_index(
                np.argmax(spellchecker_logits).item() + 1,
                namespace='target_tokens'),

             self._model.vocab.get_token_from_index(
                 np.argmax(punct_logits).item(),
                 namespace=self._model.label_namespace))

            for spellchecker_logits, punct_logits, mask in zip(predictions['spellchecker_logits'],
                                                               predictions['punct_logits'],
                                                               predictions['mask']) if mask
        ]

        entities = self.ner_model(json_dict['tokens']) if self.use_ner_model else None

        return list(self.__decode_preds(json_dict['tokens'], predictions, entities))

    @staticmethod
    def __clean(token: str) -> str:
        return token.strip(punctuation)

    @staticmethod
    def __restore_casing(token: str, casing_pattern: Union[List[bool], str]) -> str:
        if isinstance(casing_pattern, list):
            return ''.join([char.upper() if upper else char for char, upper in zip(token, casing_pattern)])

        if casing_pattern == 'same':
            return token

        if casing_pattern == 'upper':
            return token.upper()

        if casing_pattern == 'capitalized':
            return token.capitalize()

        if casing_pattern == 'lower':
            return token.lower()

        raise ValueError(f'Unknown casing pattern {casing_pattern}.')

    def __replace_digits(self, token: str) -> str:
        if self.all_digit2chars is None:
            digit2char = {value: [] for value in self.char2digit.values()}
            for key, value in self.char2digit.items():
                digit2char[value].append(key)

            self.all_digit2chars = [{key: value[idx] for idx, (key, value) in zip(indices, digit2char.items())}
                                    for indices in product(*[range(len(value)) for value in digit2char.values()])]

        for replacement_dict in self.all_digit2chars:
            candidate = token[:]
            for key, value in replacement_dict.items():
                candidate = candidate.replace(key, value)
            if candidate.strip(punctuation).lower() in self.token_vocab_lowercase:
                return candidate

        return token

    def __correct_token(self, token: str, candidate: str) -> str:
        if candidate == '<DIGIT>':
            for key, value in self.char2digit.items():
                token = token.replace(key, value)
            return token

        if not candidate.islower():
            casing_pattern = 'same'
        elif len(token) == len(candidate):
            casing_pattern = [char.isupper() for char in token]
        elif token.isupper():
            casing_pattern = 'upper'
        elif token.istitle():
            casing_pattern = 'capitalized'
        else:
            casing_pattern = 'lower'

        token = self.__replace_digits(token)

        if len(token) < self._dataset_reader.min_noise_length \
                or edit_distance(token.lower(), candidate.lower()) > min(len(token), len(candidate)) * 0.4 \
                or candidate in ('<LATIN>', self.oov_token) \
                or any(char not in self.character_vocab for char in token.lower()):
            return token

        candidate = self.__restore_casing(candidate, casing_pattern)

        return candidate

    def __decode_preds(self, tokens: List[str], preds: List[Tuple[str, str]],
                       entities: Union[List[str], None]) -> Iterator[str]:
        for i, (token, (candidate, punct_label)) in enumerate(zip(tokens, preds)):

            if all(char in punctuation for char in token):
                yield token
                continue

            left_punctuation = punctuation_to_keep.match(token).groups()[0]
            right_punctuation = punctuation_to_keep.match(token[::-1]).groups()[0][::-1]

            clean_token = self.__clean(token)
            corrected_token = self.__correct_token(clean_token, candidate) \
                if entities is None or entities[i][2:] not in self.named_entities else clean_token
            corrected_token = left_punctuation + corrected_token + right_punctuation

            if token.endswith(','):
                corrected_token += ','
            elif token.endswith(self.decoding_dict[punct_label]):
                corrected_token += self.decoding_dict[punct_label]

            yield corrected_token

    def predict(self, boxes: List[str]) -> List[str]:
        if not boxes:
            return []

        predictions = []
        for sent_tokens in self.__segment(boxes):
            sent_preds = self._predict({'tokens': sent_tokens})
            predictions.extend(sent_preds)

        return predictions
