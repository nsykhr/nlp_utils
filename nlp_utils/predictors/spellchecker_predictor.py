import os
import re
import numpy as np
from itertools import product
from string import punctuation
from overrides import overrides
from nltk import edit_distance, sent_tokenize
from typing import List, Tuple, Set, Iterator, Sequence, Union, Optional

from allennlp.models import Model
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance

punctuation += '«»—…“”№–'
punctuation_to_keep = re.compile(r'[.,]*([§!"#$%&\'()*+\-/:;<=>?@\[\]\\^_`{}|~«»—…“”№–]*)')

BEFORE_PUNCT = set('(<[{«“№')
AFTER_PUNCT = set('.,!%):;>?]}»…”')

SINGLE_CHAR_WORD = {
    'н': 'и',
    'й': 'и',
    'з': '3',
    'З': '3',
    '(': 'С',
    '0': 'о',
    '&': 'в',
    'р': 'в',
    '<': '«',
    '>': '»',
    '|': '1',
    'щ': 'и',
    'е': 'с',
    'ё': 'с'
}

DOUBLE_CHAR_WORD = {
    'ша': 'на',
    'па': 'на',
    'ее': 'её',
    '3а': 'за',
    '$а': 'за'
}


class SpellCheckerPredictor(Predictor):
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader,
                 use_postprocessing: bool = True,
                 distance_threshold: float = 0.2):
        super().__init__(model=model, dataset_reader=dataset_reader)
        self.use_postprocessing = use_postprocessing
        self.distance_threshold = distance_threshold
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
        tokens = list(map(lambda x: x.rstrip('.,').strip('"\'«»“”'), json_dict['tokens']))
        return self._dataset_reader.text_to_instance(tokens, add_noise=False, add_target=False)

    @staticmethod
    def __segment(boxes: List[str]) -> Iterator[List[str]]:
        return (sentence.split() for sentence in sent_tokenize(' '.join(boxes), 'russian'))

    @staticmethod
    def __check_punct(token: str) -> str:
        if len(token) < 2:
            return token

        if token[0] in AFTER_PUNCT:
            token = token[1:]

        if token[-1] in BEFORE_PUNCT:
            token = token[:-1]

        return token

    @staticmethod
    def __check_single_char_word(token: str) -> str:
        return SINGLE_CHAR_WORD.get(token, token)

    @staticmethod
    def __check_single_char_word_with_punct(token: str) -> str:
        no_punct_token = ''.join([c for c in token if c not in punctuation])

        if len(no_punct_token) == 1 and not no_punct_token.isdigit():
            no_punct_token = SINGLE_CHAR_WORD.get(no_punct_token, no_punct_token)
            token = ''.join([c if c in punctuation else no_punct_token for c in token])

        return token

    @staticmethod
    def __check_double_char_word(token: str) -> str:
        return DOUBLE_CHAR_WORD.get(token, token)

    def __preprocess_input(self, json_dict: JsonDict) -> JsonDict:
        tokens = []
        for token in json_dict['tokens']:
            token = token.strip()
            token = self.__check_punct(token)
            token = self.__check_single_char_word(token)
            if len(token) > 1:
                token = self.__check_single_char_word_with_punct(token)
            token = self.__check_double_char_word(token)
            tokens.append(token)

        json_dict['tokens'] = tokens

        return json_dict

    def _predict(self, json_dict: JsonDict, heights: Sequence[float]) -> List[str]:
        if not json_dict:
            return []

        json_dict = self.__preprocess_input(json_dict)
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

        return list(self.__decode_preds(json_dict['tokens'], predictions, heights)) \
            if self.use_postprocessing else [x[0].replace('ё', 'е').replace('Ё', 'Е') for x in predictions]

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

        raise ValueError(f'Unknown casing pattern: "{casing_pattern}".')

    def __replace_digits(self, token: str) -> str:
        casing_pattern = [char.isupper() for char in token]
        token_lower = token.lower()

        if self.all_digit2chars is None:
            digit2char = {value: [] for value in self.char2digit.values()}
            for key, value in self.char2digit.items():
                digit2char[value].append(key)

            self.all_digit2chars = [{k: v[idx] for idx, (k, v) in zip(indices, digit2char.items())}
                                    for indices in product(*[range(len(value)) for value in digit2char.values()])]

        for replacement_dict in self.all_digit2chars:
            candidate = token_lower[:]
            for key, value in replacement_dict.items():
                candidate = candidate.replace(key, value)
            if candidate in self.token_vocab_lowercase:
                return self.__restore_casing(candidate, casing_pattern)

        return token

    def __is_close(self, token: str, candidate: str) -> bool:
        return edit_distance(token.lower(), candidate.lower()) / len(candidate) <= self.distance_threshold

    def __correct_token(self, token: str, candidate: str, height: float, mean_height: float) -> str:
        if any(x in self.token_vocab_lowercase for x in (token.strip(punctuation).lower(),
                                                         token.strip(punctuation),
                                                         token.lower())):
            return token

        if candidate == '<DIGIT>' or token[0] in ('№', '§'):
            token = token.lower()
            for key, value in self.char2digit.items():
                token = token.replace(key, value)
            return token

        if not candidate.islower():
            casing_pattern = 'same'
        elif len(token) == len(candidate):
            casing_pattern = [char.isupper() for char in token]
        elif token.isupper() and height > 1.2 * mean_height:
            casing_pattern = 'upper'
        elif token.istitle():
            casing_pattern = 'capitalized'
        else:
            casing_pattern = 'lower'

        token = self.__replace_digits(token)

        if len(token) < self._dataset_reader.min_noise_length \
                or not self.__is_close(token, candidate) \
                or candidate in ('<LATIN>', self.oov_token) \
                or any(char not in self.character_vocab for char in token.lower()):
            return token

        candidate = self.__restore_casing(candidate, casing_pattern)

        return candidate

    def __decode_preds(self, tokens: List[str], preds: List[Tuple[str, str]], heights: Sequence[float]) -> Iterator[str]:
        mean_height = np.mean(heights).item()

        for i, (token, (candidate, _)) in enumerate(zip(tokens, preds)):

            if not token or all(char in punctuation for char in token):
                yield token
                continue

            left_punctuation = punctuation_to_keep.match(token).groups()[0]
            right_punctuation = punctuation_to_keep.match(token[::-1]).groups()[0][::-1]

            clean_token = self.__clean(token)
            corrected_token = self.__correct_token(clean_token, candidate, heights[i], mean_height)
            corrected_token = left_punctuation + corrected_token + right_punctuation

            if token.endswith((',', '.')):
                corrected_token += token[-1]

            yield corrected_token.replace('ё', 'е').replace('Ё', 'Е')

    def predict(self, boxes: List[str], heights: Sequence[float], segment: bool = False) -> List[str]:
        if all(not token for token in boxes):
            return boxes

        if segment:
            i = 0
            predictions = []
            for sent_tokens in self.__segment(boxes):
                sent_heights = heights[i:i + len(sent_tokens)]
                sent_preds = self._predict({'tokens': sent_tokens}, sent_heights)
                predictions.extend(sent_preds)
                i += len(sent_tokens)
        else:
            predictions = self._predict({'tokens': boxes}, heights)

        return predictions
