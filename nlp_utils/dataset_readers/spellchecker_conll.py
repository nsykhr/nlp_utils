import re
import numpy as np
from overrides import overrides
from typing import List, Tuple, Iterator, Dict, Union, Optional

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token, CharacterTokenizer
from allennlp.data.fields import TextField, SequenceLabelField, ArrayField
from allennlp.data.token_indexers import TokenIndexer, TokenCharactersIndexer


@DatasetReader.register('spellchecker_conll')
class SpellCheckerCONLLDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 max_sequence_length: int = 50,
                 min_noise_length: int = 1,
                 start_tokens: Optional[List[str]] = None,
                 end_tokens: Optional[List[str]] = None,
                 lazy: bool = True):
        super().__init__(lazy=lazy)

        self.start_tokens = start_tokens or []
        self.end_tokens = end_tokens or []
        self.empty_tags = ['_' for _ in range(len(start_tokens))] if start_tokens is not None else []

        assert len(self.start_tokens) == len(self.end_tokens), \
            'Start tokens and end tokens lists must be of the same length.'

        self._token_indexers = token_indexers or {
            'token_characters': TokenCharactersIndexer(character_tokenizer=CharacterTokenizer(
                start_tokens=['<SOT>'], end_tokens=['<EOT>']))
        }

        self.max_sequence_length = max_sequence_length
        self.min_noise_length = min_noise_length

        self.frequent_ocr_errors = {
            '!': ['1', '|'],
            '$': ['8', 'в'],
            '(': ['[', '{', 'с'],
            ')': [']', '}'],
            '/': ['7'],

            '0': ['о'],
            '1': ['!', '|', 'г'],
            '2': ['?', 'г'],
            '3': ['з', 'э'],
            '6': ['б', 'ь'],
            '7': ['/'],
            '8': ['$', 'в'],

            ':': [';'],
            ';': [':'],
            '<': ['«'],
            '>': ['»'],
            '?': ['2'],
            '[': ['(', '{', 'с'],
            ']': [')', '}'],
            '{': ['(', '[', 'с'],
            '|': ['!', '1'],
            '}': [')', ']'],
            '«': ['<', 'к'],
            '»': ['>', 'у', 'х'],

            'а': ['о', 'с', 'я'],
            'б': ['6', 'в', 'ь'],
            'в': ['$', '8', 'б', 'т'],
            'г': ['1', '2', 'т'],
            'д': ['а', 'л', 'п'],
            'е': ['в', 'о', 'с', 'ё'],
            'ё': ['е'],
            'ж': ['><', '}{', 'х'],
            'з': ['3', 'э'],
            'и': ['й', 'к', 'н', 'п'],
            'й': ['и', 'н'],
            'л': ['д', 'п', 'т'],
            'м': ['ч'],
            'н': ['е', 'и', 'й', 'п'],
            'о': ['0', 'н'],
            'п': ['д', 'л'],
            'с': ['(', '[', '{', 'о', 'т'],
            'т': ['в', 'г', 'е', 'м', 'о', 'с'],
            'у': ['ч'],
            'ф': ['ор'],
            'х': ['><', '}{', 'ж'],
            'ш': ['щ'],
            'щ': ['ш'],
            'ы': ['ъ|', 'ь|'],
            'ь': ['6', 'б', 'т'],
            'э': ['3', 'з'],
            'ю': ['|0', '|о', 'ь']
        }
        self.alphabet = list('абвгдежзийклмнопрстуфхцчшщъыьэюя')
        self.punctuation = list('.,')

    @overrides
    def text_to_instance(self, tokens: List[str], labels: List[str] = None,
                         add_noise: bool = True, add_target: bool = True) -> Instance:
        clean_tokens = self.clean_input(tokens)
        extra_tokens, extra_labels = [], []

        if add_noise:
            extra_tokens, extra_labels = self.__add_random_numeration(clean_tokens)
            noisy_tokens = list(self.__generate_errors(extra_tokens + clean_tokens))
            text_field = TextField([Token(t) for t in (self.start_tokens + noisy_tokens + self.end_tokens)],
                                   self._token_indexers)
            fields = {'tokens': text_field}

        else:
            text_field = TextField([Token(t) for t in (self.start_tokens + clean_tokens + self.end_tokens)],
                                   self._token_indexers)
            fields = {'tokens': text_field}

        token_lengths = self.get_token_lengths(text_field.tokens)
        fields['token_lengths'] = ArrayField(token_lengths)

        if add_target:
            target_tokens = extra_tokens + self.clean_target(tokens)
            target_field = SequenceLabelField(self.start_tokens +
                                              list(self.__mask_target_tokens(target_tokens)) + self.end_tokens,
                                              sequence_field=text_field, label_namespace='target_tokens')
            fields['target_tokens'] = target_field

        if labels is not None:
            labels = extra_labels + labels
            fields['punct_labels'] = SequenceLabelField(labels=self.empty_tags + labels + self.empty_tags,
                                                        sequence_field=text_field, label_namespace='punct_labels')

        return Instance(fields)

    @staticmethod
    def read_file(file_path: str) -> Iterator[Tuple[List[str], List[str]]]:
        with open(file_path, 'r', encoding='cp1251') as f:
            tokens, labels = [], []
            num_alphas = 0

            for line in f:
                line_stripped = line.strip()
                if not line_stripped:
                    # We are filtering out sentences that are too short or
                    # end with punctuation marks that actual sentences cannot end with.
                    if num_alphas >= 3 and labels[-1] in ('_', 'FULL_STOP') and \
                            not all(token.isupper() for token in tokens if token.isalpha()):
                        yield tokens, labels

                    tokens, labels = [], []
                    num_alphas = 0
                    continue

                token, label = line_stripped.split('\t')

                tokens.append(token)
                labels.append(label)

                if token.isalpha():
                    num_alphas += 1

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        for tokens, labels in self.read_file(file_path):
            if len(tokens) > self.max_sequence_length:
                for i in range(0, len(tokens) - self.max_sequence_length + 3, self.max_sequence_length // 2):
                    yield self.text_to_instance(
                        tokens[i:i + self.max_sequence_length],
                        labels[i:i + self.max_sequence_length]
                    )
                continue

            yield self.text_to_instance(tokens, labels)

    @staticmethod
    def __add_random_numeration(tokens: List[str]) -> Tuple[List[str], List[str]]:
        if not np.random.random() < 0.05 or tokens[0].isdigit() or tokens[0] in ('§', 'п', 'ч', 'ст', 'гл', 'т'):
            return [], []

        num_length = np.random.randint(1, 4)
        nums = [str(np.random.randint(1, 21)) for _ in range(num_length)]
        extra_labels = ['FULL_STOP' for _ in range(num_length)]

        return nums, extra_labels

    @staticmethod
    def __mask_target_tokens(tokens: List[str]) -> Iterator[str]:
        for token in tokens:
            if re.search(r'[a-z]', token, re.IGNORECASE) is not None:
                yield '<LATIN>'
            elif token.isdigit():
                yield '<DIGIT>'
            else:
                yield token

    @staticmethod
    def clean_input(tokens: List[str]) -> List[str]:
        return [token.lower() for token in tokens]

    @staticmethod
    def clean_target(tokens: List[str]) -> List[str]:
        return [token.lower().replace('ё', 'е') if not token.isupper() or len(token) == 1
                else token for token in tokens]

    @staticmethod
    def get_token_lengths(tokens: List[Union[str, Token]]) -> np.ndarray:
        return np.array([len(str(token)) for token in tokens])

    def __generate_errors(self, tokens: List[str]) -> Iterator[str]:
        for token in tokens:
            token_len = len(token)
            if token_len < self.min_noise_length or token in ('<LATIN>', '<DIGIT>'):
                yield token
                continue

            noisy_token = token[:]

            # This value approaches 1/4 when token length grows.
            error_prob = (token_len ** 2 - 3) / (token_len ** 2 * 4) if token_len > 1 else 0.1
            if np.random.random() < error_prob:
                # Number of possible errors depends on token length.
                if token_len < 4:
                    num_errors = 1
                elif token_len < 8:
                    num_errors = np.random.randint(1, 3)
                else:
                    num_errors = np.random.randint(1, 4)

                for _ in range(num_errors):
                    noisy_token = self.__add_noise_to_token(noisy_token)

            yield noisy_token

    def __add_noise_to_token(self, token: str) -> str:
        token_len = len(token)
        error_type = np.random.random()

        if error_type < 0.2 and token_len >= 2:
            # Random deletion
            delete_index = np.random.randint(token_len)
            token = token[:delete_index] + token[delete_index + 1:]

        elif error_type < 0.3:
            # Random insertion
            insert_index = np.random.randint(token_len + 1)
            insert_char = np.random.choice(self.alphabet)
            token = token[:insert_index] + insert_char + token[insert_index:]

        # Random punctuation insertion
        elif error_type < 0.4:
            insert_index = np.random.randint(1, max(2, token_len))
            insert_char = np.random.choice(self.punctuation)
            token = token[:insert_index] + insert_char + token[insert_index:]
        elif error_type < 0.45:
            token = np.random.choice(self.punctuation) + token
        elif error_type < 0.5:
            token = token + np.random.choice(self.punctuation)

        else:
            # Random replacement
            frequent_ocr_errors_indices = [i for i, char in enumerate(token) if char in self.frequent_ocr_errors]
            if frequent_ocr_errors_indices:
                index_to_replace = np.random.choice(frequent_ocr_errors_indices)

                return token[:index_to_replace] + \
                    np.random.choice(self.frequent_ocr_errors[token[index_to_replace]]) + token[index_to_replace + 1:]

            delete_index = np.random.randint(token_len)
            truncated_alphabet = [char for char in (self.alphabet + self.punctuation) if char != token[delete_index]]
            insert_char = np.random.choice(truncated_alphabet)
            token = token[:delete_index] + insert_char + token[delete_index + 1:]

        return token
