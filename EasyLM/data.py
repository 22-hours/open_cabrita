import time
from functools import partial
import json
from multiprocessing import Pool

import mlxu
from ml_collections import ConfigDict
import numpy as np

from datasets import load_dataset

# required for dataset cleaning
import nltk
import ftfy

nltk.download('stopwords')
PT_STOPWORDS = set(map(str.lower, nltk.corpus.stopwords.words('portuguese')))


class DatasetFactory(object):
    """ Datset builder class. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.type = 'huggingface'
        config.text_processor = TextProcessor.get_default_config()
        config.huggingface_dataset = HuggingfaceDataset.get_default_config()
        config.json_dataset = JsonDataset.get_default_config()

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    @classmethod
    def load_dataset(cls, config, tokenizer, **kwargs):
        config = cls.get_default_config(config)
        text_processor = TextProcessor(config.text_processor, tokenizer)
        if config.type == 'huggingface':
            return HuggingfaceDataset(config.huggingface_dataset, tokenizer,
                                      text_processor, **kwargs)
        elif config.type == 'json':
            return JsonDataset(config.json_dataset, tokenizer, text_processor,
                               **kwargs)
        elif config.type == 'huggingface_clean':
            return HuggingfaceCleanDataset(config.huggingface_dataset,
                                           tokenizer, text_processor, **kwargs)
        # TODO: Add seqio type here
        elif config.type == 'seqio':
            raise NotImplementedError('Seqio is not yet supported.')
            return SeqioTaskDataset(config.task_or_mixture, **kwargs)
        else:
            raise ValueError(f'Unknown dataset type: {config.type}')

    def __init__(self):
        raise ValueError(
            'DatasetFactory is a static class and should not be instantiated.')


class TextProcessor(object):
    """ Example processor that converts a dictionary of texts into tokens. """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.fields_from_example = ''
        config.fields = ''
        config.subfield_separator = ' '
        config.add_bos_token = True
        config.add_eos_token = True
        config.prepend_text = ''
        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer):
        self.config = self.get_default_config(config)
        assert self.config.fields != '' or self.config.fields_from_example != '', (
            'Either fields or fields_from_example must be specified.')
        self.tokenizer = tokenizer

    def __call__(self, example, has_aux=False):
        if has_aux:
            example, *aux = example
        else:
            aux = tuple()
        token_buffer = []
        loss_mask_buffer = []

        if self.config.add_bos_token:
            token_buffer.append(self.tokenizer.bos_token_id)
            loss_mask_buffer.append(0.0)

        if self.config.fields_from_example != '':
            fields = example[self.config.fields_from_example].split(',')
        else:
            fields = self.config.fields.split(',')

        for i, field in enumerate(fields):
            if field.startswith('[') and field.endswith(']'):
                # No loss for this field.
                field = field[1:-1]
                mask = 0.0
            else:
                mask = 1.0

            if field == '<|bos|>':
                token_buffer.append(self.tokenizer.bos_token_id)
                loss_mask_buffer.append(mask)
            elif field == '<|eos|>':
                token_buffer.append(self.tokenizer.eos_token_id)
                loss_mask_buffer.append(mask)
            else:
                subfields = field.split('+')
                text = self.config.subfield_separator.join(
                    [example[subfield] for subfield in subfields])
                if i == 0:
                    text = self.config.prepend_text + text
                tokens = self.tokenizer.encode(text)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend([mask for _ in range(len(tokens))])

        if self.config.add_eos_token:
            token_buffer.append(self.tokenizer.eos_token_id)
            loss_mask_buffer.append(1.0)

        return token_buffer, loss_mask_buffer, *aux


class HuggingfaceDataset(object):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = 'c4'
        config.name = 'en'
        config.split = 'train'
        config.streaming = False
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(self.config.path,
                                     name,
                                     split=split,
                                     streaming=self.config.streaming)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens':
                        np.array(token_buffer[:chunk_size],
                                 dtype=np.int32).reshape(
                                     self.config.batch_size, -1),
                        'target_tokens':
                        np.array(token_buffer[1:chunk_size + 1],
                                 dtype=np.int32).reshape(
                                     self.config.batch_size, -1),
                        'loss_masks':
                        np.array(loss_mask_buffer[1:chunk_size + 1],
                                 dtype=np.float32).reshape(
                                     self.config.batch_size, -1),
                    }
                    if self.config.always_start_with_bos:
                        batch['input_tokens'][:,
                                              0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)


class JsonDataset(object):
    """ JSON dataset, where each line of the data file contains a JSON
        dictionary with text fields.
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = ''
        config.seq_length = 1024
        config.batch_size = 8
        config.always_start_with_bos = False
        config.start_seek_loc = 0
        config.example_index_at_start = 0
        config.tokens_count_at_start = 0
        config.tokenizer_processes = 1
        config.tokenizer_parallel_chunk_size = 32
        config.tokenizer_parallel_batch_size = 1024
        config.throughput_average_window_size = 200

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        assert self.config.path != ''
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._index = self.config.example_index_at_start
        self._file_loc = self.config.start_seek_loc
        self._total_tokens = self.config.tokens_count_at_start

    def parse_json(self, line):
        if not line or line == '\n':
            return None
        try:
            data = json.loads(line)
        except json.decoder.JSONDecodeError:
            print(f'Error parsing json line:\n{line}')
            return None
        return data

    def json_iterator(self):
        with mlxu.open_file(self.config.path, 'r') as fin:
            fin.seek(self._file_loc)
            while True:
                line = fin.readline()
                self._file_loc = fin.tell()
                if not line:  # Reached EOF
                    self._index = 0
                    fin.seek(0)
                    continue

                data = self.parse_json(line)
                if data is not None:
                    # JSON parsing succeeded
                    yield data, self._file_loc, self._index
                self._index += 1

    def batched(self, iterator, batch_size):
        batch = []
        for example in iterator:
            batch.append(example)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def parallel_example_iterator(self):
        if self.config.tokenizer_processes == 1:
            for example, loc, index in self.json_iterator():
                yield self.text_processor((example, loc, index), has_aux=True)
        else:
            process_pool = Pool(self.config.tokenizer_processes)
            batched_iterator = self.batched(
                self.json_iterator(),
                self.config.tokenizer_parallel_batch_size)
            with process_pool as pool:
                map_fn = partial(self.text_processor, has_aux=True)
                next_batch = pool.map_async(
                    map_fn,
                    next(batched_iterator),
                    chunksize=self.config.tokenizer_parallel_chunk_size)
                while True:
                    current_batch = next_batch
                    next_batch = pool.map_async(
                        map_fn,
                        next(batched_iterator),
                        chunksize=self.config.tokenizer_parallel_chunk_size)
                    for example in current_batch.get():
                        yield example

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        token_buffer = []
        loss_mask_buffer = []
        last_time = 0.0
        step_times = []
        start_time = time.time()
        start_tokens = self._total_tokens
        for tokens, loss_masks, loc, index in self.parallel_example_iterator():
            token_buffer.extend(tokens)
            loss_mask_buffer.extend(loss_masks)
            while len(token_buffer) > chunk_size + 1:
                self._total_tokens += chunk_size
                step_times.append(time.time() - last_time)
                last_time = time.time()
                if len(step_times
                       ) > self.config.throughput_average_window_size:
                    step_times = step_times[-self.config.
                                            throughput_average_window_size:]
                average_throughput = chunk_size / np.mean(step_times)
                accumulated_throughput = ((self._total_tokens - start_tokens) /
                                          (time.time() - start_time))
                metrics = {
                    'dataset_file_loc': loc,
                    'dataset_example_index': index,
                    'dataset_total_tokens': self._total_tokens,
                    'dataset_accumulated_tps': accumulated_throughput,
                    'dataset_average_tps': average_throughput,
                }
                batch = {
                    'input_tokens':
                    np.array(token_buffer[:chunk_size],
                             dtype=np.int32).reshape(self.config.batch_size,
                                                     -1),
                    'target_tokens':
                    np.array(token_buffer[1:chunk_size + 1],
                             dtype=np.int32).reshape(self.config.batch_size,
                                                     -1),
                    'loss_masks':
                    np.array(loss_mask_buffer[1:chunk_size + 1],
                             dtype=np.float32).reshape(self.config.batch_size,
                                                       -1),
                }
                if self.config.always_start_with_bos:
                    batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                yield batch, metrics
                token_buffer = token_buffer[chunk_size:]
                loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(
            config=self.config,
            index=self._index,
            file_loc=self._file_loc,
            total_tokens=self._total_tokens,
        )

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))
        self._index = state_dict.get('index',
                                     self.config.example_index_at_start)
        self._file_loc = state_dict.get('file_loc', self.config.start_seek_loc)
        self._total_tokens = state_dict.get('total_tokens',
                                            self.config.tokens_count_at_start)

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def vocab_size(self):
        return len(self.tokenizer)


class HuggingfaceCleanDataset(HuggingfaceDataset):
    """ Huggingface dataset, where the dataset is loaded using the huggingface
        datasets.load_dataset() function.

    This is a modified version of the HuggingfaceDataset:
    - ftfy is used for cleaning the text
    - we apply MassiveText filters to the dataset
    - preprocessing supposes mc4 dataset
    - we supposed streaming data and shuffle the dataset
    """

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.path = '22h/mc4_pt'
        config.name = 'pt-train'
        config.split = 'train'
        config.streaming = True
        config.seq_length = 2048
        config.batch_size = 8
        config.always_start_with_bos = False
        # new configs
        config.max_examples = None
        config.shuffle = True
        config.seed = 12345
        config.shuffle_buffer_size = 10_000

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor

        self._dataset = load_dataset(self.config.path,
                                     name,
                                     split=split,
                                     streaming=self.config.streaming)
        # shuffle and slicing are different for map and iterable datasets
        if self.config.streaming:
            if self.config.shuffle:
                self._dataset = self._dataset.shuffle(
                    seed=self.config.seed,
                    buffer_size=self.config.shuffle_buffer_size)
            if self.config.max_examples is not None:
                self._dataset = self._dataset.take(self.config.max_examples)
        else:
            if self.config.shuffle:
                self._dataset = self._dataset.shuffle(seed=self.config.seed)
            if self.config.max_examples is not None:
                self._dataset = self._dataset.select(
                    range(self.config.max_examples))
        # clean and filter are the same
        self.dataset = self._dataset.map(self.clean_document)
        self._dataset = self._dataset.filter(lambda ex: not ex['any_filter'])
        self._dataset = self._dataset.remove_columns(['any_filter'])

    def clean_document(self, ex):
        """Clean and filter a single document."""
        # flake8: noqa: E501
        """Process a single document, returning a dictionary of features.

        If the document is valid, we also tokenize it and return the tokenized.
        version. If the document is invalid, we return None on 'input_ids' key.

        We add information for posterior filtering.

        # Like Maritaca AI's Sabia J, we use most of the filters from MassiveText https://arxiv.org/abs/2112.11446.
        # Much of the web also comprises social media content, which can
        # variously lack context, coherence, or substance. To remove low-quality data while minimising potential
        # for bias, we apply a number of simple, easily understood heuristic filters: we remove any document
        # that does not contain between 50 and 100,000 words, or whose mean word length is outside the
        # range of 3 to 10 characters; we remove any document with a symbol-to-word ratio greater than 0.1
        # for either the hash symbol or the ellipsis; and we remove any document with more than 90% of lines
        # starting with a bullet point, or more than 30% ending with an ellipsis. We also require that 80%
        # of words in a document contain at least one alphabetic character, and apply a "stop word" filter, to
        # remove documents that do not contain at least two of the following English words: the, be, to, of, and,
        # that, have, with; this adequately deals with ostensibly English documents that contain no coherent
        # English text.
        """
        # only save necessary data on dictionary
        text = ex['text']
        # usado no gptj, meio ruim
        # text = ftfy.fix_text(text, normalization='NFKC')
        # text = wikitext_detokenizer(text)
        text = ftfy.fix_text(text)
        words = text.split()
        word_lengths = [len(w) for w in words]
        mean_word_length = sum(word_lengths) / len(word_lengths)
        num_words = len(word_lengths)
        # we remove any document with a symbol-to-word ratio greater than 0.1
        # for either the hash symbol or the ellipsis
        num_hash = text.count('#')
        num_ellipsis = text.count('...')
        num_symbols = num_hash + num_ellipsis
        symbol_ratio = num_symbols / num_words
        # we remove any document with more than 90% of lines
        # starting with a bullet point, or more than 30% ending with an
        # ellipsis
        num_bullet = text.count('\n*')
        num_ellipsis_end = text.count('...\n')
        num_lines = text.count('\n') + 1
        bullet_ratio = num_bullet / num_lines
        ellipsis_end_ratio = num_ellipsis_end / num_lines
        # We also require that 80%
        # of words in a document contain at least one alphabetic character
        num_alpha = sum([any(c.isalpha() for c in w) for w in words])
        alpha_ratio = num_alpha / num_words
        # apply a "stop word" filter, to remove documents that do not contain at
        # least two of the following English words: the, be, to, of, and,
        # # that, have, with; this adequately deals with ostensibly English
        # documents that contain no coherent English text.
        # (WE ADAPT THIS FOR PORTUGUESE)
        num_stopwords = sum([w.lower() in PT_STOPWORDS for w in words])

        # apply filters
        num_words_filter = not (50 <= num_words <= 100_000)
        num_words_too_big_filter = num_words > 100_000
        num_words_too_small_filter = num_words < 50
        mean_word_length_filter = not (3 <= mean_word_length <= 10)
        symbol_ratio_filter = symbol_ratio > 0.1
        bullet_ratio_filter = bullet_ratio > 0.9
        ellipsis_end_ratio_filter = ellipsis_end_ratio > 0.3
        alpha_ratio_filter = alpha_ratio < 0.8
        num_stopwords_filter = num_stopwords < 2

        # combine filters
        any_filter = any([
            num_words_filter,
            num_words_too_big_filter,
            num_words_too_small_filter,
            mean_word_length_filter,
            symbol_ratio_filter,
            bullet_ratio_filter,
            ellipsis_end_ratio_filter,
            alpha_ratio_filter,
            num_stopwords_filter,
        ])

        return {
            'text': text,
            # 'num_words': num_words,
            # filters
            # 'num_words_filter': num_words_filter,
            # 'num_words_too_big_filter': num_words_too_big_filter,
            # 'num_words_too_small_filter': num_words_too_small_filter,
            # 'mean_word_length_filter': mean_word_length_filter,
            # 'symbol_ratio_filter': symbol_ratio_filter,
            # 'bullet_ratio_filter': bullet_ratio_filter,
            # 'ellipsis_end_ratio_filter': ellipsis_end_ratio_filter,
            # 'alpha_ratio_filter': alpha_ratio_filter,
            # 'num_stopwords_filter': num_stopwords_filter,
            'any_filter': any_filter,
        }


# will implement this on v2
class SeqioTaskDataset(object):
    """Dataset generated by a seqio task."""

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.task_or_mixture = 'COLOQUE_AQUI'
        config.split = 'train'  # train or validation
        config.seq_length = 2048
        config.batch_size = 8
        config.seed = 12345
        config.always_start_with_bos = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, tokenizer, text_processor):
        self.config = self.get_default_config(config)
        name = self.config.name if self.config.name != '' else None
        split = self.config.split if self.config.split != '' else None
        self._tokenizer = tokenizer
        self._text_processor = text_processor
        self._dataset = load_dataset(self.config.path,
                                     name,
                                     split=split,
                                     streaming=self.config.streaming)

    def __iter__(self):
        chunk_size = self.config.batch_size * self.config.seq_length
        total_tokens = 0
        while True:
            token_buffer = []
            loss_mask_buffer = []
            for index, example in enumerate(self._dataset):
                tokens, loss_masks = self.text_processor(example)
                token_buffer.extend(tokens)
                loss_mask_buffer.extend(loss_masks)
                while len(token_buffer) > chunk_size + 1:
                    total_tokens += chunk_size
                    metrics = {
                        'dataset_example_index': index,
                        'dataset_total_tokens': total_tokens,
                    }
                    batch = {
                        'input_tokens':
                        np.array(token_buffer[:chunk_size],
                                 dtype=np.int32).reshape(
                                     self.config.batch_size, -1),
                        'target_tokens':
                        np.array(token_buffer[1:chunk_size + 1],
                                 dtype=np.int32).reshape(
                                     self.config.batch_size, -1),
                        'loss_masks':
                        np.array(loss_mask_buffer[1:chunk_size + 1],
                                 dtype=np.float32).reshape(
                                     self.config.batch_size, -1),
                    }
                    if self.config.always_start_with_bos:
                        raise NotImplementedError(
                            'always_start_with_bos=True is not implemented')
                        # batch['input_tokens'][:, 0] = self.tokenizer.bos_token_id
                    yield batch, metrics
                    token_buffer = token_buffer[chunk_size:]
                    loss_mask_buffer = loss_mask_buffer[chunk_size:]

    def get_state_dict(self):
        return dict(config=self.config)

    def load_state_dict(self, state_dict):
        if 'config' in state_dict:
            self.config.update(ConfigDict(state_dict['config']))

    @property
    def seq_length(self):
        return self.config.seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def text_processor(self):
        return self._text_processor

    @property
    def dataset(self):
        return self._dataset

    @property
    def vocab_size(self):
        return len(self._tokenizer)
