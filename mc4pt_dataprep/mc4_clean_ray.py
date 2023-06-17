"""
This script is used to clean the C4 dataset and save it in a format that is
accpeted by the EasyLM library. It is based on the script.
"""
import multiprocessing as mp
import random
import re
from typing import Any, List, Optional
from urllib.request import urlopen

import ftfy
import nltk
import pandas as pd
import ray
from absl import app, flags, logging
from etils import epath
from ray.data.datasource.file_meta_provider import DefaultFileMetadataProvider
from ray.util.queue import Queue
import uuid

# baixa as stopwords
nltk.download('stopwords')
stopwords = set(map(str.lower, nltk.corpus.stopwords.words('portuguese')))

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None,
                    'Output path for processed files, usually a GCS path.')
flags.DEFINE_integer(
    'max_urls_to_process', None,
    'Maximum urls to process. Useful for debugging. If None, process all urls.'
)
flags.DEFINE_string('max_input_block_size', '2 gb ',
                    'Maximum amount of data to read for each reading task.')
flags.DEFINE_integer(
    'parallelism', None, 'Number of parallel processes to use.'
    ' If None, use multiprocessing.cpu_count()')
flags.DEFINE_enum('split', 'validation', ['train', 'validation'],
                  'Which dataset split to process')
# how many input files to process into a single output file
flags.DEFINE_integer(
    'output_files_per_window', None,
    'How many output files to write for each input window. '
    'Default is None, which means 1 output file per input task.')
flags.DEFINE_integer('seed', 42, 'Base random seed for shuffling')

HUGGINGFACE_BASE_URL = 'https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/{}'
TRAIN_TFRECORD_BASENAME_PATTERN = 'c4-pt.tfrecord-{0:05d}-of-01024.json.gz'
VALID_TFRECORD_PATTERN = 'c4-pt-validation.tfrecord-{0:05d}-of-00004.json.gz'


def get_total_urls(split):
    """Get the total number of urls for a given split."""
    train_urls = map(
        lambda x: HUGGINGFACE_BASE_URL.format(
            TRAIN_TFRECORD_BASENAME_PATTERN.format(x)), range(1024))
    validation_urls = map(
        lambda x: HUGGINGFACE_BASE_URL.format(VALID_TFRECORD_PATTERN.format(x)
                                              ), range(4))
    logging.info('Loading %s urls', split)
    urls = sorted({'train': train_urls, 'validation': validation_urls}[split])
    logging.info('There are %d urls', len(urls))
    return urls


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def process_document(ex):
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
    text = ftfy.fix_text(text, normalization='NFKC')
    text = wikitext_detokenizer(text)
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
    # starting with a bullet point, or more than 30% ending with an ellipsis
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
    # that, have, with; this adequately deals with ostensibly English documents
    # that contain no coherent English text. (USE PORTUGUESE FOR THIS)
    num_stopwords = sum([w.lower() in stopwords for w in words])

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
        # original data
        # 'raw_text': ex['text'],
        'text': text,
        'num_words': num_words,
        # filters
        'num_words_filter': num_words_filter,
        'num_words_too_big_filter': num_words_too_big_filter,
        'num_words_too_small_filter': num_words_too_small_filter,
        'mean_word_length_filter': mean_word_length_filter,
        'symbol_ratio_filter': symbol_ratio_filter,
        'bullet_ratio_filter': bullet_ratio_filter,
        'ellipsis_end_ratio_filter': ellipsis_end_ratio_filter,
        'alpha_ratio_filter': alpha_ratio_filter,
        'num_stopwords_filter': num_stopwords_filter,
        'any_filter': any_filter,
    }


def calculate_filter_statistics(df: pd.DataFrame) -> dict:
    """Calculate filter statistics."""
    total_examples = len(df)
    cols_to_sum = df.head(0).drop(['text'], axis=1).columns
    out = df[cols_to_sum].sum().to_dict()
    out['total_examples'] = total_examples
    return out


def clean_and_calc_filter_statistics(df: pd.DataFrame,
                                     stats_queue: Queue) -> pd.DataFrame:
    """Clean and calculate filter statistics.

    Args:
        df (pd.DataFrame): dataframe with text column
        stats_queue (Queue): queue to put statistics

    Returns:
        pd.DataFrame: dataframe with text column
    """
    stats_before = calculate_filter_statistics(df)
    df = df[~df['any_filter']]
    stats_after = calculate_filter_statistics(df)
    stats = dict(stats_before=stats_before, stats_after=stats_after)
    stats_queue.put(stats, block=False)
    df = df[['text']]
    return df


# FILE_SIZE_FETCH_PARALLELIZATION_THRESHOLD


def get_download_size(url):
    """Get the size of a file without downloading it."""
    return int(urlopen(url).headers['Content-Length'])


class UrlFileMedataProvider(DefaultFileMetadataProvider):
    """Workaround for FILE_SIZE_FETCH_PARALLELIZATION_THRESHOLD limitation
    Will fetch files sizes for each file in parallel.
    """

    def expand_paths(self, paths, *args, **kwargs):
        paths_and_sizes = (ray.data.from_items(paths).map(lambda path: (
            path, get_download_size(path))).iterator().iter_rows())
        yield from paths_and_sizes


def process_urls(output_dir: str,
                 urls: List[str],
                 stats_queue: Queue,
                 meta_queue: Queue,
                 process_document_parallelism: int,
                 bytes_per_window: int = 2 * 1024**3,
                 seed: int = 42,
                 output_files_per_input_task: Optional[int] = None):
    """Process urls.

    Args:
        output_dir (str): output directory
        urls (list): urls
        stats_queue (Queue): queue to put statistics
        meta_queue (Queue): queue to put metadata
        bytes_per_window (int, optional): bytes per window. Defaults to
            2 * 1024**3.
        process_document_parallelism (int, optional): process document
            parallelism.
        seed (int, optional): seed. Defaults to 42.
        output_files_per_input_task (int, optional): output files per input
            task. Defaults to None (no repartitioning).
    """
    ds = ray.data.read_json(urls, meta_provider=UrlFileMedataProvider())
    pipe = ds.window(bytes_per_window=bytes_per_window)
    # full shuffle
    # pipe = pipe.foreach_window(lambda window: window.repartition(
    #     process_document_parallelism).map(process_document).map_batches(
    #         clean_and_calc_filter_statistics,
    #         fn_kwargs=dict(stats_queue=stats_queue),
    #         zero_copy_batch=True).random_shuffle(
    #             seed=seed, num_blocks=output_files_per_input_task))
    # shuffle each block
    def df_shuffle(df):
        return df.sample(n=len(df), random_state=seed, ignore_index=True)

    pipe = pipe.foreach_window(lambda window: window.repartition(
        process_document_parallelism)
        .map(process_document)
        .map_batches(
            clean_and_calc_filter_statistics,
            fn_kwargs=dict(stats_queue=stats_queue), zero_copy_batch=True)
        .map_batches(df_shuffle, zero_copy_batch=True)
        .repartition(output_files_per_input_task)
        .map_batches(df_shuffle, zero_copy_batch=True))
    metadata_collector = meta_queue
    pipe.write_parquet(output_dir,
                       compression='zstd',
                       metadata_collector=metadata_collector)


def disk_space_to_bytes(disk_space_string):
    """Convert disk space string to bytes."""
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}

    pattern = r'(\d+(\.\d+)?)\s*([a-zA-Z]+)'
    match = re.match(pattern, disk_space_string)
    if match:
        size = float(match.group(1))
        unit = match.group(3).upper()
    else:
        raise ValueError(
            'Invalid format. '
            'Please provide a valid disk space string (e.g., "10.5 GB").')
    size = float(size)
    unit = unit.upper()

    if unit not in units:
        raise ValueError(
            'Invalid unit. '
            'Please provide a valid disk space unit (e.g., GB, MB, KB).')

    return int(size * units[unit])


def queue_to_list(queue: Queue) -> List[Any]:
    """Convert queue to list."""
    out = []
    while not queue.empty():
        out.append(queue.get(block=False))
    return out


def collect_stats(queue):
    # out = pd.DataFrame([queue.get(block=False) for _ in range(queue.qsize())])
    out = pd.DataFrame(queue_to_list(queue))
    # out = out.groupby(lambda x: 0).agg(toolz.curried.merge_with(sum))
    return out


def move_parquet_files_to_final_dir(
        old_dir,
        new_dir,
        parallelism=None,
        shard_format='data-{0:05d}-of-{1:05d}.parquet'):
    """Rename parquet files to final directory.

    Args:
        old_dir (str): old directory
        new_dir (str): new directory. If same as old dir, will rename files and
            keep on same directory.
        parallelism (int, optional): parallelism. Defaults to None.
        shard_format (str, optional): shard format. Defaults to
            'data-{0:05d}-of-{1:05d}.parquet'.

    Returns:
        list: new paths
    """

    def rename_one(x):
        old_path, new_path = x
        if old_path != new_path:
            old_path.rename(new_path)
        return new_path

    logging.info('Renaming parquet files to final directory')

    old_paths = sorted(epath.Path(old_dir).glob('*.parquet'))
    logging.info('Found %d parquet files', len(old_paths))
    new_dir = epath.Path(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    old_and_new_paths = ray.data.from_items(
        [(path, new_dir /
          path.with_name(shard_format.format(i, len(old_paths))).name)
         for i, path in enumerate(old_paths)],
        parallelism=parallelism or mp.cpu_count())
    new_paths = old_and_new_paths.random_shuffle().map(rename_one).take_all()
    # will trhow error if old_dir is not empty
    # remove old dir if new dir is different
    if old_dir != new_dir:
        old_dir.rmdir()
    return new_paths


def main(_):
    ctx = ray.data.DatasetContext.get_current()
    ctx.use_push_based_shuffle = False
    ctx.execution_options.verbose_progress = True
    ctx.use_streaming_executor = True
    ctx.execution_options.locality_with_output = True

    # mp.set_start_method('spawn')
    total_urls = get_total_urls(FLAGS.split)
    random.Random(FLAGS.seed).shuffle(total_urls)

    if FLAGS.max_urls_to_process:
        urls_to_process = total_urls[:FLAGS.max_urls_to_process]
    else:
        urls_to_process = total_urls

    logging.info('Will process %d of %d urls', len(urls_to_process),
                 len(total_urls))

    # Write files to disk
    stats_queue = Queue()
    meta_queue = Queue()
    meta_queue.append = meta_queue.put_nowait
    tmp_dir = epath.Path(FLAGS.output_dir) / f'tmp-{uuid.uuid4()}'
    logging.info('Writing to tmp_dir: %s', tmp_dir)
    process_urls(
        output_dir=str(tmp_dir),
        urls=urls_to_process,
        stats_queue=stats_queue,
        meta_queue=meta_queue,
        bytes_per_window=disk_space_to_bytes(FLAGS.max_input_block_size),
        process_document_parallelism=FLAGS.parallelism or mp.cpu_count(),
        seed=FLAGS.seed,
        output_files_per_input_task=FLAGS.output_files_per_window)

    stats = collect_stats(stats_queue)
    meta = queue_to_list(meta_queue)
    logging.info('Stats: %s\n', str(stats))

    final_dir = epath.Path(FLAGS.output_dir) / FLAGS.split
    logging.info('Moving files to final directory %s', final_dir)
    move_parquet_files_to_final_dir(
        old_dir=tmp_dir,
        new_dir=final_dir,
        parallelism=FLAGS.parallelism or mp.cpu_count(),
        shard_format=FLAGS.split + '{0:05d}-of-{1:05d}.parquet')

    logging.info('Finished!')


if __name__ == '__main__':
    # set log to info
    logging.set_verbosity(logging.INFO)
    app.run(main)

# python3 mc4_clean_ray.py --split=validation --output_dir=mc4_val
# --output_files_per_window=1

# time python3 mc4_clean_ray.py --split=validation --output_dir=${GCS_BUCKET:-gs://your-bucket-name}/data/parquet_mc4_clean --output_files_per_window=1
# time python3 mc4_clean_ray.py \
# --split=train \
# --output_dir=${GCS_BUCKET:-gs://your-bucket-name}/data/parquet_mc4_clean \
# --max_urls_to_process=100 \
# --max_input_block_size=10gb \
# --output_files_per_window=10

# Testes locais
# time python3 mc4_clean_ray.py --split=validation --output_dir=parquet_mc4_clean --output_files_per_window=1
# Com full shuffle
# real    0m58.676s
# user    0m24.871s
# sys     0m15.146s