"""
This script takes tokenied documents as inputs and converts them to TFRecords to
be used for training and evaluation.
"""

import datetime
import functools
import json
import math
import multiprocessing as mp
import random
import sys
from copy import deepcopy

import more_itertools
import numpy as np
import polars as pl
import tensorflow as tf
import toolz
from absl import app, flags, logging
from etils import epath

# this will also add abseil flags to the global namespace
from process_filter_mc4pt_docs import recursive_load_json

# ['output_dir', 'parallelism', 'split', 'max_input_files_per_output_file', 'seed']
# are all flags defined in the process_filter_mc4pt_docs.py script
FLAGS = flags.FLAGS
flags.DEFINE_string('input_dir', None,
                    'Input path for processed files, usually a GCS path.')
flags.DEFINE_integer(
    'max_input_files_to_process', None,
    'Maximum input files to process. Useful for debugging. If None, process all input files.'
)
flags.DEFINE_integer(
    'min_input_files_to_process', None, 'Minimum input files to process.'
    ' Will exit if less than this number of files are found.')

flags.DEFINE_integer(
    'records_per_output_file', None,
    'Records per output file. Use this to avoid big/small files. '
    ' This is a guesstimate, the actual number of records per file may vary.')
flags.DEFINE_integer(
    'max_bytes_read_at_once', None,
    'Maximum bytes to read at once. This value will be inferred from'
    ' estimated_bytes_size on input_dir metadatas. Use biggest as possible for'
    ' better shuffle and speed processing.')
flags.DEFINE_integer('sequence_length', 2049, 'Sequence length on tfrecords.')
flags.DEFINE_integer('min_unique_tokens', 200,
                     'Minimum number of unique tokens per sequence.')
# flags.DEFINE_string('output_dir', None,
#                     'Output path for processed files, usually a GCS path.')
# flags.DEFINE_integer(
#     'max_urls_to_process', None,
#     'Maximum urls to process. Useful for debugging. If None, process all urls.'
# )
# flags.DEFINE_integer(
#     'parallelism', None, 'Number of parallel processes to use.'
#     ' If None, use multiprocessing.cpu_count()')
# flags.DEFINE_enum('split', 'validation', ['train', 'validation'],
#                   'Which dataset split to process')
# # how many input files to process into a single output file
# flags.DEFINE_integer(
#     'max_input_files_per_output_file', 1,
#     'How many input files to process into a single output file')
# flags.DEFINE_integer('seed', 42, 'Base random seed for shuffling')


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(example):
    feature = {"text": _int64_feature(example)}
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()


def write_tfrecords(serialized_examples, path):
    """Write serialized examples to a TFRecord file"""
    total_bytes = 0
    with tf.io.TFRecordWriter(path) as writer:
        for ex in serialized_examples:
            writer.write(ex)
            total_bytes += len(ex)
    logging.info('Wrote %d records and %d bytes to %s',
                 len(serialized_examples), total_bytes, path)
    return total_bytes


def serialize_batch_of_examples(examples):
    return np.array([serialize_example(ex) for ex in examples], dtype=object)


def flatten_and_chunk_numpy(sequences_of_tokens,
                            sequence_length=2049,
                            seed=42,
                            min_unique_tokens=200,
                            copy=False):
    """Flatten tokens from all documents and generate sequences of fixed lengths"""
    # reusing variable out to avoiding wasting memory
    # if using to_numpy(): data is already clone
    # if using view(): data is not cloned ("use if you know what you are doing")
    if copy:
        out = deepcopy(sequences_of_tokens)
    else:
        out = sequences_of_tokens

    logging.info('Input shape: {}'.format(out.shape))
    # first we shuffle order of documents
    logging.info('Shuffling documents')
    np.random.default_rng(seed=seed).shuffle(out)
    # then we flatten all documents
    logging.info('Flattening documents')
    out = np.concatenate(out)
    logging.info('Total number of tokens: %d', len(out))
    # chunk into sequences of fixed length
    logging.info('Chunking documents into sequences of length %d',
                 sequence_length)
    n_full, trailing = divmod(len(out), sequence_length)
    logging.info('Number of full sequences: %d', n_full)
    logging.info('Number of trailing tokens (dropped): %d', trailing)
    out = out[:n_full * sequence_length].reshape(n_full, sequence_length)
    np.random.default_rng(seed=seed).shuffle(out)
    # shuffle again the flattened documents
    if min_unique_tokens:
        logging.info('Removing sequences with less than %d unique tokens',
                     min_unique_tokens)
        # remove sequences with less than min_unique_tokens
        min_unique_mask = np.array(
            [len(np.unique(x)) >= min_unique_tokens for x in out])
        out = out[min_unique_mask]
        print('Removed {} sequences with less than {} unique tokens'.format(
            (~min_unique_mask).sum(), min_unique_tokens))
    logging.info('Output shape: {}'.format(out.shape))
    return out

# Nem usa
# def np_batch_split(arr, batch_size):
#     """Split an array into batches of size batch_size. Useful splitting large
#         arrays into smaller chunks for parallel processing.
#     """
#     # at least one batch
#     n_batches = max(len(arr) // batch_size, 1)
#     splits = np.array_split(arr, n_batches)
#     logging.info('Split array of length %d into %d batches', len(arr),
#                  n_batches)
#     return splits


# too slow and complicated
def flatten_and_chunk_polars(df,
                             sequence_length=2049,
                             seed=42,
                             min_unique_tokens=200):
    df = df.sample(n=len(df), seed=seed, shuffle=True)
    # explode and group by chunks of 2049
    df = (df.select(pl.col('input_ids').flatten()).with_row_count(
        name='group_id').with_columns(pl.col('group_id') // sequence_length))
    df = (df.groupby('group_id', maintain_order=True).agg(
        pl.col('input_ids'),
        pl.count().alias('group_size')).filter(
            pl.col('group_size').eq(sequence_length)))
    df = (df.with_columns(
        pl.col('input_ids').arr.unique().arr.lengths().
        alias('unique_tokens')).filter(
            pl.col('unique_tokens').ge(min_unique_tokens)).select('input_ids'))
    df = df.sample(n=len(df), seed=seed, shuffle=True)

    return df


# %time df2 = flatten_and_chunk_polars(df, 2049, 42, 200)
# CPU times: user 3min 23s, sys: 44.2 s, total: 4min 7s
# Wall time: 2min 14s

# %time df3 = flatten_and_chunk_numpy(df['input_ids'].to_numpy(), copy=True)
# CPU times: user 33.6 s, sys: 5.15 s, total: 38.7 s
# Wall time: 38.7 s

# %time df3 = flatten_and_chunk_numpy(df['input_ids'].to_numpy(), copy=False)
# CPU times: user 28.8 s, sys: 1.06 s, total: 29.8 s
# Wall time: 29.8 s

# %time df = pl.read_parquet(epath.Path('${GCS_BUCKET:-gs://your-bucket-name}/data/mc4pt_clean_docs/train/data/data_1.parquet').read_bytes(), use_pyarrow=True)

# flatten_and_chunk_numpy(df['input_ids'].to_numpy(), copy=False)
# tokens_matrix_batches = np_batch_split(flatten_and_chunk_numpy(df['input_ids'].to_numpy(), copy=False), mp.cpu_count())


def read_parquet_input_ids_to_numpy(path):
    """Read input_ids from parquet file and return as numpy array"""
    logging.info('Starting to read %s', path)
    out = pl.read_parquet(epath.Path(path).read_bytes(),
                          use_pyarrow=True)['input_ids'].to_numpy()
    logging.info('Finished reading data with shape %s from %s', out.shape,
                 path)
    return out


def main(argv):
    output_dir = epath.Path(FLAGS.output_dir)
    input_dir = epath.Path(FLAGS.input_dir)
    df_input_metas = recursive_load_json(input_dir / 'metadata')
    assert not df_input_metas.is_empty()
    df_old_output_metas = recursive_load_json(output_dir / 'metadata')
    total_input_files = df_input_metas['output_file'].to_list()
    if df_old_output_metas.is_empty():
        logging.info('No already processed stats files found')
        files_to_process = df_input_metas['output_file'].to_list()
        current_file_idx = 1
        output_dir.joinpath('data').mkdir(parents=True, exist_ok=True)
        output_dir.joinpath('metadata').mkdir(parents=True, exist_ok=True)
    else:
        already_processed_files = set(
            df_old_output_metas.select(
                pl.col('input_files').flatten())['input_files'])
        files_to_process = [
            path for path in total_input_files
            if path not in already_processed_files
        ]
        current_file_idx = df_old_output_metas['idx_file'].max() + 1

    # shuffle files to process
    random.Random(FLAGS.seed).shuffle(files_to_process)

    logging.info('There are %d files to process', len(files_to_process))
    input_files_by_status = {
        'total': len(total_input_files),
        'processed': len(total_input_files) - len(files_to_process),
        'to_process': len(files_to_process)
    }
    logging.info('Input files by status: %s', input_files_by_status)
    if FLAGS.max_input_files_to_process:
        files_to_process = files_to_process[:FLAGS.max_input_files_to_process]
        logging.info('Processing only %d files', len(files_to_process))
    if FLAGS.min_input_files_to_process is not None and len(
            files_to_process) < FLAGS.min_input_files_to_process:
        logging.info(
            'Not enough files to process, exiting... '
            '(min_input_files_to_process=%d)',
            FLAGS.min_input_files_to_process)
        return
    if not files_to_process:
        logging.info('Nothing to process, exiting...')
        return
    files_to_process_batches = list(
        more_itertools.chunked(files_to_process,
                               FLAGS.max_input_files_per_output_file))

    # input_paths_to_token_matrix_pipe = toolz.compose_left(
    #     toolz.curried.map(read_parquet_input_ids_to_numpy),
    #     lambda x: np.concatenate(list(x)),
    #     functools.partial(flatten_and_chunk_numpy,
    #                       sequence_length=FLAGS.sequence_length,
    #                       seed=FLAGS.seed,
    #                       min_unique_tokens=FLAGS.min_unique_tokens,
    #                       copy=False),
    #     lambda x: np_batch_split(x, FLAGS.parallelism or mp.cpu_count()))

    concat_combine_flatten_pipe = toolz.compose_left(
        # concatenate arrays of input_ids
        np.concatenate,
        # flatten and chunk
        functools.partial(flatten_and_chunk_numpy,
                          sequence_length=FLAGS.sequence_length,
                          seed=FLAGS.seed,
                          min_unique_tokens=FLAGS.min_unique_tokens,
                          copy=False),
        # split into batches for parallel processing
        # functools.partial(np_batch_split,
        #                   batch_size=FLAGS.parallelism or mp.cpu_count()),
        functools.partial(np.array_split,
                           indices_or_sections=FLAGS.parallelism
                           or mp.cpu_count()))

    idx_file = current_file_idx
    with mp.Pool(FLAGS.parallelism or mp.cpu_count()) as pool:
        for input_files_batch in files_to_process_batches:
            logging.info('Processing batch of %d files:\n %s',
                         len(input_files_batch), '\n'.join(input_files_batch))
            logging.info('Reading and serializing examples...')
            # serialized_examples_batches = np.concatenate(
            #     pool.map(serialize_batch_of_examples,
            #              input_paths_to_token_matrix_pipe(input_files_batch)))

            serialized_examples_batches = np.concatenate(
                pool.map(
                    serialize_batch_of_examples,
                    concat_combine_flatten_pipe(
                        # pool.map(read_parquet_input_ids_to_numpy,
                        list(
                            map(read_parquet_input_ids_to_numpy,
                                input_files_batch)))))
            # split records into files, last file may have less records
            n_splits = math.ceil(
                len(serialized_examples_batches) /
                FLAGS.records_per_output_file)
            # iter is used to destroy the array after using it
            serialized_examples_batches = iter(
                np.array_split(serialized_examples_batches, n_splits))

            logging.info('Writing %d files to %s', n_splits, output_dir)
            for serialized_examples_batch in serialized_examples_batches:
                n_examples = len(serialized_examples_batch)
                output_file = str(
                    output_dir.joinpath(
                        f'data/data_{idx_file}_{n_examples}.tfrecords'))
                bytes_written = write_tfrecords(serialized_examples_batch,
                                                output_file)
                # save metadata
                new_meta = dict(idx_file=idx_file,
                                input_files=input_files_batch,
                                output_file=output_file,
                                n_examples=n_examples,
                                n_tokens=n_examples * FLAGS.sequence_length,
                                bytes_written=bytes_written,
                                utc_isoformat_write_ts=datetime.datetime.
                                utcnow().isoformat(),
                                input_files_count=len(input_files_batch),
                                seed=FLAGS.seed,
                                cmd_called=' '.join(sys.argv))
                logging.info('Metadata: \n%s\n', new_meta)
                output_dir.joinpath('metadata').joinpath(
                    f'meta_{idx_file}.json').write_text(json.dumps(new_meta))
                idx_file += 1
                del serialized_examples_batch


if __name__ == '__main__':
    # set log to info
    logging.set_verbosity(logging.INFO)
    app.run(main)

# train: aos poucos
"""
run-one-constantly time python3 convert_docs_to_tfrecords.py \
--max_input_files_to_process=5 \
--records_per_output_file=100000 \
--input_dir=${GCS_BUCKET:-gs://your-bucket-name}/data/mc4pt_clean_docs/train  \
--output_dir=${GCS_BUCKET:-gs://your-bucket-name}/data/mc4pt_clean_tfrecords/train \
--min_input_files_to_process=5 \
--max_input_files_per_output_file=5 \
--sequence_length=2049 \
--min_unique_tokens=200 \
--seed=42
"""
# Sem parallel reading
# real    0m6.370s
# user    0m11.303s
# sys     0m7.204s

# Com parallel reading - aumenta a comunicação
# real    10m17.746s
# user    50m46.919s
# sys     6m38.819s

# validation: Pode rodar tudo de uma vez
"""
# --max_input_files_per_output_file=10000 \
# --min_input_files_to_process=5 \
time python3 convert_docs_to_tfrecords.py \
--max_input_files_to_process=1000 \
--records_per_output_file=100000 \
--input_dir=${GCS_BUCKET:-gs://your-bucket-name}/data/mc4pt_clean_docs/validation  \
--output_dir=${GCS_BUCKET:-gs://your-bucket-name}/data/mc4pt_clean_tfrecords/validation \
--sequence_length=2049 \
--min_unique_tokens=200 \
--seed=42
"""
# time python3 device_train.py --tune-model-path=${GCS_BUCKET:-gs://your-bucket-name}/EleutherAI-pretrained-models/GPT-J-6B/step_383500/ --config=configs/mc4-clean-teste-1-no-fresh-opt.json
