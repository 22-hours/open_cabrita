import ftfy
import datasets
from EasyLM.models.llama.llama_model import LLaMAConfig
from EasyLM.models.llama.llama_model import LLaMATokenizer
import multiprocessing as mp
import polars as pl
import pandas as pd

import nltk
nltk.download('stopwords')
PT_STOPWORDS = set(map(str.lower, nltk.corpus.stopwords.words('portuguese')))

_DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/multilingual/c4-{language}{split_suffix}.tfrecord-{index:05d}-of-{n_shards:05d}.json.gz"

_N_SHARDS_PER_SPLIT = {'pt-train': 1024, 'pt-validation': 4}

TOKENIZERS_PATHS = {
    'llama_default': '/home/marcospiau/original_easylm_weights/open_llama_3b_easylm/tokenizer.model',
    'merged_pt': 'ERRO'
}



def get_url(config, shard):
    lang, split = config.split('-')
    return _DATA_URL.format(
        language=lang,
        split_suffix="-validation" if split == "validation" else "",
        index=shard,
        n_shards=_N_SHARDS_PER_SPLIT[config])

def add_count_tokens(text, tokenizer):
    """Add count of tokens to example."""
    n_tokens = tokenizer(text, return_length=True).length
    return {'n_tokens': n_tokens}

def get_llama_tokenizer(vocab_file):
    """Get tokenizer with same configs used while training."""
    # no treino isso é adicionado na mão antes de tokenizer, aqui vamos botar
    # no tokenizer pra facilitar
    padding_side = 'left'
    truncation_side = 'right'
    add_bos_token = True
    add_eos_token = True
    tokenizer = LLaMATokenizer(
        vocab_file=vocab_file,
        add_bos_token=add_bos_token,
        add_eos_token=add_eos_token,
        padding_side=padding_side,
        truncation_side=truncation_side,
    )
    return tokenizer


def clean_document(ex):
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

def get_dataset(config, shards, streaming=True):
    data_files = [get_url(config, shard) for shard in shards]
    ds = datasets.load_dataset('json', data_files=data_files, split='train')
    return ds

# tokenizer_path = 'llama_default'
# LLaMAConfig 
N_SHARDS = 10
ds = get_dataset('pt-train', range(N_SHARDS))
ds = ds.map(clean_document, remove_columns=ds.column_names, num_proc=mp.cpu_count())
# ds.select(range(10)).map(lambda x: add_count_tokens(x['text'], tokenizer), batched=True, batch_size=1000).to_pandas()


# get tokenized lengths
tokenizer = get_llama_tokenizer(TOKENIZERS_PATHS['llama_default'])
ds = ds.map(lambda x: add_count_tokens(x['text'], tokenizer), num_proc=mp.cpu_count())

df = pl.from_arrow(ds.data.table)
df = df.with_columns(pl.col('n_tokens').lt(200).alias('lt_200_unique_tokens_filter'))
df = df.with_columns((pl.col('any_filter') | pl.col('lt_200_unique_tokens_filter')).alias('any_filter'))

# token counts aggregated by filter or not
df_token_group = df.groupby('any_filter').agg(
    pl.col('n_tokens').sum().alias('sum'),
    pl.col('n_tokens').mean().alias('mean'),
    pl.col('n_tokens').std().alias('std'),
    pl.count()
)
print(df_token_group.write_csv())

# df2 = df.select(pl.col('^.*_filter$'), pl.count()).sum().to_pandas().T.sort_values(0, ascending=False)



df2 = df.select(pl.col('^.*_filter$').sum(), pl.count())
suma = df2.to_pandas().T.sort_values(0, ascending=False).rename(columns={0: 'count'})
suma_pct = suma.div(df2['count'].item()).mul(100).round(2).rename(columns={'count': 'pct'})
# lateral concatenation
suma_final = pd.concat([suma, suma_pct], axis=1)
suma_final['1024_shards'] = suma_final['count'] * (1024 / N_SHARDS)
print(pl.from_pandas(suma_final.reset_index()).write_csv())