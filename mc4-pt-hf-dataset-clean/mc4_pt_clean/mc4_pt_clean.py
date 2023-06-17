"""mC4 dataset based on Common Crawl."""


import gzip
import json

import datasets
import re
import ftfy
import nltk


logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
A colossal, cleaned version of Common Crawl's web crawl corpus.

Based on Common Crawl dataset: "https://commoncrawl.org".

This is the processed version of Google's mC4 dataset by AllenAI.
"""

_CITATION = """
@article{2019t5,
    author = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
    title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
    journal = {arXiv e-prints},
    year = {2019},
    archivePrefix = {arXiv},
    eprint = {1910.10683},
}
"""

_URL = "https://github.com/allenai/allennlp/discussions/5056"

_DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/multilingual/c4-{language}{split_suffix}.tfrecord-{index:05d}-of-{n_shards:05d}.json.gz"

# _LANGUAGES = [
#     "af",
#     "am",
#     "ar",
#     "az",
#     "be",
#     "bg",
#     "bg-Latn",
#     "bn",
#     "ca",
#     "ceb",
#     "co",
#     "cs",
#     "cy",
#     "da",
#     "de",
#     "el",
#     "el-Latn",
#     "en",
#     "eo",
#     "es",
#     "et",
#     "eu",
#     "fa",
#     "fi",
#     "fil",
#     "fr",
#     "fy",
#     "ga",
#     "gd",
#     "gl",
#     "gu",
#     "ha",
#     "haw",
#     "hi",
#     "hi-Latn",
#     "hmn",
#     "ht",
#     "hu",
#     "hy",
#     "id",
#     "ig",
#     "is",
#     "it",
#     "iw",
#     "ja",
#     "ja-Latn",
#     "jv",
#     "ka",
#     "kk",
#     "km",
#     "kn",
#     "ko",
#     "ku",
#     "ky",
#     "la",
#     "lb",
#     "lo",
#     "lt",
#     "lv",
#     "mg",
#     "mi",
#     "mk",
#     "ml",
#     "mn",
#     "mr",
#     "ms",
#     "mt",
#     "my",
#     "ne",
#     "nl",
#     "no",
#     "ny",
#     "pa",
#     "pl",
#     "ps",
#     "pt",
#     "ro",
#     "ru",
#     "ru-Latn",
#     "sd",
#     "si",
#     "sk",
#     "sl",
#     "sm",
#     "sn",
#     "so",
#     "sq",
#     "sr",
#     "st",
#     "su",
#     "sv",
#     "sw",
#     "ta",
#     "te",
#     "tg",
#     "th",
#     "tr",
#     "uk",
#     "und",
#     "ur",
#     "uz",
#     "vi",
#     "xh",
#     "yi",
#     "yo",
#     "zh",
#     "zh-Latn",
#     "zu",
# ]

# _N_SHARDS_PER_SPLIT = {
#     "af": {"train": 64, "validation": 1},
#     "am": {"train": 16, "validation": 1},
#     "ar": {"train": 1024, "validation": 4},
#     "az": {"train": 256, "validation": 1},
#     "be": {"train": 128, "validation": 1},
#     "bg": {"train": 1024, "validation": 1},
#     "bg-Latn": {"train": 4, "validation": 1},
#     "bn": {"train": 512, "validation": 1},
#     "ca": {"train": 512, "validation": 1},
#     "ceb": {"train": 8, "validation": 1},
#     "co": {"train": 8, "validation": 1},
#     "cs": {"train": 1024, "validation": 2},
#     "cy": {"train": 256, "validation": 1},
#     "da": {"train": 1024, "validation": 1},
#     "de": {"train": 2048, "validation": 16},
#     "el": {"train": 1024, "validation": 2},
#     "el-Latn": {"train": 16, "validation": 1},
#     "en": {"train": 11264, "validation": 128},
#     "eo": {"train": 32, "validation": 1},
#     "es": {"train": 2048, "validation": 16},
#     "et": {"train": 256, "validation": 1},
#     "eu": {"train": 64, "validation": 1},
#     "fa": {"train": 1024, "validation": 2},
#     "fi": {"train": 1024, "validation": 1},
#     "fil": {"train": 64, "validation": 1},
#     "fr": {"train": 2048, "validation": 16},
#     "fy": {"train": 16, "validation": 1},
#     "ga": {"train": 16, "validation": 1},
#     "gd": {"train": 16, "validation": 1},
#     "gl": {"train": 128, "validation": 1},
#     "gu": {"train": 64, "validation": 1},
#     "ha": {"train": 8, "validation": 1},
#     "haw": {"train": 2, "validation": 1},
#     "hi": {"train": 1024, "validation": 2},
#     "hi-Latn": {"train": 16, "validation": 1},
#     "hmn": {"train": 8, "validation": 1},
#     "ht": {"train": 8, "validation": 1},
#     "hu": {"train": 1024, "validation": 2},
#     "hy": {"train": 128, "validation": 1},
#     "id": {"train": 1024, "validation": 4},
#     "ig": {"train": 4, "validation": 1},
#     "is": {"train": 128, "validation": 1},
#     "it": {"train": 1024, "validation": 8},
#     "iw": {"train": 1024, "validation": 1},
#     "ja": {"train": 1024, "validation": 8},
#     "ja-Latn": {"train": 8, "validation": 1},
#     "jv": {"train": 8, "validation": 1},
#     "ka": {"train": 256, "validation": 1},
#     "kk": {"train": 256, "validation": 1},
#     "km": {"train": 64, "validation": 1},
#     "kn": {"train": 64, "validation": 1},
#     "ko": {"train": 1024, "validation": 1},
#     "ku": {"train": 16, "validation": 1},
#     "ky": {"train": 64, "validation": 1},
#     "la": {"train": 64, "validation": 1},
#     "lb": {"train": 32, "validation": 1},
#     "lo": {"train": 8, "validation": 1},
#     "lt": {"train": 512, "validation": 1},
#     "lv": {"train": 256, "validation": 1},
#     "mg": {"train": 8, "validation": 1},
#     "mi": {"train": 4, "validation": 1},
#     "mk": {"train": 128, "validation": 1},
#     "ml": {"train": 128, "validation": 1},
#     "mn": {"train": 128, "validation": 1},
#     "mr": {"train": 1024, "validation": 1},
#     "ms": {"train": 512, "validation": 1},
#     "mt": {"train": 128, "validation": 1},
#     "my": {"train": 64, "validation": 1},
#     "ne": {"train": 256, "validation": 1},
#     "nl": {"train": 1024, "validation": 4},
#     "no": {"train": 1024, "validation": 1},
#     "ny": {"train": 4, "validation": 1},
#     "pa": {"train": 32, "validation": 1},
#     "pl": {"train": 1024, "validation": 4},
#     "ps": {"train": 16, "validation": 1},
#     "pt": {"train": 1024, "validation": 4},
#     "ro": {"train": 1024, "validation": 2},
#     "ru": {"train": 4096, "validation": 32},
#     "ru-Latn": {"train": 32, "validation": 1},
#     "sd": {"train": 64, "validation": 1},
#     "si": {"train": 64, "validation": 1},
#     "sk": {"train": 512, "validation": 1},
#     "sl": {"train": 256, "validation": 1},
#     "sm": {"train": 4, "validation": 1},
#     "sn": {"train": 8, "validation": 1},
#     "so": {"train": 64, "validation": 1},
#     "sq": {"train": 128, "validation": 1},
#     "sr": {"train": 256, "validation": 1},
#     "st": {"train": 2, "validation": 1},
#     "su": {"train": 4, "validation": 1},
#     "sv": {"train": 1024, "validation": 2},
#     "sw": {"train": 32, "validation": 1},
#     "ta": {"train": 256, "validation": 1},
#     "te": {"train": 128, "validation": 1},
#     "tg": {"train": 64, "validation": 1},
#     "th": {"train": 1024, "validation": 1},
#     "tr": {"train": 1024, "validation": 4},
#     "uk": {"train": 1024, "validation": 2},
#     "und": {"train": 3072, "validation": 32},
#     "ur": {"train": 128, "validation": 1},
#     "uz": {"train": 32, "validation": 1},
#     "vi": {"train": 1024, "validation": 4},
#     "xh": {"train": 2, "validation": 1},
#     "yi": {"train": 16, "validation": 1},
#     "yo": {"train": 2, "validation": 1},
#     "zh": {"train": 1024, "validation": 2},
#     "zh-Latn": {"train": 8, "validation": 1},
#     "zu": {"train": 8, "validation": 1},
# }

_N_SHARDS_PER_SPLIT = {'pt-train': 1024, 'pt-validation': 4}


class Mc4Config(datasets.BuilderConfig):
    """BuilderConfig for mC4."""

    def __init__(self, *args, splits, **kwargs):
        """BuilderConfig for mC4.
        Args:
            languages (:obj:`List[str]`): list of languages to load
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            *args,
            name=name,
            # name="+".join(languages),
            # name="+".join(languages),
            **kwargs,
        )
        # self.splits = splits









class Mc4(datasets.GeneratorBasedBuilder):
    """mC4, a colossal, cleaned version of Common Crawl's web crawl corpus."""

    # BUILDER_CONFIGS = [Mc4Config(languages=[lang]) for lang in _LANGUAGES]
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name='pt-train'),
        datasets.BuilderConfig(name='pt-validation')
    ]
    # BUILDER_CONFIG_CLASS = Mc4Config

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "timestamp": datasets.Value("string"),
                    "url": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_urls = {}
        lang, split = self.config.name.split('-')
        n_shards = _N_SHARDS_PER_SPLIT[self.config.name]
        # for split in ["train", "validation"]:
        data_urls[split] = [
            _DATA_URL.format(
                language=lang,
                split_suffix="-validation" if split == "validation" else "",
                index=index,
                n_shards=n_shards,
            )
            # for lang in self.config.languages
            for index in range(_N_SHARDS_PER_SPLIT[self.config.name])
        ]
        # train_downloaded_files = dl_manager.download(data_urls["train"])
        # validation_downloaded_files = dl_manager.download(data_urls["validation"])
        dowloaded_files = dl_manager.download(data_urls[split])
        return [datasets.SplitGenerator(name=datasets.Split(split), gen_kwargs={"filepaths": dowloaded_files})]
    
    def _generate_examples(self, filepaths):
        """This function returns the examples in the raw (text) form by iterating on all the files."""
        id_ = 0
        for filepath in filepaths:
            logger.info("generating examples from = %s", filepath)
            with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                for line in f:
                    if line:
                        example = json.loads(line)
                        example = process_document(example)
                        if not example['any_filter']:
                            example = {
                                k: v
                                for k,v in example.items()
                                if k in self.info.features}
                            yield id_, example
                            id_ += 1


import ftfy
import re

# baixa as stopwords
nltk.download('stopwords')
stopwords = set(map(str.lower, nltk.corpus.stopwords.words('portuguese')))

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
        'url': ex['url'],
        'timestamp': ex['timestamp'],
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