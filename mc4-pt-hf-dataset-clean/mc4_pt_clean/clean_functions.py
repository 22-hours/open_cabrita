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