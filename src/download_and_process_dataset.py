import huggingface_hub
import ftfy
import cytoolz
import datasets


from tqdm import tqdm
import os

from random import sample
import glob


def chunks(sentences, n, tot_len):
    """Yield successive n-sized chunks from sentences."""
    for i in range(0, tot_len, n):
        end_i = min(len(sentences),i + n)
        yield sentences[i:end_i]["text"]

def make_sentence_files(dataset, chunksize = 10000000, data_dir = "./wiki_tokenizer"):
    """
    Make a sentence per line files, chuncsize sentences per file
    """
    # make sure data dir exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # use simple regex for sentence tokenizing
    # sent_detector = nltk.RegexpTokenizer(u'[^　！？。]*[！？。.\n]')

    # loop over the chunks
    for chunk_ind, sentence_chunk in enumerate(chunks(dataset, chunksize, len(dataset))):

        # new file for each chunk
        filename = "sent_{}.txt".format(chunk_ind)
        filepath = os.path.join(data_dir, filename)

        print("writing to ", filepath)

        with open(filepath, "w") as f:

            for sentence in tqdm(sentence_chunk):

                # remove newlines
                line = sentence.strip()

                # unicode normalize japanese spaces etc
                # unicodedata.normalize('NFKC', line)

                # tokenize into sentences
                # sentences = sent_detector.tokenize(line)

                # do not save empty items such as
                #if sentences != []:

                #    f.writelines(s + '\n' for s in sentences)

                f.write("###\n" + line + '\n\n')

def sample_and_make_tempfile(sentences_dir, num_files, data_dir):
    """ Use the set of files containing a sentence per line,
    sample num_files out of those and save as a temp file """

    sentence_files = glob.glob(sentences_dir + "/*.txt")

    # sample num_files
    sampled_files=sample(sentence_files, num_files)

    print("sampled files:")
    print(sampled_files)

    #read all the lines from sampled files and save to a list
    all_lines = []
    for filename in sampled_files:
        with open(filename) as f:
            lines = f.read().splitlines()

        all_lines.extend(lines)

    print("number of lines sampled:", len(all_lines))

    #combine into a single file and save
    tempfile_path = os.path.join(data_dir, "temp.txt")
    with open(tempfile_path, "w") as f:

                for sentence in tqdm(all_lines):

                    # remove newlines
                    line = sentence.strip()

                    # do not save empty items such as
                    if sentence != []:

                        f.writelines(sentence + '\n')

    print("Wrote to ", tempfile_path)
    return tempfile_path


huggingface_hub.snapshot_download(repo_id='marcospiau/wikipedia_pt_20230120_huggingface_dataset',
                                  repo_type='dataset',
                                  local_dir='./wiki')

dataset = datasets.load_from_disk('./wiki/train/')

ds_wiki_clean = (
    dataset
    .remove_columns(['id', 'url', 'title'])
    .map(cytoolz.curried.update_in(keys=['text'], func=ftfy.fix_text),
         num_proc=8, desc='Applying ftfy.fix_text')
    .shuffle(seed=12345)
)

make_sentence_files(ds_wiki_clean, chunksize=100000)
sample_and_make_tempfile('wiki_tokenizer', 10, '.')
