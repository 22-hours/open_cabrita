"""Create a debug version of the MC4 dataset for training."""

import datasets

if __name__ == "__main__":
    # iretable dataset
    ds = datasets.load_dataset('mc4', 'pt', split='train', streaming=True)
    ds = ds.remove_columns(['url', 'timestamp']).take(100_000)
    # convert to map-style to save to disk
    ds = datasets.Dataset.from_list(list(ds))
    ds.save_to_disk('mc4_train_debug')

    # save in json format
    ds.to_json('mc4_train_debug.jsonl')