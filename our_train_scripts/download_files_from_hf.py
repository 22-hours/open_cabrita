"""Download datasets or models from HuggingFace Hub."""

import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--repo_id',
                    type=str,
                    default=None,
                    help='HuggingFace Hub repo id to upload to')
parser.add_argument('--local_folder',
                    type=str,
                    default=None,
                    help='Path to local folder to upload to HuggingFace Hub')
args = parser.parse_args()

if __name__ == '__main__':
    download_results = snapshot_download(repo_id=args.repo_id,
                                         local_dir=args.local_folder,
                                         local_dir_use_symlinks=False)
