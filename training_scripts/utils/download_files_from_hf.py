"""Download datasets or models from HuggingFace Hub."""

import argparse
from huggingface_hub import snapshot_download
import os

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

    resolved_local_folder = os.path.expanduser(args.local_folder)
    download_results = snapshot_download(repo_id=args.repo_id,
                                         local_dir=resolved_local_folder,
                                         local_dir_use_symlinks=False)
    print('Downloaded files: ', download_results)
    print('Downloaded files to: ', resolved_local_folder)
    print(os.system(f'ls -lht {resolved_local_folder}'))
