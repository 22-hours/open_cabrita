import argparse
from huggingface_hub import HfApi

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--local_folder',
                    type=str,
                    default=None,
                    help='Path to local folder to upload to HuggingFace Hub')
parser.add_argument('--repo_id',
                    type=str,
                    default=None,
                    help='HuggingFace Hub repo id to upload to')
parser.add_argument(
    '--path_in_repo',
    type=str,
    default=None,
    help='Path to upload to in HuggingFace Hub repo. '
    'Default is None, which corresponds to the root of the repo')
parser.add_argument(
    '--repo_type',
    type=str,
    default=None,
    help='Type of repo to upload to in HuggingFace Hub repo. None is '
    'the default, corresponding to the a model repo. Other')
args = parser.parse_args()

if __name__ == '__main__':
    hf_api = HfApi()
    hf_api.create_repo(repo_id=args.repo_id,
                       exist_ok=True,
                       private=True,
                       repo_type=args.repo_type)
    hf_api.upload_folder(
        folder_path=args.local_folder,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
    )
