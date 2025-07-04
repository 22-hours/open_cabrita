python open_cabrita-feat-gemma/src/download_model.py

cd open_cabrita-feat-gemma
export PYTHONPATH="${PWD}:$PYTHONPATH"
python EasyLM/models/gemma/convert_hf_to_easylm.py --output_file ../gemma7beasylm --checkpoint_dir ../gemma7bhf/ --model_size 7b