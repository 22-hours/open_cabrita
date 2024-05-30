gcloud compute tpus tpu-vm scp ../our_train_scripts/train_gemma7b_clean_text_1.sh    v4-us-central2:   --worker=all   --zone=us-central2-b

gcloud compute tpus tpu-vm ssh v4-us-central2   --zone=us-central2-b   --worker=all   --command='mv train_gemma7b_clean_text_1.sh open_cabrita-feat-gemma/our_train_scripts'

gcloud compute tpus tpu-vm ssh v4-us-central2   --zone=us-central2-b   --worker=all   --command='cd open_cabrita-feat-gemma;export PYTHONPATH="${PWD}:$PYTHONPATH"; sh our_train_scripts/train_gemma7b_clean_text_1.sh'