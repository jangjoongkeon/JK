# Script

This folder contain useful scripts.

## test_tpu_model.sh

### Description

This script is used to automate the process of evaluating a TPU-trained model in our local GPU server.

This script do the following :

* Retrieve vocabulary file and configuration file (from pretrained version of BERT)
* Retrieve checkpoint of the TPU-trained model (from Google bucket)
* Retrieve files for SQuAD evaluation
* Setup GPU environment (to use only 1 GPU)
* Predict the evaluation sub-dataset of SQuAD using the TPU-trained model
* Evaluate the predictions made using the official SQuAD script

### Usage

`./test_tpu_model.sh <bucket_name> <bert_model> <result_path> <step> <GPU> <batch_size> <seq_max_len> <doc_stride>`

Argument description :

| Argument | Example | Description |
|---|---|---|
| `bucket_name` | `bucket_for_squad_42maru` | Name of the Google Bucket where data is stored. |
| `bert_model` | `uncased_L-24_H-1024_A-16` | Name of the pre-trained BERT model used for training. |
| `result_path` | `cola/results/demo_mrc_2` | Path (of the Google bucket) where the model is stored. |
| `step` | `32579` | Number of step of the model to load. You can find it in the name of the model file : `model.ckpt-32579.meta`. |
| `GPU` | `1` | ID of the GPU to use for evaluation [0 ~ 3]. |
| `batch_size` | `8` | Batch size to use for evaluation. |
| `seq_max_len` | `384` | Maximum sequence length (maximum size for context). |
| `doc_stride` | `128` | Document stride to use (for too big context). |

### Additional notes

* The script can be run from anywhere.
* You might need to give yourself permission to execute this script. You can do the following : `chmod +x test_tpu_model.sh`
* You might need to install `gsutil` in order to download data from Google Bucket. Use this [link](https://cloud.google.com/storage/docs/gsutil_install#linux) for installation steps. 