#!/bin/bash

# Get the path to the script
MY_PATH="`dirname \"$0\"`"              # relative
MY_PATH="`( cd \"$MY_PATH\" && pwd )`"  # absolutized and normalized
echo " ===> Files will be downloaded here : $MY_PATH/bert_model/ and here : $MY_PATH/squad/"

# Check User input
if [ "$#" -ne 8 ]
then
    echo "Usage: $0 <bucket_name> <bert_model> <result_path> <step> <GPU> <batch_size> <seq_max_len> <doc_stride>" >&2
    echo "    bucket_name : Name of the Google Bucket where data is stored. Example = bucket_for_squad_42maru" >&2
    echo "    bert_model : Name of the pre-trained BERT model used for training. Example = uncased_L-24_H-1024_A-16" >&2
    echo "    result_path : Path (of the Google bucket) where the model is stored. Example = cola/results/demo_mrc_real_tpu" >&2
    echo "    step : Number of step of the model to load. You can find it in the name of the model file : model.ckpt-32579.meta. Example = 32579" >&2
    echo "    GPU : Numero of the GPU to use for evaluation [0 ~ 3]. Example = 1" >&2
    echo "    batch_size : Batch size to use for evaluation. Example = 8" >&2
    echo "    seq_max_len : Maximum sequence length (maximum size for context). Example = 384" >&2
    echo "    doc_stride : Doc stride to use (for too big context). Example = 128" >&2
    exit 1
fi

echo " ===> Copying vocabulary file and configuration file from pretrained BERT (from Google bucket) ..."
gsutil cp gs://$1/$2/bert_config.json $MY_PATH/bert_model/bert_config.json
gsutil cp gs://$1/$2/vocab.txt $MY_PATH/bert_model/vocab.txt

echo " ===> Copying checkpoint of trained model (from Google bucket) ..."
gsutil cp gs://$1/$3/*$4* $MY_PATH/bert_model/

echo " ===> Retrieving files for evaluation from SQuAD 2.0 ..."
mkdir $MY_PATH/squad
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O $MY_PATH/squad/dev-v2.0.json
wget https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/ -O $MY_PATH/squad/evaluate-v2.0.py

echo " ===> Setting up GPU #$5 ..."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$5

echo " ===> Moving to mrc_model (to execute model's evaluation)"
cd $MY_PATH/../mrc_model/

echo " ===> Predicting eval subset using GPU and trained model ..."
python3 -m en.run_squad_42maru_single --vocab_file=$MY_PATH/bert_model/vocab.txt --bert_config_file=$MY_PATH/bert_model/bert_config.json --init_checkpoint=$MY_PATH/bert_model/model.ckpt-$4 --do_train=False --do_predict=True --predict_file=$MY_PATH/squad/dev-v2.0.json --predict_batch_size=$6 --max_seq_length=$7 --doc_stride=$8 --output_dir=$MY_PATH/squad/ --use_tpu=False

echo " ===> Moving back to original place"
cd $MY_PATH

# Evaluate
echo " ===> Evaluating predictions ..."
python $MY_PATH/squad/evaluate-v2.0.py $MY_PATH/squad/dev-v2.0.json $MY_PATH/squad/predictions.json
echo " ===> Done <==="