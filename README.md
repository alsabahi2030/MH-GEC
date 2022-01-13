# Multi-head Sequence Tagging Model for Grammatical Error Correction

This repository provides code for training and testing  models for GEC with the official PyTorch implementation of the paper.

It is mainly based on `AllenNLP` and `transformers`.
## Installation
The following command installs all necessary packages except pytorch that should be installed separately:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7 and pytorch ==1.5.1.

## Datasets
All the public GEC datasets used in the paper can be downloaded from [here](https://www.cl.cam.ac.uk/research/nl/bea2019st/#data).<br>
Synthetically created datasets can be generated/downloaded [here]().<br>
To train the model data has to be preprocessed and converted to special format with the command:
```.bash
python utils/preprocess_data_annotations.py -s SOURCE -t TARGET -o OUTPUT_FILE
```
## Train model
To train the model, simply run:
```.bash
python train.py --train_set TRAIN_SET --dev_set DEV_SET \
                --model_dir MODEL_DIR --transformer_model TRANSFORMER_MODEL
```
## Model inference
To run your model on the input file use the following command:
```.bash
CUDA_VISIBLE_DEVICES=[] python predict_new.py --model_path MODEL_PATH [MODEL_PATH ...] \
  --vocab_path VOCAB_PATH --input_file INPUT_FILE --output_file OUTPUT_FILE
                  
```
Among parameters:
- `min_error_probability` - minimum error probability 
- `additional_confidence` - confidence bias 
- `special_tokens_fix` to reproduce some reported results of pretrained models. for example, special_tokens_fix=0 with xlnet models and special_tokens_fix=1 for BERT and Roberta.
-  `early_exit1`  - with early exit of five heads(DELETE,REPLACE,APPEND,DETECT,CORRECTION).
-  `early_exit2`  - along with early_exit1 it specifies the early exit model with seven heads(DELETE,REPLACE,APPEND,DETECT, TRANSFORM,MERGE,CORRECTION).
-  `multiclassifier`  - with multi-head classifier of five heads(DELETE,REPLACE,APPEND,DETECT,CORRECTION)
-  `multiclassifier2`  - along with `multiclassifier`  it specifies the multi-head classifier model with seven heads(DELETE,REPLACE,APPEND,DETECT, TRANSFORM,MERGE,CORRECTION).
-  `evaluate`  - if specified, it do an evalaution after the inference using ERRANT or M2 score.
-  `overwrite`  - if specified, it do the inference and the evaluation again and overwite the old files.
-  `is_ensemble`  - if specified, it shows the number of ensemble models to be used.
-  `iteration_count `  - if specified, it specifies the maximun number of iterations the model can do.
-  `vocab_path` Path to the vocab files. For the base model, the path is (`./data/output_suffix5k_vocabulary`) and for the multi-head and early exit models, the path is (`./data/output_suffix5kordered_vocabulary`)
##An Example:
To do the inference using the multi-head model with seven heads. we run the following:
###On xlnet-based
```
export LC_ALL=C.UTF-8
source ~/.bashrc
CUDA_VISIBLE_DEVICES=1 python predict_new.py --model_path \
"./models/xlnet_0_epoch_0.th" \
--vocab_path=./data/output_suffix5kordered_vocabulary \
--subset valid --min_error_probability 0.66 --additional_confidence 0.35  \
 --batch_size=64 --iteration_count=5 --evaluate  --multiclassifier --multiclassifier2 --is_ensemble 1
```
###On Roberta-based
```
export LC_ALL=C.UTF-8
source ~/.bashrc
CUDA_VISIBLE_DEVICES=1 python predict_new.py --model_path \
"./models/roberta_1_epoch_0.th" \
--vocab_path=./data/output_suffix5kordered_vocabulary \
--subset valid --min_error_probability 0.62 --additional_confidence 0.3  \
 --batch_size=64 --iteration_count=5 --evaluate  --multiclassifier --multiclassifier2 --is_ensemble 1
```

For external input file:
```
CUDA_VISIBLE_DEVICES=0 python predict_new.py  \ 
--model_path "./xlnet_0_epoch_0.th" --min_error_probability 0.66 \ 
--additional_confidence 0.35 --batch_size=64 --iteration_count=5 \ 
--is_ensemble 1 --vocab_path="./data/output_suffix5kordered_vocabulary/" \
--input_file "./data/wi.dev.ori" --multiclassifier --multiclassifier2
```
Note that the `--multiclassifier --multiclassifier2 --special_tokens_fix` can be omitted if the checkpoint file's name followed the specific format `transformer_0 or 1_somethingmc2_...` here mc2 means multi-head classifier with seven heads. Example , `xlnet_0_suffixvalidpiedaev3mc2_trainallnewfinetunel11_epoch_0.th`. 
## The output files:
Basicly, after inference two files will be generated:
1. `*.cor` the corrected file.
2. `*.gedit` this file contains the edits
