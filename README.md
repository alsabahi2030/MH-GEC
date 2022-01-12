It is mainly based on `AllenNLP` and `transformers`.
## Installation
The following command installs all necessary packages except pytorch that should be installed separately:
```.bash
pip install -r requirements.txt
```
The project was tested using Python 3.7 and pytorch ==1.5.1.

## Model inference
To run your model on the input file use the following command:
```.bash
CUDA_VISIBLE_DEVICES=[] python predict_new3_ann.py --model_path MODEL_PATH [MODEL_PATH ...] \
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
-  `annotations`  - to use the model that combined the correction and annotation together.
-  `vocab_path` Path to the vocab files. For the base model, the path is (`./data/output_suffix5k_vocabulary`) and for the multi-head and early exit models, the path is (`./data/output_suffix5kordered_vocabulary`)
##An Example:
To do the inference using the multi-head model with seven heads. we run the following:
###On xlnet-based
```
export LC_ALL=C.UTF-8
source ~/.bashrc
CUDA_VISIBLE_DEVICES=1 python predict_new3_ann.py --model_path \
"./models/xlnet_0_suffixvalidpiedaev3mc2_trainallnewfinetunel11_epoch_0.th" \
--vocab_path=./data/output_suffix5kordered_vocabulary \
--subset valid --min_error_probability 0.66 --additional_confidence 0.35  \
 --batch_size=64 --iteration_count=5 --evaluate  --multiclassifier --multiclassifier2 --is_ensemble 1
```
###On Roberta-based
```
export LC_ALL=C.UTF-8
source ~/.bashrc
CUDA_VISIBLE_DEVICES=1 python predict_new3_ann.py --model_path \
"./models/roberta_1_suffixpiedaev35mc2_trainallnewfinetunel51_epoch_0.th" \
--vocab_path=./data/output_suffix5kordered_vocabulary \
--subset valid --min_error_probability 0.62 --additional_confidence 0.3  \
 --batch_size=64 --iteration_count=5 --evaluate  --multiclassifier --multiclassifier2 --is_ensemble 1
```
###Ensemble of the two models
```
CUDA_VISIBLE_DEVICES=5 python predict_new3_ann.py --model_path \
"./models/xlnet_0_suffixvalidpiedaev3mc2_trainallnewfinetunel11_epoch_0.th" \
"./models/roberta_1_suffixpiedaev35mc2_trainallnewfinetunel51_epoch_0.th" \
--subset valid   --is_ensemble 2 --additional_confidence 0.25 --min_error_probability 0.55  \
--batch_size=64 --iteration_count=5  --evaluate 
```
For external input file:
```
CUDA_VISIBLE_DEVICES=0 python predict_new3_ann.py  \ 
--model_path "./xlnet_0_suffixvalidpiedaev3mc2_trainallnewfinetunel11_epoch_0.th" --min_error_probability 0.66 \ 
--additional_confidence 0.35 --batch_size=64 --iteration_count=5 \ 
--is_ensemble 1 --vocab_path="./data/output_suffix5kordered_vocabulary/" \
--input_file "./data/wi.dev.ori" --multiclassifier --multiclassifier2
```
Note that the `--multiclassifier --multiclassifier2 --special_tokens_fix` can be omitted if the checkpoint file's name followed the specific format `transformer_0 or 1_somethingmc2_...` here mc2 means multihead classifier with seven heads. Example , `xlnet_0_suffixvalidpiedaev3mc2_trainallnewfinetunel11_epoch_0.th`. 
## The output files:
Basicly, after inference two files will be generated:
1. `*.cor` the corrected file.
2. `*.edit` this file contains the gector edits that can be fed into the explanation model.
if you are evaluating the model on the validation sets, another file will be generated for the scores with `*.report` extension . 
For evaluation use [ERRANT](https://github.com/chrisjbryant/errant).
# Pretrained models
<table>
  <tr>
    <th>Pretrained encoder</th>
    <th>additional_confidence</th>
    <th>min_error_probability</th>
    <th>BEA-2019 (dev)</th>
    <th>BEA-2019 (test)</th>
  </tr>
  <tr>
    <th>Old Model(RoBERTa)</th>
    <th>0.20</th>
    <th>0.50</th>
    <th>53.1</th>
    <th>-</th>
  </tr>
    <tr>
    <th>New model(RoBERTa)</th>
    <th>0.30</th>
    <th>0.62</th>
    <th>56.8</th>
    <th>73.8</th>
  </tr>
  <tr>
    <th>Old Model(XLNet) </th>
    <th>0.35</th>
    <th>0.66</th>
    <th>55.2</th>
    <th>-</th>
  </tr>  
  <tr>
    <th>New Model(XLNet) </th>
    <th>0.35</th>
    <th>0.66</th>
    <th>57.5</th>
    <th>74.4</th>
  </tr>  
  <tr>
    <th>Ensemble of two (XLNet + Roberta) </th>
    <th>0.25</th>
    <th>0.55</th>
    <th>58.5</th>
    <th>75.4</th>
  </tr>

</table>