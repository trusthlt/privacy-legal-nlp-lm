# Privacy-Preserving Models for Legal Natural Language Processing
This repository provides the source codes for the paper **Privacy-Preserving Models for Legal Natural Language Processing**, 
which focuses on investigating effective strategies for large-scale differential private pre-training of BERT in the legal domain.
This can potentially be extended to other transformer models and domains as well. 

## Instructions
This repository consists of three parts contained in three separate directories.
### Data Preparation
  The folder `data_prepare` includes the script `pretrain_data_generate.py` to generate the pre-training dataset.     
  * Firstly, you need prepare a bunch of text data with one paragraph/document per line in the format of txt 
  or previously saved using ``dataset.save_to_disk(dataset_path)`` from [datasets library](https://github.com/huggingface/datasets),
  and assign the directory name to the argument `dataset_dir`. In our work, we mainly collected legal text data from 
  [CaseLaw](https://case.law) and [SigmaLaw](https://osf.io/qvg8s/). Alternatively, you can simply provide a dataset name from HuggingFace Hub with the
  argument `dataset_name`.    
  * For the tokenization, you can train an in-domain vocabulary/tokenizer with your data using [tokenizers library](https://github.com/huggingface/tokenizers) 
  or just use an existing BERT tokenizer, and give the tokenizer directory or name to the argument `tokenizer_name_or_path`. 
  We provided our legal vocabulary in the directory `legal_tokenizer`. In addition, we customized some tokenize functions in the scrip `utils.py`, you can choose what you want or write one yourself and pass it to the argument `tokenize_func`.     
  * An example of running the script is as below. For more details about the optional arguments please refer to the description in the source code.
    ```
    python pretrain_data_generate.py \
      --tokenizer_name_or_path legal_tokenizer \
      --tokenize_func legal \
      --dataset_dir directory_of_prepared_data \
      --tokenize_train_file directory_to_cache_tokenized_data/train.arrow \
      --tokenize_val_file directory_to_cache_tokenized_data/valid.arrow \
      --save_path directory_to_save_dataset \
      --max_seq_length 128 \
      --shuffle \
      --shuffle_train_file shuffled_indices.arrow \
      --short_seq_prob 0.05 \
      --seed 11 \
    ```

### Pre-training with Differential Privacy (DP)
The script `pretrain/train.py` is used for training a differentially private BERT model effectively leveraging [JAX](https://github.com/google/jax) backend.
It takes the Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) tasks as training objectives. 
You can continue the training from an existing BERT model by assigning `model_name_or_path` with your pretrained model path, or a BERT model name from HuggingFace Hub.
Otherwise, you may also train a model from scratch by providing a `config_name`. 
The script will automatically restore the training state from the latest checkpoints if any exists in the `output_dir`.     
Our experiments indicate that huge batch size benefits the DP training. 
You may try to scale up the batch size by specifying very large `gradient_accumulation_steps`.    
For the DP arguments, you can either specify the privacy budget `epsilon` or the noise multiplier `sigma`. Lower `epsilon` means higher `sigma` and theoretically stronger privacy. 
The DP training is applied by default unless the `disable_dp` argument is given.
Here is an example of how to run the scrip, please check the source code for a complete description of the arguments.
```
python train.py \
    --model_name_or_path bert-base-uncased \
    --data_path directory_to_dataset \
    --output_dir directory_to_save_model \
    --logging_dir directory_to_log \
    --do_train \
    --do_eval \
    --save_steps 12 \
    --eval_steps 12 \
    --logging_steps 12 \
    --save_total_limit 4 \
    --num_train_epochs 5 \
    --per_device_eval_batch_size 64 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 8192 \
    --seed 42 \
    --mlm_probability 0.15 \
    --overwrite_output_dir \
    --learning_rate 1e-3 \
    --weight_decay 0.5 \
    --warmup_steps 25 \
    --num_train_samples 26910944 \
    --target_eps 5
```


### Evaluation with Downstream Tasks
The entry for downstream task evaluation is placed under the folder `downstream_tasks`. 
The script `train.py` is used for fine-tuning a DP pretrained BERT model on an end task and evaluating its performances.
There are 4 tasks: [Overruling, CaseHOLD, Terms of Service (ToS)](https://arxiv.org/pdf/2104.08671.pdf), 
and [LEDGAR](https://aclanthology.org/2020.lrec-1.155.pdf). We mainly take the first two tasks as representatives for the evaluation.
The following shows a running example：
```
python train.py \
  --task_name casehold \
  --pretrained_weights name_or_path_to_pretrained_model \
  --model_name type_of_the_model \
  --save_path directory_to_save_checkpoints_and_results \
  --train_batch_size 16 \
  --valid_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --lr 3e-5 \
  --log_steps 166 \
  --num_epochs 4 \
  --hist_log_dir directory_to_log_eval_scores \
  --overwrite_save_path \
```



