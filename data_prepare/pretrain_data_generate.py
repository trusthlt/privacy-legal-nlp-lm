import os
import sys
from collections import defaultdict
from utils import TokenizeProcessor, create_instance_from_document
from datasets import load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from argparse import ArgumentParser
from transformers import AutoTokenizer
from functools import partial
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level="NOTSET",
    datefmt="[%X]",
)
logger = logging.getLogger(__name__)


def args_parser():
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name_or_path', type=str, required=True,
                        help='pretrained tokenizer name or path.')
    parser.add_argument('--tokenize_func', type=str, required=True,
                        choices=['wiki', 'legal'],
                        help='select the tokenization function for the dataset.')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='path of the cached dataset')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='the name of the dataset to use (via the datasets library).')
    parser.add_argument('--dataset_config_name', type=str, default=None,
                        help='the configuration name of the dataset to use (via the datasets library).')
    parser.add_argument('--validation_split_percentage', type=int, default=5,
                        help="the percentage of the train set used as validation set "
                             "in case there's no validation split.")
    parser.add_argument('--tokenize_train_file', type=str, default=None,
                        help='file name for caching tokenized training data.')
    parser.add_argument('--tokenize_val_file', type=str, default=None,
                        help='file name for caching tokenized validation data.')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='overwrite the cached training and evaluation sets')
    parser.add_argument('--do_lower_case', action='store_false',
                        help='whether to lowercase the input text.')
    parser.add_argument('--use_fast', action='store_true',
                        help='whether to use fast tokenizer.')
    parser.add_argument('--instance_train_file', type=str, default=None,
                        help='file name for caching processed training instances.')
    parser.add_argument('--instance_val_file', type=str, default=None,
                        help='file name for caching processed validation instances.')
    parser.add_argument('--version', type=str, default='small',
                        help=""" which version of dataset to create.
                        - "small": only save input_ids of two sentences and NSP labels
                        - "normal": save 'token_type_ids', 'special_token_ids' as well""")
    parser.add_argument('--save_path', type=str, default=None,
                        help='path for save_to_disk (for the final processed datasets)')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='maximal sequence length of training input instance.')
    parser.add_argument('--short_seq_prob', type=float, default=0.1,
                        help='probability of creating a short instance.')
    parser.add_argument('--shuffle', action='store_false',
                        help='whether to shuffle the training dataset.')
    parser.add_argument('--shuffle_train_file', type=str, default=None,
                        help='file name for caching shuffled indices.')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for dataset shuffling.')

    return parser.parse_args()


def create_instances(params, examples, indices):
    instances = defaultdict(list)
    for example, idx in zip(examples['input_ids'], indices):
        instances_ = create_instance_from_document(example, idx,
                                                   datasets=params['datasets'],
                                                   version=params['version'],
                                                   max_seq_length=params['max_seq_length'],
                                                   short_seq_prob=params['short_seq_prob'],
                                                   special_tokens=params['special_tokens'], )
        for key, value in instances_.items():
            # ###append -> extend
            instances[key].append(value)
    return instances


def show_n_examples(dataset, n=3):
    for key in dataset.keys():
        logger.info(f"examples of {key} set:")
        for i in range(n):
            logger.info(f"{datasets[key][i]}")


if __name__ == "__main__":
    args = args_parser()
    logger.info(args)
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    if args.dataset_dir is not None:
        if os.path.isfile(os.path.join(args.dataset_dir, 'dataset_dict.json')):
            # Loading dataset from disk
            logger.info("Loading dataset from disk ...")
            datasets = load_from_disk(args.dataset_dir)
        else:
            # Loading dataset from a bunch of txt/csv/json files
            logger.info("Loading dataset from files ...")
            data_files = [os.path.join(args.dataset_dir, fname) for fname in os.listdir(args.dataset_dir)]
            extension = data_files[0].split(".")[-1]
            data_files = [file for file in data_files if file.split(".")[-1] == extension]
            logger.info(f"Found {len(data_files)} {extension} files")
            logger.info(f"Files: {data_files}")

            if extension == "txt":
                extension = "text"
            datasets = DatasetDict()
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        logger.info("Loading dataset from the hub ...")
        datasets = DatasetDict()
        datasets["validation"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[:{args.validation_split_percentage}%]",

        )
        datasets["train"] = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=f"train[{args.validation_split_percentage}%:]",
        )
    else:
        raise ValueError("Please specify the path of the dataset or "
                         "the name of a dataset from the HuggingFace hub.")

    logger.info(f"original datasets={datasets}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        use_fast=args.use_fast, )

    tokenize_processor = TokenizeProcessor(tokenizer, padding=False, max_seq_length=512)

    tokenize_cache_files = {}

    if args.tokenize_train_file is not None:
        tokenize_cache_files["train"] = args.tokenize_train_file
    if args.tokenize_val_file is not None:
        tokenize_cache_files["validation"] = args.tokenize_val_file

    if args.tokenize_func == 'wiki':
        tokenize_func = tokenize_processor.wiki_tokenizer
    else:
        tokenize_func = tokenize_processor.legal_tokenizer
    datasets = datasets.map(
        tokenize_func,
        input_columns=["text"],
        batched=True,
        batch_size=10000,
        remove_columns=datasets["validation"].column_names,
        load_from_cache_file=True,
        cache_file_names=None if not tokenize_cache_files else tokenize_cache_files,
        desc="Tokenizing the dataset:"
    )

    logger.info(f"tokenized datasets={datasets}")
    show_n_examples(datasets)

    if args.shuffle:
        logger.info("shuffling the training data ...")
        datasets['train'] = datasets['train'].shuffle(writer_batch_size=10000,
                                                      seed=args.seed,
                                                      indices_cache_file_name=args.shuffle_train_file)

        logger.info(f"example of shuffled datasets:")
        show_n_examples(datasets)

    special_tokens = {
        tokenizer.cls_token: tokenizer.cls_token_id,
        tokenizer.sep_token: tokenizer.sep_token_id,
        tokenizer.mask_token: tokenizer.mask_token_id,
    }

    for set_name in ['train', 'validation']:
        data_params = {'datasets': datasets[set_name],
                       'version': args.version,
                       'max_seq_length': args.max_seq_length,
                       'short_seq_prob': args.short_seq_prob,
                       'special_tokens': special_tokens}

        cache_file_name = args.instance_train_file if set_name == 'train' else args.instance_val_file
        datasets[set_name] = datasets[set_name].map(partial(create_instances, data_params),
                                                    with_indices=True,
                                                    batched=True,
                                                    load_from_cache_file=True,
                                                    cache_file_name=cache_file_name,
                                                    remove_columns=datasets[set_name].column_names,
                                                    desc=f"Generating {set_name} set:")

    logger.info(f"processed_datasets={datasets}")
    show_n_examples(datasets)

    if args.save_path is not None:
        datasets.save_to_disk(args.save_path)
