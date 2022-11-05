import pandas as pd
import os
from torch.utils.data.dataset import Dataset
import torch
import itertools
from typing import List, Union, Optional
from dataclasses import dataclass
from enum import Enum
import json
import glob
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import PreTrainedTokenizer
import csv
import tqdm
from filelock import FileLock
import logging
logger = logging.getLogger(__name__)


class OverrulingData(object):
    """
     Overruling dataset class
    """
    def __init__(self, path):
        self.data_path = path

    def train(self):
        df = pd.read_csv(os.path.join(self.data_path, 'train.csv'))
        df = df.rename(columns={'sentence1': 'txt'})
        return df

    def test(self):
        df = pd.read_csv(os.path.join(self.data_path, 'test.csv'))
        df = df.rename(columns={'sentence1': 'txt'})
        return df

    def dev(self):
        return None


class SplitDataSet:
    def __init__(self, x_train, y_train,
                 x_test, y_test,
                 x_dev=None, y_dev=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_dev = x_dev
        self.y_dev = y_dev


def split_corpus(corpus_file: str, use_dev: bool = True,
                 test_size: float = 0.2, dev_size: Union[float, None] = 0.1,
                 random_state: int = 42) -> SplitDataSet:
    x: List[str] = []
    y: List[List[str]] = []
    doc_ids: List[str] = []
    for line in open(corpus_file, encoding='utf-8'):
        labeled_provision = json.loads(line)
        x.append(labeled_provision['provision'])
        y.append(labeled_provision['label'])
        doc_ids.append(labeled_provision['source'])

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    if use_dev:
        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                                                          test_size=dev_size,
                                                          random_state=random_state)
    else:
        x_dev, y_dev = None, None

    dataset = SplitDataSet(x_train, y_train, x_test, y_test, x_dev, y_dev)
    return dataset


def multihot(labels, label_map):
    res = np.zeros(len(label_map), dtype=int)
    for lbl in labels:
        res[label_map[lbl]] = 1

    return res


class LEDGARData(object):
    """
    LEDGAR dataset class:
    Split corpus into train/test(/dev) sets, covert the text-labels into one-hot-vectors
    """

    def __init__(self, path, use_dev=True):
        self.don_data = split_corpus(path, use_dev=use_dev)
        self.all_lbls = list(sorted({
            label
            for lbls in itertools.chain(
                self.don_data.y_train,
                self.don_data.y_test,
                self.don_data.y_dev if self.don_data.y_dev is not None else []
            )
            for label in lbls
        }))
        self.label_map = {
            label: i
            for i, label in enumerate(self.all_lbls)
        }

        total = 0
        self.class_weights = np.zeros(len(self.label_map), dtype=np.float32)
        for sample in self.train()['label']:
            self.class_weights += sample
            total += 1
        self.class_weights = total / (len(self.label_map) * self.class_weights)

    def num_labels(self):
        return len(self.label_map)

    def train(self):
        return pd.DataFrame({
            'txt': self.don_data.x_train,
            'label': [multihot(lbls, self.label_map) for lbls in self.don_data.y_train]
        })

    def test(self):
        return pd.DataFrame({
            'txt': self.don_data.x_test,
            'label': [multihot(lbls, self.label_map) for lbls in self.don_data.y_test]
        })

    def dev(self):
        return pd.DataFrame({
            'txt': self.don_data.x_dev,
            'label': [multihot(lbls, self.label_map) for lbls in self.don_data.y_dev]
        })


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, task='overruling'):
        self.tokenizer = tokenizer
        self.texts = dataframe.txt
        self.targets = dataframe.label
        self.max_len = max_len
        self.task = task

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.float if self.task == 'ledgar' else torch.long)
        }


@dataclass(frozen=True)
class CaseHOLDInputExample:
    """
    A single training/test example for caseHOLD multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class EURLEXInputExample:
    """
    A single training/test example for EURLEX57k multi-labeling task
    """
    example_id: str
    header: str
    recitals: str
    main_body: List[str]
    attachments: str
    labels: Optional[List[str]]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: Union[List[List[int]], List[int]]
    attention_mask: Optional[Union[List[List[int]], List[int]]]
    token_type_ids: Optional[Union[List[List[int]], List[int]]]
    label: Optional[Union[int, List[str]]]  ####


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    full = "full"


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def convert_examples_to_features(
            self,
            examples: List[Union[CaseHOLDInputExample, EURLEXInputExample]],
            label_list: List[str],
            max_length: int,
            tokenizer: PreTrainedTokenizer,
    ) -> List[InputFeatures]:
        """Converts a collection of `InputExample`s to features for BERT fine-tuning"""
        raise NotImplementedError()


class CaseHOLDProcessor(DataProcessor):
    """Processor for the CaseHOLD dataset."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_train_examples(self):
        """See base class."""
        logger.info("LOOKING AT {} train".format(self.data_dir))
        return self._create_examples(self._read_csv(os.path.join(self.data_dir, "train.csv")), "train")

    def get_dev_examples(self):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(self.data_dir))
        return self._create_examples(self._read_csv(os.path.join(self.data_dir, "dev.csv")), "dev")

    def get_test_examples(self):
        """See base class."""
        logger.info("LOOKING AT {} test".format(self.data_dir))
        return self._create_examples(self._read_csv(os.path.join(self.data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines[1:]:  # Skip line with column names
            examples.append(
                CaseHOLDInputExample(
                    # Row index is example ID
                    example_id=line[0],
                    # No question for this task
                    question="",
                    # Citing text prompt is context
                    contexts=[line[1], line[1], line[1], line[1], line[1]],
                    # Holding statements are endings
                    endings=[line[2], line[3], line[4], line[5], line[6]],
                    label=line[12],
                )
            )
        print(examples[0].example_id, examples[0].label)
        return examples

    def convert_examples_to_features(self,
                                     examples: List[CaseHOLDInputExample],
                                     label_list: List[str],
                                     max_length: int,
                                     tokenizer: PreTrainedTokenizer,
                                     ) -> List[InputFeatures]:
        """
        Loads a data file into a list of `InputFeatures`
        """

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
            # if ex_index % 10000 == 0:
            #     logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            choices_inputs = []
            for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
                text_a = context
                if example.question.find("_") != -1:
                    # this is for cloze question
                    text_b = example.question.replace("_", ending)
                else:
                    text_b = example.question + " " + ending

                inputs = tokenizer(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_overflowing_tokens=True,
                )
                choices_inputs.append(inputs)

            if str(example.label) not in label_map.keys():
                continue

            label = label_map[example.label]

            input_ids = [x["input_ids"] for x in choices_inputs]
            attention_mask = (
                [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
            )
            token_type_ids = (
                [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
            )

            features.append(
                InputFeatures(
                    example_id=example.example_id,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label,
                )
            )

        for f in features[:2]:
            logger.info("*** Example ***")
            logger.info("feature: %s" % f)

        return features


class EURLEX57kProcessor(DataProcessor):
    """Processor for the EURLEX57k dataset."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def get_train_examples(self):
        filenames = glob.glob(os.path.join(self.data_dir, 'train', '*.json'))
        return self.read_files(filenames, 'train')

    def get_dev_examples(self):
        filenames = glob.glob(os.path.join(self.data_dir, 'dev', '*.json'))
        return self.read_files(filenames, 'dev')

    def get_test_examples(self):
        filenames = glob.glob(os.path.join(self.data_dir, 'test', '*.json'))
        return self.read_files(filenames, 'test')

    def get_labels(self):
        filename = os.path.join(self.data_dir, 'labels.json')
        with open(filename) as file:
            return json.loads(file.readline())

    def read_files(self, filenames: List[str], split=None):
        examples = []
        for fname in tqdm.tqdm(filenames, desc=split):
            with open(fname) as file:
                data = json.load(file)
                examples.append(EURLEXInputExample(
                    example_id=data['celex_id'],
                    header=data['header'],
                    recitals=data['recitals'],
                    main_body=data['main_body'],
                    attachments=data['attachments'],
                    labels=data['concepts']
                ))
        return examples

    def convert_examples_to_features(
            self,
            examples: List[EURLEXInputExample],
            label_list: List[str],
            max_length: int,
            tokenizer: PreTrainedTokenizer,
    ) -> List[InputFeatures]:

        features = []
        for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
            text = example.header + example.recitals
            inputs = tokenizer(
                text,
                None,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False
            )

            features.append(
                InputFeatures(
                    example_id=example.example_id,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
                    token_type_ids=inputs["token_type_ids"] if "token_type_ids" in inputs else None,
                    label=example.labels,
                )
            )

        for f in features[:2]:
            logger.info("*** Example ***")
            logger.info("feature: %s" % f)

        return features


processors = {"casehold": CaseHOLDProcessor, "eurlex57k": EURLEX57kProcessor}


class CustomDatasetWithCache(Dataset):
    """
    PyTorch multiple choice dataset class
    """

    features: List[InputFeatures]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            model_name: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        processor = processors[task](data_dir)
        label_list = processor.get_labels()
        self.label_map = {label: i for i, label in enumerate(label_list)}
        self.task = task
        cached_dir = os.path.join(data_dir, model_name)
        if not os.path.exists(cached_dir):
            os.makedirs(cached_dir)
        cached_features_file = os.path.join(
            cached_dir,
            "cached_{}_{}_{}".format(
                task,
                mode.value,
                str(max_seq_length),
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                if mode == Split.dev:
                    examples = processor.get_dev_examples()
                elif mode == Split.test:
                    examples = processor.get_test_examples()
                else:
                    examples = processor.get_train_examples()

                logger.info("Training examples: %s", len(examples))
                self.features = processor.convert_examples_to_features(
                    examples,
                    label_list,
                    max_seq_length,
                    tokenizer,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i): #-> InputFeatures:
        if self.task == 'casehold':
            return self.features[i]
        else:
            return {
                'input_ids': torch.tensor(self.features[i].input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(self.features[i].attention_mask, dtype=torch.long),
                # 'token_type_ids': None,
                'labels': torch.tensor(multihot(self.features[i].label, label_map=self.label_map), dtype=torch.float)
            }


def build_dataset(args, tokenizer):
    dev_set = None
    if args.task_name in ['casehold', 'eurlex57k']:
        logger.info(f"Loading {args.task_name} Dataset ...")
        args.data_path = os.path.join(args.data_path, args.task_name)
        training_set = CustomDatasetWithCache(args.data_path, tokenizer, args.task_name, args.model_name, args.max_len,
                                              overwrite_cache=args.overwrite_cache, mode=Split.train)
        testing_set = CustomDatasetWithCache(args.data_path, tokenizer, args.task_name, args.model_name, args.max_len,
                                             overwrite_cache=args.overwrite_cache, mode=Split.test)
        if args.task_name == 'eurlex57k':
            dev_set = CustomDatasetWithCache(args.data_path, tokenizer, args.task_name, args.model_name, args.max_len,
                                             overwrite_cache=args.overwrite_cache, mode=Split.dev)

    else:
        if args.task_name in ['overruling', 'ToS']:
            logger.info(f"Loading {args.task_name} Dataset ...")
            args.data_path = os.path.join(args.data_path, args.task_name)
            dataset = OverrulingData(args.data_path)
        else:
            args.data_path = os.path.join(args.data_path, 'LEDGAR_2016-2019_clean_freq400-500.jsonl')
            logger.info(f"Loading LEDGAR Dataset from {args.data_path} ...")
            dataset = LEDGARData(args.data_path)

        training_set = CustomDataset(dataset.train(), tokenizer, args.max_len, args.task_name)
        testing_set = CustomDataset(dataset.test(), tokenizer, args.max_len, args.task_name)
        if dataset.dev() is not None:
            dev_set = CustomDataset(dataset.dev(), tokenizer, args.max_len, args.task_name)

    return {
        "training_set": training_set,
        "dev_set": dev_set,
        "testing_set": testing_set
    }