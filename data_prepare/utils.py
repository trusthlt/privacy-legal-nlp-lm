import os.path
from collections import defaultdict
from random import random, randrange, randint
# import nltk
# nltk.download('punkt', download_dir='/ukp-storage-1/yin/venvs/casehold/nltk_data')
from nltk.tokenize import sent_tokenize
import logging
import glob

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level="NOTSET",
    datefmt="[%X]",
)
# Log on each process the small summary:
logger = logging.getLogger(__name__)


class TokenizeProcessor:
    def __init__(self, tokenizer, padding, max_seq_length):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_seq_length = max_seq_length

    def default_tokenizer(self, examples):
        # default tokenizer without sentence tokenization: each line is a document
        # only for training without NSP
        examples = [line for line in examples if len(line) > 0 and not line.isspace()]  # remove empty lines
        return self.tokenizer(
            examples,
            return_special_tokens_mask=True,
            padding=self.padding,
            truncation=True,
            max_length=self.max_seq_length,
        )

    def wiki_tokenizer(self, examples):
        # tokenizer for wikipedia dataset: each text contains multiple paragraphs
        tokenized_examples = defaultdict(list)
        for line in examples:
            if len(line) > 0 and not line.isspace():  # remove empty lines
                # split paragraphs and discard very short paragraphs
                paras = [para for para in line.split("\n\n") if len(para.split()) > 5]
                for para in paras:
                    # for each paragraph/document, split sentences and tokenize each sentence in a document
                    doc_dict = self.tokenizer(sent_tokenize(para),
                                              padding=self.padding,
                                              add_special_tokens=False,
                                              return_attention_mask=False,
                                              return_token_type_ids=False,
                                              truncation=True,
                                              max_length=self.max_seq_length, )
                    for key in doc_dict.keys():
                        tokenized_examples[key].append(doc_dict[key])
        return tokenized_examples

    def legal_tokenizer(self, examples):
        # tokenizer for legal dataset: each line is a document and each document contains several sentences
        tokenized_examples = defaultdict(list, {'input_ids': []})
        for line in examples:
            if len(line) < 1 or line.isspace():  # remove empty lines
                continue
            doc_dict = self.tokenizer(sent_tokenize(line),  # tokenize each sentence in a document
                                      padding=self.padding,
                                      add_special_tokens=False,
                                      return_attention_mask=False,
                                      return_token_type_ids=False,
                                      truncation=True,
                                      max_length=self.max_seq_length, )
            for key in doc_dict.keys():
                tokenized_examples[key].append(doc_dict[key])

        return tokenized_examples


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def generate_random_next(datasets, doc_idx, target_b_length):
    # This should rarely go for more than one iteration for large
    # corpora. However, just to be careful, we try to make sure that
    # the random document is not the same as the document
    # we're processing.
    ids_b = []
    for _ in range(10):
        random_doc_idx = randint(0, len(datasets) - 1)
        if random_doc_idx != doc_idx:
            break
    random_document = datasets[random_doc_idx]["input_ids"]

    random_start = randrange(0, len(random_document))
    for j in range(random_start, len(random_document)):
        ids_b.extend(random_document[j])
        if len(ids_b) >= target_b_length:
            break
    return ids_b


def create_instance_from_document(
        document, doc_idx, datasets, version,
        max_seq_length, short_seq_prob, special_tokens):
    """
    This is adapted from the equivalent function of Google BERT's repo.
    :param version: specify which features to save.
                    - "small": only save input_ids of two sentences and NSP labels
                    - "normal": save 'token_type_ids', 'special_token_ids' as well
    :param document: list of input_ids of current document [sent1_input_ids, sent2_input_ids, ...]
    :param doc_idx: index of current document
    :param datasets: the whole datasets
    :param max_seq_length: maximal sequence length of an instance
    :param short_seq_prob: probability for generating short instance
    :param special_tokens: mapping from special tokens ([CLS], [SEP], [SEP]) to their indices
    :return: an instance (dict-object) for pre-training
    """
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the input_ids from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instance = {}
    current_chunk = []
    current_length = 0

    # random start
    i = randrange(0, max(1, len(document)-1))
    # build only one example for each document
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
        if current_chunk:
            # `a_end` is how many segments from `current_chunk` go into the `A` (first) sentence.
            a_end = 1
            if len(current_chunk) >= 2:
                a_end = randrange(1, len(current_chunk))

            ids_a = []
            for j in range(a_end):
                ids_a.extend(current_chunk[j])

            ids_b = []

            # Random next
            # adjust the probability to make the number of actual and random next balanced
            if len(current_chunk) == 1 or random() < 0.25:
                ns_label = [0, 1]
                target_b_length = target_seq_length - len(ids_a)

                ids_b = generate_random_next(datasets, doc_idx, target_b_length)
                # We didn't actually use these segments so we "put them back" so
                # they don't go to waste.
                num_unused_segments = len(current_chunk) - a_end
                i -= num_unused_segments
            # Actual next
            else:
                ns_label = [1, 0]
                for j in range(a_end, len(current_chunk)):
                    ids_b.extend(current_chunk[j])
            truncate_seq_pair(ids_a, ids_b, max_num_tokens)

            # exception handling
            if len(ids_a) == 0:
                for sent in document:
                    ids_a.extend(sent)
                    if len(ids_a) >= target_seq_length:
                        break
                logger.info(f'len(ids_a) < 1! doc_idx={doc_idx}, document={document},'
                            f'ids_a={ids_a} now, len={len(ids_a)}')
                truncate_seq_pair(ids_a, ids_b, max_num_tokens)

            while len(ids_b) == 0:
                logger.info(f'len(ids_b) < 1! doc_idx={doc_idx}, ns_label={ns_label}')
                if ns_label[0] == 1:  # actual next is None
                    ns_label = [0, 1]
                target_b_length = target_seq_length - len(ids_a)
                # regenerate random next
                ids_b = generate_random_next(datasets, doc_idx, target_b_length)
                logger.info(f'ids_b = {ids_b} now, len={len(ids_b)}')
                truncate_seq_pair(ids_a, ids_b, max_num_tokens)

            if version == "normal":
                input_ids = [special_tokens["[CLS]"]] + ids_a + [special_tokens["[SEP]"]] \
                            + ids_b + [special_tokens["[SEP]"]]
                # The token_type_ids are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                token_type_ids = [0 for _ in range(len(ids_a) + 2)] + [1 for _ in range(len(ids_b) + 1)]
                special_tokens_mask = [1] + [0 for _ in range(len(ids_a))] + [1] + [0 for _ in
                                                                                    range(len(ids_b))] + [1]
                instance = {
                    "input_ids": input_ids,
                    "token_type_ids": token_type_ids,
                    "ns_label": ns_label,
                    "special_tokens_mask": special_tokens_mask}

            else:
                instance = {
                    "sent1": ids_a,
                    "sent2": ids_b,
                    "ns_label": ns_label}

    return instance


def rawtext_process(src_path, des_file):
    """
    Function for preprocessing raw texts.
    Raw texts: raw cases separated by empty line.
    """
    files = glob.glob(os.path.join(src_path, "*.txt"))
    logger.info(f"Found {len(files)} files: {[f.split('/')[-1] for f in files]}.")
    doc_counter = 0
    word_counter = 0
    writer = open(des_file, 'w')
    for file in files:
        reader = open(file)
        for line in reader.readlines():
            line = line.split()
            if len(line) > 5:  # remove very short document
                writer.write(' '.join(line)+'\n')
                doc_counter += 1
                word_counter += len(line)
        reader.close()
        logger.info(f"Finished processing of {file}")
    writer.close()
    logger.info(f"Total documents: {doc_counter}")
    logger.info(f"Total words: {word_counter}")
    logger.info("Done!")

