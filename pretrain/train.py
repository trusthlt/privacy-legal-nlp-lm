"""
Pre-training or fine-tuning a BERT model with MLM & NSP objectives on cached Datasets processed
by script "pretrain_data_generate.py".
"""
import glob
import logging
import os
import sys
import time
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
import datetime

import flax
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util, core
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForPreTraining,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    TensorType,
    TrainingArguments,
    is_tensorboard_available,
    set_seed,
)
from jax.tree_util import tree_flatten, tree_unflatten
from functools import partial
from utils import compute_epsilon, compute_target_nm

MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. "
                    "Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DPArguments:
    """
    Arguments pertaining to differential private (DP) training.
    """

    target_sigma: Optional[float] = field(
        default=None, metadata={"help": "Noise multiplier."}
    )
    target_eps: Optional[float] = field(
        default=5.0, metadata={"help": "Privacy budget, lower value means stronger privacy guarantee."}
    )
    l2_clip_norm: Optional[float] = field(
        default=1.0, metadata={"help": "Clip per-sample gradients to this norm."}
    )
    delta: Optional[float] = field(
        default=None, metadata={"help": "Target delta of privacy."}
    )
    disable_dp: bool = field(
        default=False, metadata={"help": "Disable privacy training and just train with vanilla SGD"}
    )


class TrainState(train_state.TrainState):
    """Train state with an Optax optimizer.
    Add a parameter for storing accumulated gradients.

    Args:
      accum_grads: accumulate gradients for extending batch size
    """
    accum_grads: core.FrozenDict[str, Any]


@flax.struct.dataclass
class FlaxDataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __call__(self, examples: List[Dict], pad_to_multiple_of: int) -> Dict[str, np.ndarray]:
        batch = self.tokenizer.pad(examples, pad_to_multiple_of=pad_to_multiple_of, return_tensors=TensorType.NUMPY)
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["input_ids"], batch["lm_label_ids"] = self.mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def mask_tokens(
            self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.copy()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype("bool")

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype("bool")
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype("bool") & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype("bool")
        indices_random &= masked_indices & ~indices_replaced

        random_words = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype="i4")
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def build_input(example, special_tokens):
    ids_a = example["sent1"]
    ids_b = example["sent2"]
    input_ids = [special_tokens["[CLS]"]] + ids_a + [special_tokens["[SEP]"]] \
                + ids_b + [special_tokens["[SEP]"]]
    # The token_type_ids are 0 for the [CLS] token, tokens of first sentence A and the first [SEP]
    # They are 1 for tokens of second sentence B and the final [SEP]
    token_type_ids = [0 for _ in range(len(ids_a) + 2)] + [1 for _ in range(len(ids_b) + 1)]
    special_tokens_mask = [1] + [0 for _ in range(len(ids_a))] + [1] + [0 for _ in range(len(ids_b))] + [1]
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "ns_label": example["ns_label"],
        "special_tokens_mask": special_tokens_mask}


def write_train_metric(summary_writer, train_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)
    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        step_ = step - (len(vals) // accum_steps) + 1
        if key == 'learning_rate':
            for i in range(0, len(vals), accum_steps):
                summary_writer.scalar(tag, vals[i], step_)
                step_ += 1
        else:
            for i in range(0, len(vals), accum_steps):
                summary_writer.scalar(tag, sum(vals[i:i + accum_steps]) / accum_steps, step_)
                step_ += 1


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DPArguments))

    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/wiki/wiki_small_128",
        help="Path of the cached processed datasets."
    )
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=None,
        help="Number of training samples"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Minimal (Final/Initial) learning rate of the linear learning rate scheduler"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximal training steps"
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args, dp_args, custom_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args, dp_args, custom_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level="NOTSET",
        datefmt="[%X]",
    )

    # Log on each process the small summary:
    logger = logging.getLogger(__name__)

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    initial_step = 0
    checkpoints = glob.glob(os.path.join(training_args.output_dir, "checkpoint-*"))
    if len(checkpoints) > 0:  # Starting from the latest checkpoint if exists
        logger.info(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
        model_args.model_name_or_path = checkpoints[-1]
        initial_step = int(checkpoints[-1].split('-')[-1])
    logger.info(f"model_name_or_path = {model_args.model_name_or_path}")

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir,
            do_lower_case=True, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir,
            do_lower_case=True, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    special_tokens = {
        tokenizer.cls_token: tokenizer.cls_token_id,
        tokenizer.sep_token: tokenizer.sep_token_id,
        tokenizer.mask_token: tokenizer.mask_token_id,
    }

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = FlaxDataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                        mlm_probability=custom_args.mlm_probability)

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    if model_args.model_name_or_path:
        model = FlaxAutoModelForPreTraining.from_pretrained(
            model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    else:
        model = FlaxAutoModelForPreTraining.from_config(
            config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Store some constants
    num_epochs = int(training_args.num_train_epochs)
    accum_steps = int(training_args.gradient_accumulation_steps)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    update_batch_size = train_batch_size * accum_steps
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()

    # Loading the cached processed datasets
    if custom_args.num_train_samples is not None:
        steps_per_epoch = custom_args.num_train_samples // update_batch_size
        start_epoch = initial_step // steps_per_epoch
    else:
        start_epoch = 0
    logger.info(f"Loading data from epoch-{start_epoch}...")
    datasets = load_from_disk(os.path.join(custom_args.data_path, f"epoch-{start_epoch}"))

    steps_per_epoch = len(datasets["train"]) // update_batch_size
    if start_epoch != initial_step // steps_per_epoch:
        start_epoch = initial_step // steps_per_epoch
        logger.info(f"Reloading data from epoch-{start_epoch}...")
        datasets = load_from_disk(os.path.join(custom_args.data_path, f"epoch-{start_epoch}"))

    logger.info(f"processed_datasets: {datasets}")
    for i in range(3):
        logger.info(f"sample {i}: {datasets['train'][i]}")

    total_train_steps = steps_per_epoch * num_epochs
    rest_train_steps = total_train_steps - initial_step

    # Save some configurations into file
    if not os.path.exists(os.path.join(training_args.output_dir, 'config.csv')):
        with open(os.path.join(training_args.output_dir, 'config.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['iter_batch_size', str(train_batch_size)])
            writer.writerow(['gradient_accumulation_steps', str(accum_steps)])
            writer.writerow(['update_batch_size', str(update_batch_size)])
            writer.writerow(['learning_rate', training_args.learning_rate])
            writer.writerow(['weight_decay', training_args.weight_decay])
            writer.writerow(['warmup_steps', training_args.warmup_steps])
            writer.writerow(['l2_clip_norm', dp_args.l2_clip_norm])
            writer.writerow(['num_train_examples', str(len(datasets["train"]))])
            writer.writerow(['num_train_epochs', str(num_epochs)])
            writer.writerow(['save_steps', str(training_args.save_steps)])
            writer.writerow(['initial_step', str(initial_step)])
            writer.writerow(['start_epoch', str(start_epoch)])
            writer.writerow(['total_steps', str(total_train_steps), 'rest_steps', str(rest_train_steps)])
            for key, val in vars(custom_args).items():
                writer.writerow([str(key), str(val)])

    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=custom_args.min_lr, end_value=training_args.learning_rate,
        transition_steps=training_args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=training_args.learning_rate,
        end_value=custom_args.min_lr,
        transition_steps=total_train_steps - training_args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps]
    )


    # No weight decay to bias and LayerNorm scale parameters
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)


    # Create optimizer
    if training_args.adafactor:
        optimizer = optax.adafactor(
            learning_rate=linear_decay_lr_schedule_fn,
        )
    else:
        optimizer = optax.adamw(
            learning_rate=linear_decay_lr_schedule_fn,
            b1=training_args.adam_beta1,
            b2=training_args.adam_beta2,
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
            mask=decay_mask_fn,
        )

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__,
                              params=model.params,
                              tx=optimizer,
                              accum_grads=None)

    # Load optimizer state from checkpoint if the initial step is not 0
    if initial_step > 0:
        state = restore_checkpoint(ckpt_dir=training_args.output_dir,
                                   target=state,
                                   step=initial_step,
                                   prefix='state_ckpt_')
        logger.info(f"initial step = {state.step}, opt_state = {state.opt_state}")

    # Define gradient update step fn
    def train_step(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params):
            lm_labels = batch.pop("lm_label_ids")
            nsp_labels = batch.pop("ns_label")
            label_mask = jnp.where(lm_labels > 0, 1.0, 0.0)  # compute loss, ignore padded input tokens
            lm_logits, nsp_logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[:2]
            lm_loss = optax.softmax_cross_entropy(lm_logits, onehot(lm_labels, lm_logits.shape[-1])) * label_mask
            nsp_loss = optax.softmax_cross_entropy(nsp_logits, nsp_labels)
            loss = lm_loss.sum() / label_mask.sum() + nsp_loss.sum() / nsp_labels.shape[0]
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")
        # accumulate the gradients
        new_state = state.replace(accum_grads=grad if state.accum_grads is None
        else jax.tree_multimap(lambda x, y: x + y, state.accum_grads, grad))

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return new_state, metrics, new_dropout_rng

    def train_step_dp(state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def loss_fn(params, input_ids, attention_mask, token_type_ids, lm_labels, nsp_labels):
            lm_logits, nsp_logits = state.apply_fn(input_ids=input_ids, attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids, params=params,
                                                   dropout_rng=dropout_rng, train=True)[:2]
            label_mask = jnp.where(lm_labels > 0, 1.0, 0.0)
            lm_loss = optax.softmax_cross_entropy(lm_logits, onehot(lm_labels, lm_logits.shape[-1])) * label_mask
            nsp_loss = optax.softmax_cross_entropy(nsp_logits, nsp_labels)
            loss = lm_loss.sum() / (label_mask.sum() + 1e-24) + nsp_loss.sum()   # loss for one sample
            return loss

        def clipping_grad(l2_norm_clip, loss, params, input_ids, attention_mask, token_type_ids, lm_labels, nsp_labels):
            loss_, grads = jax.value_and_grad(loss)(params, input_ids, attention_mask, token_type_ids, lm_labels,
                                                    nsp_labels)
            nonempty_grads, tree_def = tree_flatten(grads)
            total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
            divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
            normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
            return tree_unflatten(tree_def, normalized_nonempty_grads), loss_

        vmap_clipped_grad = jax.vmap(partial(clipping_grad, dp_args.l2_clip_norm, loss_fn), (None, 1, 1, 1, 1, 1))
        batch = {key: jnp.expand_dims(val, axis=0) for key, val in batch.items()}
        grad, loss = vmap_clipped_grad(state.params, batch['input_ids'], batch['attention_mask'],
                                       batch['token_type_ids'], batch['lm_label_ids'], batch['ns_label'])

        # aggregate the clipped per-sample gradients from one batch
        grad = jax.tree_map(partial(jnp.sum, axis=0), grad)
        loss = jnp.mean(loss)
        grad = jax.lax.pmean(grad, "batch")
        # accumulate the gradients
        state = state.replace(accum_grads=grad if state.accum_grads is None
        else jax.tree_multimap(lambda x, y: x + y, state.accum_grads, grad))

        metrics = jax.lax.pmean(
            {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}, axis_name="batch"
        )

        return state, metrics, new_dropout_rng

    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

    # Define eval fn
    def eval_step(params, batch):
        lm_labels = batch.pop('lm_label_ids')
        nsp_labels = batch.pop('ns_label')
        label_mask = jnp.where(lm_labels > 0, 1.0, 0.0)  # compute loss, ignore padded input tokens

        lm_logits, nsp_logits = model(**batch, params=params, train=False)[:2]
        lm_loss = optax.softmax_cross_entropy(lm_logits, onehot(lm_labels, lm_logits.shape[-1])) * label_mask
        nsp_loss = optax.softmax_cross_entropy(nsp_logits, nsp_labels)
        loss = lm_loss.sum() / label_mask.sum() + nsp_loss.sum() / nsp_labels.shape[0]
        # compute accuracy
        accuracy_lm = jnp.equal(jnp.argmax(lm_logits, axis=-1), lm_labels) * label_mask
        accuracy_lm = accuracy_lm.sum() / label_mask.sum()
        accuracy_nsp = jnp.equal(jnp.argmax(nsp_logits, axis=-1), jnp.argmax(nsp_labels, axis=-1)).sum() / \
                       nsp_labels.shape[0]

        # summarize metrics
        metrics = {"loss": loss, "accuracy_lm": accuracy_lm, "accuracy_nsp": accuracy_nsp}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return metrics

    p_eval_step = jax.pmap(eval_step, "batch", donate_argnums=(0,))


    def update_grad(state, dropout_rng):
        grad = jax.tree_multimap(lambda x: x / accum_steps, state.accum_grads)
        state = state.apply_gradients(grads=grad)
        state = state.replace(accum_grads=None)
        snr = {"noise": [0], "grad": [0], "snr": [0]}
        snr = jax.lax.pmean(snr, axis_name="batch")
        return state, dropout_rng, snr


    def update_grad_dp(state, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        # add noise to the aggregated gradient and normalize over batch
        grad, tree_def = tree_flatten(state.accum_grads)
        rngs = jax.random.split(dropout_rng, len(grad))
        noise = [dp_args.l2_clip_norm * sigma * jax.random.normal(r, g.shape) for r, g in zip(rngs, grad)]
        noise_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in noise])
        grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in grad])
        grad_snr = grad_norm / noise_norm
        grad = [(g + n) / update_batch_size for g, n in zip(grad, noise)]
        grad = tree_unflatten(tree_def, grad)
        # update the gradients
        state = state.apply_gradients(grads=grad)
        state = state.replace(accum_grads=None)
        snr = {"noise": noise_norm, "grad": grad_norm, "snr": grad_snr}
        snr = jax.lax.pmean(snr, axis_name="batch")
        return state, new_dropout_rng, snr


    p_optimizer_step = jax.pmap(update_grad, "batch", donate_argnums=(0,))

    # Replicate the train state on each device
    state = jax_utils.replicate(state)

    num_train_samples = len(datasets["train"]) - (len(datasets["train"]) % update_batch_size)

    epochs = tqdm(range(start_epoch, num_epochs), desc=f"Epoch ... ({start_epoch}/{num_epochs})", position=0)

    if not dp_args.disable_dp:
        p_train_step = jax.pmap(train_step_dp, "batch", donate_argnums=(0,))
        p_optimizer_step = jax.pmap(update_grad_dp, "batch", donate_argnums=(0,))
        delta = dp_args.delta if dp_args.delta is not None else 1 / (1.1 * num_train_samples)
        sigma = compute_target_nm(target_eps=dp_args.target_eps,
                                  steps=total_train_steps,
                                  sample_rate=update_batch_size / num_train_samples,
                                  delta=delta)
        if dp_args.target_sigma:
            sigma = dp_args.target_sigma

        eps, alpha = compute_epsilon(steps=total_train_steps,
                                     noise_multiplier=sigma,
                                     sample_rate=update_batch_size / num_train_samples,
                                     delta=delta)
        with open(os.path.join(training_args.output_dir, 'config.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['tf_epsilon', str(eps), 'tf_delta', str(delta), 'sigma', str(sigma)])
            writer.writerow([f"[{datetime.datetime.now()}]- Achieves ({eps}, {delta})-DP"])
        epochs.write(f"For target epsilon {dp_args.target_eps}: sigma={sigma}, alpha={alpha}")
        epochs.write(f"******* achieves ({eps}, {delta})-DP *******")

    train_time = 0
    total_step = 0
    total_loss = 0
    losses = []
    snrs = []
    loss_path = os.path.join(training_args.output_dir, 'losses.npy')
    snr_path = os.path.join(training_args.output_dir, 'SNR.npy')
    if os.path.exists(loss_path):
        losses = list(np.load(loss_path))
        total_step = len(losses) * accum_steps
        total_loss = losses[-1] * total_step

    if os.path.exists(snr_path):
        snrs = list(np.load(snr_path))


    cur_step = initial_step
    start_step = (initial_step % steps_per_epoch) * accum_steps

    if cur_step > 0 and (cur_step % training_args.eval_steps == 0 or cur_step == total_train_steps):
        # ======================== Evaluating ==============================
        num_eval_steps = len(datasets["validation"]) // eval_batch_size
        eval_metrics = []
        for val_step in tqdm(range(num_eval_steps), desc="Evaluating ...", position=2):
            batch_idx = range(val_step * eval_batch_size, (val_step + 1) * eval_batch_size)
            examples = [build_input(datasets['validation'][int(idx)], special_tokens) for idx in batch_idx]
            model_inputs = data_collator(examples, pad_to_multiple_of=16)
            # Model forward
            model_inputs = shard(model_inputs.data)
            metrics = p_eval_step(state.params, model_inputs)
            eval_metrics.append(metrics)

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            write_eval_metric(summary_writer, eval_metrics, cur_step)

        # Update progress bar
        epochs.desc = f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, " \
                      f"Acc_MLM: {eval_metrics['accuracy_lm']}, " \
                      f"Acc_NSP: {eval_metrics['accuracy_nsp']})"
        with open(os.path.join(training_args.output_dir, 'config.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([f"[{datetime.datetime.now()}]-Step... ({cur_step} | Loss: {eval_metrics['loss']}, "
                             f"Acc_MLM: {eval_metrics['accuracy_lm']}, "
                             f"Acc_NSP: {eval_metrics['accuracy_nsp']})"])

    logger.info(f"init_mean_loss: {total_loss / (total_step + 1e-24)}, total_step = {total_step}, cur_step = {cur_step}"
                f"Training batch size: {update_batch_size}, Number samples: {num_train_samples},"
                f"Rest training steps: {rest_train_steps}")

    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()
        train_metrics = []
        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        num_steps = steps_per_epoch * accum_steps

        if epoch > start_epoch:
            epoch_path = os.path.join(custom_args.data_path, f'epoch-{epoch}')
            logger.info(f"Loading data for new epoch from {epoch_path} ...")
            datasets = load_from_disk(epoch_path)
            start_step = 0
        logger.info(f"start_step: {start_step}")

        with tqdm(total=(num_steps - start_step) // accum_steps, desc="Iteration", position=1) as pbar:
            for step in range(start_step, num_steps):
                batch_idx = range(step * train_batch_size, (step + 1) * train_batch_size)
                examples = [build_input(datasets['train'][idx], special_tokens) for idx in batch_idx]
                model_inputs = data_collator(examples, pad_to_multiple_of=16)
                model_inputs = shard(model_inputs.data)
                # Model forward
                state, train_metric, dropout_rngs = p_train_step(state, model_inputs, dropout_rngs)
                train_metrics.append(train_metric)
                total_loss += train_metric['loss'][0]
                total_step += 1

                if (step + 1) % accum_steps == 0:
                    state, dropout_rngs, snr = p_optimizer_step(state, dropout_rngs)

                    mean_loss = total_loss / total_step
                    losses.append(mean_loss)
                    snrs.append(snr['snr'][0])
                    pbar.set_postfix_str(f"Loss:{mean_loss: .5f}, "
                                         f"SNR: {snr['snr']}, "
                                         f"Learning rate: {train_metric['learning_rate']}, ")
                    pbar.update(1)
                    cur_step += 1

                    if cur_step % training_args.logging_steps == 0 or cur_step == total_train_steps:
                        # Save metrics
                        train_metric = jax_utils.unreplicate(train_metric)
                        train_time += time.time() - train_start
                        if has_tensorboard and jax.process_index() == 0:
                            write_train_metric(summary_writer, train_metrics, train_time, cur_step)

                        epochs.write(
                            f"Step... ({cur_step} | Loss: {mean_loss}, Learning Rate: {train_metric['learning_rate']})"
                        )
                        with open(os.path.join(training_args.output_dir, 'config.csv'), 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([f"[{datetime.datetime.now()}]-Step... ({cur_step} | Loss: {mean_loss}, "
                                             f"Learning Rate: {train_metric['learning_rate']})"])

                        train_metrics = []

                        # Plot the loss and SNR
                        plt.plot(range(1, len(losses) + 1), losses)
                        plt.xlabel("Iteration")
                        plt.ylabel("Loss")
                        plt.savefig(os.path.join(training_args.output_dir, 'losses.png'))
                        plt.clf()
                        plt.plot(range(1, len(snrs) + 1), snrs)
                        plt.xlabel("Iteration")
                        plt.ylabel("Gradient-SNR")
                        plt.savefig(os.path.join(training_args.output_dir, 'SNR.png'))
                        plt.clf()
                        np.save(os.path.join(training_args.output_dir, 'losses.npy'), np.asarray(losses))
                        np.save(os.path.join(training_args.output_dir, 'SNR.npy'), np.asarray(snrs))

                    if cur_step % training_args.save_steps == 0 or cur_step == total_train_steps:
                        # save checkpoint after each epoch and push checkpoint to the hub
                        save_path = os.path.join(training_args.output_dir, f"checkpoint-{cur_step}")
                        if jax.process_index() == 0:
                            model.save_pretrained(
                                save_path,
                                params=jax.device_get(jax.tree_map(lambda x: x[0], state.params)),
                                push_to_hub=training_args.push_to_hub,
                                commit_message=f"Saving weights and logs of step {cur_step}",
                            )
                            tokenizer.save_pretrained(save_path)
                            save_checkpoint(ckpt_dir=training_args.output_dir,
                                            target=jax_utils.unreplicate(state),
                                            step=cur_step,
                                            prefix='state_ckpt_')

                    if cur_step % training_args.eval_steps == 0 or cur_step == total_train_steps:
                        # ======================== Evaluating ==============================
                        num_eval_steps = len(datasets["validation"]) // eval_batch_size
                        eval_metrics = []
                        for val_step in tqdm(range(num_eval_steps), desc="Evaluating ...", position=2):
                            batch_idx = range(val_step * eval_batch_size, (val_step + 1) * eval_batch_size)
                            examples = [build_input(datasets['validation'][int(idx)], special_tokens) for idx in
                                        batch_idx]
                            model_inputs = data_collator(examples, pad_to_multiple_of=16)
                            # Model forward
                            model_inputs = shard(model_inputs.data)
                            metrics = p_eval_step(state.params, model_inputs)
                            eval_metrics.append(metrics)

                        # normalize eval metrics
                        eval_metrics = get_metrics(eval_metrics)
                        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)

                        # Save metrics
                        if has_tensorboard and jax.process_index() == 0:
                            write_eval_metric(summary_writer, eval_metrics, cur_step)

                        # Update progress bar
                        epochs.desc = f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, " \
                                      f"Acc_MLM: {eval_metrics['accuracy_lm']}, " \
                                      f"Acc_NSP: {eval_metrics['accuracy_nsp']})"
                        with open(os.path.join(training_args.output_dir, 'config.csv'), 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [f"[{datetime.datetime.now()}]-Step... ({cur_step} | Loss: {eval_metrics['loss']}, "
                                 f"Acc_MLM: {eval_metrics['accuracy_lm']}, "
                                 f"Acc_NSP: {eval_metrics['accuracy_nsp']})"])

                    if custom_args.max_train_steps is not None and cur_step == custom_args.max_train_steps:
                        sys.exit()
