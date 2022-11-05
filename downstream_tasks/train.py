import logging
import sys
from argparse import ArgumentParser
import numpy as np
import os
import csv
import glob
import shutil
import torch
from torch import cuda
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
import time
import statistics
from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import default_data_collator
from models import BERTForMultiLabelClassification
from data_loader import build_dataset
from tqdm import tqdm
from sklearn import metrics
from utils import tune_threshs, apply_threshs
import matplotlib.pyplot as plt

device = 'cuda' if cuda.is_available() else 'cpu'
logger = logging.getLogger(__name__)
logger.info(f"***** Device: {device} ********")


def argsparse():
    parser = ArgumentParser()
    # general parameters
    parser.add_argument('--task_name', type=str, required=True,
                        choices=['casehold', 'overruling', 'ToS', 'ledgar'],
                        help='name of the downstream task to train.')
    parser.add_argument('--pretrained_weights', type=str, default='bert-base-uncased',
                        help='name or path of the pretrained BERT model.')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='name or type of the pretrained BERT model.')
    parser.add_argument('--eval', action='store_true',
                        help='whether only evaluate a model (otherwise, train a model).')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='path to the model checkpoint if training from a checkpoint or evaluate a model.')
    parser.add_argument('--data_path', type=str, default='data', help='path to the dataset.')
    parser.add_argument('--save_path', type=str, default='results', help='path to save models')
    parser.add_argument('--overwrite_save_path', action='store_true', default=False,
                        help='whether overwrite the save path if it already exits')
    parser.add_argument('--seed', type=int, default=42, help='random seed.')
    parser.add_argument('--overwrite_cache', action='store_true', default=False,
                        help='For casehold task, whether overwrite the cached tokenized data')
    parser.add_argument('--log_steps', type=int, default=None,
                        help='interval steps to evaluate and log the performance')
    parser.add_argument('--max_save_limit', type=int, default=2,
                        help='maximal number of checkpoints to save')
    parser.add_argument('--hist_log_dir', type=str, default='base',
                        help='name of the directory to save history evaluation results for a specific model or task.')

    # training parameters
    parser.add_argument('--max_len', type=int, default=128, help='maximal length of one tokenized document')
    parser.add_argument('--train_batch_size', type=int, default=8, help='number of sample for one training step.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='number of steps to actually accumulate the gradients.')
    parser.add_argument('--valid_batch_size', type=int, default=8, help='validating batch size.')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-06, help='initial learning rate for training.')
    parser.add_argument('--active_bert_layers', type=int, default=None,
                        help='number of BERT layer to tune (None means all of them, no freeze).')
    return parser.parse_args()


def train():
    model.train()
    class_weights = torch.from_numpy(dataset.class_weights).float().to(device) if hasattr(dataset, 'class_weights') else None
    total_step = 0
    total_loss = 0
    losses = []
    times = []
    best_metrics = {"macro f1": 0.0,
                    "micro f1": 0.0,
                    "accuracy": 0.0,
                    "counter": 0}
    if args.log_steps is None:
        args.log_steps = len(training_loader) - 1
    for epoch in range(args.num_epochs):
        logger.info(f"******** Training Epoch {epoch} ********")
        start = time.time()
        with tqdm(total=len(training_loader), desc=f"Epoch {epoch}") as pbar:
            for step, data in enumerate(training_loader, 0):
                ids = data['input_ids'].to(device)
                mask = data['attention_mask'].to(device)
                token_type_ids = data['token_type_ids'].to(device) if 'token_type_ids' in data else None
                targets = data['labels'].to(device)
                if class_weights is not None:
                    outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets,
                                    class_weights=class_weights)
                else:
                    outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets,)
                loss = outputs[0]
                loss.backward()

                losses.append(loss.item())
                total_loss += loss.item()
                mean_loss = total_loss / (epoch * len(training_loader) + step+1)
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(training_loader) - 1:
                    total_step += 1
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    if total_step % args.log_steps == 0:
                        # evaluate the model
                        best_metrics = evaluate(total_step, best_metrics)
                        # early stopping for CaseHOLD & Ledgar task
                        if (args.task_name in ['casehold', 'ledgar'] and best_metrics["counter"] >= 2) \
                                or (args.task_name in ['overruling', 'ToS'] and best_metrics["counter"] >= 4):
                            logger.info(f"******** Early stopping at step {total_step} ******** ")
                            break
                        ckpt_path = os.path.join(save_path, f"checkpoint_{total_step}.pt")
                        logger.info(f"******** Saving model to {ckpt_path} ******** ")
                        torch.save(model.state_dict(), ckpt_path)
                        checkpoints = glob.glob(os.path.join(save_path, "checkpoint_*.pt"))
                        # remove older checkpoints if the total number of checkpoints exceed the save limit
                        if len(checkpoints) > args.max_save_limit:
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                            for ckpt in checkpoints[:-args.max_save_limit]:
                                os.remove(ckpt)
                        model.train()

        end = time.time()
        times.append(end-start)

        if best_metrics["counter"] >= 4:
            break

    logger.info(f"Times: {times}")
    logger.info(f"Median1 = {statistics.median(times)}")

    # plot the training losses over step
    plt.plot(range(1, len(losses)+1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, 'losses.png'))
    # save results to history evaluation file
    log_history(best_metrics)

    logger.info("******** error analysis on the best model********")
    model.load_state_dict(torch.load(os.path.join(args.save_path, f"best_model_lr{args.lr}.pt")))
    test_pred, test_true = predict(testing_loader)
    f1_score_macro = round(metrics.f1_score(test_true, test_pred, average='macro'), 3)
    error_analysis(test_pred, test_true, os.path.join(args.save_path, f"prediction_lr{args.lr}_f{f1_score_macro}.csv"))


def predict(dataloader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device) if 'token_type_ids' in data else None
            targets = data['labels'].to(device)
            outputs = model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            if args.task_name =='ledgar':
                y_pred.extend(torch.sigmoid(outputs[0]).cpu().detach().numpy().tolist())
            else:
                y_pred.extend(torch.argmax(outputs[0], dim=1).cpu().detach().numpy().tolist())
            y_true.extend(targets.cpu().detach().numpy().tolist())
    return np.array(y_pred), np.array(y_true)


def compute_log_metrics(step, y_true, y_pred, best_metrics):
    accuracy = round(metrics.accuracy_score(y_true, y_pred), 3)
    f1_score_micro = round(metrics.f1_score(y_true, y_pred, average='micro'), 3)
    f1_score_macro = round(metrics.f1_score(y_true, y_pred, average='macro'), 3)
    logger.info(f"Accuracy Score = {accuracy}")
    logger.info(f"F1 Score (Micro) = {f1_score_micro}")
    logger.info(f"F1 Score (Macro) = {f1_score_macro}")
    if best_metrics["macro f1"] < f1_score_macro:
        best_metrics["macro f1"] = f1_score_macro
        best_metrics["micro f1"] = f1_score_micro
        best_metrics["accuracy"] = accuracy
        best_metrics["counter"] = 0
        best_path = os.path.join(args.save_path, f"best_model_lr{args.lr}.pt")
        logger.info(f"******** Find new best, save to {best_path} ******** ")
        torch.save(model.state_dict(), best_path)
    else:
        best_metrics["counter"] += 1
    with open(os.path.join(save_path, 'eval_results.csv'), 'a') as f:
        logger.info("******** Saving evaluation results to file********")
        writer = csv.writer(f)
        row = [step, accuracy, f1_score_micro, f1_score_macro]
        writer.writerow(row)

    return best_metrics


def log_history(best_metrics):
    hist_eval_path = args.hist_log_dir
    if not os.path.exists(hist_eval_path):
        os.makedirs(hist_eval_path)

    with open(os.path.join(hist_eval_path, 'bests.txt'), 'a') as f1:
        f1.write(str(best_metrics['macro f1'])+" ")

    hist_eval_path = os.path.join(hist_eval_path,
                                  f'eval_results_{args.lr}_{args.train_batch_size*args.gradient_accumulation_steps}.csv')
    with open(hist_eval_path, 'w') as f:
        writer = csv.writer(f)
        reader1 = csv.reader(open(os.path.join(save_path, 'eval_results.csv'), 'r'))
        reader2 = csv.reader(open(os.path.join(save_path, 'config.csv'), 'r'))
        writer.writerows(reader1)
        writer.writerow(['best', best_metrics['accuracy'], best_metrics['micro f1'], best_metrics['macro f1']])
        writer.writerows(reader2)
    logger.info(f"******** Saved results to {hist_eval_path} ********")


def evaluate(step, best_metrics):
    logger.info(f"******** Evaluating at step {step} ********")
    if args.task_name == "ledgar":
        logger.info("******** Tuning clf thresholds on dev set ********")
        dev_pred, dev_true = predict(dev_loader)
        threshs = tune_threshs(dev_pred, dev_true)
    test_pred, test_true = predict(testing_loader)
    if args.task_name == "ledgar":
        logger.info("******** Applying thresholds ********")
        test_pred = apply_threshs(test_pred, threshs)

    best_metrics = compute_log_metrics(step, test_true, test_pred, best_metrics)
    return best_metrics


def error_analysis(pred, true, save_dir):
    with open(save_dir, 'w') as f:
        f.write("id,isCorrect,pred,true\n")
        for i in range(len(pred)):
            f.write(f"{i},{pred[i]==true[i]},{pred[i]},{true[i]}\n")
    f.close()


if __name__ == "__main__":
    args = argsparse()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # create path for saving the models, configuration and evaluation results
    save_path = os.path.join(args.save_path, args.task_name)
    if os.path.exists(save_path) and args.overwrite_save_path:
        try:
            shutil.rmtree(save_path)
        except OSError as e:
            logger.info(f"Error: {e.filename} - {e.strerror}.")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise ValueError("The save path is already existing, please enable overwrite_save_path or assign a new path.")

    # show and save the arguments setting
    with open(os.path.join(save_path, 'config.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'value'])
        logger.info("******** Training Configuration ********")
        for key, val in vars(args).items():
            logger.info(f"{key}={val}")
            writer.writerow([str(key), str(val)])

    # create the file for saving the evaluation results
    with open(os.path.join(save_path, 'eval_results.csv'), 'w') as f:
        writer = csv.writer(f)
        row = ['step', 'accuracy', 'f1_micro', 'f1_macro']
        writer.writerow(row)

    # initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights, do_lower_case=True)
    from_tf = os.path.exists(os.path.join(args.pretrained_weights, 'tf_model.h5'))
    from_flax = os.path.exists(os.path.join(args.pretrained_weights, 'flax_model.msgpack'))
    print("from tf:", from_tf, "from flax:", from_flax)

    if args.task_name in ['overruling', 'ToS']:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_weights, from_flax=from_flax, from_tf=from_tf, num_labels=2)
    elif args.task_name == 'ledgar':
        model = BERTForMultiLabelClassification(
            args.pretrained_weights, from_flax=from_flax, from_tf=from_tf, num_labels=98)
    elif args.task_name == 'casehold':
        # Default fast tokenizer is buggy on CaseHOLD task, switch to legacy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights, do_lower_case=True, use_fast=False)
        model = AutoModelForMultipleChoice.from_pretrained(
            args.pretrained_weights, from_flax=from_flax, from_tf=from_tf, num_labels=5)
    else:
        raise ValueError("Non-existing task name.")
    model.to(device)

    # create dataset and build dataloader
    dataset = build_dataset(args, tokenizer)
    training_loader = DataLoader(dataset['training_set'], args.train_batch_size, shuffle=True,
                                 collate_fn=default_data_collator if args.task_name == 'casehold' else None)

    testing_loader = DataLoader(dataset['testing_set'], args.valid_batch_size, shuffle=False,
                                collate_fn=default_data_collator if args.task_name == 'casehold' else None)
    if dataset['dev_set'] is not None:
        dev_loader = DataLoader(dataset['dev_set'], args.valid_batch_size, shuffle=False,
                                collate_fn=default_data_collator if args.task_name == 'casehold' else None)

    logger.info(f"Size of training set: {dataset['training_set'].__len__()}, size of testing set: {dataset['testing_set'].__len__()}")
    logger.info(f"Number of batches for training: {len(training_loader)}")
    count = 0
    for batch in training_loader:
        logger.info(f"sample batch: {batch}")
        logger.info(f"labels: {batch['labels'].shape}")
        count += 1
        if count >= 3:
            break

    if args.ckpt_path:
        logger.info(f"Loading pretrained weights from {args.ckpt_path}")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(args.ckpt_path))
        else:
            model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))

    # freeze some bert layers and only tune the last few layers
    if args.active_bert_layers:
        trainable_layers = [model.bert.pooler, model.classifier]
        trainable_layers.extend(model.bert.encoder.layer[-args.active_bert_layers:])
        logger.info(f"Number of trainable layers:{len(trainable_layers)}")
        total_params = 0
        trainable_params = 0

        for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()

        logger.info(f"Total parameters count: {total_params}")
        logger.info(f"Trainable parameters count: {trainable_params}")

    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(training_loader) // args.gradient_accumulation_steps * args.num_epochs)

    train()
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pt"))
    logger.info("******** Training completed. ********")