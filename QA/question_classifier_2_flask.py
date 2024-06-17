import json
import torch
from sklearn.metrics import matthews_corrcoef
import numpy as np
import pandas as pd
import time
import datetime
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
import random
from flask import Flask, jsonify, request
import matplotlib.pyplot as plt
import seaborn as sns

# 预训练BERT模型存储路径
bert_model = 'E:/project/TrainKGQA/chinese-bert-wwm-ext'

# 微调BERT模型作为问题分类器的存储路径
MODEL_PATH = 'model/classifier.bin'

# 数据路径
data_path = 'json/data.json'
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained(bert_model)

types_tag = ('1', '2', '3', '4', '5', '11', '12', '18', '19', '20')
# 标签到idx的映射
tag2idx = {tag: idx for idx, tag in enumerate(types_tag)}
# idx到标签的映射
idx2tag = {idx: tag for idx, tag in enumerate(types_tag)}
print(tag2idx)
num_labels = len(tag2idx.keys())


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def tokenize(text):
    all_input_ids = []
    all_input_mask = []
    MAX_SEQ_LEN = 64
    for sentence in text:
        tokens = tokenizer.tokenize(sentence)

        # limit size to make room for special tokens
        if MAX_SEQ_LEN:
            tokens = tokens[0:(MAX_SEQ_LEN - 2)]

        # add special tokens
        tokens = [tokenizer.cls_token, *tokens, tokenizer.sep_token]

        # convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # create mask same size of input
        input_mask = [1] * len(input_ids)

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)

    # pad up to max length
    # up to max_seq_len if provided, otherwise the max of current batch
    max_length = MAX_SEQ_LEN if MAX_SEQ_LEN else max([len(ids) for ids in all_input_ids])

    all_input_ids = torch.LongTensor([i + [tokenizer.pad_token_id] * (max_length - len(i))
                                      for i in all_input_ids])
    all_input_mask = torch.FloatTensor([m + [0] * (max_length - len(m)) for m in all_input_mask])

    return all_input_ids, all_input_mask


def dataset(data_path):
    sentences = []
    labels = []

    with open(data_path, 'r', encoding='utf-8') as fp:
        jsondata = json.load(fp)
        for item in jsondata['train'] + jsondata['dev']:
            sentences.append(item['question'])
            labels.append(str(item['question_type']))

    tags = [tag2idx[label] for label in labels]
    targets = torch.tensor(tags)
    all_input_ids, all_input_mask = tokenize(sentences)
    print('Original: ', sentences[0])
    print('Token IDs:', all_input_ids[0])
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(all_input_ids, all_input_mask, targets)

    # Create a 90-10 train-validation split.

    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader


def train(train_dataloader, validation_dataloader):
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    model = BertForSequenceClassification.from_pretrained(
        bert_model,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
        return_dict=False
    )

    # Tell pytorch to run this model on the GPU.
    model.to(device)
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                            num_warmup_steps = 0, # Default value in run_glue.py
    #                                            num_training_steps = total_steps)

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # print (batch)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the`to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels,
                                 return_dict=False
                                 )

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            # scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")
    # Save the model
    # torch.save(model.state_dict(), MODEL_PATH)
    torch.save(model.state_dict(), "model/classifier_4.bin")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    """保存训练概要到本地"""
    # 保留 2 位小数
    pd.set_option('precision', 2)
    # 加载训练统计到 DataFrame 中
    df_stats = pd.DataFrame(data=training_stats)
    # 使用 epoch 值作为每行的索引
    df_stats = df_stats.set_index('epoch')
    # 展示表格数据
    df_stats.to_csv("model/training_stats_4.csv")

    """绘制性能图"""
    # 绘图风格设置
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)
    # 绘制学习曲线
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    plt.savefig("model/loss_4.jpg")
    plt.show()


def test(MODEL_PATH, bert_model, data_path):
    test_sentences = []
    test_labels = []
    with open(data_path, 'r', encoding='utf-8') as fp:
        jsondata = json.load(fp)
        for item in jsondata['test']:
            test_sentences.append(item['question'])
            test_labels.append(str(item['question_type']))

    tags = [tag2idx[label] for label in test_labels]
    targets = torch.tensor(tags)
    all_input_ids, all_input_mask = tokenize(test_sentences)
    print('Original: ', test_sentences[0])
    print('Token IDs:', all_input_ids[0])

    # Set the batch size.
    batch_size = 32

    # Create the DataLoader.
    prediction_data = TensorDataset(all_input_ids, all_input_mask, targets)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(all_input_ids)))
    model = BertForSequenceClassification.from_pretrained(
        bert_model,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
        return_dict=False

    )
    # tokenizer = BertTokenizer.from_pretrained(bert_model)
    model.load_state_dict(torch.load(MODEL_PATH))
    # Put model in evaluation mode
    #
    #
    # # Tell pytorch to run this model on the GPU.
    # classifier_model = torch.load(MODEL_PATH)
    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in classifier_model.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    #     # load params
    # model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to CPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        print(label_ids)
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('DONE.')

    matthews_set = []

    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch...
    for i in range(len(true_labels)):
        # The predictions for this batch are a 2-column ndarray (one column for "0"
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        print(pred_labels_i)
        # Calculate and store the coef for this batch.
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print('Total MCC: %.3f' % mcc)

    true_tag = []
    predicate_tag = []
    for true_label in flat_true_labels:
        true_tag.append(idx2tag[true_label])
    for flat_prediction in flat_predictions:
        predicate_tag.append(idx2tag[flat_prediction])

    # Calculate the MCC
    mcc_tag = matthews_corrcoef(true_tag, predicate_tag)

    print('Total MCC: %.3f' % mcc_tag)
    print(predicate_tag[0])
    print(true_tag[0])


class SentencePredictionModel(object):
    def __init__(self, model_path, bert_model):
        # 创建并加载模型
        self.model = BertForSequenceClassification.from_pretrained(
            bert_model,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            return_dict=False
        )

        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def predict(self, content):
        """
        功能：转换问题为标准的输入格式,test_sentences，all_input_ids
        """
        sentences = []
        sentences.append(content)

        all_input_ids, all_input_mask = tokenize(sentences)
        print('Original: ', sentences[0])
        print('Token IDs:', all_input_ids[0])

        batch_size = 1

        # Create the DataLoader.
        prediction_data = TensorDataset(all_input_ids, all_input_mask)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(all_input_ids)))
        model = BertForSequenceClassification.from_pretrained(
            bert_model,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            return_dict=False
        )
        model.load_state_dict(torch.load(MODEL_PATH))
        # Put model in evaluation mode
        model.to(device)
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to CPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            # Store predictions and true labels
            predictions.append(logits)

        pred_labels = np.argmax(predictions[0], axis=1).flatten()
        predicate_tag = idx2tag[pred_labels[0]]
        return predicate_tag


HOST = 'localhost'
PORT = 5000

app = Flask(__name__)
pred_model = SentencePredictionModel(MODEL_PATH, bert_model)


@app.route('/task', methods=['GET'])
def get_task():
    question = request.args.get('question')
    result_type = int(pred_model.predict(question))
    return jsonify({'type': result_type})


if __name__ == "__main__":
    # train_dataloader, validation_dataloader = dataset(data_path)
    # train(train_dataloader, validation_dataloader)

    # test(MODEL_PATH, bert_model, data_path)
    # text = "从上海西站到南京南站的高铁在几点发车"
    # type = '11'

    # predictions = pred_model.predict(text)
    # print(predictions)
    # print(true_labels)

    app.run(host=HOST, port=PORT)
