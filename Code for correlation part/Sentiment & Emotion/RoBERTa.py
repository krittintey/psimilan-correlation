import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skmultilearn.model_selection import iterative_train_test_split
from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word, expand_contraction, remove_special_character
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
import torch.nn.functional as F
import pickle
from sklearn.metrics import classification_report

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df = pd.read_csv('../PSIMILAN/goemotion_sentiments_multilabel.csv')
# df = pd.read_csv('../PSIMILAN/goemotion_ekman_emotions_multilabel.csv')

X = df['text'].values
y = df.iloc[:, 9:].values

X_2d = np.array([[text] for text in X])

np.random.seed(42)
X_train, y_train, X_test, y_test = iterative_train_test_split(X_2d, y, test_size=0.2)
X_train, y_train, X_val, y_val = iterative_train_test_split(X_train, y_train, test_size=0.25)

X_train_new = np.array([text for sub in X_train for text in sub]) 
X_test_new = np.array([text for sub in X_test for text in sub]) 
X_val_new = np.array([text for sub in X_val for text in sub]) 

def text_preprocessing(text):
    preprocess_functions = [to_lower, remove_email, remove_url, expand_contraction]
    preprocessed_text = preprocess_text(text, preprocess_functions)
    return preprocessed_text

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def preprocessing_for_roberta(data):

    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  
            add_special_tokens=True,        
            max_length=MAX_LEN,
            truncation=True,             
            padding='max_length',         
            #return_tensors='pt',           
            return_attention_mask=True      
        )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

MAX_LEN = 512

print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_roberta(X_train_new)
val_inputs, val_masks = preprocessing_for_roberta(X_val_new)

train_labels = torch.FloatTensor(y_train)
val_labels = torch.FloatTensor(y_val)

# batch size = 16, 32
batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

def initialize_model(epochs=10):
  
    roberta_classifier = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4, output_attentions=False, output_hidden_states=False)
    roberta_classifier = nn.DataParallel(roberta_classifier)
    roberta_classifier.to(device)

    optimizer = AdamW(roberta_classifier.parameters(),
                      lr=2e-5,    
                      betas=(0.9, 0.98), 
                      eps=1e-6,
                      weight_decay=0.1    
                )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps
                )
    
    return roberta_classifier, optimizer, scheduler

# Specify loss function
num_labels = 4
loss_fn = nn.BCELoss()
m = nn.Sigmoid()
training_stats = list()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=10, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            outputs = model(b_input_ids, b_attn_mask)
            logits = outputs.logits

            # Compute loss and accumulate the loss values
            loss = loss_fn(m(logits), b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)

            training_stats.append(
                {
                    'Epoch': epoch_i + 1,
                    'Training_Loss': avg_train_loss,
                    'Valid_Loss': val_loss,
                    'Valid_Accuracy': val_accuracy,
                    'Time_Elapsed': time_elapsed,
                }
            )

            torch.save(model.state_dict(), f'../PSIMILAN/models/Fine_Tuning_RoBERTa_GoEmotion_Sentiment_epoch{epoch_i+1}.h5')
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            outputs = model(b_input_ids, b_attn_mask)
            logits = outputs.logits

        # Compute loss
        #print(logits)
        #print(b_labels)
        loss = loss_fn(m(logits), b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        # preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        # accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        accuracy = accuracy_thresh(logits.view(-1, num_labels), b_labels.view(-1, num_labels))
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def accuracy_thresh(y_pred, y_true, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: 
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.byte()).float().mean().item()

set_seed(42)    # Set seed for reproducibility
roberta_classifier, optimizer, scheduler = initialize_model(epochs=4)
train(roberta_classifier, train_dataloader, val_dataloader, epochs=4, evaluation=True)

def roberta_predict(model, test_dataloader):
    """Perform a forward pass on the trained RoBERTa model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            outputs = model(b_input_ids, b_attn_mask)
            logits = outputs.logits
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = all_logits.sigmoid().cpu().numpy()

    return probs

# Run `preprocessing_for_bert` on the test set
print('Tokenizing data...')
test_inputs, test_masks = preprocessing_for_roberta(X_test_new)

# Create the DataLoader for our test set
test_dataset = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

probs = roberta_predict(roberta_classifier, test_dataloader)

with open('../PSIMILAN/Probs/Fine_Tuning_RoBERTa_GoEmotion_Sentiment_probs.pickle', 'wb') as prob:
    pickle.dump(probs, prob)

df_label_columns = df.columns[9:]
label_names = list(df_label_columns)
report = classification_report(y_test, np.round(probs), target_names=label_names, zero_division=0, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('../PSIMILAN/roberta_sentiments_report.csv')

# Sample Validation
df_sample = pd.read_csv('../PSIMILAN/sentiment_sample_validate_dataset_trans.csv')
# df_sample = pd.read_csv('../PSIMILAN/ekman_emotions_sample_validate_dataset_trans.csv')
# df_sample = pd.read_csv('../PSIMILAN/suicide_sample_validate_dataset_trans.csv')

X_sample = df_sample['Translate'].values

print('Tokenizing data...')
sample_inputs, sample_masks = preprocessing_for_roberta(X_sample)

sample_dataset = TensorDataset(sample_inputs, sample_masks)
sample_sampler = SequentialSampler(sample_dataset)
sample_dataloader = DataLoader(sample_dataset, sampler=sample_sampler, batch_size=32)

sample_probs = roberta_predict(roberta_classifier, sample_dataloader)

encoding = {
    'pos': [1, 0, 0, 0],
    'neg': [0, 1, 0, 0],
    'q': [0, 0, 1, 0],
    'neu': [0, 0, 0, 1]
}

# encoding = {
#     'anger': [1, 0, 0, 0, 0, 0, 0],
#     'disgust': [0, 1, 0, 0, 0, 0, 0],
#     'fear': [0, 0, 1, 0, 0, 0, 0],
#     'joy': [0, 0, 0, 1, 0, 0, 0],
#     'sadness': [0, 0, 0, 0, 1, 0, 0],
#     'surprise': [0, 0, 0, 0, 0, 1, 0],
#     'neutral': [0, 0, 0, 0, 0, 0, 1]
# }

y_sample = df_sample['sentiment'].values
y_sample_encoded = [encoding[sentiment] for sentiment in y_sample]

df_label_columns = df.columns[9:]
label_names = list(df_label_columns)
y_sample_new = torch.FloatTensor(y_sample_encoded)

report = classification_report(y_sample_new, np.round(sample_probs), target_names=label_names, zero_division=0, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('../PSIMILAN/roberta_sentiments_sample_validation_report.csv')

