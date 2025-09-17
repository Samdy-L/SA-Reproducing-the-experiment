import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
from transformers import pipeline
import os
from sklearn.metrics import accuracy_score
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
#os.environ['CURL_CA_BUNDLE'] = ''
#os.environ['REQUESTS_CA_BUNDLE'] = ''
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import pandas as pd
import datasets
from datasets import Dataset
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
#from huggingface_hub import login
#login(token=YOUR_TOKEN)

#==========================================================================
# Use accuarcy for evaluation metrics 
def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy':accuracy_score(predictions,labels)}

#==========================================================================
# LLM fine-tuning for text classification
def fine_tune (training_path, checkpoint, bs, n_epochs, output_dir):
    df = pd.read_csv(training_path, usecols=['text', 'label'])
    df_tr, df_te = train_test_split(df, test_size=0.2,shuffle=False)
    train_dataset = Dataset.from_dict(df_tr)
    test_dataset = Dataset.from_dict(df_te)
    my_dataset_dict = datasets.DatasetDict({"train":train_dataset,
                                        "test":test_dataset})

    tokenizer=AutoTokenizer.from_pretrained(checkpoint)

    if (checkpoint =='gpt2'):
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_text = my_dataset_dict.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}


    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, 
                                                               num_labels=3,
                                                               id2label=id2label, 
                                                               label2id=label2id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

  
    if (checkpoint =='gpt2'):
        model.config.pad_token_id = model.config.eos_token_id

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=16,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        save_strategy="epoch"
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_text["train"],
        eval_dataset=tokenized_text["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics)

    trainer.train()

    trainer.save_model()
    
#==========================================================================
# 7B models Fine-tuning

class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights,
                                              dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')

        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    
def fine_tune7B (training_path, checkpoint, bs, n_epochs, output_dir):
    df = pd.read_csv(training_path, usecols=['text', 'label'])
    df_tr, df_te = train_test_split(df, test_size=0.2,shuffle=False)
    train_dataset = Dataset.from_dict(df_tr)
    test_dataset = Dataset.from_dict(df_te)
    my_dataset_dict = datasets.DatasetDict({"train":train_dataset,
                                        "test":test_dataset})
    # Tokenizer definition
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)

    # Function to tokenize the data
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)
    # Use the function
    tokenized_text = my_dataset_dict.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



    # Define labels
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=3,
                                                               id2label=id2label, 
                                                               label2id=label2id,
                                                               device_map='auto')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False


        
  
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=16,
        gradient_checkpointing=True,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        logging_steps = 25,
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="no")

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_text["train"],
        eval_dataset=tokenized_text["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics)
    
    trainer.train()
    trainer.save_model()
    
#==========================================================================    
# Model testing     
def model_test (testing_path, checkpoint, cuda, output_name):
    p = Path('./Results/')
    p.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv (testing_path)
    text = np.array(test['text'])
    y_true = np.array(test['label'])
    
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    
    # Define labels
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    
    if (cuda):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, 
                                                               num_labels=3,
                                                               id2label=id2label, 
                                                               label2id=label2id)
        model.to('cuda')
    
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=3,
                                                               id2label=id2label, 
                                                               label2id=label2id,
                                                               device_map='auto')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (cuda):
        sentiment_task = pipeline("sentiment-analysis", 
                          model=model, 
                          tokenizer=tokenizer,
                          device=device)
    else:
        sentiment_task = pipeline("sentiment-analysis", 
                          model=model, 
                          tokenizer=tokenizer)

    y_pred = []
    for i in range (len(test)):
        if (len(text[i]) > 1024):
            y_true = np.delete(y_true, i)
            pass
        else:
            res = sentiment_task(text[i])
            label = res[0]['label']
            if (label=='neutral'):
                y_pred.append(1)
            elif (label=='positive'):
                y_pred.append(2)
            elif (label=='negative'):
                y_pred.append(0)
        print(i,'/',len(test))
    target_names = ['negative', 'neutral', 'positive']
    print(classification_report(y_true, y_pred, target_names=target_names))
    report = classification_report(y_true, y_pred, output_dict=True, target_names=target_names)

    # Convert the report to a DataFrame
    df = pd.DataFrame(report).transpose()

    # Save the DataFrame to a CSV file
    df.to_csv(output_name)






