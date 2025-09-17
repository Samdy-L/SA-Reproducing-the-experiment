from transformers import AutoTokenizer
import numpy as np
from transformers import pipeline
import os
from pathlib import Path
from transformers import AutoModelForSequenceClassification
import pandas as pd
import shap
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
#os.environ['CURL_CA_BUNDLE'] = ''
#os.environ['REQUESTS_CA_BUNDLE'] = ''
#from huggingface_hub import login
#login(token='YOUR_TOKEN')

#==========================================================================
# Define the sentiment model
def sentiment_model (checkpoint,fine_tuned):
    tokenizer=AutoTokenizer.from_pretrained(fine_tuned)
    
    
    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
        

    if (checkpoint =='gpt2'):
        tokenizer.pad_token = tokenizer.eos_token
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if (checkpoint=="bert-base-uncased"):
        model = AutoModelForSequenceClassification.from_pretrained(fine_tuned, 
                                                           num_labels=3,
                                                           id2label=id2label, 
                                                           label2id=label2id)
        model.to("cuda")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(fine_tuned, 
                                                           num_labels=3,
                                                           id2label=id2label, 
                                                           label2id=label2id,
                                                           device_map='auto')
  
    if tokenizer.pad_token is None:
       tokenizer.add_special_tokens({'pad_token': '[PAD]'})
       model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    
    if (checkpoint=='bert-base-uncased'):
        sentiment_task = pipeline("sentiment-analysis", 
                            model=model, 
                            tokenizer=tokenizer,
                            device='cuda')
    else:
        sentiment_task = pipeline("sentiment-analysis", 
                            model=model, 
                            tokenizer=tokenizer)
    return sentiment_task


#==========================================================================
# Number of biased instances 
def bias (model, path, name):
    text = pd.read_csv(path, header=None)
    text = text.values.tolist()
    results = []
    for i in range (len(text)):
        a = model(text[i])
        print (a)
        results.append([text[i][0],a[0]['label'],a[0]['score']])
    
    results = np.array(results)    
    biased = []
    l = 0 
    r = 1 
    for i in range (int(len(results)/2)):
        if(results[l,1]==results[r,1]):
            pass
        else:
            biased.append([str(results[l,0]), results[l,1], results[l,2]])
            biased.append([str(results[r,0]), results[r,1], results[r,2]])
        l = l+2
        r = r+2
        
    df = pd.DataFrame(biased,columns=['text','sentiment','score'])
    df.to_csv(name)
    return len(biased)/2

#==========================================================================
# Number of instances for all models
def bias_matrix (fine_tuned, cp, path_prompts, output_name):
    bias_mat = np.zeros(len(cp))
    p = Path('./Results/')
    p.mkdir(parents=True, exist_ok=True)
    for j in range(len(cp)):
        print(fine_tuned[j])
        model = sentiment_model(checkpoint=cp[j],
                                    fine_tuned= fine_tuned[j])
        print(fine_tuned[j])
        bp = Path('./Results/'+fine_tuned[j])
        bp.mkdir(parents=True, exist_ok=True)
        instances = bias(model,
                         path_prompts, 
                        './Results/'+fine_tuned[j]+'/'+'instances.csv')
        bias_mat[j] = instances
    df_bias = pd.DataFrame(bias_mat, 
                           index=['BERT'],#, 'GPT-2', 'Falcon', 'MistralAI'
                           columns=['Energy'])
    df_bias.to_csv(output_name)
    


#==========================================================================   
# explain the model sentiment using SHAP
def explain_trans (text_path, fine_tuned, cp, mytitle, title):
    p = Path('./SHAP/'+fine_tuned)
    p.mkdir(parents=True, exist_ok=True)
    model = sentiment_model(cp, fine_tuned)
    global text
    text = pd.read_csv(text_path, encoding='cp1252')
    text = text.values.tolist()   
    for j in range(len(text)):
        sentiment = model(text[j])
        explainer = shap.Explainer(model)
        shap_values = explainer(text[j])
        f1 = plt.figure()
        ax1 = f1.add_subplot()
        shap.plots.bar(shap_values[0,:,sentiment[0]['label']], 
                       max_display=6, 
                       clustering=False,
                       clustering_cutoff=1.0,
                       show=False,
                       show_data=False)
        if (mytitle):
            plt.title(title, loc='center')
        else:
            plt.title(text[j][0]+' '+'('+sentiment[0]['label']+')', loc='center')
        plt.savefig('./SHAP/'+fine_tuned+'/'+str(j)+'.png', dpi=500, bbox_inches='tight')
 
        # 添加这行代码来关闭图形，释放内存
        plt.close(f1)  # 关闭当前图形





   


 
