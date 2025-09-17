# Article
Mohammed I. Radaideh, O Hwang Kwon, and Majdi I. Radaideh, "Fairness and Social Bias Quantification in Large Language Models for Sentiment Analysis", [Knowledge Based Systems](https://www.sciencedirect.com/journal/knowledge-based-systems), 2025, [DOI](https://doi.org/10.1016/j.knosys.2025.113569). 

# Installation 
The best way to run the codes is by using Anaconda. Create an Anaconda environment with Python 3.11 and install the required packages using:
```bash  
conda create -n NAME python=3.11
pip install -r requirments.txt
```
Replace NAME in the first line with any name. Sufficient GPU memory is crucial for fine-tuning and testing. Check whether Nvidia-cuda was installed using 
```bash
import torch
print(torch.cuda.is_available())
```
If this prints ```False```, you can download CUDA from [Pytorch](https://pytorch.org/get-started/locally/) website.

# Notes for Error Handling
If you try to run the codes from clusters, you may get errors when importing models from huggingface, and you need to define environment variables. I could handle some of the errors using the following: 
```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_P2P_DISABLE"]="1"
os.environ["NCCL_IB_DISABLE"]="1"
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
```
These lines are in the files in ```./src/```. Currently, the last two are comments; you can uncomment them if needed. I also needed at some point to use the following linux command to avoid an importing error: 
```bash
$ export HF_ENDPOINT=https://hf-mirror.com
```
Please note that these errors are related to the computer you are using and the model you are importing. The codes will most likely work on personal computers but not clusters, which have additional security protocols for imports.  

In worst-case scenario, the models can be downloaded from HaggingFace and locally imported if errors persist. 

# Fine-tuning and Testing
Due to X/Twitter policy, we cannot publicly share the tweets related to nuclear energy, but they can be sent via request to the Corresponding author [Majdi](radaideh@umich.edu). Fine-tuning of the models using 4,000 general tweets (```./data/zero-shot.csv```) from this Kaggle [dataset](https://www.kaggle.com/datasets/daniel09817/twitter-sentiment-analysis) can be done by running: 

```bash 
python run_fine_tuning.py
```
The models fine-tuned using 80\% of this Kaggle dataset (```./data/train_gen.csv```) are available on my [page](https://huggingface.co/kumo24) on huggingface. For example, the fine-tuned BERT model has the checkpoint ```kumo24/bert-sentiment```. Additionally, models fine-tuned using nuclear energy tweets are also available, e.g., the BERT model has a checkpoint ```kumo24/bert-sentiment-nuclear```.  

To test the fine-tuned models on 3,000 samples (```./data/sample.csv```), use: 

```bash 
python run_testing.py
```

The classification accuracy is 97-98\%. You can also test the models using 20\% of the [dataset](https://www.kaggle.com/datasets/daniel09817/twitter-sentiment-analysis) (```./data/test_gen.csv```), but the classification accuracy will also be 98\%. The classification report is written to ```./Results/``` file (e.g., ```./Resutls/bert_cr.csv```). Both ```run_fine_tuning.py``` & ```run_testing.py``` have comments that describe each input parameter. The models used are BERT, GPT-2, LLaMA-2-7B, Falcon-7B, and MistralAI-7B. The fine-tuned models with the highest classification accuracy are on my [page](https://huggingface.co/kumo24) on HuggingFace. 

**The model and the results for LLaMA-2 are excluded because of the restrictions on its use.** For LLaMA-2 fine-tuning, you must have a huggingface account, and you need to grant access from Meta; you will get it in a few hours. If you want to import the model from huggingface, you need to generate a token from your account and use it by huggingface package (```huggingface_hub```). The lines of code below show how to do that in Python. If you want to avoid this last step, download the model and keep it locally on your machine after you get permission to access the files.

```bash
from huggingface_hub import login
login(token='YOUR TOKEN')
```
These two lines are among the top lines of files in ``` ./src```; you can uncomment them and use your token. For testing, use the directory of your fine-tuned model as a checkpoint. 

# Bias Instances
The 100 simple energy prompts are located in ```./prompts```. To get the energy bias instances for all models except LLaMA-2, run: 

```bash 
python run_bias.py
```
The numbers of instances are written in ```./Results/``` for models fine-tuned on the general tweets (```./general_instaces.csv```) and nuclear tweets (```./nuclear_instances.csv```). The instances for each model are written in separate directories in ```./Results/kumo24```. 
# Using SHAP
To use SHAP on BERT fine-tuned on general tweets run: 

```bash 
python run_shap.py
```
SHAP bar plots are written to ```./SHAP/kumo24/bert-sentiment/```. The instances for all cases considered in our paper are in an Excel sheet located at ```./SHAP/Supplementary_Material.xlsx```. 
