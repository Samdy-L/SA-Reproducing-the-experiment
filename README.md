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

# My experiment
I reproduced the BERT-only experiment:  
I ran the scripts on GPU to obtain results and plots.
