from src.fine_tuning import fine_tune, fine_tune7B



# BERT
fine_tune (training_path='./data/zero_shot.csv', # path to the csv file
           checkpoint='bert-base-uncased',       # base model checkpoint on huggingface
           bs = 32,                              # Batch size, reduce it if you got CUDA out of memory error
           n_epochs = 5,                         # Number of epochs 
           output_dir='bert')                    # Path to save the fine-tuned model

#GPT-2           
fine_tune (training_path='./data/zero_shot.csv', 
           checkpoint='gpt2',
           bs = 32,
           n_epochs = 5, 
           output_dir='gpt2')

#Falcon           
fine_tune7B (training_path='./data/zero_shot.csv', 
           checkpoint='tiiuae/falcon-7b', 
           bs = 24,
           n_epochs = 5,
           output_dir='falcon')

#MistralAI           
fine_tune7B (training_path='./data/zero_shot.csv', 
           checkpoint='mistralai/Mistral-7B-Instruct-v0.2', 
           bs = 24,
           n_epochs = 5,
           output_dir='mistral')



'''         
#LLaMA-2
# Please check my notes on how to use LLaMA-2
# Uncomment lines 21 and 22 in src/fine_tuning.py
# Make sure to bring a token from huggingface
# if you have the model locally, change the parameter checkpoint to the directory containing LLaMA-2             
fine_tune7B (training_path='./data/zero_shot.csv', 
           checkpoint='meta-llama/Llama-2-7b-hf',
           bs = 24,
           n_epochs = 5, 
           output_dir='llama2')
'''




          
