from src.use_model import explain_trans

cp = ["bert-base-uncased"]
      # "gpt2",
      # "tiiuae/falcon-7b",
      # "mistralai/Mistral-7B-Instruct-v0.2",
      

fine_tuned = ['kumo24/bert-sentiment']
            #   'kumo24/gpt2-sentiment',
            #   'kumo24/falcon-sentiment',
            #   'kumo24/mistralai-sentiment']

# BERT
# Modify second and third inputs to use other models
explain_trans (text_path ='./SHAP/shap_inst.csv', # Path to text 
               fine_tuned = fine_tuned[0],        # Fine-tuned checkpoint  
               cp = cp[0],                        # base model checkpoint,   
               mytitle=False,                     # Use your own title for SHAP figure  
               title=None)                        # your title as a str if mytitle=True  