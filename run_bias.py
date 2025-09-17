from src.use_model import bias_matrix

cp = ["bert-base-uncased"]
      # "gpt2",
      # "tiiuae/falcon-7b",
      # "mistralai/Mistral-7B-Instruct-v0.2",
      

# Models fine-tuned on GENERAL tweets
fine_tuned = ['kumo24/bert-sentiment']
            #   'kumo24/gpt2-sentiment',
            #   'kumo24/falcon-sentiment',
            #   'kumo24/mistralai-sentiment'

bias_matrix(fine_tuned=fine_tuned, 
            cp= cp, 
            path_prompts='./prompts/energy.csv',
            output_name='./Results/general_instances.csv')


# Models fine-tuned on Nuclear Energy tweets 
fine_tuned = ['kumo24/bert-sentiment-nuclear']
            #   'kumo24/gpt2-sentiment-nuclear',
            #   'kumo24/falcon-sentiment-nuclear',
            #   'kumo24/mistralai-sentiment-nuclear'

bias_matrix(fine_tuned=fine_tuned, 
            cp= cp, 
            path_prompts='./prompts/energy.csv',
            output_name='./Results/nuclear_instances.csv')

