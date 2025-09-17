from src.fine_tuning import model_test

# Classification accuracy on testing data 
# BERT 
model_test (testing_path = './data/sample.csv',       # Path to testing file, chane to ./data/test_gen.csv if you want to test on more samples.           
            checkpoint = 'kumo24/bert-sentiment',     # Fine-tuned model checkpoint on huggingface.
            cuda = True,                              # Use True for BERT, and False for the others. BERT was the first model and does not support device_map = 'auto'; you can only set the model on GPU. 
            output_name='./Results/bert_cr.csv')      # Name of the Classification Report output csv file.  

# # GPT-2
# model_test (testing_path = './data/sample.csv', 
#             checkpoint = 'kumo24/gpt2-sentiment', 
#             cuda = False,
#             output_name='./Results/gpt2_cr.csv')

# # Falcon            
# model_test (testing_path = './data/sample.csv', 
#             checkpoint = 'kumo24/falcon-sentiment', 
#             cuda = False,
#             output_name='./Results/falcon_cr.csv')

# # MistralAI            
# model_test (testing_path = './data/sample.csv', 
#             checkpoint = 'kumo24/mistralai-sentiment', 
#             cuda = False,
#             output_name='./Results/mistral_cr.csv')
            
