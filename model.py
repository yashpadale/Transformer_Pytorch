from Transformer_Pytorch.LanguageModel.init import *
summary_writer = TransformerModel(
    file_path=r"../Data/SARVAM/classification.txt",
    max_seq_length=100, d_model=512, num_heads=16, num_layers=4, d_ff=324, dropout=0.1,
    learning_rate=0.0001,
    save_model_dir="../pt/summary", pickle_path='../pickle/summary.pkl',custom_tokens=None)

summary_writer.train(epochs=100, batch_size=4)
summary_writer.save_model()
with open(r"../pickle/summary.pkl", 'rb') as file:
    loaded_dict = pickle.load(file)

summary_writer.load_model(dictionary=loaded_dict)

def call_summary_model(prompt:str):
    result = summary_writer.generate_text(prompt,dictionary=loaded_dict,raw_output=False)
    return result
while True:
    i=input("Enter: ")
    result = call_summary_model(i)
    print(result)

# Should we be maintaining social distancing even after Covid-19? Write summary
