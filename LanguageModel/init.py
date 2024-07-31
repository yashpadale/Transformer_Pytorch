from Transformer import *
from general import *
file_path= r"/LanguageModel/train.txt"
max_seq_length=35
X, Y ,dictionary=preprocess(file_path,maxlen=max_seq_length)
dropout = 0.1
d_model = 512
num_heads = 32
num_layers = 4
d_ff = 1024
src_vocab_size=max(  list(dictionary.values() ) )+1
device = torch.device("cuda" )
transformer=LanguageModel(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
transformer.Train(X, Y, 15,25)
# sample prompt ="Summarize the main points discussed in the documentary film Food, Inc."
transformer.save_model("My_First_Transformer")
model=transformer.load_model("My_First_Transformer")
while True:
    i=input("Enter Prompt: ")
    i=model.generate_text(i,dictionary=dictionary)
    result = extract_between_tokens(i)
    print(result)




