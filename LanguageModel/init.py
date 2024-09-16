

import torch
import pickle
from Transformer import *
from general import *


class TransformerModel:
    def __init__(self, file_path, max_seq_length, d_model, num_heads, num_layers, d_ff, dropout,learning_rate,custom_tokens,
                 save_model_dir="My_First_Transformer", pickle_path='output_response.pkl'):
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate=learning_rate
        self.save_model_dir = save_model_dir
        self.pickle_path = pickle_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = None
        self.dictionary = None
        self.X = None
        self.Y = None
        self.src_vocab_size=None
        self.custom_tokens=custom_tokens



    def load_data(self):
        self.X, self.Y, self.dictionary = preprocess(self.file_path, maxlen=self.max_seq_length,custom_tokens=self.custom_tokens)

        if self.X is None or self.Y is None:
            raise ValueError("Failed to load data. Ensure the preprocessing function returns valid training data.")

        src_vocab_size = max(list(self.dictionary.values())) + 1
        self.src_vocab_size=src_vocab_size
        self.transformer = LanguageModel(src_vocab_size, self.d_model, self.num_heads, self.num_layers, self.d_ff,
                                         self.max_seq_length, self.dropout,self.learning_rate,self.custom_tokens)

    def train(self,epochs, batch_size):
        self.load_data()

        if self.transformer is None or self.X is None or self.Y is None:
            raise ValueError("Model is not properly initialized or data is not loaded.")
        self.transformer.Train(self.X, self.Y,  batch_size,epochs)

    def save_model(self):
        if self.transformer is None:
            raise ValueError("Model is not loaded. Cannot save the model.")
        self.transformer.save_model(self.save_model_dir)
        with open(self.pickle_path, 'wb') as file:
            pickle.dump(self.dictionary, file)

    def load_model(self, dictionary=None):
        # Use the provided dictionary if available, otherwise use self.dictionary
        use_dictionary = dictionary if dictionary is not None else self.dictionary

        if use_dictionary is None:
            raise ValueError("No dictionary available for text generation.")

        with open(self.pickle_path, 'rb') as file:
            self.dictionary = pickle.load(file)

        self.src_vocab_size = max(list(self.dictionary.values())) + 1


        # Set up the model with the current parameters
        self.transformer = LanguageModel(self.src_vocab_size, self.d_model, self.num_heads, self.num_layers,
                                         self.d_ff, self.max_seq_length, self.dropout, self.learning_rate,self.custom_tokens)

        # Load the model weights
        self.transformer.load_model(self.save_model_dir)

        # Load the dictionary


    def generate_text(self, prompt, dictionary=None,raw_output=False):
        if self.transformer is None:
            raise ValueError("Model is not loaded. Call load_model() before generating text.")

        # Use the provided dictionary if available, otherwise use self.dictionary
        use_dictionary = dictionary if dictionary is not None else self.dictionary

        if use_dictionary is None:
            raise ValueError("No dictionary available for text generation.")
        # print(prompt)
        generated_text = self.transformer.generate_text(prompt, dictionary=use_dictionary)
        if raw_output!=False:
            print("raw_output : ",generated_text)
            result = extract_between_tokens(generated_text)
            return result
        else:
            result = extract_between_tokens(generated_text)
            return result



