
def number_to_words(predictions, dictionary):
    predictions=[predictions.tolist()]
    # Invert the dictionary
    inverted_dictionary = {v: k for k, v in dictionary.items()}

    predicted_sentences = []

    for prediction_row in predictions:
        words_row = []
        for index in prediction_row:
            # Check if the index exists in the inverted dictionary
            word = inverted_dictionary.get(index)
            if word is not None:
                words_row.append(word)
        predicted_sentence = ' '.join(words_row)
        predicted_sentences.append(predicted_sentence)

    return predicted_sentences


def split_list(lst, percentage: float):
    len_= int(len(lst) * percentage)
    first_list = lst[:len_]
    second_list = lst[len_:]
    return first_list, second_list

def rearrange(batches):
    X = []
    Y = []
    for i in range(len(batches) - 1):
        X.append(batches[i])
        Y.append(batches[i + 1])
    return X, Y

def return_dict(unique_words:list):
    dictionary={}
    for i in range(len(unique_words)):
        dictionary[unique_words[i]]=i+1
    return dictionary

from nltk.tokenize import word_tokenize

def pad_segments(content: str, maxlen: int):
    maxlen = maxlen - 1  # Adjust maxlen to account for the <start> and <end> tokens
    segments = content.split('<end>')
    padded_segments = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Ensure <start> and <end> are treated as single tokens
        segment = segment.replace('<start>', 'STARTTOKENPLACEHOLDER')
        segment = segment.replace('<end>', 'ENDTOKENPLACEHOLDER')

        # Tokenize the segment
        tokens = word_tokenize(segment)

        # Replace the placeholders with original tokens
        tokens = ['<start>' if token == 'STARTTOKENPLACEHOLDER' else token for token in tokens]
        tokens = ['<end>' if token == 'ENDTOKENPLACEHOLDER' else token for token in tokens]

        # Ensure the segment starts with '<start>'
        if tokens[0] != '<start>':
            print(tokens)
            print(segment)
            raise ValueError("Segment does not start with '<start>'")

        # Remove '<start>' token for padding calculation
        start_token = tokens.pop(0)

        # Check if the segment exceeds maxlen - 1 (considering <start> and <end>)
        if len(tokens) > maxlen - 1:
            tokens = tokens[:maxlen - 1]

        padding_needed = maxlen - len(tokens) - 1
        tokens.extend(['<space>'] * padding_needed)

        # Reinsert '<start>' token at the beginning
        tokens.insert(0, start_token)

        # Add '<end>' token at the end
        tokens.append('<end>')

        padded_segments.append(' '.join(tokens))

        # Join all padded segments into the final output
    final_output = ' '.join(padded_segments)
    return final_output


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def split_and_sort(string):
    string = string.replace('<start>', 'STARTTOKENPLACEHOLDER')
    string = string.replace('<end>', 'ENDTOKENPLACEHOLDER')
    string = string.replace('<space>', 'SPACETOKENPLACEHOLDER')
    tokens = word_tokenize(string)
    tokens = ['<start>' if token == 'STARTTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<end>' if token == 'ENDTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<space>' if token == 'SPACETOKENPLACEHOLDER' else token for token in tokens]
    unique_words_list = sorted(set(tokens))
    return unique_words_list


def return_order(dict_, content: str):
    # Replace special tokens with placeholders
    content = content.replace('<start>', 'STARTTOKENPLACEHOLDER')
    content = content.replace('<end>', 'ENDTOKENPLACEHOLDER')
    content = content.replace('<space>', 'SPACETOKENPLACEHOLDER')

    # Tokenize the content
    tokens = word_tokenize(content)

    # Replace placeholders back to original tokens
    tokens = ['<start>' if token == 'STARTTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<end>' if token == 'ENDTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<space>' if token == 'SPACETOKENPLACEHOLDER' else token for token in tokens]

    # Map tokens to their corresponding values in the dictionary
    order = [dict_[token] for token in tokens if token in dict_]

    return order


def returns_batches(order, n):
    batches = [order[i:i + n] for i in range(len(order) - n + 1)]
    return batches

def rearrange(batches):
    X = []
    Y = []
    for i in range(len(batches) - 1):
        X.append(batches[i])
        Y.append(batches[i + 1])
    return X, Y