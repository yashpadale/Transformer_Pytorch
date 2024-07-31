
from utils import *

def preprocess(file_name,maxlen):
    content=read_file(file_name)
    content = pad_segments(content=content, maxlen=maxlen)
    # print("content")
    # print(content)
    u = split_and_sort(string=content)
    # print("u")
    # print(u)
    dictionary = return_dict(unique_words=u)
    vocab_size=max(  list(dictionary.values() ) )+1
    order = return_order(dict_=dictionary, content=content)
    batches = returns_batches(order=order, n=maxlen)
    X, Y = rearrange(batches=batches)
    # x_train, x_test = split_list(lst=X, percentage=per)
    # y_train, y_test = split_list(lst=Y, percentage=per)
    return  X, Y ,dictionary

def process(query,maxlen,dictionary):
    query='<start> '+query+' <end> '

    content = pad_segments(content=query, maxlen=maxlen)
    # print("content")
    # print(content)
    order = return_order(dict_=dictionary, content=content)
    # print("order")
    # print(order)
    return order






