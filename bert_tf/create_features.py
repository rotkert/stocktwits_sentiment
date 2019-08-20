import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import pickle

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

OUTPUT_DIR = 'output_unb'#@param {type:"string"}
#@markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = False #@param {type:"boolean"}
#@markdown Set USE_BUCKET and BUCKET if you want to (optionally) store model output on GCP bucket.

if DO_DELETE:
  try:
    tf.gfile.DeleteRecursively(OUTPUT_DIR)
  except:
    # Doesn't matter if the directory didn't exist
    pass
tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

data = pd.read_pickle('labeled_dataset.pickle')
data_small = data.sample(1200000)
data_small = data_small.replace({'sentiment_declared' : {'Bearish' : 0, 'Bullish' : 1}})
data_small = data_small.sample(frac=1).reset_index(drop=True)

train = data_small[:1000000]
test = data_small[1000000:]
label_list = [0, 1]

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x.message, 
                                                                   text_b = None, 
                                                                   label = x.sentiment_declared), axis = 1)
test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x.message, 
                                                                   text_b = None, 
                                                                   label = x.sentiment_declared), axis = 1)

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

MAX_SEQ_LENGTH = 64
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

with open('train_features_unb.pickle', 'wb') as file:
    pickle.dump(train_features, file)

with open('dev_features_unb.pickle', 'wb') as file:
    pickle.dump(test_features, file)
