encoding='utf-8'
import collections
import math
import os
import random
import zipfile
import gc
#from datalist1 import vocabulary

#from datalicense8 import vocabulary8
from nltk.tokenize import word_tokenize 

import numpy as np
import urllib
import tensorflow as tf
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import pprint
print(tf.__version__)

'''
filename = 'F:/Word2vec/text8.zip'

embedding_save_path = "F:/Word2vec/embeddings.txt" 

# Read the data into a list of strings.并生成单词表
def read_data(filename):
  """
  Extract the first file enclosed in a zip file as a list of words.可以直接使用zipfile进行文件的读取，然后使用tf自带的as_str_any方法将其还原成字符串表示
  """
  #wf= open('F:/Word2vec/test8.txt','w',encoding='utf-8')
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()#tf.compat.as_str将二进制转换为字符串，f.namelist()[0]解压后的第一个文件
    
   # wf.write(data)
  return data

vocabulary = read_data(filename)
print(vocabulary[:1000])
print('Data size', len(vocabulary))
'''

embedding_save_path = "H:/Experiments-dataset/ennl/nl-vector.txt" 


file='H:/Experiments-dataset/ennl/nltexts1.txt'

with open(file,'r',encoding='utf-8')as rf:
	lines=rf.readlines()
	'''
	for line in lines:
		text=line.strip('\n')
		#print(text)
		
		#text=parser(text)
		
		stop_words = set(stopwords.words('english'))
		word_tokens = word_tokenize(text)
		vocabulary = []
		english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\n*','\b']
		for w in word_tokens:
			if w not in stop_words:
				if w not in english_punctuations:
					vocabulary.append(w.encode("GBK", 'ignore'))
	'''
stop_words = set(stopwords.words('english'))
filtered_sentence = []
for line in lines:
	
	text=line.strip('\n')
	
	word_tokens = word_tokenize(line)
		
	english_punctuations = (',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\\n*','\b','\n','\\')
	for w in word_tokens:
		if w not in stop_words and w not in english_punctuations:
				filtered_sentence.append(w)
		#print('over')
		#print(text.encode("GBK", 'ignore'))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 100000#词典大小（50000）


def build_dataset(words, n_words):
  """Process raw inputs into a dataset.选择词频前5000个单词作为单词列表，其它的不在列表里面的作为unknown 
  data是单词的index，dictionary是正向的word–>index的字典，reverse_dictionary是反向的index–>word的字典"""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  
  reversed_dictionary = dict(zip(dictionary.values(),
                                 dictionary.keys()))
  return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(filtered_sentence,
                                                            vocabulary_size)

'''
result=open('dataresult1.py','w',encoding='utf-8')
result.write('data = ' + pprint.pformat(data,width=1000,compact=True))
result.close()

result=open('countresult1.py','w',encoding='utf-8')
result.write('count = ' + pprint.pformat(count,width=1000,compact=True))
result.close()

result=open('dictresult1.py','w',encoding='utf-8')
result.write('dict = ' + pprint.pformat(dictionary,width=1000,compact=True))
result.close()

result=open('redictresult1.py','w',encoding='utf-8')
result.write('redict =' + pprint.pformat(reverse_dictionary,width=1000,compact=True))
result.close()
                                       
'''
#reverse_dictionary=collections.defaultdict(list,reverse_dictionary)
del filtered_sentence#vocabulary  # Hint to reduce memory.
gc.collect()
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10],
      [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2,
                               skip_window=1)
for i in range(8):
  print(batch[i], reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])


# Step 4: Build and train a skip-gram model.

batch_size = 500  #一个batch中的训练数据的个数（128）
embedding_size = 300  # Dimension of the embedding vector.向量维度（128）
skip_window = 1       # How many words to consider left and right.
num_skips = 2         
# How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16    # 抽取的验证单词数
# Random set of words to evaluate similarity on.
valid_window = 100  #验证单词只从频数最高的100个单词中抽取
# Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size,replace=False)
# 从np.arange（valid_window）中选valid——size个
num_sampled = 64    # Number of negative examples to sample.负样本的单词数量

graph = tf.Graph()

with graph.as_default():

  # Input data.输入数据
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of 
  # missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))#随机生成词向量
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Construct the variables for the NCE loss构造NCE损失的变量
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  # Compute the average NCE loss for the batch.计算批处理的平均NCE损失
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=embed,
                     num_sampled=num_sampled,
                     num_classes=vocabulary_size))#使用NCE loss

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.计算小批量示例和所有嵌入之间的余弦相似性
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.添加变量初始化
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print('Initialized')

  average_loss = 0
  for step in range(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 2000 == 0:
      if step > 0:
        average_loss /= 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step ', step, ': ', average_loss)
      average_loss = 0

    # Note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8  # number of nearest neighbors最近邻数
        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log_str = '%s %s,' % (log_str, close_word)
        #print(log_str)
      final_embedding = normalized_embeddings.eval()
  final_embeddings = normalized_embeddings.eval()
  with open(embedding_save_path,'w',encoding = 'utf-8') as file:
	  for i in range(len(final_embedding)): 
		  word = reverse_dictionary[i] 
		  vector = [] 
		  for j in range(len(final_embedding[i])): 
			  vector.append(final_embedding[i][j]) 
		  file.writelines(word + '：' + str(vector) + '\n')

  
  
# Step 6: Visualize the embeddings.可视化向量


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt

  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in range(plot_only)]
  plot_with_labels(low_dim_embs, labels)

except ImportError:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')


