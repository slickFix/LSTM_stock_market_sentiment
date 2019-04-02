
# coding: utf-8

# # Modeling Stock Market Sentiment with LSTM 

# In[ ]:


# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# importing libraries

import os
import re
import string
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import  matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter


# In[ ]:


# Display

pd.set_option('max_colwidth', 800)
pd.options.display.max_columns = None
pd.options.display.max_rows = None


# In[ ]:


# current directory
os.getcwd()


# ## Processing Data
# 
# 
# Data used here is from **StockTwits.com** which is a social media network for traders and investors to share their views about the stock market. When a user posts a message, they tag the relevant stock ticker [$SPY in our case which is for S&P 500 index fund] and have option to tag the message with their sentiment - "bullish" or "bearish"

# #### Read and view data

# In[ ]:


# read data from csv file
data = pd.read_csv('StockTwits_SPY_Sentiment_2017.gz',encoding='utf-8',index_col=0)


# In[ ]:


data.head()


# In[ ]:


# Defining text messages and their labels

messages = data.message.values
labels = data.sentiment.values


# #### Preprocess messages
# 
# Preprocessing the raw text data to normalize for the context. Normalizing for known unique 'entities' that carry similar contextual meaning. 
# 
# Therefore replacing the references to 
# * specific stock ticker ($SPY), 
# * user names, 
# * url links,
# * numbers with special tokenidentifying the entity 
# 
# Converting text into lower case and removing punctuations.               

# In[ ]:


def preprocess_messages(text):
    
    
    # SAVING REGEX PATTERNS
    REGEX_PRICE_SIGN = re.compile(r'\$(?!\d*\.?\d+%)\d*\.?\d+|(?!\d*\.?\d+%)\d*\.?\d+\$')
    REGEX_PRICE_NOSIGN = re.compile(r'(?!\d*\.?\d+%)(?!\d*\.?\d+k)\d*\.?\d+')
    REGEX_TICKER = re.compile('\$[a-zA-Z]+')
    REGEX_USER = re.compile('\@\w+')
    REGEX_LINK = re.compile('https?:\/\/[^\s]+')
    REGEX_HTML_ENTITY = re.compile('\&\w+')
    REGEX_NON_ACSII = re.compile('[^\x00-\x7f]')
    
    #string.punctuation - '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    #string.punctuation.replace('<', '').replace('>', '')
    #--> '!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~'
    #re.escape(string.punctuation.replace('<', ''))
    #--> '\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~'
    
    REGEX_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation.replace('<', '').replace('>', '')))
    REGEX_NUMBER = re.compile(r'[-+]?[0-9]+')
    
    
    # CONVERTING TO LOWERCASE
    text = text.lower()
    
    # REPLACE ST "ENTITITES" WITH A UNIQUE TOKEN
    text = re.sub(REGEX_TICKER, ' <TICKER> ', text)
    text = re.sub(REGEX_USER, ' <USER> ', text)
    text = re.sub(REGEX_LINK, ' <LINK> ', text)
    text = re.sub(REGEX_PRICE_SIGN, ' <PRICE> ', text)
    text = re.sub(REGEX_PRICE_NOSIGN, ' <NUMBER> ', text)
    text = re.sub(REGEX_NUMBER, ' <NUMBER> ', text)
    # REMOVE EXTRANEOUS TEXT DATA
    text = re.sub(REGEX_HTML_ENTITY, "", text)
    text = re.sub(REGEX_NON_ACSII, "", text)
    text = re.sub(REGEX_PUNCTUATION, "", text)
    
    # Tokenizing and removing < and > that are not in special tokens
    words = " ".join(token.replace("<", "").replace(">", "")
                     if token not in ['<TICKER>', '<USER>', '<LINK>', '<PRICE>', '<NUMBER>']
                     else token
                     for token in text.split())

    return words


# In[ ]:


messages = np.array([preprocess_messages(msg) for msg in messages])


# #### Generate Vocab to index mapping
# 
# Encoding words to numbers for the alogrithm to work with inputs by encoding each word to a unique index.

# In[ ]:


vocab = " ".join(messages).split()


# In[ ]:


len(vocab)


# In[ ]:


len(set(vocab))


# In[ ]:


word_idx = {word:idx for idx,word in enumerate(sorted(set(vocab)),1)}
idx_word = {idx:word for word,idx in word_idx.items()}    


# #### Checking messages length

# In[ ]:


message_len = [len(msg) for msg in messages]

print('Minimum length : ',min(message_len))
print('Maximum length : ',max(message_len))
print('Mean length : ',np.mean(message_len))


# In[ ]:


min_idx = [i  for i in range(len(message_len)) if message_len[i]==0]
print("Indexes where message length is 0 :",min_idx)


# In[ ]:


print('messages length: ',len(messages))
print('no of labels: ',len(labels))


# In[ ]:


# dropping zero message length message

messages = np.delete(messages,min_idx)
labels = np.delete(labels,min_idx)


# In[ ]:


print('messages length after removing of zero length messages: ',len(messages))
print('no of labels after removing of zero length messages: ',len(labels))


# #### Encoding Messages and Labels to the indexes

# In[ ]:


def encode_messages(messages,word_idx):
    encoded_msg = [] 
    for msg in messages:
        encoded_msg.append([word_idx[word] for word in msg.split()])
    
    return np.array(encoded_msg)


# In[ ]:


encoded_msg = encode_messages(messages,word_idx)
encoded_msg


# In[ ]:


data.sentiment.nunique()


# In[ ]:


data.sentiment.value_counts()


# In[ ]:


def encode_labels(labels):
    return np.array([0 if label=='bullish' else 1 for label in labels ])


# In[ ]:


encoded_label = encode_labels(labels)
encoded_label


# #### Zero Padding the messages

# In[ ]:


#finding the maximum sentence

len_encoded_msg = [len(i) for i in encoded_msg]
seq_len1 = max(len_encoded_msg)
seq_len1


# In[ ]:


print('Minimum length : ',min(len_encoded_msg))
print('Maximum length : ',max(len_encoded_msg))
print('Mean length : ',np.mean(len_encoded_msg))


# In[ ]:


# plt.hist(len_encoded_msg)
# plt.show()


# In[ ]:


# padding the encoded_messages

padd_msg = np.zeros((len(encoded_msg),seq_len1))

for i,message in enumerate(encoded_msg):
    padd_msg[i,seq_len1-len(message):] = message


# In[ ]:


padd_msg.shape


# #### Train,Test,Validation split

# In[ ]:


# creating x and test split

x, x_test, y, y_test = train_test_split(padd_msg, encoded_label, test_size=0.1, random_state=42)


# In[ ]:


# printing the shapes of the respective sets

print("Shape of x : ",x.shape)
print("Shape of y : ",y.shape)
print("Shape of x_test set : ",x_test.shape)
print("Shape of y_test set : ",y_test.shape)


# In[ ]:


# creating train and validation split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)


# In[ ]:


# printing the shapes of the respective sets

print("Shape of x_train : ",x_train.shape)
print("Shape of y_train : ",y_train.shape)
print("Shape of x_val set : ",x_val.shape)
print("Shape of y_val set : ",y_val.shape)


# ## Building and Training LSTM network

# In[ ]:


def create_placeholders(batch_size,lstm_neurons_li):
    'creating placeholders'
    
    with tf.variable_scope('Placeholders'):
        x_ph = tf.placeholder(tf.int32,[None,None],name ='x_ph')
        y_ph = tf.placeholder(tf.int32,None, name='y_ph')
        keep_prob_ph = tf.placeholder(tf.float32,name='keep_prob_ph')
            
    with tf.variable_scope('LSTM_Placeholders'):        
        init_state_l1_ph = tf.placeholder(tf.float32,[2,batch_size,lstm_neurons_li[0]],name='l1_state')
        init_state_l2_ph = tf.placeholder(tf.float32,[2,batch_size,lstm_neurons_li[1]],name='l2_state')
    
    return x_ph,y_ph,keep_prob_ph,init_state_l1_ph,init_state_l2_ph


# In[ ]:


def forward_propagation(x_ph,vocab_size,embed_size,lstm_neurons_li,keep_prob_ph,batch_size,
                        init_state_l1_ph,init_state_l2_ph):
    
    # creating embedding layer
    with tf.variable_scope('Embedding_layer'):
        embedding = tf.Variable(tf.random_uniform((vocab_size,embed_size),minval=-1,maxval=1))
        embed_layer = tf.nn.embedding_lookup(embedding,x_ph)
    
    # creating LSTM layer
    with tf.variable_scope('LSTM_layer'):
        # creating lstm cells
        lstms = [tf.contrib.rnn.BasicLSTMCell(size,name='lstm_cell') for size in lstm_neurons_li]
        # adding dopout to the cells
        drops = [tf.contrib.rnn.DropoutWrapper(lstm,output_keep_prob = keep_prob_ph) for lstm in lstms]
        # stacking multiple LSTM layers
        cell = tf.contrib.rnn.MultiRNNCell(drops,state_is_tuple=True)
        
        
#         # getting initial state of all zeros
#         init_state = cell.zero_state(batch_size,tf.float32)
#         #init_state = tf.identity(init_state, name="init_state")

        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(init_state_l1_ph[0], init_state_l1_ph[1]),
             tf.contrib.rnn.LSTMStateTuple(init_state_l2_ph[0], init_state_l2_ph[1]) ]
             )
        
        
        lstm_outputs,final_state = tf.nn.dynamic_rnn(cell,embed_layer,initial_state=rnn_tuple_state)
    
    # creating sigmoid fc layer
    with tf.variable_scope('FC_layer'):
        a_output = tf.contrib.layers.fully_connected(lstm_outputs[:,-1],1,activation_fn=tf.sigmoid)
        
        tf.summary.histogram('Predictions',a_output)
    
    return a_output,cell,final_state


# In[ ]:


def compute_cost(a_output,y_ph):
    
    with tf.variable_scope('Loss'):
        cost = tf.losses.mean_squared_error(y_ph,a_output)
        
        tf.summary.scalar("Loss", cost)
    
    return cost


# In[ ]:


# accuracy function

def acc_fn(a_output,y_ph):
    
    with tf.variable_scope('Accuracy'):
        correct_pred = tf.equal(tf.cast(tf.round(a_output),tf.int32),y_ph)
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        
        tf.summary.scalar("Accuracy", accuracy)
    
    return accuracy


# In[ ]:


def get_batches(x,y,batch_size = 100):
    
    n_batches = len(x)//batch_size
    
    # removing left out records
    x,y = x[:n_batches*batch_size],y[:n_batches*batch_size]
    
    for i in range(0,len(x),batch_size):
        yield x[i:i+batch_size],y[i:i+batch_size]


# In[ ]:


# model_training

def model_train(x_train,y_train,x_val,y_val,vocab_size,hparam,
                embed_size=300,lstm_neurons_li=[128,64],
                keep_prob=0.5,learning_rate=1e-1,epochs=50,batch_size=256):
    
	
    print(str(hparam).center(50,'-'))
    
    # reset default graph
    tf.reset_default_graph()
    
    # create placeholder
    x_ph,y_ph,keep_prob_ph,init_state_l1_ph,init_state_l2_ph = create_placeholders(batch_size,lstm_neurons_li)
    
    # forward propogation
    a_output,cell,final_state = forward_propagation(x_ph,vocab_size,
                                                    embed_size,
                                                    lstm_neurons_li,
                                                    keep_prob_ph,
                                                    batch_size,
                                                    init_state_l1_ph,
                                                    init_state_l2_ph)    
    
    # cost calculation
    cost = compute_cost(a_output,y_ph)
    
    # optimizers calculation
    hparam_li = hparam.split('_')
    
    if hparam_li[-1] == 'AdadeltaOptimizer':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
        
    if hparam_li[-1] == 'AdamOptimizer':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
         
    
    # accuracy definition
    accuracy = acc_fn(a_output,y_ph)
    
    saver = tf.train.Saver()
    
    print("Adding all trainable tensors to tensorboard visualisation".center(50,'-'))
    
    #print()
    #print("LSTM WEIGHTS")
    #print()

    #[print(n.name)for n in tf.trainable_variables('LSTM_layer')]
    
    [tf.summary.histogram(n.name, n)for n in tf.trainable_variables('LSTM_layer')]
    
    #print()
    #print()
    #print("FC WIEGHTS")
    #print()
    
    #[print(n.name)for n in tf.trainable_variables('FC_layer')]
    
    [tf.summary.histogram(n.name, n)for n in tf.trainable_variables('FC_layer')]
    
    print()
    print("TRAINING STARTS".center(50,'-'))
    
    
    summ = tf.summary.merge_all()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        n_batches = len(x_train)//batch_size
        
        writer_train = tf.summary.FileWriter('./tb/'+hparam+'/train/',sess.graph)
        writer_val = tf.summary.FileWriter('./tb/'+hparam+'/val/')
        
        global_step = 0
        
        for epoch in range(epochs):
            startime = datetime.now()
            
            state_l1 = np.zeros((2,batch_size,lstm_neurons_li[0]))
            state_l2 = np.zeros((2,batch_size,lstm_neurons_li[1]))
            
            train_acc = []
            for step,(x,y) in enumerate(get_batches(x_train,y_train,batch_size),1):
                
                global_step+=1
                
                print('.',end=' ')
                
                feed = {x_ph:x,
                        y_ph:y.reshape(-1,1),
                        keep_prob_ph:keep_prob,
                        init_state_l1_ph:state_l1,
                        init_state_l2_ph:state_l2}
                
                loss_,state,_,batch_acc, s = sess.run([cost,final_state,optimizer,accuracy,summ],feed_dict=feed)
                
                state_l1 = state[0]
                state_l2 = state[1]
                
                writer_train.add_summary(s,global_step) # writing summary real time
                
                train_acc.append(batch_acc)
                
                
                
                if (step) % 17 == 0:
                    
                    val_acc = []
                    #val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
                    
                    for xx, yy in get_batches(x_val, y_val, batch_size):
                        feed_val = {x_ph: xx,
                                y_ph: yy.reshape(-1,1),
                                keep_prob_ph: 1,
                                init_state_l1_ph:state_l1,
                                init_state_l2_ph:state_l2}  
                            #initial_state: val_state}

                        val_batch_acc,s = sess.run([accuracy, summ], feed_dict=feed_val)                                                                
                        val_acc.append(val_batch_acc)
                        
                    writer_val.add_summary(s,global_step) # writing summary real time
                    
                # after the last batch is used for training i.e. after every epoch of training, evaluating result
                if (step)%n_batches == 0:
                    
                    val_acc = []            
                    
                    for xx,yy in get_batches(x_val,y_val,batch_size):
                        feed_val = {x_ph:xx,
                                    y_ph:yy.reshape(-1,1),
                                    keep_prob_ph:1,
                                    init_state_l1_ph:state_l1,
                                    init_state_l2_ph:state_l2}                               
                        
                        val_batch_acc = sess.run([accuracy],feed_dict=feed_val)
                        val_acc.append(val_batch_acc)
                    
                    stoptime = datetime.now()
                    print()
                    print("Epoch: {}/{}...".format(epoch+1, epochs),
                          "Batch: {}/{}...".format(step, n_batches),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Accuracy: {:.3f}".format(np.mean(val_acc)),
                          "Epoch time: {}".format(str(stoptime-startime)))
            
            #saver.save(sess,'./model_save/sentiment.ckpt',global_step = epoch+1)
            
            


# ## Hyper parameter search

# In[ ]:


vocab_size = len(word_idx)+1

#hyperparameter search
learning_rate_li = [0.01,0.1]
embedding_size_li = [300,650]
batch_size_li = [128,256]
optimizer_li = ['AdadeltaOptimizer','AdamOptimizer']


for lr in learning_rate_li:
    for embed_size in embedding_size_li:
        for batch_size in batch_size_li:
            for opti in optimizer_li:                
                    
                    
                    hparam = "lr"+"_"+str(lr)+"_"+"embed"+"_"+str(embed_size)+"_"+"batch"+"_"+str(batch_size)+"_"+str(opti)

                    model_train(x_train,y_train,x_val,y_val,
                                vocab_size,
                                hparam,
                                embed_size=embed_size,
                                lstm_neurons_li=[128,64],
                                keep_prob=0.5,
                                learning_rate=lr,
                                epochs=50,
                                batch_size=batch_size)


