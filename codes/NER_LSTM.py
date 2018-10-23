
# coding: utf-8

# # Recognize named entities on Twitter with LSTMs
# 
# A recurrent neural network is used to solve Named Entity Recognition (NER) problem (recognizing named entities from Twitter). NER is a common task in natural language processing systems. It serves for extraction such entities from the text as persons, organizations, locations, etc.

# In[1]:


import sys
sys.path.append("..")


# ### Load the Twitter Named Entity Recognition corpus
# 
# We will work with a corpus, which contains twits with NE tags. Every line of a file contains a pair of a token (word/punctuation symbol) and a tag, separated by a whitespace. Different tweets are separated by an empty line.
# 
# The function *read_data* reads a corpus from the *file_path* and returns two lists: one with tokens and one with the corresponding tags. This function also replaces a user's nickname to `<USR>` token and any URL to `<URL>` token (thinking that a URL and a nickname are just strings which start with *http://* or *https://* in case of URLs and a *@* symbol for nicknames).

# In[2]:


def read_data(file_path):
    tokens = []
    tags = []
    
    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            # Replace all urls with <URL> token
            # Replace all users with <USR> token

            
            if(token.startswith('@')):
                token = '<USR>'
            elif(token.startswith('http://') or token.startswith('https://')):
                token = '<URL>'
            tweet_tokens.append(token)
            tweet_tags.append(tag)
            
    return tokens, tags


# And now, load three separate parts of the dataset:
#  - *train* data for training the model;
#  - *validation* data for evaluation and hyperparameters tuning;
#  - *test* data for final evaluation of the model.

# In[3]:


train_tokens, train_tags = read_data('data/train.txt')
validation_tokens, validation_tags = read_data('data/validation.txt')
test_tokens, test_tags = read_data('data/test.txt')


# You should always understand what kind of data you deal with. For this purpose, you can print the data running the following cell:

# In[4]:


for i in range(3):
    for token, tag in zip(train_tokens[i], train_tags[i]):
        print('%s\t%s' % (token, tag))
    print()


# ### Prepare dictionaries
# 
# To train a neural network, we will use two mappings: 
# - {token}$\to${token id}: address the row in embeddings matrix for the current token;
# - {tag}$\to${tag id}: one-hot ground truth probability distribution vectors for computing the loss at the output of the network.
# 
# Now you need to implement the function *build_dict* which will return {token or tag}$\to${index} and vice versa.

# In[5]:


from collections import defaultdict


# In[6]:


def build_dict(tokens_or_tags, special_tokens):
    """
        tokens_or_tags: a list of lists of tokens or tags
        special_tokens: some special tokens
    """
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []
    
    # Create mappings from tokens to indices and vice versa
    # Add special tokens to dictionaries
    # The first special token must have index 0
    
    
    idx = 0
    for token in special_tokens:
        idx2tok.append(token)
        tok2idx[token] = idx
        idx += 1

    for token_list in tokens_or_tags:
        for token in token_list:
            if token not in tok2idx:
                idx2tok.append(token)
                tok2idx[token] = idx
                idx += 1
    
    return tok2idx, idx2tok


# After implementing the function *build_dict* you can make dictionaries for tokens and tags. Special tokens in our case will be:
#  - `<UNK>` token for out of vocabulary tokens;
#  - `<PAD>` token for padding sentence to the same length when we create batches of sentences.

# In[7]:


special_tokens = ['<UNK>', '<PAD>']
special_tags = ['O']

# Create dictionaries 
token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
tag2idx, idx2tag = build_dict(train_tags, special_tags)


# The next additional functions will help you to create the mapping between tokens and ids for a sentence. 

# In[8]:


def words2idxs(tokens_list):
    return [token2idx[word] for word in tokens_list]

def tags2idxs(tags_list):
    return [tag2idx[tag] for tag in tags_list]

def idxs2words(idxs):
    return [idx2token[idx] for idx in idxs]

def idxs2tags(idxs):
    return [idx2tag[idx] for idx in idxs]


# ### Generate batches
# 
# Neural Networks are usually trained with batches. It means that weight updates of the network are based on several sequences at every single time. The tricky part is that all sequences within a batch need to have the same length. So we will pad them with a special `<PAD>` token. It is also a good practice to provide RNN with sequence lengths, so it can skip computations for padding parts. We provide the batching function *batches_generator* readily available for you to save time. 

# In[9]:


def batches_generator(batch_size, tokens, tags,
                      shuffle=True, allow_smaller_last_batch=True):
    """Generates padded batches of tokens and tags."""
    
    n_samples = len(tokens)
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        order = np.arange(n_samples)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        max_len_token = 0
        for idx in order[batch_start: batch_end]:
            x_list.append(words2idxs(tokens[idx]))
            y_list.append(tags2idxs(tags[idx]))
            max_len_token = max(max_len_token, len(tags[idx]))
            
        # Fill in the data into numpy nd-arrays filled with padding indices.
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * token2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * tag2idx['O']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths


# ## Build a recurrent neural network
# 
# This is the most important part of the assignment. Here we will specify the network architecture based on TensorFlow building blocks. It's fun and easy as a lego constructor! We will create an LSTM network which will produce probability distribution over tags for each token in a sentence. To take into account both right and left contexts of the token, we will use Bi-Directional LSTM (Bi-LSTM). Dense layer will be used on top to perform tag classification.  

# In[10]:


import tensorflow as tf
import numpy as np


# In[11]:


class BiLSTMModel():
    pass


# First, we need to create [placeholders](https://www.tensorflow.org/versions/master/api_docs/python/tf/placeholder) to specify what data we are going to feed into the network during the execution time.  For this task we will need the following placeholders:
#  - *input_batch* — sequences of words (the shape equals to [batch_size, sequence_len]);
#  - *ground_truth_tags* — sequences of tags (the shape equals to [batch_size, sequence_len]);
#  - *lengths* — lengths of not padded sequences (the shape equals to [batch_size]);
#  - *dropout_ph* — dropout keep probability; this placeholder has a predefined value 1;
#  - *learning_rate_ph* — learning rate; we need this placeholder because we want to change the value during training.
# 
# It could be noticed that we use *None* in the shapes in the declaration, which means that data of any size can be feeded. 
# 
# You need to complete the function *declare_placeholders*.

# In[12]:


def declare_placeholders(self):
    """Specifies placeholders for the model."""

    # Placeholders for input and ground truth output.
    self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch') 
    self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')
  
    # Placeholder for lengths of the sequences.
    self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths')
    
    # Placeholder for a dropout keep probability. If we don't feed
    # a value for this placeholder, it will be equal to 1.0.
    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
    
    # Placeholder for a learning rate (tf.float32).
    self.learning_rate_ph = tf.placeholder_with_default(1e4, shape=[]) 


# In[13]:


BiLSTMModel.__declare_placeholders = classmethod(declare_placeholders)


# Now, let us specify the layers of the neural network. First, we need to perform some preparatory steps: 
#  
# - Create embeddings matrix with [tf.Variable](https://www.tensorflow.org/api_docs/python/tf/Variable). Specify its name (*embeddings_matrix*), type  (*tf.float32*), and initialize with random values.
# - Create forward and backward LSTM cells. TensorFlow provides a number of [RNN cells](https://www.tensorflow.org/api_guides/python/contrib.rnn#Core_RNN_Cells_for_use_with_TensorFlow_s_core_RNN_methods) ready for you. We suggest that you use *BasicLSTMCell*, but you can also experiment with other types, e.g. GRU cells. [This](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) blogpost could be interesting if you want to learn more about the differences.
# - Wrap your cells with [DropoutWrapper](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper). Dropout is an important regularization technique for neural networks. Specify all keep probabilities using the dropout placeholder that we created before.
#  
# After that, you can build the computation graph that transforms an input_batch:
# 
# - [Look up](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup) embeddings for an *input_batch* in the prepared *embedding_matrix*.
# - Pass the embeddings through [Bidirectional Dynamic RNN](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn) with the specified forward and backward cells. Use the lengths placeholder here to avoid computations for padding tokens inside the RNN.
# - Create a dense layer on top. Its output will be used directly in loss function.  
#  
# Fill in the code below. In case you need to debug something, the easiest way is to check that tensor shapes of each step match the expected ones. 
#  

# In[14]:


def build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
    """Specifies bi-LSTM architecture and computes logits for inputs."""
    
    # Create embedding variable (tf.Variable) with dtype tf.float32
    initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
    embedding_matrix_variable = tf.Variable(initial_embedding_matrix, name='embeddings_matrix', dtype=tf.float32)
    
    # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units 
    # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.
    forward_cell = tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn, forget_bias=3.0),
        input_keep_prob=self.dropout_ph,
        output_keep_prob=self.dropout_ph,
        state_keep_prob=self.dropout_ph
    )
    backward_cell = tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn, forget_bias=3.0),
        input_keep_prob=self.dropout_ph,
        output_keep_prob=self.dropout_ph,
        state_keep_prob=self.dropout_ph
    )

    # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).
    # Shape: [batch_size, sequence_len, embedding_dim].
    embeddings = tf.nn.embedding_lookup(embedding_matrix_variable, self.input_batch)
    
    # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
    # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn]. 
    # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
    (rnn_output_fw, rnn_output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw= forward_cell, cell_bw= backward_cell,
        dtype=tf.float32,
        inputs=embeddings,
        sequence_length=self.lengths
    )
    rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

    # Dense layer on top.
    # Shape: [batch_size, sequence_len, n_tags].   
    self.logits = tf.layers.dense(rnn_output, n_tags, activation=None)


# In[15]:


BiLSTMModel.__build_layers = classmethod(build_layers)


# To compute the actual predictions of the neural network, you need to apply [softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) to the last layer and find the most probable tags with [argmax](https://www.tensorflow.org/api_docs/python/tf/argmax).

# In[16]:


def compute_predictions(self):
    """Transforms logits to probabilities and finds the most probable tags."""
    
    # Create softmax (tf.nn.softmax) function
    softmax_output = tf.nn.softmax(self.logits)
    
    # Use argmax (tf.argmax) to get the most probable tags
    # Don't forget to set axis=-1
    # otherwise argmax will be calculated in a wrong way
    self.predictions = tf.argmax(softmax_output, axis=-1)


# In[17]:


BiLSTMModel.__compute_predictions = classmethod(compute_predictions)


# During training we do not need predictions of the network, but we need a loss function. We will use [cross-entropy loss](http://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy), efficiently implemented in TF as 
# [cross entropy with logits](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits). Note that it should be applied to logits of the model (not to softmax probabilities!). Also note,  that we do not want to take into account loss terms coming from `<PAD>` tokens. So we need to mask them out, before computing [mean](https://www.tensorflow.org/api_docs/python/tf/reduce_mean).

# In[18]:


def compute_loss(self, n_tags, PAD_index):
    """Computes masked cross-entopy loss with logits."""
    
    # Create cross entropy function function (tf.nn.softmax_cross_entropy_with_logits)
    ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, n_tags)
    loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_tags_one_hot, logits=self.logits)
    
    # Create loss function which doesn't operate with <PAD> tokens (tf.reduce_mean)
    mask = tf.cast(tf.not_equal(loss_tensor, PAD_index), tf.float32)
    self.loss =  tf.reduce_mean(tf.reduce_sum(tf.multiply(loss_tensor, mask), axis=-1) / tf.reduce_sum(mask, axis=-1))


# In[19]:


BiLSTMModel.__compute_loss = classmethod(compute_loss)


# The last thing to specify is how we want to optimize the loss. 
# We suggest that you use [Adam](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) optimizer with a learning rate from the corresponding placeholder. 
# You will also need to apply [clipping](https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_clipping) to eliminate exploding gradients. It can be easily done with [clip_by_norm](https://www.tensorflow.org/api_docs/python/tf/clip_by_norm) function. 

# In[20]:


def perform_optimization(self):
    """Specifies the optimizer and train_op for the model."""
    
    # Create an optimizer (tf.train.AdamOptimizer)
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
    self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
    
    # Gradient clipping (tf.clip_by_norm) for self.grads_and_vars
    # Pay attention that you need to apply this operation only for gradients 
    # because self.grads_and_vars contains also variables.
    # list comprehension might be useful in this case.
    clip_norm = tf.cast(1.0, tf.float32)
    self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in self.grads_and_vars]
    
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)


# In[21]:


BiLSTMModel.__perform_optimization = classmethod(perform_optimization)


# You have specified all the parts of your network. You may have noticed, that we didn't deal with any real data yet, so what you have written is just recipes on how the network should function.
# Now we will put them to the constructor of our Bi-LSTM class to use it in the next section. 

# In[22]:


def init_model(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
    self.__declare_placeholders()
    self.__build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
    self.__compute_predictions()
    self.__compute_loss(n_tags, PAD_index)
    self.__perform_optimization()


# In[23]:


BiLSTMModel.__init__ = classmethod(init_model)


# ## Train the network and predict tags

# [Session.run](https://www.tensorflow.org/api_docs/python/tf/Session#run) is a point which initiates computations in the graph that we have defined. To train the network, we need to compute *self.train_op*, which was declared in *perform_optimization*. To predict tags, we just need to compute *self.predictions*. Anyway, we need to feed actual data through the placeholders that we defined before. 

# In[24]:


def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability):
    feed_dict = {self.input_batch: x_batch,
                 self.ground_truth_tags: y_batch,
                 self.learning_rate_ph: learning_rate,
                 self.dropout_ph: dropout_keep_probability,
                 self.lengths: lengths}
    
    session.run(self.train_op, feed_dict=feed_dict)


# In[25]:


BiLSTMModel.train_on_batch = classmethod(train_on_batch)


# Implement the function *predict_for_batch* by initializing *feed_dict* with input *x_batch* and *lengths* and running the *session* for *self.predictions*.

# In[26]:


def predict_for_batch(self, session, x_batch, lengths):
    
    predictions = session.run(self.predictions, feed_dict={self.input_batch:x_batch, self.lengths:lengths})
    return predictions


# In[27]:


BiLSTMModel.predict_for_batch = classmethod(predict_for_batch)


# We finished with necessary methods of our BiLSTMModel model and almost ready to start experimenting.
# 
# ### Evaluation 
# To simplify the evaluation process we provide two functions for you:
#  - *predict_tags*: uses a model to get predictions and transforms indices to tokens and tags;
#  - *eval_conll*: calculates precision, recall and F1 for the results.

# In[28]:


from evaluation import precision_recall_f1


# In[29]:


def predict_tags(model, session, token_idxs_batch, lengths):
    """Performs predictions and transforms indices to tokens and tags."""
    
    tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)
    
    tags_batch, tokens_batch = [], []
    for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
        tags, tokens = [], []
        for tag_idx, token_idx in zip(tag_idxs, token_idxs):
            tags.append(idx2tag[tag_idx])
            tokens.append(idx2token[token_idx])
        tags_batch.append(tags)
        tokens_batch.append(tokens)
    return tags_batch, tokens_batch
    
    
def eval_conll(model, session, tokens, tags, short_report=True):
    """Computes NER quality measures using CONLL shared task script."""
    
    y_true, y_pred = [], []
    for x_batch, y_batch, lengths in batches_generator(1, tokens, tags):
        tags_batch, tokens_batch = predict_tags(model, session, x_batch, lengths)
        if len(x_batch[0]) != len(tags_batch[0]):
            raise Exception("Incorrect length of prediction for the input, "
                            "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
        predicted_tags = []
        ground_truth_tags = []
        for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]): 
            if token != '<PAD>':
                ground_truth_tags.append(idx2tag[gt_tag_idx])
                predicted_tags.append(pred_tag)

        # We extend every prediction and ground truth sequence with 'O' tag
        # to indicate a possible end of entity.
        y_true.extend(ground_truth_tags + ['O'])
        y_pred.extend(predicted_tags + ['O'])
        
    results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
    return results


# ## Run your experiment

# Create *BiLSTMModel* model with the following parameters:
#  - *vocabulary_size* — number of tokens;
#  - *n_tags* — number of tags;
#  - *embedding_dim* — dimension of embeddings, recommended value: 200;
#  - *n_hidden_rnn* — size of hidden layers for RNN, recommended value: 200;
#  - *PAD_index* — an index of the padding token (`<PAD>`).
# 
# Set hyperparameters. You might want to start with the following recommended values:
# - *batch_size*: 32;
# - 4 epochs;
# - starting value of *learning_rate*: 0.005
# - *learning_rate_decay*: a square root of 2;
# - *dropout_keep_probability*: try several values: 0.1, 0.5, 0.9.
# 
# However, feel free to conduct more experiments to tune hyperparameters and earn extra points for the assignment.

# In[30]:


len(idx2token)


# In[31]:


len(idx2tag)


# In[32]:


tf.reset_default_graph()

model = BiLSTMModel(20505, 21, 200, 200, token2idx['<PAD>'])

batch_size = 32
n_epochs = 4
learning_rate = 0.005
learning_rate_decay = 1.414
dropout_keep_probability = 0.5


# If you got an error *"Tensor conversion requested dtype float64 for Tensor with dtype float32"* in this point, check if there are variables without dtype initialised. Set the value of dtype equals to *tf.float32* for such variables.

# Finally, run the training!

# In[33]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start training... \n')
for epoch in range(n_epochs):
    # For each epoch evaluate the model on train and validation data
    print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
    print('Train data evaluation:')
    eval_conll(model, sess, train_tokens, train_tags, short_report=True)
    print('Validation data evaluation:')
    eval_conll(model, sess, validation_tokens, validation_tags, short_report=True)
    
    # Train the model
    for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags):
        model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)
        
    # Decaying the learning rate
    learning_rate = learning_rate / learning_rate_decay
    
print('...training finished.')


# Full quality reports for the final model on train, validation, and test sets. To give you a hint whether you have implemented everything correctly, you might expect F-score about 40% on the validation set.
# 

# In[34]:


print('-' * 20 + ' Train set quality: ' + '-' * 20)
train_results = eval_conll(model, sess, train_tokens, train_tags, short_report=False)

print('-' * 20 + ' Validation set quality: ' + '-' * 20)
validation_results = eval_conll(model, sess, validation_tokens, validation_tags, short_report=False)

print('-' * 20 + ' Test set quality: ' + '-' * 20)
test_results = eval_conll(model, sess, test_tokens, test_tags, short_report=False)


# ### Conclusions
# 
# Nowadays, Bi-LSTM is one of the state of the art approaches for solving NER problem and it outperforms other classical methods. Despite the fact that we used small training corpora (in comparison with usual sizes of corpora in Deep Learning), our results are quite good. In addition, in this task there are many possible named entities and for some of them we have only several dozens of trainig examples, which is definately small. However, the implemented model outperforms classical CRFs for this task. Even better results could be obtained by some combinations of several types of methods, e.g. see [this](https://arxiv.org/abs/1603.01354) paper if you are interested.
