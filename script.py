############################################################################################################
#####                                                                                                  #####
#####                                             EMOJIFIER                                            #####
#####                                      Created on: 2025-02-27                                      #####
#####                                       Updated on 2025-03-11                                      #####
#####                                                                                                  #####
############################################################################################################

############################################################################################################
#####                                             PACKAGES                                             #####
############################################################################################################

# Clear the whole environment
globals().clear()

# Load the libraries
import numpy as np
import emoji
import matplotlib.pyplot as plt
import numpy as np
import tensorflow
np.random.seed(0)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding, SimpleRNN
np.random.seed(1)

# Set up the right directory
import os 
os.chdir('C:/Users/Admin/Documents/Python Projects/emojifier')
from general_functions import *


############################################################################################################
#####                                       LOAD EMOJISET DATASET                                      #####
############################################################################################################

# Load the X/Y train/test datasets
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/test_emoji.csv')

# One-hot encod Y into 5 classes (since we have 5 possible emojis) to be able to apply to softmax later
Y_train_oh = convert_to_one_hot(Y_train, 5)
Y_test_oh = convert_to_one_hot(Y_test, 5)

# Print the shape of the X/Y train/test sets
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(Y_train_oh.shape)
print(Y_test_oh.shape)

# Keep the length of the sentence with the longest length (biggest number of words)
max_len = len(max(X_train, key=lambda x: len(x.split())).split())

# Take 10 examples, print the X sentence and map the Y label (int) to the right emoji
for idx in range(10):
    print(X_train[idx], code_to_emoji(Y_train[idx]))


############################################################################################################
#####                              PREPROCESS DATA FOR EMOJIFIER-NN MODEL                              #####
############################################################################################################

# Print an example of a sentence, its Y value and the correspondign emoji
idx = 21
print(f"Sentence '{X_train[idx]}' has label index {Y_train[idx]}, which is emoji {code_to_emoji(Y_train[idx])}", )

# Print an example of a Y and its encoded Y
print(f"Label index {Y_train[idx]} in one-hot encoding format is {Y_train_oh[idx]}")

# Load the GloVe 50 dimensional embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vectors('data/glove.6B.50d.txt')

# word_to_index: dictionary mapping from a word to its index in the vocabulary (400k words)
type(word_to_index) ; len(word_to_index) ; word_to_index['mountain']
# index_to_word: dictionary mapping from an index to its corresponding word in the vocabulary
type(index_to_word) ; len(index_to_word) ; index_to_word[250836]
# word_to_vec_map: dictionary mapping a word to its GloVe vector representation
word_to_vec_map['mountain']

# Create the function to determine the average word vector of word embedding for the sentence
def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its vector representation
    
    Returns:
    avg_vector -- average vector encoding information about the sentence, numpy-array of shape (n_h,) wit n_h the dimension of the word embedding 
    """
    # Get a valid word contained in the word_to_vec_map
    any_word = next(iter(word_to_vec_map.keys()))

    # Split sentence into list of lower case words
    words = sentence.lower().split()

    # Initialize the average word vector, should have the same shape as any word vector
    avg_vector = np.zeros((word_to_vec_map[any_word].shape))

    # Initialize count to 0
    count = 0
    
    # Average the word vectors. Loop over the words in the "words" list (of the sentence)
    for w in words:
        # Check if that word exists in word_to_vec_map
        if w in word_to_vec_map:
            avg_vector += word_to_vec_map[w]
            # In that case, increment the count
            count +=1
    
    # Get the average (only if count > 0)
    if count > 0:
        avg_vector = avg_vector/count

    return avg_vector


############################################################################################################
#####                               IMPLEMENT & TRAIN EMOJIFIER-NN MODEL                               #####
############################################################################################################

# Create the function Emojify_NN_model to implement the basic NN model (1st model)
def Emojify_NN_model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
    """
    NN model to predict the emoji from a sentence
    
    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m,)
    Y -- target data, numpy array of int, of shape (m,number_classes)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm (batch size of 1)
    num_iterations -- number of iterations
    
    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the hidden layer, of shape (n_y, n_h)
    b -- bias matrix of the hidden layer, of shape (n_y,)
    """
    # Get a valid word contained in the word_to_vec_map 
    any_word = next(iter(word_to_vec_map.keys()))
        
    # Define the number of training examples
    m = Y.shape[0]
    # Define the number of classes 
    n_y = len(np.unique(Y))
    # Define the dimensions of the GloVe vectors (h for hidden layer)
    n_h = word_to_vec_map[any_word].shape[0]
    
    # Initialize weight matrix using Xavier initialization (follows a normal distribution with mean 0 and variance 2/(n_h + n_y) so sd=sqrt())
    W = np.random.randn(n_y, n_h) * np.sqrt(2. / (n_h + n_y))
    
    # Initialize bias matrix to 0 (no need for specific initialization)
    b = np.zeros((n_y,))
    
    # Encod Y (with n_y classes)
    Y_oh = convert_to_one_hot(Y, n_y) 
    
    # Optimization loop (loop over the number of iterations, for each iteration we go through all the X set)
    for t in range(num_iterations): 
        
        # Initialize loss, gradient of the loss function with respect to the weights and to the bias to 0
        loss = 0
        dW = 0
        db = 0
        
        # Loop over the training examples (we use minni-batch of size 1, that's why we are in a SGD case)
        for i in range(m):
            
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer (using forward propagation formula)
            z = np.dot(W,avg)+b
            a = softmax(z)

            # Add the loss using the i'th training label's one hot representation and "a" (the output of the softmax, being the probabilities of each class)
            # Cross-Entropy Loss
            loss += -np.dot(Y_oh[i],np.log(a))
            
            # Compute gradients (using backward propagation formula)
            dz = a - Y_oh[i]
            dW += np.dot(dz.reshape(n_y,1), avg.reshape(1, n_h))
            db += dz

            # Update parameters with Stochastic Gradient Descent (batch size of 1)
            W = W - learning_rate * dW
            b = b - learning_rate * db
        
        # Average/normalize the loss across all training examples
        loss /= m

        # Print the loss every 100 epochs
        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- loss = " + str(loss))
            pred = predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


# Train the model and learn the softmax parameters (W, b)
# Fix a seed to have reproducible results
np.random.seed(1)
pred, W, b = Emojify_NN_model(X_train, Y_train, word_to_vec_map)


############################################################################################################
#####                                    EVALUATE EMOJIFIER-NN MODEL                                   #####
############################################################################################################

# Get the train and test accuracy
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)
# The training set was pretty small, so test results are pretty good

# Test with other examples
X_my_sentences = np.array(["i treasure you", "i love you", "funny lol", "lets play with a ball", "food is ready", "today is not good"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])
pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)
# The model has not seen the word "treasure" but because its embedding is close to "love", the model succeeds in getting the right emoji
# This model ignores word ordering (for example "today is not good" is not understood by the model)

# Print the confusion matrix (can help to understand which classes are more difficult for the model)
plot_confusion_matrix(Y_test, pred_test)


############################################################################################################
#####                         EMBEDDING LAYER FOR EMOJIFIER RNN & LSTM MODELS                          #####
############################################################################################################

# Get the index of each word in the sentence (1 sentence example)
for idx, val in enumerate(["I", "like", "learning"]):
    print(idx, val)

# Create the function to transform sentences to indexes (1 index per word of the sentence)
def sentences_to_indexes(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indexes corresponding to words in the sentences.
    The output shape should be such that it can be given to Embedding() layer. 
    
    Arguments:
    X -- array of sentences (strings), of shape (m,)
    word_to_index -- a dictionary containing each word (key) mapped to its index (value)
    max_len -- maximum number of words in a sentence
    
    Returns:
    X_indexes -- array of indexes corresponding to words in the sentences from X, of shape (m, max_len)
    """

    # Get the number of training examples
    m = X.shape[0]
    
    # Initialize X_indexes as a numpy matrix of zeros and the correct shape (zero-pad so that the length of each sentence is the length of the longest sentence)
    X_indexes = np.zeros((m, max_len))

    # Loop over training examples
    for i in range(m):
        
        # Convert the ith training sentence to lower case and split it into words (to get a list of words)
        sentence_words = X[i].lower().split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            # If w exists in the word_to_index dictionary
            if w in word_to_index:
                # Set the (i,j)th entry of X_indexes to the index of the word
                X_indexes[i, j] = word_to_index[w]
                # Increment j to j + 1 (we don't consider words that are not in the vocabulary)
                j = j+1
                
    return X_indexes


# Get an example to understand sentences_to_indexes() function
X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indexes = sentences_to_indexes(X1, word_to_index, max_len=5)
print(X1_indexes)


# Transform X_train and X_test to an array of indexes (each list has max_len character, with indexes of the word in the sentence and 0 to fill the sentence)
X_train_indexes = sentences_to_indexes(X_train, word_to_index, max_len)
X_test_indexes = sentences_to_indexes(X_test, word_to_index, max_len = max_len)


# Create the function to build the embedding layer (using pre-trained word vectors: GloVe)
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation
    word_to_index -- dictionary mapping from words to their indexes in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    # Add 1 to the vocabulary size to fit "unknown" words
    vocab_size = len(word_to_index) + 1

    # Take any word from the dictionnary
    any_word = next(iter(word_to_vec_map.keys()))

    # Get the dimension of the GloVe word vectors
    emb_dim = word_to_vec_map[any_word].shape[0]
      
    # Initialize the embedding matrix as a numpy array of zeros. Each row will store the vector representation of one word
    emb_matrix = np.zeros((vocab_size,emb_dim))
    
    # Set each row "idx" of the embedding matrix to be the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct input and output size and make it non-trainable.
    # trainable = True allows the optimization algorithm to modify the values of the word embeddings (in this case we don't want)
    # our dataset is small so it's not worth trying to train a large pre-trained set of embeddings
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer 
    # None means that the batch size is not fixed 
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# Get a first look at the embedding_layer
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][1] =", embedding_layer.get_weights()[0][1][1])
print("Input_dim: ", embedding_layer.input_dim)
print("Output_dim: ",embedding_layer.output_dim)
print(len(embedding_layer.get_weights()))
print(embedding_layer.get_weights()[0].shape)


############################################################################################################
#####                              IMPLEMENT & TRAIN EMOJIFIER-RNN MODEL                               #####
############################################################################################################

# Create the function to implement the simple RNN model (2nd model)
def Emojify_RNN(input_shape, word_to_vec_map, word_to_index):
    """
    Creates the Emojify-RNN model using simple RNN (2nd model)
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indexes in the vocabulary

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indexes as the input, it should be of shape input_shape and dtype 'int32' (as it contains indexes, which are integers)
    sentence_indexes = Input(input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indexes through your embedding layer
    embeddings = embedding_layer(sentence_indexes)   
    
    # Propagate the embeddings through an RNN layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences
    X = SimpleRNN(units=128, return_sequences=True)(embeddings)
    
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X) 

    # Propagate X through another RNN layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences
    X = SimpleRNN(units=128, return_sequences=False)(X)

    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)

    # Propagate X through a Dense layer with 5 units (since 5 classes to predict)
    X = Dense(units=5)(X)

    # Add a softmax activation
    X = Activation(activation='softmax')(X)
    
    # Create Model instance which converts sentence_indexes into X
    model = Model(inputs=sentence_indexes, outputs=X)
        
    return model


# Create the model
Emojify_RNN_model = Emojify_RNN((max_len,), word_to_vec_map, word_to_index)

# Get the summary of the model
Emojify_RNN_model.summary()

# Compile the model
Emojify_RNN_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
Emojify_RNN_model.fit(X_train_indexes, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)


############################################################################################################
#####                                   EVALUATE EMOJIFIER-RNN MODEL                                   #####
############################################################################################################

# Evaluate on the test set
pred = Emojify_RNN_model.predict(X_test_indexes)
loss, acc = Emojify_RNN_model.evaluate(X_test_indexes, Y_test_oh)
print("Test accuracy = ", acc)

# Check the mislabelled examples
for i in range(len(X_test)):
    y_pred = np.argmax(pred[i])
    if(y_pred != Y_test[i]):
        print('Sentence: ' + X_test[i] + 'Expected emoji:' + code_to_emoji(Y_test[i]) + ' vs Predicted emoji:'+ code_to_emoji(y_pred))

# Check with our own sentence (make sure all the words are in the Glove embeddings)
new_sentence = np.array(['not feeling happy'])
new_sentence_indexes = sentences_to_indexes(new_sentence, word_to_index, max_len)
print(new_sentence[0] +' '+  code_to_emoji(np.argmax(Emojify_RNN_model.predict(new_sentence_indexes))))


############################################################################################################
#####                              IMPLEMENT & TRAIN EMOJIFIER-LSTM MODEL                              #####
############################################################################################################

# Create the function to implement the LSTM model (3rd model)
def Emojify_LSTM(input_shape, word_to_vec_map, word_to_index):
    """
    Creates the Emojify-LSTM model (3rd model)
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indexes in the vocabulary

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indexes as the input, it should be of shape input_shape and dtype 'int32' (as it contains indexes, which are integers)
    sentence_indexes = Input(input_shape,dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # Propagate sentence_indexes through your embedding layer (sentence_indexes contains indexes of each word, embedding_layer contains word vectors of each word/index)
    embeddings = embedding_layer(sentence_indexes)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences
    X = LSTM(units=128,return_sequences=True)(embeddings)
    
    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X) 

    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences
    X = LSTM(units=128,return_sequences=False)(X)

    # Add dropout with a probability of 0.5
    X = Dropout(rate=0.5)(X)

    # Propagate X through a Dense layer with 5 units (since 5 classes to predict)
    X = Dense(units=5)(X)

    # Add a softmax activation
    X = Activation(activation='softmax')(X)
    
    # Create Model instance which converts sentence_indexes into X
    model = Model(inputs=sentence_indexes, outputs=X)
        
    return model


# Create the model
Emojify_LSTM_model = Emojify_LSTM((max_len,), word_to_vec_map, word_to_index)

# Get the summary of the model
Emojify_LSTM_model.summary()
# Input layer as an output of 10, meaning that max_len=10 was chosen (because all sentences in the dataset are less than 10 words)
# Our architecture uses 20,223,927 parameters (of which 20,000,050 [from the word embeddings] are non-trainable [400001 words * 50 vector length])

# Compile the Model 
Emojify_LSTM_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the Keras model on indexes and Y_train_oh, using 50 epochs and 32 as batch size
Emojify_LSTM_model.fit(X_train_indexes, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)


############################################################################################################
#####                                   EVALUATE EMOJIFIER-LSTM MODEL                                  #####
############################################################################################################

# Evaluate on the test set
X_test_indexes = sentences_to_indexes(X_test, word_to_index, max_len = max_len)
pred = Emojify_LSTM_model.predict(X_test_indexes)
loss, acc = Emojify_LSTM_model.evaluate(X_test_indexes, Y_test_oh)
print("Test accuracy = ", acc)

# Check the mislabelled examples
for i in range(len(X_test)):
    y_pred = np.argmax(pred[i])
    if(y_pred != Y_test[i]):
        print('Sentence: ' + X_test[i] + 'Expected emoji:' + code_to_emoji(Y_test[i]) + ' vs Predicted emoji:'+ code_to_emoji(y_pred))


# Check with our own sentence (make sure all the words are in the Glove embeddings)
new_sentence = np.array(['not feeling happy'])
new_sentence_indexes = sentences_to_indexes(new_sentence, word_to_index, max_len)
print(new_sentence[0] +' '+  code_to_emoji(np.argmax(Emojify_LSTM_model.predict(new_sentence_indexes))))