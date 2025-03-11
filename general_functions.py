# Load the libraries
import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Create the function to read the GloVe word embeddings
def read_glove_vectors(glove_file):
    """
    Read the GloVe word embeddings
    
    Arguments:
    glove_file -- file containing the GloVe embeddings
    
    Returns:
    words_to_index -- a dictionnary mapping each word (key) to a specific index (value) (ABC ordered)
    index_to_words -- a dictionnary mapping an index (key) to each word (value) (basically inverting keys and values from words_to_index)
    word_to_vec_map -- a dictionnary mapping the word embedding (value) to each word (key)
    """

    # Open the file
    with open(glove_file, 'r', encoding="utf8") as f:
        # Create the words empty set and word_to_vec_map empty dictionnary
        words = set()
        word_to_vec_map = {}

        # Iterate on each line, split it and take the 1st element as the word (to add to the words set) 
        # and the rest as the word vector (to add to the dictionnary with the key being the word and the value being the array)
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        # Initialize i to 1
        i = 1

        # Create empty dictionnaries for words_to_index and index_to_words
        words_to_index = {}
        index_to_words = {}

        # Iterate on the words (sorted, so ABC order)
        for w in sorted(words):
            # Fill the words_to_index dictionnary with giving a different index (value) to each word (key)
            words_to_index[w] = i
            # Fill the index_to_words dictionnary as the reverse of words_to_index dictionnary (each index [key] gets a different word [value])
            index_to_words[i] = w
            # Augment i at each iteration to ensure to have a different index per word and vice versa
            i = i + 1

    return words_to_index, index_to_words, word_to_vec_map


# Create the function to implement the softmax (will be used as activation function for the very simple NN model)
def softmax(x):
    """
    Compute softmax values for each sets of scores in x (it transforms the x vector into probabilities)
    Formula: s_i = exp(x_i)/sum(exp(x_i))

    Arguments:
    x -- vector (with shape of the number of classes to be predicted from)
    
    Returns:
    s -- the softmax values for each value of x (we get the probabilities for each class)
    """
    s = np.exp(x)/sum(np.exp(x))

    return s


# Create the function to read a csv file and create the X and Y dataset
def read_csv(file):
    """
    Read a csv file and create the X and Y dataset
    
    Arguments:
    file -- a csv file (containing 2 columns, later used as X and Y)
    
    Returns:
    X -- array containing the sentence we will use to predict Y
    Y -- array containing the coded emoji (that we want to predict)
    """
    # Create empty lists
    phrase = []
    emoji = []

    # Open the csv file
    with open (file) as csv_data_file:
        csv_reader = csv.reader(csv_data_file)

        # Iterate on each row
        for row in csv_reader:
            # Take the 1st element as the phrase (X) and the 2nd element as the coded emoji (Y)
            phrase.append(row[0])
            emoji.append(row[1])

    # Transform phrase to an array and name it X
    X = np.asarray(phrase)
    # Transform emoji to an array (with integer type since the values taken here go from 0 to 4) and name it Y
    Y = np.asarray(emoji, dtype=int)

    return X, Y


# Create a function to transform an array to a one-hot array with n classes
def convert_to_one_hot(array, n):
    """
    Convert an array to a one-hot array with n classes
    
    Arguments:
    array -- array with max n different values
    n -- number of classes (int)
    
    Returns:
    encoded_array -- one-hot version of the input array
    """
    # np.eye(n): creates an identity matrix (so filled with 0, and 1 on the diagonal) with dimension (n,n)
    # [array.reshape(-1)]: flattens the array into 1D vector
    # We keep the xth row from np.eye(n) where x corresponds to the value of the flatten array (and do that for each value of the array)
    encoded_array = np.eye(n)[array.reshape(-1)]
    
    return encoded_array


# Emoji dictionnary used for this project (5 different emojis with value from 0 to 4)
emoji_dictionary = {"0": ":red_heart:", "1": ":baseball:", "2": ":grinning_face:", "3": ":disappointed_face:", "4": ":fork_and_knife:"}


# Create a function to convert a code into the corresponding emoji
def code_to_emoji(code):
    """
    Converts a code (int) into the corresponding emoji

    Arguments:
    code -- code (int) [between 0 and 4 here]
    
    Returns:
    emoji_output -- emoji (string behind)
    """
    # Take the label corresponding to the code
    emoji_string = emoji_dictionary[str(code)]
    # Find the emoji from the label
    emoji_output = emoji.emojize(emoji_string)

    return emoji_output


# Create a function to print each X and its prediction
def print_predictions(X, pred):
    """
    Print each Xs and its prediction

    Arguments:
    X -- array with input X sentences
    pred -- array with predictions (from the X)
    """
    # Iterate over the X items
    for i in range(X.shape[0]):
        print(X[i], code_to_emoji(int(pred[i])))


# Create a function to plot the confusion matrix
def plot_confusion_matrix(true_y, pred_y, title='Confusion matrix', cmap='Blues'):
    """
    Plot the confusion matrix between the true_y and the pred_y

    Arguments:
    true_y -- true/actual values of Y
    pred_y -- predicted values of Y
    title -- title of the plot (by default 'Confusion matrix')
    cmap -- colormap of the plot (by default 'Blues')
    """
    # Transform the predictions to integer and reshape to a row array
    pred_y = pred_y.reshape(pred_y.shape[0],).astype(int)

    # Compute the confusion matrix using a crosstab between both vectors
    # margins=False: because we don't want the row and column totals
    confusion_matrix = pd.crosstab(true_y, pred_y, rownames=['Actual'], colnames=['Predicted'], margins=False)
    
    # Plot the confusion matrix (with numbers)
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues', cbar=False)
    plt.show()


# Create a function to predict emojis and compute the accuracy of the model over the given set with the given (trained) parameters (will be used for the simple NN)
def predict(X, Y, W, b, word_to_vec_map):
    """
    Given X (sentences) and Y (emoji codes), predict emojis and compute the accuracy of the model over the given set with the given parameters
    
    Arguments:
    X -- input data containing sentences, numpy array
    Y -- labels, containing code of the label emoji, numpy array
    W -- weights matrix, numpy array
    b -- bias matrix, numpy array
    word_to_vec_map -- dictionnary containing the words as key and the word vectors as value
    
    Returns:
    pred -- numpy array of shape (m, 1) with the predictions
    """
    # Get the number of items in X
    m = X.shape[0]

    # Create the prediction vector, of size m (number of items in X) full of 0
    pred = np.zeros((m, 1))

    # Take a random word (in this case the 1st one) from the word embeddings to know the dimensions
    any_word = list(word_to_vec_map.keys())[0]
    
    # Dimension of the embedding
    n_h = word_to_vec_map[any_word].shape[0] 
    
    # Loop over the X items
    for j in range(m):
        
        # Split jth example (sentence) into list of lower case words
        words = X[j].lower().split()
        
        # Average the word vectors of the words in the sentence by creating an avg vector full of 0, adding all the word vectors together and dividing by the number of words in the word_to_vec_map dictionnary
        avg = np.zeros((n_h,))
        count = 0
        for w in words:
            if w in word_to_vec_map:
                avg += word_to_vec_map[w]
                count += 1
        if count > 0:
            avg = avg / count

        # Forward propagation with avg being the input and softmax the activation function
        Z = np.dot(W, avg) + b
        A = softmax(Z)

        # Take the max index of A to determine the prediction
        pred[j] = np.argmax(A)
        
    # Print the accuracy by taking the % of predicted values being equal to the true values
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred