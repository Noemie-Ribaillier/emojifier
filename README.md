# Emojifier

## Project description 
The goal of this project is to build an emojifier, ie to find the most appropriate emoji among a list of 5 emojis (heart, baseball, smile, disappointed and fork_and_knife) to add to a sentence. 
When a person types a text with his/her phone for example, the phone offers to add/replace the word "love" with a heart for example, but it may not be the case with a very similar word like "adore" (as the phone only learnt that the "heart" emoji is linked to "love" and "heart"). We want to implement a model helping for that.

Using word vectors definitely helps with that because even if the model doesn't learn on a specific word (didn't see that word with an emoji in the training fase), it will be able to give an emoji to this word by seeing to which word, that the model learnt on, this new word is close to. The algorithm is able to generalize and associate additional words in the test set to the same emoji. This allows to build an accurate classifier mapping from sentences to emojis, even using a small training set. 

We will build 3 models:
* a model (Emojifier-NN) using a simple NN with 1 hidden layer (only using the average of the embedding vectors of each word for a specific sentence)
* a model (Emojifier-RNN) using a simple RNN, so that it succeeds in getting the word order in the sentence
* a model (Emojifier-LSTM) using LSTM, so that it succeeds in getting the word order in the sentence and is less prone to gradient vanishing/exploding


## Dataset
The dataset is already split into train and test datasets (csv files). The train set contains 132 rows and the test set contains 56 rows. Both sets have 2 columns (that we will call X and Y), the sentences and the corresponding coded emoji. Possible emojis are: 0 (heart), 1 (baseball), 2 (smile), 3 (disappointed) and 4 (fork_and_knife).
We one-hot encod the Y set to be accepted by a softmax layer/function.

We will use the pre-trained 50-dimensional GloVe embeddings. The GloVe embeddings aims at highlighting the relationship between words based on the contexts they are used together.


## Model Emojifier-NN

### Neural Network (NN)

A neural network is a computational model inspired by the way biological neural networks in the human brain process information. It consists of layers of nodes (also called neurons) connected to each other. Each node processes information and passes it to the next layer. The layers in a neural network typically include:
* Input layer: takes the input features (data) that the model will process
* Hidden layers: layers between the input and output where computations occur. These layers help the network learn patterns and relationships in the data.
* Output layer: provides the final prediction or output
Each connection between neurons has a weight that adjusts during training to minimize the error. A bias is also added to each neuron to help the model make better predictions.

The goal of the forward propagation is to make predictions by passing the input through the network. This is the process:
* The input is passed to the first layer
* Each neuron in a layer computes a weighted sum of its inputs (from the previous layer), applies a bias, and passes the result through an activation function (eg ReLU, sigmoid, ...)
* The output of one layer is passed as input to the next layer
Finally, the output layer generates the prediction or output of the network. The predicted values (output) are computed and used for loss calculation.

The goal of the backward propagation is to adjust the weights of the network in order to reduce the error in the predictions. This is the process:
* Compute the loss/error: using a loss function (e.g., MSE for regression, cross-entropy for classification) by comparing the predicted output to the true labels
* Backpropagate error: the error is propagated backward through the network, starting from the output layer and moving towards the input layer. The error tells the network how far off its predictions are.
* Compute gradients: using the chain rule of calculus, the gradients (partial derivatives) of the loss function with respect to each weight are calculated (this tells how much each weight contributed to the error)
* Update weights: using the gradients, the weights are updated to minimize the loss


### Script explanation
We start with a naive model, a simple NN (only gets the meaning of the words, not the order), with: 
* Input of the model: a string corresponding to a sentence (e.g. "I love you").
* Output of the model: a probability vector of shape (1,5). The chosen emoji is the index of the highest probability.

Details of the model:
* Initialize the weight matrix using Xavier intialization (to maintain the variance of the activations across layers and to prevent issues like vanishing gradients [activations becoming too small] or exploding gradients [activations becoming too large] during training). Xavier initialization sets up the weight matrix to be initialized from a normal distribution with mean 0 and variance 2/(n_in + n_out), here n_in is the dimension of embedding vector and n_out is the number of classes we want to predict.
* Convert each word in the input sentence into its word vector representation
* Compute the average of the word vectors (of this specific sentence)
* Pass the average vector through forward propagation
* Compute the loss using cross entropy loss (classification case)
* Backpropagate to update the weights and baises
* Update the weight and bias matrices by iterating a few hundreds of times on the training sample and by minimizing the loss

We get a not too bad accuracy on the test set (even for a very small training set). This is due to the generalization power that word vectors gives us. One thing the model is not able to do is understand the order of words and the combination of words in the sentence. The model doesn't understand "not feeling happy today" for example.


## Model Emojifier-RNN

### Recurrent Neural Network (RNN)
Traditional neural networks can't handle sequences. RNN address this issue, they handle very well sequences, they are networks with loops in them, allowing information to persist. RNN remember the previous information and use it for processing the current input. RNN model:
* at time 0: x_0 gives h_0
* at time 1: x_1 and H_0 gives x_1
* at time t: x_t and h_t-1 gives h_t
The network is influenced at time t by the input it receives and by what happened before.


### Script explanation
We build another model, a simple RNN (it takes into account words ordering).
It also uses pre-trained word embeddings to represent words. We'll feed word embeddings into an RNN, and the RNN will learn to predict the most appropriate emoji. 

Details of the model:
* Pad all the sentences to the same length. Indeed, for vectorization to work, all sentences should have the same length. We set a maximum sequence length and we pad (with 0) all sequences to have this same length. If a sentence is longer than the max length we fixed, the sentence will be truncated (so we just fix max length as the number of words of the longest sentence in the training set).
Padding is an important step when dealing with sequences to use mini-batches (all sequences must be of the same length).
* Build the embedding layer: it maps word indices to embedding vectors. The embedding vectors are dense vectors (meaning that most of their values are non-zero) of fixed size. The embedding matrix can be derived in 2 ways:
    * Training a model to derive the embeddings from scratch
    * Using a pretrained embedding. In this case we will use the GLoVe50 pre-trained embeddings and we will not update them because our training set is quite small
    The embedding matrix has:
    * Inputs: integer matrix of size (batch size, max input length) corresponding to sentences converted into lists of indices (integers)
    * Outputs: an array of shape (batch size, max input length, dimension of word vectors), it returns the word embeddings for a sentence
* Feed the embedding layer's output to a SimpleRNN layer
* Feed the previous output to a dropout layer (with probability 0.5). It means that 50% of the neurons are randomly "dropped" (set to zero) during each forward pass in training. It aims at regularizing the model by preventing overfitting, making the network more robust and capable of generalizing better to new data. During the test fase, dropout is disabled, but the outputs are scaled to maintain consistency.
* Feed the previous output to a SimpleRNN layer with 128 hidden units
* Feed the previous output to a dropout layer (with probability 0.5)
* Feed the previous output to a dense layer that reduces the number of neurons to the number of classes we want to predict
* Feed the previous output to a softmax activation layer (to get the probability for each class)
* Create the model with input and output

This model gets the following inputs/outputs:
* intput: X_train transformed to indexes (the sentence is split, then each word is transformed to its index) and Y encoded (to 5 classes here)
* output: the predictions

This model succeed in understanding the "I'm not happy" that the simple NN model didn't get.


## Model Emojifier-LSTM

### Long-Short-Term Memory (LSTM)
RNN are good at using recent past information but are not so good at using long past information, this is where LSTM come handy (remembering information for long periods of time is practically their default behavior). LSTM is a RNN excelling at capturing long-term dependencies. LSTM solves the vanishing and exploding gradients problem faced by RNN.

#### Architecture
The LSTM architecture consists of one unit (called memory unit or LSTM unit).
The LSTM unit has 4 components:
* forget gate: used to perform deletion of information from memory, whether the information coming from the previous time is to be remembered or irrelevant so to be forgotten
* input gate: used to add new information in memory, it tries to learn new information from the input
* candidate memory cell: used to create new candidate information to be inserted into the memory
* output gate: used to output information present in memory, it passes the updated information from time t to t+1
The 3 gates are used to create selector vectors (with values between 0 and 1, near these 2 extremes, since they use sigmoid function as the activation function in the output layer). 

The LSTM has the following inputs and outputs:
* inputs: 
    * vector cell state at time t-1 (c_t-1) representing the long term memory
    * vector hidden state at time t-1 (h_t-1) representing the short term memory
    * vector X at time t (x_t) representing the new input
* outputs: 
    * hidden state for time t+1 (h_t+1): it uses long term memory (cell state c) to update the short term memory (hidden state h)
    * cell state for time t+1 (c_t+1): it uses recent past information (hidden state H) and new information coming from outside (input vector x) to update the long term memory (cell state c)

Dimensions of the cell state and the hidden state are the same and determined by the number of LSTM units.

#### Step by step

##### Forget gate
Goal: based on the inputs, what information should be removed from the cell state vector coming from time t-1 (c_t-1)
Inputs: x_t and h_t-1
Formula: f_t = sigma(W_f*[h_t-1, x_t] + b_f) wit W_f the weights that connect both the previous hidden state (h_t-1) and the current input (x_t) to the forget gate
Output: a selector vector, of length c_t-1, with values between 0 and 1 because the activation function use the sigmoid function. 
Later, the selector vector is multiplied element-wise with the cell state vector (c_t-1) received as input, so:
* a position where the selector vector has a value 0 completely eliminates the information included in the same position in the cell state ("forget it") 
* a position where the selector vector has a value 1 leaves unchanged the information included in the same position in the cell state ("keep it").

##### Input gate
Goal: based on the inputs, how much of the new candidate memory (c_tilde_t) should be allowed to modify the cell state
Inputs: x_t and h_t-1
Formula: i_t = sigma(W_i*[h_t-1, x_t] + b_i)
Output: a selector vector of length c_t-1, with values between 0 and 1 because the activation function use the sigmoid function. 

##### Candidate memory cell
Goal: based on the inputs, it creates a candidate vector of new candidates that could be added to the cell state
Inputs: x_t and h_t-1
Formula: c_tilde_t = tanh(W_c*[h_t-1, x_t] + b_c)
Output: a candidate vector of length c_t-1, with values between -1 and 1 because the tanh function is used as activation function. Tanh is a smooth function that normalizes values between -1 and 1 and behave like a linear function for small values so it makes the model more stable.

Later, the selector vector (from input gate) and the candidate vector (from candidate memory cell) are multiplied with each other, element wise. This means that:
* a position where the selector vector has a value equal to 0 completely eliminates the information included in the same position in the candidate vector
* a position where the selector vecor has a value equal to 1 leaves unchanged the information included in the same position in the candidate vector 

Later, the result of the multiplication between the candidate vector and the selector vector is added to the cell state vector, this adds new information to the cell state (after some information has been removed by the forget gate).
Formula: c_t = f_t * c_t-1 + i_t * c_tilde_t
* f_t affects how much the previous state should influence the current state
* candidate memory cell (c_tilde_t): can have both positive and negative values. A negative value for the candidate memory cell means that the LSTM wants to decrease certain parts of the cell state, but it does so after the forget gate has determined how much of the previous cell state should be retained, and after it's been controlled by the input gate (which determines how much of it should be stored).

##### Output gate
Goal: based on the inputs, it determines the value of the hidden state outputted by the LSTM at time t (output, h_t) and received by the LSTM at time t+1 (input, h_t+1).
Inputs: x_t and h_t-1
Formula: o_t = sigma(W_o*[h_t-1, x_t] + b_o)
Output: a selector vector, of length c_t-1, with values between 0 and 1 because the activation function use the sigmoid function. 

Later, the selector vector is multiplied to the candidate vector (tanh(c_t)). The tanh gives values between -1 and 1, to make it possible to control the stability of the network over time. 
A position where the selector vector has a value equal to 0 completely eliminates the information included in the same position in the candidate vector. A position where the selector vector has a value equal to 1 leaves unchanged the information included in the same position in the candidate vector.
Formula: h_t = o_t * tanh(c_t), so hidden state is a function of long term memory (c_t) and the current output.

##### LSTM solves vanishing and exploding gradients
Vanishing and exploding gradients are common issues while training RNNs. It comes from the way gradients are propagated through the layers of the network during backpropagation.
* Vanishing gradients:
    * problem: problem occurs when gradients become very small as they are propagated backward through the network causing the weights in earlier layers to stop updating effectively. This happens mainly with networks dealing with long-term dependencies (or deep networks) where the gradients tend to shrink exponentially as they are backpropagated over many layers, making it difficult for the model to learn long-term dependencies.
    * solution: LSTM address the vanishing gradient problem by using the cell state, which allows gradients to flow more easily through time steps. The gating mechanisms also help regulate the information flow, preventing gradients from shrinking to zero over time.
* Exploding gradients:
    * problem: it happens when gradients become very large as they are backpropagated through the network. This can cause the model parameters (weights) to grow uncontrollably, leading to numerical instability and causing the training process to fail.
    * solution: LSTMs mitigate exploding gradients by using sigmoid and tanh activations for their gates and outputs, which naturally constrain the gradient values. Additionally, gradient clipping is often employed to prevent instability during training.

## Script explanation
We build another model, a RNN using LSTM (takes into account words ordering and prevent gradient vanishing/exploding problems).
It also uses pre-trained word embeddings to represent words. We'll feed word embeddings into an LSTM, and the LSTM will learn to predict the most appropriate emoji. 

Details of the model are the same than with RNN except that we use 2 LSTM layers instead of the SimpleRNN layer.
This model succeeds in getting the "I'm not happy" that simple NN model didn't get and the etst accuracy is better for the LSTM NN than for the simple RNN.


## References
This script is coming from the Deep Learning Specialization course. I enriched it to this new version.
