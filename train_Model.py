import keras
import keras.backend as K
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from preprocess_utils import preprocess, preprocess_input
import matplotlib.pyplot as plt



def load_preprocess(filename):
    with open(filename, mode='rb') as in_file:
        return pickle.load(in_file)
    
train_inputs, test_inputs, vocab_to_int, int_to_vocab = load_preprocess('preprocess.p')
transfer_values_train = load_preprocess('encoded_images_vgg16.p')
transfer_values_train = np.array(transfer_values_train)


transfer_values_test = load_preprocess('encoded_images_test_vgg16.p')
transfer_values_test = np.array(transfer_values_test)


def get_random_tokens(idx):
    result = []
    
    for i in idx:
        j = np.random.choice(len(train_inputs[i]))
        
        result.append(train_inputs[i][j])
        
    return result


def batch_generator(batch_size):
    while True:
        idx = np.random.randint(len(train_inputs), size= batch_size)
        transfer_values = transfer_values_train[idx]
        tokens = get_random_tokens(idx)
        
        token_lengths = [len(t) for t in tokens]
        max_len = np.max(token_lengths)
        
        tokens_padded = pad_sequences(tokens,
                                      maxlen= max_len,
                                      padding='post',
                                      truncating='post',
                                      value= vocab_to_int['<PAD>'])
        
        
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]
        
        
        x_data =  {'decoder_input': decoder_input_data,
                   'transfer_values_input': transfer_values}
        
        y_data ={'decoder_output': decoder_output_data}
        
        yield(x_data, y_data)


state_size = 512
embedding_size = 128
transfer_values_size = transfer_values_train[0].shape[0]
num_words = len(int_to_vocab) + 1
batch_size = 1024


generator = batch_generator(batch_size=batch_size)


transfer_values_input = keras.layers.Input(shape=(transfer_values_size,), name='transfer_values_input')

decoder_transfer_map = keras.layers.Dense(state_size,
                                          activation='tanh',
                                          name='decoder_transfer_map')

decoder_input = keras.layers.Input(shape=(None, ), name='decoder_input')

decoder_embedding = keras.layers.embeddings.Embedding(input_dim=num_words,
                                                      output_dim=embedding_size,
                                                      name='decoder_embedding')

decoder_gru1 = keras.layers.GRU(state_size, name='decoder_gru1',
                                return_sequences=True)
decoder_gru2 = keras.layers.GRU(state_size, name='decoder_gru2',
                                return_sequences=True)
decoder_gru3 = keras.layers.GRU(state_size, name='decoder_gru3',
                                return_sequences=True)

decoder_dense = keras.layers.Dense(num_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(transfer_values):
    
    initial_state = decoder_transfer_map(transfer_values)

    net = decoder_input
    net = decoder_embedding(net)
    
    # Connect all the GRU layers.
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    
    net = keras.layers.Dropout(0.5)(net)
    net = decoder_gru3(net, initial_state=initial_state)
    
    net = keras.layers.Dropout(0.5)(net)
    decoder_output = decoder_dense(net)
    
    return decoder_output

decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = keras.models.Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])


def sparse_cross_entropy(y_true, y_pred):

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    loss_mean = tf.reduce_mean(loss)
    return loss_mean

optimizer = keras.optimizers.RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target])

try:
    decoder_model.load_weights('Captioning.h5')
except Exception:
    print('Trained Model not found, retraining model')
    
decoder_model.fit_generator(generator=generator,
                            steps_per_epoch=30,
                            epochs=20)

