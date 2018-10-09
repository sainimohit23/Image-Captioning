import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import keras
from keras.preprocessing import image
from preprocess_utils import preprocess, preprocess_input


token = 'Flickr8k_text/Flickr8k.token.txt'
captions = open(token, 'r').read().strip().split('\n')

d = {}
for i, row in enumerate(captions):
    row = row.split('\t')
    row[0] = row[0][:len(row[0])-2]
    if row[0] in d:
        d[row[0]].append(row[1])
    else:
        d[row[0]] = [row[1]]
        
images = 'Flickr8k_Dataset/Flicker8k_Dataset/'
img = glob.glob(images+'*.jpg')



def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp


train_images_file = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))
train_img = split_data(train_images)

test_images_file = 'Flickr8k_text/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))
test_img = split_data(test_images)


image_model = keras.applications.vgg16.VGG16(include_top=True, input_shape=(224,224,3))
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = keras.models.Model(inputs=image_model.input,
                             outputs=transfer_layer.output)


def encode(image):
    image = preprocess(image)
    temp_enc = image_model_transfer.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc


encoding_train = []
for img_1 in tqdm(train_img):
    encoding_train.append(encode(img_1))

with open("encoded_images_vgg16.p", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle) 



encoding_test = []
for img_1 in tqdm(test_img):
    encoding_test.append(encode(img_1))
    
with open("encoded_images_test_vgg16.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle) 











vocab_to_int = {}
i = 0
for key, capts in d.items():
    for capt in capts:
        words = capt.strip().split()
        for word in words:
            if word not in vocab_to_int:
                vocab_to_int[word] = i
                i += 1

codes = ['<PAD>','<EOS>','<UNK>','<GO>']
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)+1

int_to_vocab = {}
int_to_vocab = {num: word for word, num in vocab_to_int.items()}



train_data_pre  = []
for img_1 in tqdm(train_img):
    train_data_pre.append(d[img_1[len(images):]])

test_data_pre = []
for img_1 in tqdm(test_img):
    test_data_pre.append(d[img_1[len(images):]])

train_data = []
for i in train_data_pre:
    tokentoken = []
    for line in i:
        tokens = []
        for word in line.strip().split():
            tokens.append(vocab_to_int[word])
        
        tokentoken.append(tokens)
            
    train_data.append(tokentoken)

test_data = []
for i in test_data_pre:
    tokentoken = []
    for line in i:
        tokens = []
        for word in line.strip().split():
            tokens.append(vocab_to_int[word])
        
        tokentoken.append(tokens)
            
    test_data.append(tokentoken)



pickle.dump((train_data, test_data, vocab_to_int, int_to_vocab),
            open('preprocess.p', 'wb'))







