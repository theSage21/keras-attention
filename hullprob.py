import numpy as np
from keras.layers import Input, LSTM
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from models.custom_recurrents import AttentionDecoder
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from itertools import islice


def ohe(i, ml):
    v = [0] * ml
    v[i] = 1
    return v


def hulltask(maxlen):
    x = np.random.random((maxlen, 2))
    hull = [ohe(i, maxlen) for i in ConvexHull(x).vertices]
    padlen = (maxlen // len(hull)) + 1
    hull = (hull * padlen)[:maxlen]
    return x, np.array(hull)


def datagen(batch_size, maxlen):
    while True:
        x, y = [], []
        for _ in range(batch_size):
            a, b = hulltask(maxlen)
            x.append(a.tolist())
            y.append(b.tolist())
        yield np.array(x), np.array(y)


maxlen = 10
inp_dim = 2
hidden_state = 5
batch_size = 50
opt = 'adam'
LOSS = 'categorical_crossentropy'

# -------------model
i = Input(shape=(maxlen, inp_dim))
enc = Bidirectional(LSTM(hidden_state, return_sequences=True),
                    merge_mode='concat')(i)
dec = AttentionDecoder(hidden_state, maxlen, return_probabilities=True)(enc)
model = Model(inputs=i, outputs=dec)
model.summary()
model.compile(opt, LOSS)


# --------------Training
x, y = list(islice(datagen(batch_size, maxlen), 2))[0]
print(x.shape, y.shape)
l, vl = [], []
print('init | Loss | Validation loss')
print('-----+------+----------------')
for init in range(0, 5000, 10):
    history = model.fit_generator(datagen(batch_size, maxlen), 2,
                                  epochs=init+10, initial_epoch=init,
                                  validation_data=datagen(batch_size, maxlen),
                                  validation_steps=1, verbose=1)
    l.extend(history.history['loss'])
    vl.extend(history.history['val_loss'])
    loss = round(history.history['loss'][-1], 3)
    valloss = round(history.history['val_loss'][-1], 3)
    print('{:5}|{:5} |{:5}'.format(init, loss, valloss), end='\r')
    if init % 100 == 0:
        model.save('trained_model')


# ---------------Plotting

plt.plot(l, label='loss')
plt.plot(vl, label='Val_loss')
plt.legend()
plt.savefig('loss.png')
