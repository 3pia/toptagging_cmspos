import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import utils
layers = keras.layers

# ---------------------------------------------------------
# Boilerplate. You can ignore this part.
# ---------------------------------------------------------
folder = utils.get_submit_folder("4vectors")  # do not remove this line!

# ---------------------------------------------------------
# Load and plot data
# ---------------------------------------------------------
jets, labels = utils.load_data(type='4vectors', data='train')  # load data
print("shape of images", jets.shape)
print("shape of labels", labels.shape)

# ---------------------------------------------------------
# Build your model
# ---------------------------------------------------------
inp = layers.Input(shape=(200, 6, 1))
y = layers.Convolution2D(8, (3, 6), activation='relu', padding='valid')(inp)
y = layers.Convolution2D(16, (3, 1), activation='relu', padding='valid')(y)
y = layers.MaxPooling2D((3, 1))(y)
y = layers.Convolution2D(32, (3, 1), activation='relu', padding='valid')(y)
y = layers.Flatten()(y)
y = layers.Dense(2, activation="softmax")(y)

model = keras.models.Model(inputs=inp, outputs=y)
print(model.summary())  # print model details

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(0.001), metrics=["accuracy"])

model.fit(jets, labels, epochs=20, validation_split=0.1, batch_size=128, verbose=2,
          callbacks=[keras.callbacks.CSVLogger(folder + '/history.csv')])

# ---------------------------------------------------------
# Plot your losses and results
# ---------------------------------------------------------
history = np.genfromtxt(folder+'/history.csv', delimiter=',', names=True)

fig, ax = plt.subplots(1)
ax.plot(history['epoch'], history['loss'],     label='training')
ax.plot(history['epoch'], history['val_loss'], label='validation')
ax.legend()
ax.set(xlabel='epoch', ylabel='loss')
fig.savefig(folder+'/loss.png')

fig, ax = plt.subplots(1)
ax.plot(history['epoch'], history['acc'],     label='training')
ax.plot(history['epoch'], history['val_acc'], label='validation')
ax.legend()
ax.set(xlabel='epoch', ylabel='acc')
fig.savefig(folder+'/acc.png')

jets_test, truth = utils.load_data(type='4vectors', data='test')  # load test data
truth_top = truth.astype(np.bool)
predictions = model.predict(jets_test)
fig, ax = plt.subplots(1)
ax.hist(predictions[:, 1][~truth_top], label='qcd', alpha=0.6, bins=np.linspace(0, 1, 40))
ax.hist(predictions[:, 1][truth_top], label='top', alpha=0.6, bins=np.linspace(0, 1, 40))
ax.set_xlim(0, 1)
ax.legend()
ax.set(xlabel='output', ylabel='#')
fig.savefig(folder+'/discrimination.png')
