import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import utils
layers = keras.layers

# ---------------------------------------------------------
# Boilerplate. You can ignore this part.
# ---------------------------------------------------------
folder = utils.get_submit_folder()  # do not remove this line!

# ---------------------------------------------------------
# Load and plot data
# ---------------------------------------------------------
images, labels = utils.load_data(type='images', data='train')  # load data
print("shape of images", images.shape)
print("shape of labels", labels.shape)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for i, ax in enumerate(axes):
    idx = np.random.choice(np.where(labels == i)[0])
    norm = LogNorm(10**-4, images.max(), clip='True')
    im = ax.imshow(images[idx][..., -1], norm=norm)
    ax.set_xlabel('eta')
    ax.set_ylabel('phi')
axes[0].set_title("qcd")
axes[1].set_title("top")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig(folder+'/random_images.png')

# ---------------------------------------------------------
# Build your model
# ---------------------------------------------------------
inp = layers.Input(shape=(40, 40, 1))
y = layers.Conv2D(8, 3, activation='relu', padding='same')(inp)
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Conv2D(16, 5, activation='relu', padding='same')(y)
y = layers.MaxPooling2D(pool_size=(2, 2))(y)
y = layers.Conv2D(32, 5, activation='relu', padding='same')(y)
y = layers.Flatten()(y)
y = layers.Dropout(0.1)(y)
y = layers.Dense(2, activation="softmax")(y)

model = keras.models.Model(inputs=inp, outputs=y)
print(model.summary())  # print model details

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(0.001), metrics=["accuracy"])

model.fit(images, labels, epochs=10, validation_split=0.1, batch_size=128, verbose=2,
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

images_test, truth = utils.load_data(type='images', data='test')  # load test data
truth_top = truth.astype(np.bool)
predictions = model.predict(images_test)
fig, ax = plt.subplots(1)
ax.hist(predictions[:, 1][~truth_top], label='qcd', alpha=0.6, bins=np.linspace(0, 1, 40))
ax.hist(predictions[:, 1][truth_top], label='top', alpha=0.6, bins=np.linspace(0, 1, 40))
ax.set_xlim(0, 1)
ax.legend()
ax.set(xlabel='output', ylabel='#')
fig.savefig(folder+'/discrimination.png')
