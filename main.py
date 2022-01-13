import tensorflow
import os
import numpy as np
from model import ResLT_model
from utils import ResLT_generator, plot_cm
from read_data import load_mnist_lt


# avoid memory allocation error
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

bs = 32
num_epochs = 30
alpha = 0.5

np.random.seed(2389)

# loading the data
x_train, y_train, x_test, y_test, x_train_lt, y_train_lt =  load_mnist_lt(outdir = '.', b = 6)

gen_ = ResLT_generator(x_train_lt, y_train_lt, bs)

model, pred_model = ResLT_model()

model.summary()

model.compile(loss= [tensorflow.keras.losses.SparseCategoricalCrossentropy(), tensorflow.keras.losses.SparseCategoricalCrossentropy(), tensorflow.keras.losses.SparseCategoricalCrossentropy(), tensorflow.keras.losses.SparseCategoricalCrossentropy()],
            optimizer= tensorflow.keras.optimizers.Adam(),
            loss_weights=[1 - alpha, alpha, alpha, alpha],
            metrics='acc')

training_output = model.fit(gen_,
                            batch_size = 3*bs, 
                            epochs =  num_epochs, steps_per_epoch =  1000)

y_pred = pred_model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

norm_acc = plot_cm(y_test, y_pred)