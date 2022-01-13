import numpy as np
from collections import Counter
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn

def plot_cm(y_test, y_pred, outdir = '.'):
    
    """ Plots the confusion matrix and returns the normalized accuracy
        Inputs:
            y_test: groundtruth labels (n,1)
            y_pred: predicted labels after argmax (n,1)
            outdir: folder to save the Confusion Matrix
        Returns:
            norm_acc: normalized accuracy
    """
    
    cm = confusion_matrix(y_test, y_pred)
    cm = cm/np.sum(cm, axis=1)
    cm = np.round(cm, 2)

    df_cm = pd.DataFrame(cm, index = [i for i in "0123456789"],
              columns = [i for i in "0123456789"])

    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})

    plt.savefig(os.path.join(outdir, 'confusion_matrix.png'))

    norm_acc = 100*accuracy_score(y_test, y_pred, normalize= True)	
    print('Norm. Acc. %.2f' % (norm_acc))
    
    return norm_acc
# generating data
def data_generator(x, y, batch_size):
    
    """ data generator
        Inputs:
            x: input data (n,h,w,c)
            y: labels (n,1)
            batch_size: the number of samples in a batch
        Returns:
            batches (x,y) at each iteration
    """
    
    while True:

        x_batch, y_batch  = [], []

        c = list(zip(x, y))
        np.random.shuffle(c)
        x, y = zip(*c)

        for x_,y_ in zip(x,y):

            x_batch.append(x_)
            y_batch.append(y_)

            if len(x_batch) == batch_size:

                x_batch, y_batch = np.array(x_batch), np.array(y_batch)

                yield(x_batch,y_batch)

                x_batch, y_batch = [], []
                
# combining the generators
def ResLT_generator(x_train_lt,  y_train_lt, bs):
    
    """ data generator for head(h), medium(m) and tail(t) classes
        Inputs:
            x_train_lt: input data (n,h,w,c)
            y_train_lt: labels (n,1)
            bs: the number of samples in a batch
        Returns:
            batches ([x1, x2, x3],[y1, y1, y2, y3]) at each iteration. 1 : hmt, 2: mt, 3: t
    """
    
    w = Counter(y_train_lt)
    nums, num_samples = np.array(list(w.keys())), np.array(list(w.values()))

    tail_cls = nums[num_samples <= 100]
    medium_cls = nums[(num_samples > 100) & (num_samples <= 1000)]
    head_cls = nums[(num_samples > 1000)]

    x_train_t = x_train_lt[np.isin(y_train_lt, tail_cls)]
    x_train_mt = x_train_lt[np.isin(y_train_lt, np.concatenate([medium_cls,tail_cls]))]

    y_train_t = y_train_lt[np.isin(y_train_lt, tail_cls)]
    y_train_mt = y_train_lt[np.isin(y_train_lt, np.concatenate([medium_cls,tail_cls]))]

    gen_hmt = data_generator(x_train_lt, y_train_lt, bs)
    gen_mt = data_generator(x_train_mt, y_train_mt, bs)
    gen_t = data_generator(x_train_t, y_train_t, bs)

    while True:

        x1, y1 = next(gen_hmt)
        x2, y2 = next(gen_mt)
        x3, y3 = next(gen_t)
        
        x_batch,y_batch = [x1, x2, x3], [y1, y1, y2, y3]
        yield(x_batch,y_batch)