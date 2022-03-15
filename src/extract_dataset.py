import pickle, gzip, numpy

def extract_dataset():
    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    u = pickle._Unpickler( f )
    u.encoding = 'latin1'
    train_set, valid_set, test_set = u.load()
    f.close()
    return train_set, valid_set, test_set

