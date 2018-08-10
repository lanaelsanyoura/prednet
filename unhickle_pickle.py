import os
import numpy as np
np.random.seed(123)
import hickle as hkl
import h5py
from six.moves import cPickle as pickle

TEMP_DATA_DIR = './kitti_data'

# Data files
train_file = os.path.join(TEMP_DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(TEMP_DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(TEMP_DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(TEMP_DATA_DIR, 'sources_val.hkl')
test_sources = os.path.join(TEMP_DATA_DIR, 'sources_test.hkl')
test_x = os.path.join(TEMP_DATA_DIR, 'X_test.hkl')

p_train_file = os.path.join(TEMP_DATA_DIR, 'X_train_pickle.hkl')
p_train_sources = os.path.join(TEMP_DATA_DIR, 'sources_train_pickle.hkl')
p_val_file = os.path.join(TEMP_DATA_DIR, 'X_val_pickle.hkl')
p_val_sources = os.path.join(TEMP_DATA_DIR, 'sources_val_pickle.hkl')
p_test_sources = os.path.join(TEMP_DATA_DIR, 'sources_test_pickle.hkl')
p_test_x = os.path.join(TEMP_DATA_DIR, 'X_test_pickle.hkl')

# Unhickle The file
# Pickle the file
train_ = hkl.load(train_file)
source_train_ = hkl.load(train_sources)
val_ = hkl.load(val_file)
source_val_ = hkl.load(val_sources)
test_source = hkl.load(test_sources)
test_val = hkl.load(test_x)

with open(p_train_file, 'wb') as pickle_train:
    pickle.dump(train_, pickle_train)
with open(p_val_file, 'wb') as pickle_val:
    pickle.dump(val_, pickle_val)

with open(p_train_sources, 'wb') as pickle_source_train:
    pickle.dump(source_train_, pickle_source_train)
with open(p_val_sources, 'wb') as pickle_source_val:
    pickle.dump(source_val_, pickle_source_val)

with open(p_test_x, 'wb') as pickle_val_test:
    pickle.dump(test_val, pickle_val_test)
with open(p_test_sources, 'wb') as pickle_source_test:
    pickle.dump(test_source, pickle_source_test)

pickle_train.close()
pickle_val.close()
pickle_source_train.close()
pickle_source_val.close()
pickle_source_test.close()
pickle_val_test.close()
