#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import randrange

# Does this need to be selected randomly?

# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = round(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        count = 0;
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
            count += 1
        dataset_split.append(fold)
    return dataset_split


# In[ ]:


def cross_validation(k, dataset):
    folds = cross_validation_split(dataset, k)
    
    for i in range(k):
        train = folds.copy()
        test = folds[i]
        del train[i]
        # Enter code for fitting

