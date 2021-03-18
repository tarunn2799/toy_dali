#!/usr/bin/env python
# coding: utf-8

# In[7]:


import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
import os
import random
import torch
import numpy as np
import pandas as pd
import glob
from PIL import Image
import itertools
import nvidia.dali.fn as fn
import cupy
import ast


# # External Source Example
# 

# In[20]:


class MultiClassIterator(object):
    def __init__(self, datasets, batch_size):
        len_df = [len(df) for df in datasets]
        self.largest_dataset = max(len_df) #length of csv file with the most number of datapoints
        self.largest_dataset_idx = len_df.index(self.largest_dataset) #identifier for which one the largest is
        self.counter_index = {i: 0 for i, dataset in enumerate(datasets)} #a counter to see how many datapoints have been taken from each dataset
        self.counter_dataset = 0 #identifier for which dataset we're currently on
        self.iterable_datasets = itertools.cycle(datasets)
        self.batch_size = batch_size
        self.datasets = datasets

    def customroundrobin(self, iterable, index): 
        '''
        Performs a round robin over the smaller dataset. In the case where one CSV is smaller than the other, the
        smaller one is iterated through again till we reach the length of the largest CSV dataset.
        '''
        start_over = 0
        if index >= len(iterable):
            start_over += 1
        while True:
            for i, element in enumerate(iterable):
                if i >= index or start_over:
                    if i == len(iterable) - 1:
                        start_over += 1
                    yield element
                    
    def __iter__(self):
        return self
        
    def __next__(self):
        batch_crops = []
        batch_labels = [np.array(i) for i in range(4)] #Ignore this, just a dummy label being generated. 
        print('Counter Index Updated', self.counter_index)
        if self.counter_index[self.largest_dataset_idx] == self.largest_dataset: 
            #if we've gone through one iteration of the entire dataset, resets counters to 0. 
            self.counter_index = {i: 0 for i, dataset in enumerate(self.datasets)}
            
        if self.counter_index[self.largest_dataset_idx] < self.largest_dataset:
            #takes a row from our CSV file dataset, and appends the result values to a list. Value is yielded.
                batch_counter = 0
                cur_iterable = self.customroundrobin(self.iterable_datasets.__next__(), self.counter_index[self.counter_dataset])
                while batch_counter < self.batch_size:
                    self.counter_index[self.counter_dataset] += 1
                    data_point = cur_iterable.__next__()
                    crops = data_point["labels_crops"]
                    crops = ast.literal_eval(crops)
                    crops = [crops[0], crops[1], crops[2] - crops[0], crops[3] - crops[1]] #Format to support ops.slice
                    batch_crops.append(np.array([crops], dtype=np.int32))
                    batch_counter += 1
                self.counter_dataset += 1
                if self.counter_dataset == len(datasets):
                    self.counter_dataset = 0
                yield (batch_crops, batch_labels)
        

    @property
    def size(self):
        return self.largest_dataset


# In[21]:


class ExternalSourcePipeline(Pipeline):
    def __init__(self, file_list, batch_size, num_threads, device_id, external_data):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_list= file_list)
        self.label = ops.ExternalSource()
        self.crops = ops.ExternalSource()
#         self.counter = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.cast = ops.Cast(device="cpu", dtype=types.INT32)
        self.external_data = external_data
        self.iterator = iter(self.external_data)
        
    def define_graph(self):
        jpegs, dummy_labels = self.input()
        self.labels = self.label()
        self.crop_dim = self.crops()
#         self.counter_idx = self.counter()
        anchor =  fn.reshape(fn.slice(self.crop_dim, 0, 2, axes=[1]), shape=[-1])
        shape = fn.reshape(fn.slice(self.crop_dim, 2, 2, axes = [1]), shape= [-1])
        anchor = self.cast(anchor)
        shape = self.cast(shape)
        images = self.decode(jpegs)
        images = self.res(images)

        #self.num_classes =  len(self.std_train_config["classes_train"][self.std_train_config["identifier"][self.identifier_index]])

#       decode and slicing
        jpegs = fn.slice(jpegs, anchor, shape, axes= [0,1], device= 'gpu')
        jpegs = self.res(jpegs)

        return (images, self.labels, self.crop_dim)

    def iter_setup(self):
            print('Entering iter_setup func')
            crops, labels = list(next(self.iterator))[0]
            print(crops, labels)
            self.feed_input(self.labels, labels)
            self.feed_input(self.crop_dim, crops)


# In[22]:


datasets = [pd.read_csv(df, index_col= 'Unnamed: 0').to_dict(orient='records') for df in glob.glob('*.csv')]


multi_iter = MultiClassIterator(datasets, 4)


# In[23]:


pipe = ExternalSourcePipeline(file_list = 'single_image.txt' ,batch_size= 4, num_threads=2, device_id=0,
                                  external_data=multi_iter)


# In[24]:


pii = DALIGenericIterator(pipe, output_map=['data', 'label', 'crops'], auto_reset = True, size = multi_iter.size)


# In[ ]:




