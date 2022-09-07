#!/usr/bin/env python

import numpy as np
import math
import re
from utilities import Load_2a3a_traindata, Load_2a3a_testdata, Load_2c2d3a3d_traindata
from utilities import gaussian_probability_density, mean, variance, gaussian_naive_bayes,leave_one_out_gnb


#path for train data and test data
str_path_2a3a_train = 'data_train_2a3a.txt'
str_path_2a3a_test = 'data_test_2a3a.txt'
str_path_2c2d3c3d_program = 'data_2c2d3c3d_program.txt'

#Performing Gaussian Naive Bayes LOO for using 2 train features(height,weight)
features,labels=Load_2c2d3a3d_traindata(str_path_2c2d3c3d_program, remove_age = True)
perf_2f = leave_one_out_gnb(features,labels)
print('Gaussian Naive Bayes-LOO-2 features-%of true predictions: ' ,perf_2f)

#3c Performing Gaussian Naive Bayes LOO for using 3 train features(height,weight,age)
features,labels=Load_2c2d3a3d_traindata(str_path_2c2d3c3d_program, remove_age = False)
perf_3f = leave_one_out_gnb(features,labels)
print('Gaussian Naive Bayes-LOO-3 features-%of true predictions: ',perf_3f)

