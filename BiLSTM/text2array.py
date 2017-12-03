##################################
#author          :Vishwajit
#date            :2017 09
#owner           :Cogknit Semantics Pvt Ltd
##################################

# coding: utf-8
import os
import numpy as np


# In[4]:

pwd


# In[5]:

base = '/home/blazingboosh/EDGE_IN/speech/BiLSTM/Parameter_Matrices/'
dir_list = [base + 'input_layer_to_BLSTM/', base + 'Fully_connected_layer/', base + 'BLSTM_to_BLSTM/']
params= dict()


# In[17]:

for dir_ in dir_list:
    for filename in os.listdir(dir_):
        with open(dir_+filename) as infile:
            print str(filename)[:-4]
            params[str(filename)[:-4]]= [map(float, line.split()) for line in infile]
        


# In[18]:

params = np.array(params)


# In[19]:

pwd 


# In[20]:

np.save('BiLSTM_parameters', params)



