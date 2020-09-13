#########################
# Script to check CNN performance accross energy ranges 
# 10th September 2020
# 
# Evaluates net for given model and plots
# Takes in ONE file to Test on, can compare to old reco
# Runs Energy, Zenith, Track length (1 variable energy or zenith, 2 = energy then zenith, 3 = EZT)
#   Inputs:
#       -i input_file:  name of ONE file 
#       -d path:        path to input files
#       -o ouput_dir:   path to output_plots directory
#       -n name:        name for folder in output_plots that has the model you want to load
#       -e epochs:      epoch number of the model you want to load
#       --variables:    Number of variables to train for (1 = energy or zenith, 2 = EZ, 3 = EZT)
#       --first_variable: Which variable to train for, energy or zenith (for num_var = 1 only)
#       --compare_reco: boolean flag, true means you want to compare to a old reco (pegleg, retro, etc.)
#       -t test:        Name of reco to compare against, with "oscnext" used for no reco to compare with
####################################

import numpy as np
import h5py
import time
import os, sys
import random
from collections import OrderedDict
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",type=str,default=None,
                    dest="input_file", help="names for test only input file")
parser.add_argument("-d", "--path",type=str,default='/data/icecube/jmicallef/processed_CNN_files/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/home/users/jmicallef/LowEnergyNeuralNetwork/',
                    dest="output_dir", help="path to output_plots directory, do not end in /")
parser.add_argument("-n", "--name",type=str,default=None,
                    dest="name", help="name for output directory and where model file located")
parser.add_argument("-e","--epoch", type=int,default=None,
                    dest="epoch", help="which model number (number of epochs) to load in")
parser.add_argument("--variables", type=int,default=1,
                    dest="output_variables", help="1 for track")
parser.add_argument("--first_variable", type=str,default="track",
                    dest="first_variable", help = "adding support for track")
parser.add_argument("--compare_reco", default=False,action='store_true',
                        dest='compare_reco',help="use flag to compare to old reco vs. NN")
parser.add_argument("-t","--test", type=str,default="oscnext",
                        dest='test',help="name of reco")
parser.add_argument("--mask_zenith", default=False,action='store_true',
                        dest='mask_zenith',help="mask zenith for up and down going")
parser.add_argument("--z_values", type=str,default=None,
                        dest='z_values',help="Options are gt0 or lt0")
parser.add_argument("--emax",type=float,default=100.,
                        dest='emax',help="Factor to multiply energy by")
parser.add_argument("--efactor",type=float,default=100.,
                        dest='efactor',help="ENERGY TO MULTIPLY BY!")
args = parser.parse_args()

test_file = args.path + args.input_file
output_variables = args.output_variables
filename = args.name
compare_reco = args.compare_reco
print("Comparing reco?", compare_reco)

print("checkpoint 1")

dropout = 0.2
learning_rate = 1e-3
DC_drop_value = dropout
IC_drop_value =dropout
connected_drop_value = dropout
min_energy = 5
max_energy = args.emax
energy_factor = args.efactor

print("checkpoint 2")

mask_zenith = args.mask_zenith
z_values = args.z_values

print("checkpoint 3")

save = True
save_folder_name = "%soutput_plots/%s/"%(args.output_dir,filename)
if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
load_model_name = "%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,epoch) 
use_old_weights = True

print("checkpoint 4")

if args.first_variable == "track" or args.first_variable == "Track" or args.first_variable == "t" or args.first_variable == "T":
    first_var = "track"
    first_var_index = 8
    print("testing with track as the first index")
    
print("checkpoint 5")


reco_name = args.test
save_folder_name += "/%s_%sepochs/"%(reco_name.replace(" ",""),epoch)
if os.path.isdir(save_folder_name) != True:
    os.mkdir(save_folder_name)

print("checkpoint 6")
    
#Load in test data
print("Testing on %s"%test_file)
f = h5py.File(test_file, 'r')
Y_test_use = f['Y_test'][:]
X_test_DC_use = f['X_test_DC'][:]
X_test_IC_use = f['X_test_IC'][:]

print("checkpoint 7")

energy_col = f[:,0]

mask1_uplim = energy_col <= 25000000000

mask2_lowlim = energy_col > 25000000000
mask2_uplim = energy_col <= 50000000000

mask3_lowlim = energy_col > 50000000000
mask3_uplim = energy_col <= 75000000000

mask4_lowlim = energy_col > 75000000000

Y_test_use_1 = Y_test_use[mask1_uplim]
X_test_DC_use_1 = X_test_DC_use[mask1_uplim]
X_test_IC_use_1 = X_test_IC_use[mask1_uplim]

Y_test_use_2 = Y_test_use[mask2_lowlim]
Y_test_use_2 = Y_test_use[mask2_uplim]
X_test_DC_use_2 = X_test_DC_use[mask2_uplim]
X_test_DC_use_2 = X_test_DC_use[mask2_lowlim]
X_test_IC_use_2 = X_test_IC_use[mask2_uplim]
X_test_IC_use_2 = X_test_IC_use[mask2_lowlim]

Y_test_use_3 = Y_test_use[mask3_lowlim]
Y_test_use_3 = Y_test_use[mask3_uplim]
X_test_DC_use_3 = X_test_DC_use[mask3_uplim]
X_test_DC_use_3 = X_test_DC_use[mask3_lowlim]
X_test_IC_use_3 = X_test_IC_use[mask3_uplim]
X_test_IC_use_3 = X_test_IC_use[mask3_lowlim]

Y_test_use_4 = Y_test_use[mask4_uplim]
X_test_DC_use_4 = X_test_DC_use[mask4_uplim]
X_test_IC_use_4 = X_test_IC_use[mask4_uplim]

# cat_1 = f[mask1_uplim]
# cat_2 = f[mask2_lowlim]
# cat_2 = f[mask2_uplim]
# cat_3 = f[mask3_lowlim]
# cat_3 = f[mask3_uplim]
# cat_4 = f[mask4_lowlim]

print("checkpoint 7.a")


# if compare_reco:
    # reco_test_use = f['reco_test'][:]
# f.close
# del f
print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_use.shape, "checkpoint 8")




### category 1
#Make network and load model
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from cnn_model import make_network
model_DC = make_network(X_test_DC_use_1,X_test_IC_use_1,output_variables,DC_drop_value,IC_drop_value,connected_drop_value)
model_DC.load_weights(load_model_name)
print("Loading model %s"%load_model_name)

print("checkpoint 9")

# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.losses import mean_squared_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import binary_crossentropy

if first_var == "track":
def TrackLoss(y_truth, y_predicted):
    return binary_crossentropy(y_truth[:,0],y_predicted[:,0])

model_DC.compile(loss=TrackLoss,
            optimizer=Adam(lr=learning_rate),
            metrics=[TrackLoss])

print("track first")

print("checkpoint 10")

t0 = time.time()

print("checkpoint 11")

Y_test_predicted_1 = model_DC.predict([X_test_DC_use_1,X_test_IC_use_1])

print("checkpoint 12")

t1 = time.time()

print("checkpoint 13")

f = h5py.File("%s/prediction_values%s_class1.hdf5"%(save_folder_name,epoch))
print("Saving output file: %s/prediction_values%s_class1.hdf5"%(save_folder_name,epoch))
f.create_dataset("Y_test_use", data=Y_test_use_1)
f.create_dataset("Y_predicted", data=Y_test_predicted_1)
f.close()
print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))
print(X_test_DC_use_1.shape,X_test_IC_use_1.shape,Y_test_predicted_1.shape,Y_test_use_1.shape)

print("checkpoint")

### category 2
model_DC = make_network(X_test_DC_use_2,X_test_IC_use_2,output_variables,DC_drop_value,IC_drop_value,connected_drop_value)
model_DC.load_weights(load_model_name)
print("Loading model %s"%load_model_name)

print("checkpoint 9")

if first_var == "track":
def TrackLoss(y_truth, y_predicted):
    return binary_crossentropy(y_truth[:,0],y_predicted[:,0])

model_DC.compile(loss=TrackLoss,
            optimizer=Adam(lr=learning_rate),
            metrics=[TrackLoss])

print("track first")

print("checkpoint 10")

t0 = time.time()

print("checkpoint 11")

Y_test_predicted_2 = model_DC.predict([X_test_DC_use_2,X_test_IC_use_2])

print("checkpoint 12")

t1 = time.time()

print("checkpoint 13")

f = h5py.File("%s/prediction_values%s_class2.hdf5"%(save_folder_name,epoch))
print("Saving output file: %s/prediction_values%s_class2.hdf5"%(save_folder_name,epoch))
f.create_dataset("Y_test_use", data=Y_test_use_2)
f.create_dataset("Y_predicted", data=Y_test_predicted_2)
f.close()
print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))
print(X_test_DC_use_2.shape,X_test_IC_use_2.shape,Y_test_predicted_2.shape,Y_test_use_2.shape)

print("checkpoint")

### category 3
model_DC = make_network(X_test_DC_use_3,X_test_IC_use_3,output_variables,DC_drop_value,IC_drop_value,connected_drop_value)
model_DC.load_weights(load_model_name)
print("Loading model %s"%load_model_name)

print("checkpoint 9")

if first_var == "track":
def TrackLoss(y_truth, y_predicted):
    return binary_crossentropy(y_truth[:,0],y_predicted[:,0])

model_DC.compile(loss=TrackLoss,
            optimizer=Adam(lr=learning_rate),
            metrics=[TrackLoss])

print("track first")

print("checkpoint 10")

t0 = time.time()

print("checkpoint 11")

Y_test_predicted_3 = model_DC.predict([X_test_DC_use_3,X_test_IC_use_3])

print("checkpoint 13")

t1 = time.time()

print("checkpoint 13")

f = h5py.File("%s/prediction_values%s_class3.hdf5"%(save_folder_name,epoch))
print("Saving output file: %s/prediction_values%s_class3.hdf5"%(save_folder_name,epoch))
f.create_dataset("Y_test_use", data=Y_test_use_3)
f.create_dataset("Y_predicted", data=Y_test_predicted_3)
f.close()
print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))
print(X_test_DC_use_3.shape,X_test_IC_use_3.shape,Y_test_predicted_3.shape,Y_test_use_3.shape)

print("checkpoint")

### category 4
model_DC = make_network(X_test_DC_use_4,X_test_IC_use_4,output_variables,DC_drop_value,IC_drop_value,connected_drop_value)
model_DC.load_weights(load_model_name)
print("Loading model %s"%load_model_name)

print("checkpoint 9")

if first_var == "track":
def TrackLoss(y_truth, y_predicted):
    return binary_crossentropy(y_truth[:,0],y_predicted[:,0])

model_DC.compile(loss=TrackLoss,
            optimizer=Adam(lr=learning_rate),
            metrics=[TrackLoss])

print("track first")

print("checkpoint 10")

t0 = time.time()

print("checkpoint 11")

Y_test_predicted_4 = model_DC.predict([X_test_DC_use_4,X_test_IC_use_4])

print("checkpoint 14")

t1 = time.time()

print("checkpoint 14")

f = h5py.File("%s/prediction_values%s_class4.hdf5"%(save_folder_name,epoch))
print("Saving output file: %s/prediction_values%s_class4.hdf5"%(save_folder_name,epoch))
f.create_dataset("Y_test_use", data=Y_test_use_4)
f.create_dataset("Y_predicted", data=Y_test_predicted_4)
f.close()
print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))
print(X_test_DC_use_4.shape,X_test_IC_use_4.shape,Y_test_predicted_4.shape,Y_test_use_4.shape)

print("checkpoint")