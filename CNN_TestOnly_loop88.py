#########################
# Version of CNN on 12 May 2020
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

for i in range(25,78):

    test_file = args.path + args.input_file
    output_variables = args.output_variables
    filename = args.name
    epoch = 2*i
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

    # if args.first_variable == "Zenith" or args.first_variable == "zenith" or args.first_variable == "Z" or args.first_variable == "z":
        # first_var = "zenith"
        # first_var_index = 1
        # print("Assuming Zenith is the only variable to test for")
        # assert output_variables==1,"DOES NOT SUPPORT ZENITH FIRST + additional variables"
    # elif args.first_variable == "energy" or args.first_variable == "energy" or args.first_variable == "e" or args.first_variable == "E":
        # first_var = "energy"
        # first_var_index = 0
        # print("testing with energy as the first index")
    if args.first_variable == "track" or args.first_variable == "Track" or args.first_variable == "t" or args.first_variable == "T":
        first_var = "track"
        first_var_index = 8
        print("testing with track as the first index")
        
    print("checkpoint 5")
    # else:
        # first_var = "energy"
        # first_var_index = 0
        # print("only supports energy and zenith right now! Please choose one of those. Defaulting to energy")
        # print("testing with energy as the first index")

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

    if compare_reco:
        reco_test_use = f['reco_test'][:]
    f.close
    del f
    print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_use.shape, "checkpoint 8")

    #mask_energy_train = np.logical_and(np.array(Y_test_use[:,0])>min_energy/max_energy,np.array(Y_test_use[:,0])<1.0)
    #Y_test_use = np.array(Y_test_use)[mask_energy_train]
    #X_test_DC_use = np.array(X_test_DC_use)[mask_energy_train]
    #X_test_IC_use = np.array(X_test_IC_use)[mask_energy_train]
    #if compare_reco:
    #    reco_test_use = np.array(reco_test_use)[mask_energy_train]
    # if compare_reco:
        # print("TRANSFORMING ZENITH TO COS(ZENITH)")
        # reco_test_use[:,1] = np.cos(reco_test_use[:,1])
    # if mask_zenith:
        # print("MANUALLY GETTING RID OF HALF THE EVENTS (UPGOING/DOWNGOING ONLY)")
        # if z_values == "gt0":
            # maxvals = [max_energy, 1., 0.]
            # minvals = [min_energy, 0., 0.]
            # mask_zenith = np.array(Y_test_use[:,1])>0.0
        # if z_values == "lt0":
            # maxvals = [max_energy, 0., 0.]
            # minvals = [min_energy, -1., 0.]
            # mask_zenith = np.array(Y_test_use[:,1])<0.0
        # Y_test_use = Y_test_use[mask_zenith]
        # X_test_DC_use = X_test_DC_use[mask_zenith]
        # X_test_IC_use = X_test_IC_use[mask_zenith]
        # if compare_reco:
            # reco_test_use = reco_test_use[mask_zenith]

    #Make network and load model
    from keras.optimizers import SGD
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint

    from cnn_model import make_network
    model_DC = make_network(X_test_DC_use,X_test_IC_use,output_variables,DC_drop_value,IC_drop_value,connected_drop_value)
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

    # Run prediction
    #Y_test_compare = Y_test_use[:,first_var_index]
    #score = model_DC.evaluate([X_test_DC_use,X_test_IC_use], Y_test_compare, batch_size=256)
    #print("Evaluate:",score)

    t0 = time.time()

    print("checkpoint 11")

    Y_test_predicted = model_DC.predict([X_test_DC_use,X_test_IC_use])

    print("checkpoint 12")

    t1 = time.time()

    print("checkpoint 13")

    print("Saving output file: %s/prediction_values%s.hdf5"%(save_folder_name,epoch))
    f = h5py.File("%s/prediction_values%s.hdf5"%(save_folder_name,epoch))
    f.create_dataset("Y_test_use", data=Y_test_use)
    f.create_dataset("Y_predicted", data=Y_test_predicted)
    f.close()
    print("This took me %f seconds for %i events"%(((t1-t0)),Y_test_predicted.shape[0]))
    print(X_test_DC_use.shape,X_test_IC_use.shape,Y_test_predicted.shape,Y_test_use.shape)

