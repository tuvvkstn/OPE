# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:52:29 2015

@author: dhbk
"""
import sys, os, shutil
mypath = ['./common', './New1ML-OPE','./New2ML-OPE','./New1Online-OPE',
          './New2Online-OPE', './New1Streaming-OPE','./New2Streaming-OPE']
for temp in mypath:
    sys.path.insert(0, temp)
import utilities

import run_New1ML_OPE
import run_New2ML_OPE
import run_New1Online_OPE
import run_New2Online_OPE
import run_New1Streaming_OPE
import run_New2Streaming_OPE

def main():
    # Check input
    if len(sys.argv) != 6:
        print"usage: python run.py [method name] [train file] [setting file] [model folder] [test data folder]"
        exit()
    # Get environment variables
    method_name = sys.argv[1]
    train_file = sys.argv[2]
    setting_file = sys.argv[3]
    model_folder = sys.argv[4]
    test_data_folder = sys.argv[5]
    tops = 10#int(sys.argv[5])
    # Create model folder if it doesn't exist
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.makedirs(model_folder)
    # Read settings
    print'reading setting ...'
    settings = utilities.read_setting(setting_file)
    # Read data for computing perplexities
    print'read data for computing perplexities ...'
    test_data = utilities.read_data_for_perpl(test_data_folder)
    # Check method and run algorithm
    methods = ['new1ml-ope','new2ml-ope', 'new1online-ope','new2online-ope', 'new1streaming-ope', 'new2streaming-ope']
    method_low = method_name.lower()    
    if method_low == 'new1ml-ope':        
        run_new1_mlope = run_New1ML_OPE.runNew1MLOPE(train_file, settings, model_folder, test_data, tops)
        run_new1_mlope.run()
    elif method_low == 'new2ml-ope':
        run_new2_mlope = run_New2ML_OPE.runNew2MLOPE(train_file, settings, model_folder, test_data, tops)
        run_new2_mlope.run()
    elif method_low == 'new1online-ope':
        run_new1_onlineope = run_New1Online_OPE.runNew1OnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_new1_onlineope.run()
    elif method_low == 'new2online-ope':
        run_new2onlineope = run_New2Online_OPE.runNew2OnlineOPE(train_file, settings, model_folder, test_data, tops)
        run_new2onlineope.run()
        
    elif method_low == 'new1streaming-ope':
        run_new1_streamingope = run_New1Streaming_OPE.runNew1StreamingOPE(train_file, settings, model_folder, test_data, tops)
        run_new1_streamingope.run()
    elif method_low == 'new2streaming-ope':
        run_new2streamingope = run_New2Streaming_OPE.runNew2StreamingOPE(train_file, settings, model_folder, test_data, tops)
        run_new2streamingope.run()
    else:
        print '\ninput wrong method name: %s\n'%(method_name)
        print 'list of methods:'
        for method in methods:
            print '\t\t%s'%(method)
        exit()
        
if __name__ == '__main__':
    main()