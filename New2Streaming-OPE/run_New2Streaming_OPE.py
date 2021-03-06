#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import New2Streaming_OPE

sys.path.insert(0, '../common')
import utilities

class runNew2StreamingOPE:
    
    def __init__(self, train_file, settings, model_folder, test_data, tops):
        self.train_file = train_file
        self.settings = settings
        self.model_folder = model_folder
        self.test_data = test_data
        self.tops = tops
        
    def run(self):
        # Initialize the algorithm
        print'initialize the algorithm ...'
        new2streaming_ope = New2Streaming_OPE.New2StreamingOPE(self.settings['num_terms'], self.settings['num_topics'], self.settings['alpha'], 
                                                   self.settings['eta'], self.settings['iter_infer'])
        # Start
        print'start!!!'
        i = 0
        while i < self.settings['iter_train']:
            i += 1
            print'\n***iter_train:%d***\n'%(i)
            datafp = open(self.train_file, 'r')
            j = 0
            while True:
                j += 1
                (wordids, wordcts) = utilities.read_minibatch_list_frequencies(datafp, self.settings['batch_size'])
                # Stop condition
                if len(wordids) == 0:
                    break
                # 
                print'---num_minibatch:%d---'%(j)
                (time_e, time_m, theta) = new2streaming_ope.static_online(wordids, wordcts)
                # Compute sparsity
                sparsity = utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
                # Compute perplexities
                LD2 = utilities.compute_perplexities_vb(new2streaming_ope._lambda, self.settings['alpha'], self.settings['eta'], 
                                                        self.settings['iter_infer'], self.test_data)
                # Search top words of each topics
                list_tops = utilities.list_top(new2streaming_ope._lambda, self.tops)
                # Write files
                utilities.write_file(i, j, new2streaming_ope._lambda, time_e, time_m, theta, sparsity, LD2, list_tops, self.tops, 
                                     self.model_folder)
            datafp.close()
        # Write settings
        print'write setting ...'
        file_name = '%s/setting.txt'%(self.model_folder)
        utilities.write_setting(self.settings, file_name)
        # Write final model to file
        print'write final model ...'
        file_name = '%s/beta_final.dat'%(self.model_folder)
        utilities.write_topics(new2streaming_ope._lambda, file_name)
        # Finish
        print'done!!!'
