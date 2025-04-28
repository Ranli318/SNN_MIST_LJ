# SNN_library.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core SNN Model Library Using Brian2
Provides: Initialization, Training, Testing, Evaluation, and Visualization
"""

# --------------------------------------------------- Libraries ---------------------------------------------------
import os
import random
import pickle
import warnings
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from brian2 import *
from sklearn.metrics import confusion_matrix
from tqdm import tqdm, tqdm_gui

warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------------------------------- Model Class ---------------------------------------------------

class Model:
    def __init__(self, debug=False, **kwargs):
        self.setup_defaults()
        self.override_params(**kwargs)
        self.setup_network(debug)

    def setup_defaults(self):
        self.size_img = [28, 28]
        self.n_input = 784
        self.n_e = 400
        self.n_i = 400

        self.gl = 0.0001 * uS
        self.tau_m_ex = 300 * ms
        self.tau_m_in = 10 * ms
        self.tau_e = 6 * ms
        self.tau_i = 10 * ms
        self.tau_t_h = 1e7 * ms

        self.learning_rate = 1
        self.train_items = 59000
        self.assign_items = 500
        self.eval_items = 500
        self.num_epochs = 5

        self.v_rest_e = -65. * mV
        self.v_reset_e = -65. * mV
        self.v_thresh_e = -50. * mV
        self.v_thresh_e_inc = 0.05 * mV

        self.v_rest_i = -65. * mV
        self.v_reset_i = -43. * mV
        self.v_thresh_i = -38. * mV

        self.refrac_exc = 55 * ms
        self.refrac_inh = 10 * ms

        self.taupre = 20.10 * ms
        self.taupost = 20.70 * ms
        self.dApre = 117520 * uS
        self.dApost = 71840 * uS

        self.gmin = 0.1 * uS
        self.gmax = 100 * uS
        self.g_var = 0.1

        self.check_min_spikes = 5
        self.run = 'nominal'
        self.MC_iteration = 0
        self.custom_folder = False

    def override_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if not self.custom_folder:
            self.results_folder = os.path.join('..', '..', 'results', f'SNN_{self.n_input}x{self.n_e}x{self.n_i}')
        os.makedirs(self.results_folder, exist_ok=True)

    def setup_network(self, debug):
        app = {}
        defaultclock.dt = 0.1 * ms

        app['PG'] = PoissonGroup(self.n_input, rates=np.zeros(self.n_input) * Hz, name='PG')

        exc_eqs = '''
            dv/dt = (ge*(0*mV-v)/(20*uS) + gi*(-100*mV-v)/(0.2*uS) + (v_rest_e-v)) / tau_m_ex : volt
            dge/dt = -ge / tau_e : siemens
            dgi/dt = -gi / tau_i : siemens
            dv_th/dt = -v_th / tau_t_h : volt
        '''
        reset_exc = '''
            v = v_reset_e
            v_th += v_thresh_e_inc
        '''
        app['EG'] = NeuronGroup(self.n_e, exc_eqs, threshold='v>v_thresh_e+v_th', method='euler', reset=reset_exc, refractory=self.refrac_exc, name='EG')
        app['EG'].v = self.v_rest_e - 20. * mV

        if debug:
            app['ESP'] = SpikeMonitor(app['EG'])
            app['ESM'] = StateMonitor(app['EG'], ['v'], record=True)
            app['ERM'] = PopulationRateMonitor(app['EG'])

        inh_eqs = '''
            dv/dt = (ge*(0*mV-v)/(0.5*uS) + gi*(-85*mV-v)/(10*uS) + (v_rest_i-v)) / tau_m_in : volt
            dge/dt = -ge / tau_e : siemens
            dgi/dt = -gi / tau_i : siemens
        '''
        app['IG'] = NeuronGroup(self.n_i, inh_eqs, threshold='v>v_thresh_i', method='euler', reset='v=v_reset_i', refractory=self.refrac_inh, name='IG')
        app['IG'].v = self.v_rest_i - 20. * mV

        if debug:
            app['ISP'] = SpikeMonitor(app['IG'])
            app['ISM'] = StateMonitor(app['IG'], ['v'], record=True)
            app['IRM'] = PopulationRateMonitor(app['IG'])

        stdp_eqs = '''
            w : siemens
            lr : 1 (shared)
            dApre/dt = -Apre / taupre : siemens (event-driven)
            dApost/dt = -Apost / taupost : siemens (event-driven)
        '''
        pre_eq = '''
            ge_post += w
            Apre += dApre
            w = clip(w - lr*Apost, gmin, gmax)
        '''
        post_eq = '''
            Apost += dApost
            w = clip(w + lr*Apre, gmin, gmax)
        '''

        app['S1'] = Synapses(app['PG'], app['EG'], model=stdp_eqs, on_pre=pre_eq, on_post=post_eq, name='S1')
        app['S1'].connect()
        app['S1'].w = 'rand() * gmax * 0.2'
        app['S1'].lr = 1

        app['S2'] = Synapses(app['EG'], app['IG'], model='w:siemens', on_pre='ge += w')
        app['S2'].connect('i==j')
        app['S2'].w = 1.2 * uS

        app['S3'] = Synapses(app['IG'], app['EG'], model='w:siemens', on_pre='gi += w')
        app['S3'].connect('i!=j')
        app['S3'].w = 1.5 * uS

        self.net = Network(app.values())
        self.net.run(0 * ms)

    def __getitem__(self, key):
        return self.net[key]

############ ############ ############ ############ ############ ############ ############ ############ ############ ############ ############ ############ ############ Still Testing
 if self.custom_folder==0:
            self.results_folder = os.path.join('..','..','results','SNN_'+str(self.size_img[0]*self.size_img[1])+'x'+str(self.n_e)+'x'+str(self.n_i)+'')                               
                
        os.makedirs(self.results_folder,exist_ok=True)
        
        n_input = self.n_input
        n_e = self.n_e
        n_i = self.n_i 
        gl = self.gl
        tau_m_ex = self.tau_m_ex
        tau_m_in = self.tau_m_in
        tau_e = self.tau_e
        tau_i = self.tau_i
        tau_t_h = self.tau_t_h

        v_rest_e = self.v_rest_e
        v_reset_e = self.v_reset_e
        v_thresh_e = self.v_thresh_e 
        v_thresh_e_inc = self.v_thresh_e_inc 
        
        v_rest_i = self.v_rest_i 
        v_reset_i = self.v_reset_i 
        v_thresh_i = self.v_thresh_i
        
        taupre = random.gauss(self.taupre,self.g_var*self.taupre)
        taupost = random.gauss(self.taupost,self.g_var*self.taupost)
        dApre = random.gauss(self.dApre,self.g_var*self.dApre)
        dApost = random.gauss(self.dApost,self.g_var*self.dApost)
        gmin = self.gmin
        gmax = self.gmax        
        
        defaultclock.dt=0.1*ms
        
        # input images as rate encoded Poisson generators
        app['PG'] = PoissonGroup(n_input, rates=np.zeros(n_input)*Hz, name='PG')

        # excitatory group
        neuron_e = '''
            dv/dt = (ge*(0*mV-v)/(20*uS) + gi*(-100*mV-v)/(0.2*uS) + (v_rest_e-v)) / tau_m_ex : volt
            dge/dt = -ge / tau_e : siemens
            dgi/dt = -gi / tau_i: siemens
            dv_th/dt = -v_th / tau_t_h : volt # adaptive threshold
            '''
        reset_eqs_e = '''
            v=v_reset_e
            v_th+=v_thresh_e_inc
        '''
            
        #with 50ms for the refractory period it worked ok for 100 and 400 neurons (o seo creo porque no vi el spike count de esos casos)
        app['EG'] = NeuronGroup(n_e, neuron_e, threshold='v>v_thresh_e+v_th', method='euler', refractory=self.refrac_exc, reset=reset_eqs_e, name='EG')
        app['EG'].v = v_rest_e - 20.*mV
        
        if (debug):
            app['ESP'] = SpikeMonitor(app['EG'], name='ESP')
            app['ESM'] = StateMonitor(app['EG'], ['v'], record=True, name='ESM')
            app['ERM'] = PopulationRateMonitor(app['EG'], name='ERM')
        
        # ibhibitory group
        neuron_i = '''
            dv/dt = (ge*(0*mV-v)/(0.5*uS) + gi*(-85*mV-v)/(10*uS) + (v_rest_i-v)) / tau_m_in : volt
            dge/dt = -ge / tau_e : siemens
            dgi/dt = -gi / tau_i : siemens
            '''
            
        # Apre and Apost - presynaptic and postsynaptic traces, lr - learning rate
        stdp='''
            w :                             siemens
            lr :                            1 (shared)
            dApre/dt = -Apre / taupre :     siemens (event-driven)
            dApost/dt = -Apost / taupost :  siemens (event-driven)
            '''
        
        pre='''
            ge_post += w
            Apre += dApre
            w = clip(w - lr*Apost, gmin, gmax)
            '''
            #Apre += (dApre+randn()*dApre/20)
        post='''
            Apost += dApost
            w = clip(w + lr*Apre, gmin, gmax)
            '''
            #Apost += (dApost+randn()*dApost/20)

        #with 10ms for the refractory period it worked ok for 100 and 400 neurons (o seo creo porque no vi el spike count de esos casos)
        app['IG'] = NeuronGroup(n_i, neuron_i, threshold='v>v_thresh_i', method='euler', refractory=self.refrac_inh, reset='v=v_reset_i', name='IG')
        app['IG'].v = v_rest_i - 20.*mV

        if (debug):
            app['ISP'] = SpikeMonitor(app['IG'], name='ISP')
            app['ISM'] = StateMonitor(app['IG'], ['v'], record=True, name='ISM')
            app['IRM'] = PopulationRateMonitor(app['IG'], name='IRM')
        
        # poisson generators one-to-all excitatory neurons with plastic connections 
        app['S1'] = Synapses(app['PG'], app['EG'], stdp, on_pre=pre, on_post=post, name='S1')
        app['S1'].connect()
        app['S1'].w = 'rand()*gmax*0.2' # random weights initialisation
        app['S1'].lr = 1 # enable stdp        
        
        if (debug):
            # some synapses
            app['S1M'] = StateMonitor(app['S1'], ['w', 'Apre', 'Apost'], record=app['S1'][380,:25], name='S1M') 
       
        # excitatory neurons one-to-one inhibitory neurons
        app['S2'] = Synapses(app['EG'], app['IG'], 'w : siemens', on_pre='ge += w', name='S2')
        app['S2'].connect('i==j')
        #app['S2'].delay = 'rand()*10*ms'
        app['S2'].w = 1.2*uS # very strong fixed weights to ensure corresponding inhibitory neuron will always fire

        # inhibitory neurons one-to-all-except-one excitatory neurons
        app['S3'] = Synapses(app['IG'], app['EG'], 'w : siemens', on_pre='gi += w', name='S3')
        app['S3'].connect('i!=j')
        #app['S3'].delay = 'rand()*5*ms'
        app['S3'].w = 1.5*uS # weights are selected in such a way as to maintain a balance between excitation and ibhibition
            
        self.net = Network(app.values())
        self.net.run(0*second)
        
    def __getitem__(self, key):
        return self.net[key]
    
    def create_var_dict(self, **kwargs):

        var_dict = {'size_img': self.size_img,
                    'n_input': self.n_input,
                    'n_e': self.n_e,
                    'n_i': self.n_i,
                    'tau_m_ex': self.tau_m_ex,
                    'tau_m_in': self.tau_m_in,
                    'tau_e': self.tau_e,
                    'tau_i': self.tau_i,
                    'tau_t_h': self.tau_t_h,                                       
                    'v_rest_e': self.v_rest_e,
                    'v_reset_e': self.v_reset_e,
                    'v_thresh_e': self.v_thresh_e,
                    'v_thresh_e_inc': self.v_thresh_e_inc,
                    'v_rest_i': self.v_rest_i,
                    'v_reset_i': self.v_reset_i,
                    'v_thresh_i': self.v_thresh_i,
                    'refrac_exc': self.refrac_exc,
                    'refrac_inh': self.refrac_inh,                    
                    'taupre': self.taupre,
                    'taupost': self.taupost,
                    'dApre': self.dApre,
                    'dApost': self.dApost,
                    'gmin': self.gmin,
                    'gl': self.gl,
                    'gmax': self.gmax,
                    'g_var': self.g_var
                    }        
        return var_dict
       
    def train(self,
              train_dataset,
              gen_plot='yes',
              restore='yes',
              store='yes', 
              epoch=1, 
              **kwargs):        

        '''
        Feed train set to SNN with STDP
        Freeze STDP
        Feed train set to SNN again and collect generated features
        Train RandomForest on the top of these features and labels provided
        Feed test set to SNN and collect new features
        Predict labels with RandomForest and calculate accuacy score
        '''
        pid = 0
        str2dump = ''            

        for key, value in kwargs.items():
            if key=='learning_rate':
                self.learning_rate=value
            elif key=='results_folder':
                self.results_folder=value
            elif key=='train_items':
                self.train_items=value
            elif key=='assign_items':
                self.assign_items=value
            elif key=='eval_items':
                self.eval_items=value
            elif key=='check_min_spikes':
                self.check_min_spikes=value   
            elif key=='pid':
                pid=value
            elif key=='run':
                self.run=value       
            elif key=='MC_iteration':
                self.MC_iteration=value       
                
        tqdm_text = 'MC it. '+str(pid)+''                
                
        n_input = self.n_input
        n_e = self.n_e
        n_i = self.n_i 
        gl = self.gl
        tau_m_ex = self.tau_m_ex
        tau_m_in = self.tau_m_in
        tau_e = self.tau_e
        tau_i = self.tau_i
        tau_t_h = self.tau_t_h

        v_rest_e = self.v_rest_e
        v_reset_e = self.v_reset_e
        v_thresh_e = self.v_thresh_e 
        v_thresh_e_inc = self.v_thresh_e_inc 
        
        v_rest_i = self.v_rest_i 
        v_reset_i = self.v_reset_i 
        v_thresh_i = self.v_thresh_i
        
        taupre = random.gauss(self.taupre,self.g_var*self.taupre)
        taupost = random.gauss(self.taupost,self.g_var*self.taupost)
        dApre = random.gauss(self.dApre,self.g_var*self.dApre)
        dApost = random.gauss(self.dApost,self.g_var*self.dApost)
        gmin = self.gmin
        gmax = self.gmax                            
                
        X=train_dataset[:self.train_items]
        
        #print('->Training network model '+str(self.n_input)+'x'+str(self.n_e)+'x'+str(self.n_i)+'')   
        
        new_instance_variables=self.create_var_dict()

        list_saved_runs = os.listdir(self.results_folder)

        curr_run_folder = ''
        if len(list_saved_runs)!=0:
            for run_folder_idx, run_folder in enumerate(list_saved_runs):
                if os.path.exists(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl")):
                    aux_file = open(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl"), "rb")
                    SNN_variable_dict = pickle.load(aux_file)    
                    
                    if new_instance_variables==SNN_variable_dict:
                        curr_run_folder = run_folder
                        break                        
            
        if curr_run_folder == '':
            curr_run_folder = 'run_'+str(len(list_saved_runs)+1)+''
            os.makedirs(os.path.join(self.results_folder,curr_run_folder), exist_ok=True)
            aux_file = open(os.path.join(self.results_folder,curr_run_folder,"SNN_settings.pkl"), "wb")
            pickle.dump(new_instance_variables, aux_file)
            aux_file.close()         
        
        
        if self.run == 'nominal':
            figures_folder_str = os.path.join(self.results_folder, curr_run_folder, 'nominal','figs')            
            w_inp_exc_folder_str = os.path.join(self.results_folder, curr_run_folder, 'nominal','W_inp-exc')
            log_training_proc_str = os.path.join(self.results_folder, curr_run_folder, 'nominal','history.log')   
        elif self.run == 'monte-carlo':
            figures_folder_str = os.path.join(self.results_folder, curr_run_folder, 'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','figs')            
            w_inp_exc_folder_str = os.path.join(self.results_folder, curr_run_folder, 'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','W_inp-exc')
            log_training_proc_str = os.path.join(self.results_folder, curr_run_folder, 'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','history.log')
            
        os.makedirs(figures_folder_str,exist_ok=True)
        os.makedirs(w_inp_exc_folder_str,exist_ok=True)

        if os.path.exists(os.path.join(self.results_folder, curr_run_folder, 'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','MC_run_settings.pkl')):
            aux_file = open(os.path.join(self.results_folder, curr_run_folder, 'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','MC_run_settings.pkl'), "rb")
            MC_run_settings = pickle.load(aux_file)           
            locals().update(MC_run_settings)
        else:
            aux_file = open(os.path.join(self.results_folder, curr_run_folder, 'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','MC_run_settings.pkl'), "wb")
            pickle.dump({'taupre': taupre, 'taupost': taupost, 'dApre': dApre, 'dApost': dApost}, aux_file)
            aux_file.close()              
            
        self.net['S1'].lr = self.learning_rate # stdp on
        idx2restore=0
        epoch2restore=0
        if restore=='yes':
            for ep in range(epoch):
                for idx in range(len(X)+1):
                    state_str='train_'+str(idx)+'_img_'+str(ep+1)+'_epochs'
                    results_file=os.path.join(w_inp_exc_folder_str,''+state_str+'.b2')
                    if os.path.exists(results_file):
                        idx2restore=idx
                        epoch2restore=ep
                        self.net.restore(state_str, results_file)

        for ep in range(epoch2restore, epoch, 1):
            #with alive_bar(len(X), theme='smooth') as bar:
            with tqdm(total=len(X), desc=tqdm_text, position=pid) as pbar:
            #with tqdm.gui.tqdm(total=len(X), desc=tqdm_text, position=pid+1) as pbar:
                for idx in range(len(X)):
                    #bar()
                    pbar.update(1)
                    if idx>idx2restore:
                        max_spike_count=-1
                        counter_repeat=0
                        while max_spike_count<self.check_min_spikes and counter_repeat<=10:
                            # rate monitor to count spikes
                            mon = SpikeMonitor(self.net['EG'], name='RM', record=False)
                            self.net.add(mon)
                            
                            # active mode
                            self.net['PG'].rates = (X[idx].ravel('F')*(1+counter_repeat/2))*Hz
                            self.net.run(0.35*second)
    
                            # spikes per neuron foreach image                        
                            spike_count=np.array(mon.count, dtype=int8)
                            max_spike_count=np.sum(spike_count)
                            
                            # passive mode
                            self.net['PG'].rates = np.zeros(n_input)*Hz
                            self.net.run(0.15*second)
                            """
                            print(Fore.RED, end ="")
                            print('Spike count='+str(max_spike_count)+', Freq.='+str(255/4*(1+counter_repeat/2))+' Hz.', end ="")
                            print(Style.RESET_ALL)
                            """
                            
                            str2dump = str2dump + '-> Image '+str(idx)+': Spike count='+str(max_spike_count)+', Freq.='+str(255/4*(1+counter_repeat/2))+' Hz.\n'
                            
                            counter_repeat=counter_repeat+1
                            self.net.remove(self.net['RM'])
                            
                        state_str='train_'+str(idx)+'_img_'+str(ep+1)+'_epochs'
                        results_file=os.path.join(w_inp_exc_folder_str,''+state_str+'.b2')
    
                        if (mod(idx,1000)==0 or idx==1 or idx==100 or idx ==500) and store=='yes':
                            self.net.store(state_str, results_file)
                            if gen_plot=='yes':
                                self.plot_w_matrix_evolution(num_images=idx, interval=100)                            

                        if mod(idx,100)==0:
                            log_training_proc = open(log_training_proc_str, "a+")
                            log_training_proc.write(str2dump)
                            log_training_proc.close()   
                            str2dump = ''
                            
#                           self.net.restore('Train_'+str(idx)+'_img_'+str(ep+1)+'_epochs', 'train_'+str(idx)+'_img_'+str(ep+1)+'_epochs.b2')
    
                            """
                            if ep==0:
                                if idx in [100, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10e3, 20e3, 30e3, 40e3, 50e3, 59999]:
                                    weight_copy=np.copy(self['S1'].w)
                                    #self.plot_w_matrix_single(idx, ep+1)
                                    self.plot_w_matrix_evolution(num_images=idx, interval=500)
                            else:
                                if idx in [10e3, 20e3, 30e3, 40e3, 50e3, len(X)-1]:
                                    weight_copy=np.copy(self['S1'].w)
                                    self.plot_w_matrix_single(idx, ep+1)   
                            """
            idx2restore=0                   
        #plt.show()

    def test(self, 
             x_train, 
             y_train, 
             x_test, 
             y_test, 
             **kwargs):
        '''
        Feed train set to SNN with STDP
        Freeze STDP
        Feed train set to SNN again and collect generated features
        Train RandomForest on the top of these features and labels provided
        Feed test set to SNN and collect new features
        Predict labels with RandomForest and calculate accuacy score
        '''
        num_epochs=1, 
        overwrite_training_spikes='no', 
        overwrite_testing_spikes='no', 
        idx2restore_input=0, 
        epoch2restore_input=0, 
        idx_interval=10e3, 
        figure_plt=[], 
        show_accuracy_plot='no', 
        color_plot='r', 
        plot_cm='no',
        pid = 0
        str2dump = ''            

        for key, value in kwargs.items():
            if key=='learning_rate':
                self.learning_rate=value
            elif key=='results_folder':
                self.results_folder=value
            elif key=='train_items':
                self.train_items=value
            elif key=='assign_items':
                self.assign_items=value
            elif key=='eval_items':
                self.eval_items=value
            elif key=='check_min_spikes':
                self.check_min_spikes=value  
            elif key=='num_epochs':
                num_epochs=value
            elif key=='overwrite_training_spikes':
                overwrite_training_spikes=value
            elif key=='overwrite_testing_spikes':
                overwrite_testing_spikes=value
            elif key=='idx2restore_input':
                idx2restore_input=value
            elif key=='epoch2restore_input':
                epoch2restore_input=value                  
            elif key=='idx_interval':
                idx_interval=value
            elif key=='figure_plt':
                figure_plt=value
            elif key=='show_accuracy_plot':
                show_accuracy_plot=value
            elif key=='color_plot':
                color_plot=value
            elif key=='plot_cm':
                plot_cm=value                      
            elif key=='pid':
                pid=value
            elif key=='run':
                self.run=value       
            elif key=='MC_iteration':
                self.MC_iteration=value       

        tqdm_text = "#" + "{}".format(pid).zfill(3)      

        n_input = self.n_input
        n_e = self.n_e
        n_i = self.n_i 
        gl = self.gl
        tau_m_ex = self.tau_m_ex
        tau_m_in = self.tau_m_in
        tau_e = self.tau_e
        tau_i = self.tau_i
        tau_t_h = self.tau_t_h

        v_rest_e = self.v_rest_e
        v_reset_e = self.v_reset_e
        v_thresh_e = self.v_thresh_e 
        v_thresh_e_inc = self.v_thresh_e_inc 
        
        v_rest_i = self.v_rest_i 
        v_reset_i = self.v_reset_i 
        v_thresh_i = self.v_thresh_i
        
        taupre = self.taupre
        taupost = self.taupost
        dApre = self.dApre
        dApost = self.dApost
        gmin = self.gmin
        gmax = self.gmax  

        new_instance_variables=self.create_var_dict()

        list_saved_runs = os.listdir(self.results_folder)

        curr_run_folder = ''
        if len(list_saved_runs)!=0:
            for run_folder_idx, run_folder in enumerate(list_saved_runs):
                if os.path.exists(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl")):
                    aux_file = open(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl"), "rb")
                    SNN_variable_dict = pickle.load(aux_file)
                
                    if new_instance_variables==SNN_variable_dict:
                        curr_run_folder = run_folder
                        break                        
            
        if curr_run_folder == '':
            printf(print('->Error, this network does not exist, and it thereby it was not trained, exiting', end='')  )
            return 1


        if self.run == 'nominal':
            figures_folder_str = os.path.join(self.results_folder,curr_run_folder,'nominal','figs')            
            w_inp_exc_folder_str = os.path.join(self.results_folder,curr_run_folder,'nominal','W_inp-exc')
            train_spikes_str = os.path.join(self.results_folder,curr_run_folder,'nominal','recorded_spikes','train')
            test_spikes_str = os.path.join(self.results_folder,curr_run_folder,'nominal','recorded_spikes','test')            
        elif self.run == 'monte-carlo':
            figures_folder_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','figs')            
            w_inp_exc_folder_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','W_inp-exc')
            train_spikes_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','recorded_spikes','train')
            test_spikes_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','recorded_spikes','test')   
            
        os.makedirs(figures_folder_str,exist_ok=True)
        os.makedirs(w_inp_exc_folder_str,exist_ok=True)
        os.makedirs(test_spikes_str,exist_ok=True)
        os.makedirs(train_spikes_str,exist_ok=True)
            
        if idx_interval>0:
            epoch2restore=range(0, epoch2restore_input, 1)
            idx2restore=range(0, idx2restore_input, idx_interval)
        else:
            epoch2restore=[epoch2restore_input-1]
            idx2restore=[idx2restore_input]
            
        accuracy_train_vec=[]
        accuracy_test_vec=[]
        image_vec=[]

        #print("\n" * (pid*5))
        print('Testing Monte-Carlo iteration '+str(pid)+'', end='\n')
        with tqdm_gui(total=len(idx2restore)*len(epoch2restore), desc='Testing Monte-Carlo iteration '+str(pid)+'', leave=False) as pbar:       
            for epoch2restore_idx, epoch2restore_val in enumerate(epoch2restore):
                for idx2restore_idx, idx2restore_val in enumerate(idx2restore):    

                    pbar.update(1)                    
                    #print('->Searching trained network...', end='')            
        
                    state_str='train_'+str(idx2restore_val)+'_img_'+str(epoch2restore_val+1)+'_epochs'
                    results_file=os.path.join(w_inp_exc_folder_str,''+state_str+'.b2')
                    if os.path.exists(results_file):
                        #idx2restore=idx2restore_val
                        #epoch2restore=epoch2restore_val
                        self.net.restore(state_str, results_file)
                        #print('done! ('+str(epoch2restore_val+1)+' epochs, '+str(idx2restore_val)+' images)')
                    
                        rec_spikes_filename_train=os.path.join(train_spikes_str,'SNN_'+str(epoch2restore_val+1)+'_eps_'+str(idx2restore_val)+'_imgs_RFC_'+str(self.assign_items)+'_imgs.npy')
                        rec_spikes_filename_test=os.path.join(test_spikes_str,'SNN_'+str(epoch2restore_val+1)+'_eps_'+str(idx2restore_val)+'_imgs_RFC_'+str(self.eval_items)+'_imgs.npy')                
                        
                        if overwrite_training_spikes=='yes' or os.path.exists(rec_spikes_filename_train)==False:
                            #print('->Generating spike train for Random Forest training ('+str(self.assign_items)+' images)...', end='\n')            
                            f_train = self.evaluate(x_train[:self.assign_items], progressbar_pos = pid, desc = 'MC it. '+str(pid)+' Generating spike train for training classifier  - '+str(idx2restore_val)+'')
                            np.save(rec_spikes_filename_train, f_train, allow_pickle=True)
                            #print('->Done!', end='\n')
                        else:
                            #print('->Loading spike train for Random Forest training ('+str(self.assign_items)+' images)...', end='\n')      
                            f_train = np.load(rec_spikes_filename_train, allow_pickle=True)
                            #print('->Done!', end='\n')
                            
                        if overwrite_testing_spikes=='yes' or os.path.exists(rec_spikes_filename_test)==False:
                            #print('->Generating spike train for Random Forest testing ('+str(self.eval_items)+' images)...', end='\n')  
                            f_test = self.evaluate(x_test[:self.eval_items], progressbar_pos = pid, desc = 'MC it. '+str(pid)+' Generating spike train for testing classifier  - '+str(idx2restore_val)+'')
                            np.save(rec_spikes_filename_test, f_test, allow_pickle=True)
                            #print('->Done!', end='\n')
                        else:
                            #print('->Loading spike train for Random Forest testing ('+str(eval_items)+' images)...', end='')  
                            f_test = np.load(rec_spikes_filename_test, allow_pickle=True)
                            #print('->Done!', end='\n')

                        #print('###############################################################################################', end='\n')                                
   
                        
                        """ 
                        accuracy_train_vec=np.append(accuracy_train_vec,accuracy_train)
                        accuracy_test_vec=np.append(accuracy_test_vec,accuracy_test)
                        image_vec=np.append(image_vec, idx2restore_val+epoch2restore_val*60000)
                        """

    def plot_test_metrics(self, 
                          x_train, 
                          y_train, 
                          x_test, 
                          y_test, 
                          **kwargs):
        '''
        Feed train set to SNN with STDP
        Freeze STDP
        Feed train set to SNN again and collect generated features
        Train RandomForest on the top of these features and labels provided
        Feed test set to SNN and collect new features
        Predict labels with RandomForest and calculate accuacy score
        '''
        num_epochs=1 
        overwrite_training_spikes='no' 
        overwrite_testing_spikes='no' 
        idx2restore_input=0
        epoch2restore_input=0 
        idx_interval=1e3 
        figure_plt=[] 
        show_accuracy_plot='no'
        color_plot='r'
        plot_cm='yes'
        pid = 0
        str2dump = ''            

        for key, value in kwargs.items():
            if key=='MC_iterations':
                MC_iterations=value
            if key=='idx2restore_input':
                idx2restore_input=value
            if key=='epoch2restore_input':
                epoch2restore_input=value                
            if key=='idx_interval':
                idx_interval=value
            if key=='show_accuracy_plot':
                show_accuracy_plot=value

                
        tqdm_text = "#" + "{}".format(pid).zfill(3)      

        n_input = self.n_input
        n_e = self.n_e
        n_i = self.n_i 
        gl = self.gl
        tau_m_ex = self.tau_m_ex
        tau_m_in = self.tau_m_in
        tau_e = self.tau_e
        tau_i = self.tau_i
        tau_t_h = self.tau_t_h

        v_rest_e = self.v_rest_e
        v_reset_e = self.v_reset_e
        v_thresh_e = self.v_thresh_e 
        v_thresh_e_inc = self.v_thresh_e_inc 
        
        v_rest_i = self.v_rest_i 
        v_reset_i = self.v_reset_i 
        v_thresh_i = self.v_thresh_i
        
        taupre = self.taupre
        taupost = self.taupost
        dApre = self.dApre
        dApost = self.dApost
        gmin = self.gmin
        gmax = self.gmax  

        new_instance_variables=self.create_var_dict()

        list_saved_runs = os.listdir(self.results_folder)

        curr_run_folder = ''
        if len(list_saved_runs)!=0:
            for run_folder_idx, run_folder in enumerate(list_saved_runs):
                if os.path.exists(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl")):
                    aux_file = open(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl"), "rb")
                    SNN_variable_dict = pickle.load(aux_file)
                
                    if new_instance_variables==SNN_variable_dict:
                        curr_run_folder = run_folder
                        break                        
            
        if curr_run_folder == '':
            printf(print('->Error, this network does not exist, and it thereby it was not trained, exiting', end='')  )
            return 1

        if idx_interval>0:
            epoch2restore=range(0, epoch2restore_input, 1)
            idx2restore=range(0, idx2restore_input, int(idx_interval))
        else:
            epoch2restore=[epoch2restore_input-1]
            idx2restore=[idx2restore_input]
        
        acc_train_mat = np.zeros((len(idx2restore),len(range(MC_iterations))))
        acc_test_mat = np.zeros((len(idx2restore),len(range(MC_iterations))))
        image_vec = np.zeros((len(idx2restore),1))        

        with tqdm_gui(total=len(range(MC_iterations)), desc='Reading data form '+str(MC_iterations)+' MC runs', leave=False) as pbar:  
            with pd.ExcelWriter('/home/aguirrfl/confusion_matrix.xls') as writer:                     
                for MC_it_idx in range(MC_iterations):
                
                    self.MC_iteration = MC_it_idx
                
                    if self.run == 'nominal':
                        figures_folder_str = os.path.join(self.results_folder,curr_run_folder,'nominal','figs')            
                        w_inp_exc_folder_str = os.path.join(self.results_folder,curr_run_folder,'nominal','W_inp-exc')
                        train_spikes_str = os.path.join(self.results_folder,curr_run_folder,'nominal','recorded_spikes','train')
                        test_spikes_str = os.path.join(self.results_folder,curr_run_folder,'nominal','recorded_spikes','test')            
                    elif self.run == 'monte-carlo':
                        figures_folder_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','figs')            
                        w_inp_exc_folder_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','W_inp-exc')
                        train_spikes_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','recorded_spikes','train')
                        test_spikes_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','recorded_spikes','test')   
                    
                    os.makedirs(figures_folder_str,exist_ok=True)
                    os.makedirs(w_inp_exc_folder_str,exist_ok=True)
                    os.makedirs(test_spikes_str,exist_ok=True)
                    os.makedirs(train_spikes_str,exist_ok=True)
                    
                    #print("\n" * (pid*5))
                    print('Testing Monte-Carlo iteration '+str(pid)+'', end='\n')
                    with tqdm_gui(total=len(idx2restore)*len(epoch2restore), desc='Testing Monte-Carlo iteration '+str(pid)+'', leave=False) as pbar:       
                        for epoch2restore_idx, epoch2restore_val in enumerate(epoch2restore):
                            for idx2restore_idx, idx2restore_val in enumerate(idx2restore):    
        
                                pbar.update(1)                    
                                #print('->Searching trained network...', end='')            
                
                                state_str='train_'+str(idx2restore_val)+'_img_'+str(epoch2restore_val+1)+'_epochs'
                                results_file=os.path.join(w_inp_exc_folder_str,''+state_str+'.b2')
                                if os.path.exists(results_file):
                            
                                    rec_spikes_filename_train=os.path.join(train_spikes_str,'SNN_'+str(epoch2restore_val+1)+'_eps_'+str(idx2restore_val)+'_imgs_RFC_'+str(self.assign_items)+'_imgs.npy')
                                    rec_spikes_filename_test=os.path.join(test_spikes_str,'SNN_'+str(epoch2restore_val+1)+'_eps_'+str(idx2restore_val)+'_imgs_RFC_'+str(self.eval_items)+'_imgs.npy')                
                                
                                    #print('->Loading spike train for Random Forest training ('+str(self.assign_items)+' images)...', end='\n')      
                                    f_train = np.load(rec_spikes_filename_train, allow_pickle=True)
                                    #print('->Done!', end='\n')
                                
                                    #print('->Loading spike train for Random Forest testing ('+str(eval_items)+' images)...', end='')  
                                    f_test = np.load(rec_spikes_filename_test, allow_pickle=True)
                                    #print('->Done!', end='\n')
        
                                    #print('###############################################################################################', end='\n')                                
        
                                    assignament=self.assign_labels(f_train,y_train[:self.assign_items], desc = 'Assigning labels to MC it. '+str(pid)+', '+str(idx2restore_val)+' images')
                                    predicted_vals_train=self.classify_spikes(f_train,assignament, desc = 'Classifying train images, MC it. '+str(pid)+', '+str(idx2restore_val)+' images')
                                    predicted_vals_test=self.classify_spikes(f_test,assignament, desc = 'Classifying test images, MC it. '+str(pid)+', '+str(idx2restore_val)+' images')
                                                            
                                    #accuracy_train = clf.score(f_train, y_train[:self.assign_items])
                                    accuracy_train = np.sum(np.reshape(y_train[:self.assign_items],(self.assign_items,1))==predicted_vals_train)/self.assign_items
                                    #y_pred_train = clf.predict(f_train)        
                                    cm_train = confusion_matrix(predicted_vals_train, y_train[:self.assign_items])        
                                
                                    #accuracy_test = clf.score(f_test, y_test[:self.eval_items])
                                    accuracy_test = np.sum(np.reshape(y_test[:self.eval_items],(self.eval_items,1))==predicted_vals_test)/self.eval_items
                                    #y_pred_test = clf.predict(f_test)
                                    #cm_test = confusion_matrix(y_pred_test, y_test[:self.eval_items])        
                                    cm_test = confusion_matrix(predicted_vals_test, y_test[:self.eval_items])        
    
                                    acc_train_mat[idx2restore_idx][MC_it_idx] = accuracy_train
                                    acc_test_mat[idx2restore_idx][MC_it_idx] = accuracy_test
                                    image_vec[idx2restore_idx] = idx2restore_val
                                
                                    """
                                    print(Fore.GREEN, end ="")
                                    print('Training Accuracy: '+"{:.2f}".format(accuracy_train)+', Test Accuracy: '+"{:.2f}".format(accuracy_test)+'')
                                    print(Style.RESET_ALL)
                                
                                    print('Confussion Matrix - Training dataset:')        
                                    print(cm_train)
                                    print('Confussion Matrix - Test dataset:')        
                                    print(cm_test)        
                                    """
                                
                                    """
                                    accuracy_train_vec=np.append(accuracy_train_vec,accuracy_train)
                                    accuracy_test_vec=np.append(accuracy_test_vec,accuracy_test)
                                    image_vec=np.append(image_vec, idx2restore_val+epoch2restore_val*60000)
                                    """

                    data_cm = cm_test/cm_test.astype(np.float).sum(axis=0)
                    cm_dF = pd.DataFrame(data_cm)
                    """
                    cm_folder = '/home/aguirrfl/confussion_matrix/Monte-Carlo='+str(MC_it_idx)+''
                    os.makedirs(cm_folder, exist_ok=True)
                    """
                    cm_dF.to_excel(writer, sheet_name=''+str(MC_it_idx)+'_MC')

        if plot_cm=='yes':
            fig =plt.figure(figsize=(8,8))
            im = plt.imshow(cm_test/cm_test.astype(np.float).sum(axis=0), interpolation='nearest', vmin=0, vmax=1, cmap=plt.get_cmap('CMRmap_r')) 
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
	    #plt.rcParams['font.size'] = 12
            cbar=plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Accuracy [a.u.]', size=15)
            cbar.ax.tick_params(labelsize='large')
            #plt.title('Training images: '+str(num_images)+', Epochs: '+str(train_epochs)+'', fontsize = 12.0)
            plt.xlabel("Presented digit", fontsize=15)
            plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize=15)
            plt.ylabel("Predicted digit", fontsize=15)
            plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize=15)
            plt.show()

        print('------------------------------------------------------------------------------------')
                                                          

        conf_interval_train = np.zeros((len(idx2restore),2))
        conf_interval_test = np.zeros((len(idx2restore),2))

        for idx2restore_idx, idx2restore_val in enumerate(idx2restore):    
            conf_interval_train[idx2restore_idx,:]=st.norm.interval(alpha=0.9, 
                                                                    loc=np.mean(acc_train_mat[idx2restore_idx,:]), 
                                                                    scale=np.std(acc_train_mat[idx2restore_idx,:]))                            
 
            conf_interval_test[idx2restore_idx,:]=st.norm.interval(alpha=0.9, 
                                                                   loc=np.mean(acc_test_mat[idx2restore_idx,:]), 
                                                                   scale=np.std(acc_test_mat[idx2restore_idx,:]))                            
                           


        avg_acc_train_vec = np.mean(acc_train_mat, axis = 1)
        avg_acc_test_vec = np.mean(acc_test_mat, axis = 1)
        std_acc_train_vec = np.std(acc_train_mat, axis = 1)
        std_acc_test_vec = np.std(acc_test_mat, axis = 1)
            
        data_accuracy = pd.DataFrame(acc_train_mat)
        data_accuracy.to_excel('/home/aguirrfl/data_accuracy.xls')

        if figure_plt==[]:
            self.fig, self.axis = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(6.5,8))
        else:
            self.axis=figure_plt

        """
        self.axis.fill_between(image_vec[:,0],
                               conf_interval_train[:,0],
                               conf_interval_train[:,1],
                               alpha=0.25)
    
        self.axis.plot(image_vec[:,0],
                      conf_interval_train[:,0],
                      linestyle='--',
                      c='#1f77b4')
    
        self.axis.plot(image_vec[:,0],
                       conf_interval_train[:,1],
                       linestyle='--',
                       c='#1f77b4')

        self.axis.fill_between(image_vec[:,0],
                               conf_interval_test[:,0],
                               conf_interval_test[:,1],
                               alpha=0.25)
    
        self.axis.plot(image_vec[:,0],
                      conf_interval_test[:,0],
                      linestyle='--',
                      c='#1f77b4')
    
        self.axis.plot(image_vec[:,0],
                       conf_interval_test[:,1],
                       linestyle='--',
                       c='#1f77b4')
    
        self.axis.plot(image_vec[:,0],
                       avg_acc_train_vec, 
                       c=color_plot,
                       marker = 'o',
                       markersize = 12,
                       markeredgecolor = color_plot,
                       markerfacecolor = 'w',
                       linestyle='--',
                       linewidth=2,
                       label=''+str(self.n_e)+' neuron - Train accuracy')

        self.axis.plot(image_vec[:,0],
                       avg_acc_test_vec, 
                       c=color_plot,
                       marker = 's',
                       markersize = 12,
                       markeredgecolor = color_plot,
                       markerfacecolor = 'w',
                       linestyle='-',
                       linewidth=2,
                       label=''+str(self.n_e)+' neuron - Test accuracy')
        """
  
        self.axis.errorbar(image_vec[:,0],
                      	   avg_acc_train_vec, 
			   std_acc_train_vec,
                       	   c='red',
                       	   marker = 'o',
                           markersize = 8,
                           markeredgecolor = 'red',
                       	   markerfacecolor = 'w',
			   ecolor='red',
                           linestyle='--',
			   capsize=4, 
                           linewidth=2)#,
                           #label=''+str(self.n_e)+' neuron - Train accuracy')
        """
        self.axis.errorbar(image_vec[:,0],
                           avg_acc_test_vec,  
			   std_acc_test_vec,
                           c='blue',
                           marker = 's',
                           markersize = 8,
                           markeredgecolor = 'blue',
                           markerfacecolor = 'w',
			   ecolor='blue',
                           linestyle='-',
			   capsize=4,
                           linewidth=2,
                           label=''+str(self.n_e)+' neuron - Test accuracy')
        """
        self.axis.set_xlabel("Training images (x10k) [#]", fontsize=15)
        self.axis.set_ylabel("Accuracy [arbitrary units]", fontsize=15)
        self.axis.set_xlim(0, 60)
        self.axis.set_ylim(0, 1.1)
        self.axis.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1])
        self.axis.set_xticks([0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000])
        self.axis.set_xticklabels(['0', '5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60'], fontsize=15)
        self.axis.tick_params(axis='y', which='major', labelsize=15)
        self.axis.legend(fontsize=15, handlelength=4, loc=4)

        print(acc_train_mat.max())	
        print(acc_test_mat.max())
        print(acc_train_mat)	
        print(acc_test_mat)

        if show_accuracy_plot=='yes':
            plt.show()

        
    def evaluate(self, X, **kwargs):    
        
        for key, value in kwargs.items():
            if key=='check_min_spikes':
                self.check_min_spikes=value  
            if key=='progressbar_pos':
                progressbar_pos=value  
            if key=='desc':
                desc=value   
                
        self.net['S1'].lr = 0  # stdp off

        n_input = self.n_input
        n_e = self.n_e
        n_i = self.n_i 
        gl = self.gl
        tau_m_ex = self.tau_m_ex
        tau_m_in = self.tau_m_in
        tau_e = self.tau_e
        tau_i = self.tau_i
        tau_t_h = self.tau_t_h

        v_rest_e = self.v_rest_e
        v_reset_e = self.v_reset_e
        v_thresh_e = self.v_thresh_e 
        v_thresh_e_inc = self.v_thresh_e_inc 
        
        v_rest_i = self.v_rest_i 
        v_reset_i = self.v_reset_i 
        v_thresh_i = self.v_thresh_i
        
        taupre = self.taupre
        taupost = self.taupost
        dApre = self.dApre
        dApost = self.dApost
        gmin = self.gmin
        gmax = self.gmax    
            
        tqdm_text = "#" + "{}".format(progressbar_pos).zfill(3)      
        
        #features = []
        #with alive_bar(len(X), theme='smooth') as bar:
        with tqdm_gui(total=len(X), desc=desc, leave=False) as pbar:     
            for idx in range(len(X)):
                # rate monitor to count spikes
                   
                max_spike_count=-1
                counter_repeat=0    
                while max_spike_count<self.check_min_spikes and counter_repeat<=10:
                    
                    mon = SpikeMonitor(self.net['EG'], name='RM', record=False)
                    self.net.add(mon)
                    
                    # active mode
                    self.net['PG'].rates = (X[idx].ravel('F')*(1+counter_repeat/2))*Hz
                    self.net.run(0.35*second)
                    
                    # spikes per neuron foreach image
                    spike_count=np.reshape(np.array(mon.count, dtype=int8),(1,len(np.array(mon.count, dtype=int8))))
                    max_spike_count=np.sum(spike_count)
                    
                    # passive mode
                    self.net['PG'].rates = np.zeros(n_input)*Hz
                    self.net.run(0.15*second)
                    
                    #print(Fore.RED, end ="")
                    #print('Spike count='+str(max_spike_count)+', Freq.='+str(255/4*(1+counter_repeat/2))+' Hz.', end ="")
                    #print(Style.RESET_ALL)
                       
                    self.net.remove(self.net['RM'])
                    counter_repeat=counter_repeat+1

                # spikes per neuron foreach image
                if 'features' in locals():
                    features=np.append(features,spike_count,axis=0)
                else:
                    features=spike_count
                pbar.update(1)                  
                #bar()
        return features

    def assign_labels(self, spikes, labels, desc = ''):
        
        if spikes.shape[0]!=len(labels):
            print('Error!')
        else:
            assignaments=np.ones(spikes.shape)*nan
            assignaments_weights=np.ones(spikes.shape)*nan
            neuron_ordering=np.zeros((spikes.shape[1],1))
            #print('->Assigning labels to neurons:') 
            with tqdm(total=spikes.shape[1], desc=desc, leave=True) as pbar:    
            #with alive_bar(spikes.shape[1], theme='smooth') as bar:
                for neuron_idx in range(spikes.shape[1]):
                    aux_var=nan
                    max_spikes=amax(spikes[:,neuron_idx])
                    
                    aux_spikes=np.clip(spikes[:,neuron_idx],0,1).astype(float)
                    aux_spikes[aux_spikes == 0] = np.nan
                    assignaments[:,neuron_idx]=labels*aux_spikes
                    assignaments_weights[:,neuron_idx]=spikes[:,neuron_idx]/max_spikes
                    
                    neuron_response = assignaments[:,neuron_idx]
                    neuron_response_weights = assignaments_weights[:,neuron_idx]
    
                    firing_images = ~numpy.isnan(neuron_response)                               
                    
                    reseponse_probability = np.bincount(neuron_response[firing_images].astype(int), weights=neuron_response_weights[firing_images])
                    if reseponse_probability.size>0:
                        neuron_ordering[neuron_idx] = reseponse_probability.argmax()
                    else:
                        neuron_ordering[neuron_idx] = random.randint(0, 9)
                    pbar.update(1)                  
   
        return neuron_ordering
    
    def classify_spikes(self, spikes, assignaments, desc = ''):
        
        predictions=np.ones(spikes.shape)*nan
        predictions_weights=np.ones(spikes.shape)*nan
        predicted_images=np.zeros((spikes.shape[0],1))
    
        #print('->Classifying patterns:') 
        with tqdm(total=spikes.shape[1], desc=desc, leave=True) as pbar:    
            for image_idx in range(spikes.shape[0]):
                max_spikes=amax(spikes[image_idx,:])
                aux_var=nan
                
                aux_spikes=np.clip(spikes[image_idx,:],0,1).astype(float)
                aux_spikes[aux_spikes == 0] = np.nan
                predictions[image_idx,:]=assignaments.T*aux_spikes
                predictions_weights[image_idx,:]=spikes[image_idx,:]/max_spikes
                
                neuron_response = predictions[image_idx,:]
                neuron_response_weights = predictions_weights[image_idx,:]
    
                firing_images = ~numpy.isnan(neuron_response)                               
                
                if np.all(firing_images == False)==False:
                    reseponse_probability = np.bincount(neuron_response[firing_images].astype(int), weights=neuron_response_weights[firing_images])
                    predicted_images[image_idx] = reseponse_probability.argmax()
                pbar.update(1)                  
        return predicted_images

    def plot_w_matrix_single(self, num_images, train_epochs, silent_plot='no', **kwargs):
        
        for key, value in kwargs.items():
            if key=='max_val':
                max_val=value
            elif key=='min_val':
                min_val=value
        
        weight_copy=np.copy(self['S1'].w)
        population = self.n_e
        sqrt_population=np.sqrt(population)
        array_2d_length = self.size_img[0] * int(sqrt_population)
        array_2d_width = self.size_img[0] * int(sqrt_population)
        weight_2d_array = np.zeros((array_2d_width, array_2d_length))
        for i, item in enumerate(weight_copy):
            x = int((i % population) % sqrt_population) * self.size_img[0] + (i // population) % self.size_img[0]
            y = int((i % population) // sqrt_population) * self.size_img[0] + (i // population) // self.size_img[0]
            weight_2d_array[x,y] = item
    
        fig =plt.figure(figsize=(8,8))
        if 'max_val' in kwargs and 'min_val' not in kwargs: 
            im = plt.imshow(weight_2d_array, interpolation='nearest', vmax=max_val, cmap=plt.get_cmap('CMRmap_r')) 
        elif 'min_val' in kwargs and 'max_val' not in kwargs:
            im = plt.imshow(weight_2d_array, interpolation='nearest', vmin=min_val, cmap=plt.get_cmap('CMRmap_r')) 
        elif 'min_val' in kwargs and 'max_val' in kwargs:
            im = plt.imshow(weight_2d_array, interpolation='nearest', vmin=min_val, vmax=max_val, cmap=plt.get_cmap('CMRmap_r')) 
        elif 'min_val' not in kwargs and 'max_val' not in kwargs:
            im = plt.imshow(weight_2d_array, interpolation='nearest', cmap=plt.get_cmap('CMRmap_r')) 
            
        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)
        #plt.rcParams['font.size'] = 12
        cbar=plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Conductance [S]', size=15)
        ax = plt.gca()
        cbar.ax.tick_params(labelsize='large')

        plt.title('Training images: '+str(num_images)+', Epochs: '+str(train_epochs)+'', fontsize = 12.0)
        if silent_plot=='no':
            plt.show(block=False)                       
            plt.pause(2)     
        #plt.show()    
        return plt

    def plot_w_matrix_evolution(self, results_folder='.', interval=1000, num_images=60000):
        epoch=1
        num_images=60000
        max_val=0
        min_val=9e99
    
        new_instance_variables=self.create_var_dict()

        list_saved_runs = os.listdir(self.results_folder)

        curr_run_folder = ''
        if len(list_saved_runs)!=0:
            for run_folder_idx, run_folder in enumerate(list_saved_runs):
                if os.path.exists(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl")):
                    aux_file = open(os.path.join(self.results_folder,run_folder,"SNN_settings.pkl"), "rb")
                    SNN_variable_dict = pickle.load(aux_file)
                
                    if new_instance_variables==SNN_variable_dict:
                        curr_run_folder = run_folder
                        break                        
            
        if curr_run_folder == '':
            printf(print('->Error, this network does not exist, and it thereby it was not trained, exiting', end='')  )
            return 1

        if self.run == 'nominal':
            figures_folder_str = os.path.join(self.results_folder,curr_run_folder,'nominal','figs')            
            w_inp_exc_folder_str = os.path.join(self.results_folder,curr_run_folder,'nominal','W_inp-exc')
            train_spikes_str = os.path.join(self.results_folder,curr_run_folder,'nominal','recorded_spikes','train')
            test_spikes_str = os.path.join(self.results_folder,curr_run_folder,'nominal','recorded_spikes','test')            
        elif self.run == 'monte-carlo':
            figures_folder_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','figs')            
            w_inp_exc_folder_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','W_inp-exc')
            train_spikes_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','recorded_spikes','train')
            test_spikes_str = os.path.join(self.results_folder,curr_run_folder,'monte-carlo','MC_iteration='+str(self.MC_iteration)+'','recorded_spikes','test')   

        os.makedirs(figures_folder_str, exist_ok=True)
        os.makedirs(w_inp_exc_folder_str, exist_ok=True)
        
        model = self
    
        #print('->Searching for the training points...')         
        for ep in range(epoch):
            #with alive_bar(num_images, theme='smooth') as bar:
            for idx in range(num_images):
                #bar() 
                state_str='train_'+str(idx)+'_img_'+str(ep+1)+'_epochs'
                results_file=os.path.join(w_inp_exc_folder_str,''+state_str+'.b2')
                if (mod(idx,interval)==0 or idx==1 or idx==100 or idx==500) and os.path.exists(results_file):
                    
                    model.net.restore(state_str, results_file)
                    weight_copy=np.copy(model['S1'].w)
                    if max_val<np.amax(weight_copy):
                        max_val=np.amax(weight_copy)
                    if min_val>np.amax(weight_copy):
                        min_val=np.amin(weight_copy)

        """
        fig_hist, axs_hist = plt.subplots(4, 5)
        fig_hist.set_figheight(12.5)
        fig_hist.set_figwidth(19)
        fig_hist.subplots_adjust(top=0.84)
        current_axis=0
        """

        idx=0        
        #print('->Generating .gif file...') 
        with imageio.get_writer(os.path.join(figures_folder_str,'G_evolution.gif'), mode='I', fps=1) as writer:                        
            for ep in range(epoch):
                #with alive_bar(num_images, theme='smooth') as bar:
                for idx in range(num_images):
                    #bar() 
                    state_str='train_'+str(idx)+'_img_'+str(ep+1)+'_epochs'
                    results_file=os.path.join(w_inp_exc_folder_str,''+state_str+'.b2')
                    aux_figure=os.path.join(figures_folder_str,''+state_str+'.png')
                    aux_figure_svg=os.path.join(figures_folder_str,''+state_str+'.svg')

                    if (mod(idx,interval)==0 or idx==1 or idx==100 or idx==500 or idx==59000) and os.path.exists(results_file):
                        
                        model.net.restore(state_str, results_file)
                        plot=model.plot_w_matrix_single(idx, ep+1, max_val=max_val, min_val=min_val, silent_plot='yes')
                        
                        plot.savefig(aux_figure)
                        plot.savefig(aux_figure_svg)
                        plot.close()
                        
                        image = imageio.imread(aux_figure)
                        writer.append_data(image)
                        #os.remove(aux_figure)
                        """
                        (row_fig, col_fig) = divmod(current_axis, 5)
                        plot_w_matrix(weight_copy, idx, row_fig, col_fig, len(model.net['EG']), axs_hist)
                        current_axis=current_axis+1
                        """
        """
        fig_hist.tight_layout(pad=0.25)          
        """

def plot_w_matrix(weight_copy, num_images, row_fig, col_fig, population, axs_train):
    
    sqrt_population=np.sqrt(population)
    array_2d_length = size_img[0] * int(sqrt_population)
    array_2d_width = size_img[0] * int(sqrt_population)
    weight_2d_array = np.zeros((array_2d_width, array_2d_length))
    for i, item in enumerate(weight_copy):
        x = int((i % population) % sqrt_population) * size_img[0] + (i // population) % size_img[0]
        y = int((i % population) // sqrt_population) * size_img[0] + (i // population) // size_img[0]
        weight_2d_array[x, y] = item

    im = axs_train[row_fig, col_fig].imshow(weight_2d_array/1e-6, interpolation='nearest', vmin=0, cmap=plt.get_cmap('CMRmap_r')) 
    axs_train[row_fig, col_fig].set_title(''+str(num_images)+' Imgs.', fontsize = 10.0)
    axs_train[row_fig, col_fig].tick_params(
                                            axis='both',       # changes apply to the x-axis
                                            which='both',      # both major and minor ticks are affected
                                            bottom=False,      # ticks along the bottom edge are off
                                            top=False,         # ticks along the top edge are off
                                            left=False,        # ticks along the bottom edge are off
                                            right=False,       # ticks along the top edge are off
                                            labelbottom=False,
                                            labelleft=False)   # labels along the bottom edge are off
    #axs_train[row, col].rcParams['font.size'] = 12
    cbar=plt.colorbar(im, ax=axs_train[row_fig, col_fig], fraction=0.046, pad=0.04)
    cbar.set_label("Conductance [\u03BCS]")   

def plot_w_and_signals(S1M, PSM, ESM, neuron_in, neuron_out):
    cnt = -150000 # tail
    plt.rcParams["figure.figsize"] = (20,10)

    plt.subplot(311)
    plt.plot(S1M.t/ms, S1M.w.T[neuron_in,neuron_out]/gmax)
    plt.ylabel('w / wmax')

    plt.subplot(312)
    plt.plot(ESM.t[cnt:]/ms, ESM.v[neuron_out][cnt:]/mV, label='exc', color='r')
    plt.ylabel('Excitatory neuron')
    
    plt.subplot(313)
    plt.plot(PSM.t[cnt:]/ms, PSM.v[neuron_in][cnt:]/mV, label='exc', color='k')
    plt.ylabel('Input generator')
    
    plt.tight_layout()
    plt.show()    

def plot_w(S1M):
    plt.rcParams["figure.figsize"] = (20,10)
    plt.subplot(311)
    plt.plot(S1M.t/ms, S1M.w.T/gmax)
    plt.ylabel('w / wmax')
    plt.subplot(312)
    plt.plot(S1M.t/ms, S1M.Apre.T)
    plt.ylabel('apre')
    plt.subplot(313)
    plt.plot(S1M.t/ms, S1M.Apost.T)
    plt.ylabel('apost')
    plt.tight_layout()
    plt.show()
    
def plot_v(ESM, ISM, neuron=25):
    plt.rcParams["figure.figsize"] = (20,6)
    cnt = -150000 # tail
    plt.subplot(211)
    for idx in range(neuron):
        plt.plot(ESM.t[cnt:]/ms, ESM.v[idx][cnt:]/mV)
    plt.axhline(y=v_thresh_e/mV, color='pink', label='v_thresh_e')
    plt.legend()
    plt.ylabel('Membrane potential Exc.')    
    plt.subplot(212)
    for idx in range(neuron):
        plt.plot(ISM.t[cnt:]/ms, ISM.v[idx][cnt:]/mV)        
    plt.axhline(y=v_thresh_i/mV, color='silver', label='v_thresh_i')
    plt.legend()
    plt.ylabel('Membrane potential Ihn.') 
    
    plt.show()
    
def plot_rates(ERM, IRM):
    plt.rcParams["figure.figsize"] = (20,6)
    plt.plot(ERM.t/ms, ERM.smooth_rate(window='flat', width=0.1*ms)*Hz, color='r')
    plt.plot(IRM.t/ms, IRM.smooth_rate(window='flat', width=0.1*ms)*Hz, color='b')
    plt.ylabel('Rate')
    plt.show();
    
def plot_spikes(ESP, ISP):
    plt.rcParams["figure.figsize"] = (20,6)
    plt.plot(ESP.t/ms, ESP.i, '.r')
    plt.plot(ISP.t/ms, ISP.i, '.b')
    plt.ylabel('Neuron index')
    plt.show()

def plot_W_maps(results_folder='.', interval=1000):
    epoch=1
    num_images=60000
    max_val=0
    min_val=9e99

    os.makedirs(os.path.join(results_folder,'figs'),exist_ok=True)
    os.makedirs(os.path.join(results_folder,'W_inp-exc'),exist_ok=True)
    
    model = Model()

    print('->Searching for the training points...')         
    for ep in range(epoch):
        with alive_bar(num_images, theme='smooth') as bar:
            for idx in range(num_images):
                bar() 
                state_str='train_'+str(idx)+'_img_'+str(ep+1)+'_epochs'
                results_file=os.path.join(results_folder,'W_inp-exc',''+state_str+'.b2')
                if mod(idx,100)==0 and os.path.exists(results_file):
                    
                    model.net.restore(state_str, results_file)
                    weight_copy=np.copy(model['S1'].w)
                    if max_val<np.amax(weight_copy):
                        max_val=np.amax(weight_copy)
                    if min_val>np.amax(weight_copy):
                        min_val=np.amin(weight_copy)

    fig_hist, axs_hist = plt.subplots(4, 5)
    fig_hist.set_figheight(12.5)
    fig_hist.set_figwidth(19)
    fig_hist.subplots_adjust(top=0.84)
    idx=0
    current_axis=0
    
    print('->Generating .gif file...') 
    with imageio.get_writer(os.path.join(results_folder,'figs','G_evolution.gif'), mode='I', fps=1) as writer:                        
        for ep in range(epoch):
            with alive_bar(num_images, theme='smooth') as bar:
                for idx in range(num_images):
                    bar() 
                    state_str='train_'+str(idx)+'_img_'+str(ep+1)+'_epochs'
                    results_file=os.path.join(results_folder,'W_inp-exc',''+state_str+'.b2')
                    aux_figure=os.path.join(results_folder,'figs',''+state_str+'.png')
                    aux_figure_svg=os.path.join(results_folder,'figs',''+state_str+'.svg')

                    if (mod(idx,interval)==0 or idx==59000) and os.path.exists(results_file):
                        
                        model.net.restore(state_str, results_file)
                        weight_copy=np.copy(model['S1'].w)
                        plot=plot_w_matrix_single(weight_copy, idx, len(model.net['EG']), ep+1, max_val=max_val, min_val=min_val, silent_plot='yes')
                        
                        plot.savefig(aux_figure)
                        plot.savefig(aux_figure_svg)
                        plot.close()
                        
                        image = imageio.imread(aux_figure)
                        writer.append_data(image)
                        #os.remove(aux_figure)
                        
                        (row_fig, col_fig) = divmod(current_axis, 5)
                        plot_w_matrix(weight_copy, idx, row_fig, col_fig, len(model.net['EG']), axs_hist)
                        current_axis=current_axis+1
    fig_hist.tight_layout(pad=0.25)                  
"""                        
def assign_labels(spikes, labels):
    
    if len(spikes)!=len(labels):
        print('Error!')
    else:
        assignaments=np.ones(spikes.shape)*nan
        with alive_bar(spikes.shape[0]*spikes.shape[1], theme='smooth') as bar:
            for image_idx in range(spikes.shape[0]):
                max_spikes=amax(spikes[image_idx,:])
                for neuron_idx in range(spikes.shape[1]):
                    if spikes[image_idx][neuron_idx]==max_spikes and max_spikes>0:
                        assignaments[image_idx][neuron_idx]=labels[image_idx]
                    bar()
        assignaments=np.round(np.nanmean(assignaments,0))

    return assignaments
"""

def test0(train_items=30, size_img=[28,28], population=100, learning_rate=1, num_epochs=1):
    '''
    STDP visualisation
    '''
    seed(0)
    
    model = Model(debug=True)
    model.train(x_train[:train_items], epoch=num_epochs, learning_rate=learning_rate, gen_plot='no', restore='no', store='no')
    
    plot_w(model['S1M'])
    #plot_w_and_signals(model['S1M'], model['PSM'], model['ESM'], 380,12)
    plot_v(model['ESM'], model['ISM'])
    plot_rates(model['ERM'], model['IRM'])
    plot_spikes(model['ESP'], model['ISP'])

    weight_copy=np.copy(model['S1'].w)
    sqrt_population=np.sqrt(population)
    array_2d_length = size_img[0] * int(sqrt_population)
    array_2d_width = size_img[0] * int(sqrt_population)
    weight_2d_array = np.zeros((array_2d_width, array_2d_length))
    for i, item in enumerate(weight_copy):
        x = int((i % population) % sqrt_population) * size_img[0] + (i // population) % size_img[0]
        y = int((i % population) // sqrt_population) * size_img[0] + (i // population) // size_img[0]
        weight_2d_array[x,y] = item

    fig =plt.figure(figsize=(18,18))
    im = plt.imshow(weight_2d_array, interpolation='nearest', vmin=0, cmap=plt.get_cmap('CMRmap_r')) 
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.rcParams['font.size'] = 34
    plt.colorbar(im)
    plt.show()    
    """
    weights=np.reshape(model['S1'].w/uS,(784,100))    
    plt.figure(figsize=(10,10))
    for img in range(100):
        plt.subplot(10,10,1+img)
        #plt.title(y_train[img])        
        plt.imshow(np.reshape(weights[:,img],(28,28)))
        plt.axis('off')
    plt.show()
    """

def test2(train_items=2000, assign_items=2000, eval_items=1000, size_img=[28,28]):
    '''
    Freeze STDP at start
    Feed train set to SNN and collect generated features
    Train RandomForest on the top of these features and labels provided
    Feed test set to SNN and collect new features
    Predict labels with RandomForest and calculate accuacy score
    '''
    seed(0)
    
    model = Model()
        
    f_train = model.evaluate(X_train[:assign_items])
    clf = RandomForestClassifier(max_depth=4, random_state=0)
    clf.fit(f_train, y_train[:assign_items])
    print(clf.score(f_train, y_train[:assign_items]))

    f_test = model.evaluate(X_test[:eval_items])
    y_pred = clf.predict(f_test)
    print(accuracy_score(y_pred, y_test[:eval_items]))

    cm = confusion_matrix(y_pred, y_test[:eval_items])
    print(cm)

def test3(train_items=2000, eval_items=1000, size_img=[28,28]):
    '''
    Train and evaluate RandomForest without SNN
    '''
    seed(0)
    
    clf = RandomForestClassifier(max_depth=4, random_state=0)
    
    train_features = X_train[:train_items].reshape(-1,28*28)
    clf.fit(train_features, y_train[:train_items])
    print(clf.score(train_features, y_train[:train_items]))
    
    test_features = X_test[:eval_items].reshape(-1,28*28)
    y_pred = clf.predict(test_features)
    print(accuracy_score(y_pred, y_test[:eval_items]))

    cm = confusion_matrix(y_pred, y_test[:eval_items])
    print(cm)