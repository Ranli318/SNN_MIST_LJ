

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SNN Simulation with 784 Input, 400 Excitatory, and 400 Inhibitory Neurons
Multithreaded Monte Carlo Training and Testing
"""

# --------------------------------------------------- Libraries ---------------------------------------------------
import os
import random
import time
import warnings
from multiprocessing import Pool, freeze_support, RLock

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from alive_progress import alive_bar
from brian2 import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras
from tqdm import tqdm

import SNN_brian2_library

warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------------------------------- SNN Functions ---------------------------------------------------

def run_snn_training(pid, x_train, g_var):
    seed()
    model = SNN_brian2_library.Model(
        size_img=size_img, n_e=n_e, n_i=n_i, gl=gl,
        tau_m_ex=tau_m_ex, tau_m_in=tau_m_in,
        tau_e=tau_e, tau_i=tau_i, tau_t_h=tau_t_h,
        learning_rate=learning_rate,
        train_items=train_items, assign_items=assign_items, eval_items=eval_items,
        num_epochs=num_epochs, v_rest_e=v_rest_e, v_reset_e=v_reset_e,
        v_thresh_e=v_thresh_e, v_thresh_e_inc=v_thresh_e_inc,
        v_rest_i=v_rest_i, v_reset_i=v_reset_i, v_thresh_i=v_thresh_i,
        refrac_exc=refrac_exc, refrac_inh=refrac_inh,
        taupre=taupre, taupost=taupost, dApre=dApre, dApost=dApost,
        gmin=gmin, gmax=gmax, g_var=g_var, run='monte-carlo')
    model.train(x_train, pid=pid, MC_iteration=pid)


def run_snn_testing(pid, x_train, g_var, interval):
    seed()
    model = SNN_brian2_library.Model(
        size_img=size_img, n_e=n_e, n_i=n_i, gl=gl,
        tau_m_ex=tau_m_ex, tau_m_in=tau_m_in,
        tau_e=tau_e, tau_i=tau_i, tau_t_h=tau_t_h,
        learning_rate=learning_rate,
        train_items=train_items, assign_items=assign_items, eval_items=eval_items,
        num_epochs=num_epochs, v_rest_e=v_rest_e, v_reset_e=v_reset_e,
        v_thresh_e=v_thresh_e, v_thresh_e_inc=v_thresh_e_inc,
        v_rest_i=v_rest_i, v_reset_i=v_reset_i, v_thresh_i=v_thresh_i,
        refrac_exc=refrac_exc, refrac_inh=refrac_inh,
        taupre=taupre, taupost=taupost, dApre=dApre, dApost=dApost,
        gmin=gmin, gmax=gmax, g_var=g_var, run='monte-carlo')

    model.test(
        x_train, y_train, x_test, y_test,
        population=n_e, learning_rate=learning_rate, num_epochs=num_epochs,
        epoch2restore_input=1, idx2restore_input=train_items,
        idx_interval=interval, check_min_spikes=5,
        color_plot='b', show_accuracy_plot='no', plot_cm='no',
        pid=pid, MC_iteration=pid)


# --------------------------------------------------- Parameters ---------------------------------------------------

# Image and neuron settings
size_img = [28, 28]
n_input = size_img[0] * size_img[1]
n_e = 400
n_i = n_e

# Neuron model parameters
gl = 0.0001 * uS
tau_m_ex = 300 * ms
tau_m_in = 10 * ms
tau_e = 6 * ms
tau_i = 10 * ms
tau_t_h = 1e7 * ms

v_rest_e, v_reset_e, v_thresh_e, v_thresh_e_inc = -65.*mV, -65.*mV, -50.*mV, 0.05*mV
v_rest_i, v_reset_i, v_thresh_i = -65.*mV, -43.*mV, -38.*mV

refrac_exc, refrac_inh = 55 * ms, 13 * ms

# Training settings
learning_rate = 1
train_items = 59000
assign_items = 500
eval_items = 500
idx_interval = 1000
num_epochs = 1
MC_runs = 50
num_processes = 50

# STDP synapse parameters
taupre = 20.10 * ms # 
taupost = 20.70 * ms # 
dApre = 117520 * uS # 
dApost = 71840 * uS # 
gmin = 0.1 * uS # test
gmax = 100 * uS # test
g_var = 0.2 # test

# --------------------------------------------------- Dataset Preparation ---------------------------------------------------

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.array([cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC) for img in x_train])
x_test = np.array([cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC) for img in x_test])

x_train, x_test = x_train / 4, x_test / 4  # Normalize to ~63 Hz

# --------------------------------------------------- Main Execution ---------------------------------------------------

if __name__ == "__main__":
    freeze_support()

    # Monte Carlo Training
    with Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        for i in range(MC_runs):
            pool.apply_async(run_snn_training, args=(i, x_train, g_var))
        pool.close()
        pool.join()

    # Monte Carlo Testing
    with Pool(processes=num_processes, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        for i in range(MC_runs):
            pool.apply_async(run_snn_testing, args=(i, x_train, g_var, idx_interval))
            time.sleep(2)
        pool.close()
        pool.join()

    # Post-analysis
    model = SNN_brian2_library.Model(
        size_img=size_img, n_e=n_e, n_i=n_i, gl=gl,
        tau_m_ex=tau_m_ex, tau_m_in=tau_m_in, tau_e=tau_e, tau_i=tau_i, tau_t_h=tau_t_h,
        learning_rate=learning_rate, train_items=train_items, assign_items=assign_items,
        eval_items=eval_items, num_epochs=num_epochs,
        v_rest_e=v_rest_e, v_reset_e=v_reset_e, v_thresh_e=v_thresh_e,
        v_thresh_e_inc=v_thresh_e_inc, v_rest_i=v_rest_i, v_reset_i=v_reset_i, v_thresh_i=v_thresh_i,
        refrac_exc=refrac_exc, refrac_inh=refrac_inh,
        taupre=taupre, taupost=taupost, dApre=dApre, dApost=dApost,
        gmin=gmin, gmax=gmax, g_var=g_var, run='monte-carlo')

    model.plot_test_metrics(
        x_train, y_train, x_test, y_test,
        MC_iterations=MC_runs, idx2restore_input=train_items,
        epoch2restore_input=1, idx_interval=idx_interval,
        show_accuracy_plot='yes')

    model.MC_iteration = 31
    model.plot_w_matrix_evolution()
