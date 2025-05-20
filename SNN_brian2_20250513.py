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
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

# --------------------------------------------------- Model Class ---------------------------------------------------

class Model:
    def __init__(self, debug=False, **kwargs):
        self.set_default_params()
        self.update_from_kwargs(**kwargs)
        self.prepare_results_folder()
        self.setup_network(debug)

    def set_default_params(self):
        self.size_img = [28, 28]                      # Input image size (rows, cols)
        self.n_input = self.size_img[0] * self.size_img[1]  # Total number of input neurons = 784
        self.n_e = 400                                # Number of excitatory neurons
        self.n_i = 400                                # Number of inhibitory neurons

        self.gl = 0.0001 * uS                         # Leak conductance
        self.tau_m_ex = 0.01 * ms                      # Membrane time constant for excitatory neurons (RC of device)
        self.tau_m_in = 0.02 * ms                       # Membrane time constant for inhibitory neurons (RC of device)  
        self.tau_e = 10 * ms                           # Excitatory synaptic time constant (Spike)
        self.tau_i = 5 * ms                          # Inhibitory synaptic time constant (Spike)
        self.tau_t_h = 1e13 * ms                       # Adaptive threshold decay time constant  reset after a spike.

        self.learning_rate = 1                        # STDP learning rate
        self.train_items = 60000                      # Number of training images
        self.assign_items = 500                       # Images used for label assignment
        self.eval_items = 500                         # Images used for final evaluation
        self.num_epochs = 200                         # Number of training epochs

        # Excitatory neuron voltages
        self.v_rest_e = -50. * mV                     # Resting membrane potential
        self.v_reset_e = -100. * mV                    # Reset voltage after spike 
        self.v_thresh_e = -40. * mV                   # Static base firing threshold
        self.v_thresh_e_inc = 0.05 * mV               # Increment to adaptive threshold after spike

        # Inhibitory neuron voltages
        self.v_rest_i = -50. * mV                     # Resting potential for inhibitory neuron
        self.v_reset_i = -150. * mV                   # Reset voltage after spike 
        self.v_thresh_i = -30. * mV                   # Firing threshold

        # Refractory periods
        self.refrac_exc = 0 * ms                    # Refractory period for excitatory neurons
        self.refrac_inh = 0 * ms                    # Refractory period for inhibitory neurons

        # STDP learning dynamics
        self.taupre = 20.7 * ms                      # Decay time for pre-synaptic trace
        self.taupost = 20.1 * ms                     # Decay time for post-synaptic trace
        self.dApre = 117520 * uS                     # Pre-synaptic trace increment
        self.dApost = 71830 * uS                     # Post-synaptic trace increment

        # Synaptic weight constraints
        self.gmin = 0.01 * uS                        # Minimum synaptic weight
        self.gmax = 100 * uS                         # Maximum synaptic weight
        self.g_var = 0                               # Synaptic variability (unused)

        # Training condition
        self.check_min_spikes = 5                    # Minimum number of spikes needed to accept learning step
        self.custom_folder = False
        self.run = 'nominal'

        #  MC iterations
        self.MC_iteration = 0 

        # STDP nonlinearity parameters
        self.beta_p = 0.90                            # Potentiation nonlinearity parameter
        self.beta_d = 0.57                            # Depression nonlinearity parameter
        self.alpha_p = 1.0                            # Potentiation amplitude factor
        self.alpha_d = 1.0                            # Depression amplitude factor
        self.n_norm = 50                              # Normalization factor for STDP

    def update_from_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key == 'results_folder':
                self.custom_folder = True
                self.results_folder = value

     def prepare_results_folder(self):
        if not self.custom_folder:
            name = f"SNN_{self.n_input}x{self.n_e}x{self.n_i}"
            self.results_folder = os.path.join("..", "..", "results", name)
        os.makedirs(self.results_folder, exist_ok=True)

    def setup_network(self, debug=False):
        defaultclock.dt = 0.1 * ms
        self.net = Network()

        self.net.add(PoissonGroup(self.n_input, rates=np.zeros(self.n_input) * Hz, name="PG"))

        EG = NeuronGroup(self.n_e, '''
            dv/dt = (ge*(0*mV - v)/(15*uS) + gi*(-100*mV - v)/(0.2*uS) + (v_rest_e - v))/tau_m_ex : volt
            dge/dt = -ge/tau_e : siemens
            dgi/dt = -gi/tau_i : siemens
            dv_th/dt = -v_th/tau_t_h : volt''',
            threshold='v>v_thresh_e+v_th',
            reset="v=v_reset_e; v_th+=v_thresh_e_inc",
            refractory=self.refrac_exc, method='euler', name='EG')

        IG = NeuronGroup(self.n_i, '''
            dv/dt = (ge*(0*mV - v)/(0.5*uS) + gi*(-90*mV - v)/(10*uS) + (v_rest_i - v))/tau_m_in : volt
            dge/dt = -ge/tau_e : siemens
            dgi/dt = -gi/tau_i : siemens''',
            threshold='v>v_thresh_i', reset="v=v_reset_i", refractory=self.refrac_inh,
            method='euler', name='IG')
        IG.v = self.v_rest_i - 20.*mV
        self.net.add(IG)

        # STDP Synapses
        # Define STDP synapses between input and excitatory neurons
        S1 = Synapses(
            self.net['PG'], EG,
            model="""
            w : siemens
            dApre/dt = -Apre / taupre : siemens (event-driven)
            dApost/dt = -Apost / taupost : siemens (event-driven)
            """,
            # Post-before-pre → LTD (weaken synapse)
            on_pre="""     
            ge_post += w
            Apre += dApre
            w = clip(w - lr * (alpha_d / n_norm) * exp(-beta_d * (gmax - w) / (gmax - gmin)) * Apost, gmin, gmax)
            """,
            # Pre-before-post → LTP (strengthen synapse)
            on_post="""
            Apost += dApost
            w = clip(w + lr * (alpha_p / n_norm) * exp(-beta_p * (w - gmin) / (gmax - gmin)) * Apre, gmin, gmax)
            """,
            name="S1"
        )
        S1.connect()
        S1.w = 'rand() * gmax * 0.2'  

        # Assign parameters outside of model
        S1.lr = self.learning_rate
        S1.alpha_p = self.alpha_p
        S1.alpha_d = self.alpha_d
        S1.beta_p = self.beta_p
        S1.beta_d = self.beta_d
        S1.gmin = self.gmin
        S1.gmax = self.gmax
        S1.n_norm = self.n_norm
        self.net.add(S1)


        # Input, Excitatory 
        S1 = Synapses(self.net['PG'], EG, model=stdp_eqs, on_pre=pre_eq, on_post=post_eq, name="S1")
        S1.connect() # Fully connects 
        S1.w = 'rand() * gmax * 0.2' # Monte Carlo run: Initializes weights randomly between 0~0.2
        S1.lr = 1
        self.net.add(S1)
        S1.alpha = self.alpha
        S1.beta = self.beta
        S1.gmin = self.gmin
        S1.gmax = self.gmax
        # Fixed Inhibition
        S2 = Synapses(EG, IG, model='w:siemens', on_pre='ge += w', name="S2")
        S2.connect('i == j')
        S2.w = 1.2 * uS
        self.net.add(S2)
        # Lateral Inhibition
        S3 = Synapses(IG, EG, model='w:siemens', on_pre='gi += w', name="S3")
        S3.connect('i != j')
        S3.w = 1.5 * uS
        self.net.add(S3)

        self.net.run(0 * ms)

      def train(self, x_train, pid=0, MC_iteration=0, learning_rate=1):
        self.learning_rate = learning_rate
        self.MC_iteration = MC_iteration

        # Update learning rate on Synapses
        self.net['S1'].lr = self.learning_rate

        print(f"[Training] Starting training process for MC iteration {MC_iteration}")
        with tqdm(total=len(x_train), desc=f'Training MC {MC_iteration}', position=pid) as pbar:
            for idx, img in enumerate(x_train):
                mon = SpikeMonitor(self.net['EG'])
                self.net.add(mon)

                max_spike_count = -1
                counter_repeat = 0
                while max_spike_count < self.check_min_spikes and counter_repeat <= 10:
                    self.net['PG'].rates = (img.ravel('F') * (1 + counter_repeat / 2)) * Hz  # Poisson input rates
                    self.net.run(0.35 * second)  # STDP active phase
                    spike_count = np.array(mon.count)
                    max_spike_count = np.sum(spike_count)

                    self.net['PG'].rates = np.zeros(self.n_input) * Hz  # Pause input
                    self.net.run(0.15 * second)
                    counter_repeat += 1

                self.net.remove(mon)
                pbar.update(1)

                if idx % 1000 == 0 or idx in [1, 100, 500]:
                    filename = os.path.join(self.results_folder, f"train_{idx}_MC_{MC_iteration}.b2")
                    self.net.store(filename)

        print(f"[Training] Finished training for MC iteration {MC_iteration}")


    def test(self, x_train, y_train, x_test, y_test, pid=0, MC_iteration=0, idx_interval=1000):
        print(f"[Testing] Starting test process for MC iteration {MC_iteration}")
        self.MC_iteration = MC_iteration

        results_path = os.path.join(self.results_folder, f"MC_{MC_iteration}")
        os.makedirs(results_path, exist_ok=True)

        accuracy_scores = []
        with tqdm(total=len(x_test), desc=f'Testing MC {MC_iteration}', position=pid) as pbar:
            for idx, img in enumerate(x_test):
                mon = SpikeMonitor(self.net['EG'])
                self.net.add(mon)

                self.net['PG'].rates = img.ravel('F') * Hz
                self.net.run(0.35 * second)

                spikes = np.array(mon.count)
                self.net.remove(mon)

                self.net['PG'].rates = np.zeros(self.n_input) * Hz
                self.net.run(0.15 * second)

                prediction = np.argmax(spikes)
                true_label = y_test[idx] if idx < len(y_test) else -1
                accuracy = int(prediction == true_label)
                accuracy_scores.append(accuracy)

                pbar.update(1)

        final_acc = np.mean(accuracy_scores)
        print(f"[Testing] Accuracy: {final_acc:.4f}")


    def evaluate(self, X, check_min_spikes=None, progressbar_pos=0, desc="Evaluating"):
        if check_min_spikes is not None:
            self.check_min_spikes = check_min_spikes

        self.net['S1'].lr = 0  # Freeze learning during evaluation
        features = []

        with tqdm(total=len(X), desc=desc, position=progressbar_pos) as pbar:
            for idx, img in enumerate(X):
                max_spike_count = -1
                counter_repeat = 0

                while max_spike_count < self.check_min_spikes and counter_repeat <= 10:
                    mon = SpikeMonitor(self.net['EG'])
                    self.net.add(mon)

                    self.net['PG'].rates = (img.ravel('F') * (1 + counter_repeat / 2)) * Hz
                    self.net.run(0.35 * second)

                    spike_count = np.array(mon.count)
                    max_spike_count = np.sum(spike_count)

                    self.net.remove(mon)
                    self.net['PG'].rates = np.zeros(self.n_input) * Hz
                    self.net.run(0.15 * second)
                    counter_repeat += 1

                features.append(spike_count)
                pbar.update(1)

        return np.array(features)


    def assign_labels(self, feature_matrix, true_labels):
        n_neurons = feature_matrix.shape[1]
        label_map = np.zeros(n_neurons, dtype=int)

        for neuron_idx in range(n_neurons):
            neuron_spikes = feature_matrix[:, neuron_idx]
            if np.sum(neuron_spikes) == 0:
                label_map[neuron_idx] = -1
            else:
                label_counts = {}
                for i in range(len(true_labels)):
                    label = true_labels[i]
                    label_counts[label] = label_counts.get(label, 0) + neuron_spikes[i]
                label_map[neuron_idx] = max(label_counts, key=label_counts.get)

        return label_map


    def classify_spikes(self, features, label_map):
        predictions = []

        for sample in features:
            label_votes = {}
            for neuron_idx, spike_count in enumerate(sample):
                label = label_map[neuron_idx]
                if label < 0:
                    continue
                label_votes[label] = label_votes.get(label, 0) + spike_count

            predictions.append(max(label_votes, key=label_votes.get) if label_votes else -1)

        return predictions


    def plot_test_metrics(self, x_train, y_train, x_test, y_test, MC_iterations=10,
                          idx2restore_input=59000, epoch2restore_input=1, idx_interval=1000,
                          show_accuracy_plot='yes'):
        acc_all = []
        confusion_all = []

        for mc in range(MC_iterations):
            restore_name = f"train_{idx2restore_input}_MC_{mc}.b2"
            restore_path = os.path.join(self.results_folder, restore_name)
            if not os.path.exists(restore_path):
                print(f"[Warning] Checkpoint not found: {restore_path}")
                continue

            self.net.restore(restore_path)

            train_feat = self.evaluate(x_train, desc=f"Assign MC {mc}")
            label_map = self.assign_labels(train_feat, y_train)

            test_feat = self.evaluate(x_test, desc=f"Classify MC {mc}")
            preds = self.classify_spikes(test_feat, label_map)

            acc = np.mean(np.array(preds) == np.array(y_test))
            cm = confusion_matrix(y_test, preds, labels=np.unique(y_test))

            acc_all.append(acc)
            confusion_all.append(cm)

            print(f"[MC {mc}] Accuracy: {acc:.4f}")

        if show_accuracy_plot == 'yes':
            plt.figure()
            plt.plot(acc_all, marker='o')
            plt.title("Accuracy over MC iterations")
            plt.xlabel("MC Iteration")
            plt.ylabel("Accuracy")
            plt.grid(True)
            plt.show()


    def plot_w_matrix_single(self, weight_matrix=None):
        if weight_matrix is None:
            weight_matrix = self.net['S1'].w[:].reshape((self.n_input, self.n_e))

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(weight_matrix, cmap='viridis', aspect='auto')
        ax.set_title("Weight matrix: Input → Excitatory")
        ax.set_xlabel("Excitatory Neurons")
        ax.set_ylabel("Input Neurons")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()


    def plot_w_matrix_evolution(self, snapshots=None, interval=1000):
        if snapshots is None:
            snapshots = [1, 100, 500, 1000, 2000, 3000]

        weight_matrices = []
        for idx in snapshots:
            path = os.path.join(self.results_folder, f"train_{idx}_MC_{self.MC_iteration}.b2")
            if not os.path.exists(path):
                continue
            self.net.restore(path)
            W = self.net['S1'].w[:].reshape((self.n_input, self.n_e))
            weight_matrices.append(W)

        image_paths = []
        for i, W in enumerate(weight_matrices):
            fig, ax = plt.subplots()
            im = ax.imshow(W, cmap='viridis', aspect='auto')
            ax.set_title(f"Weights at checkpoint {snapshots[i]}")
            filename = f"frame_{i}.png"
            plt.colorbar(im, ax=ax)
            plt.savefig(filename)
            plt.close()
            image_paths.append(filename)

        images = [imageio.imread(p) for p in image_paths]
        gif_path = os.path.join(self.results_folder, "W_evolution.gif")
        imageio.mimsave(gif_path, images, duration=interval / 1000)
        print(f"[Saved] Evolution animation: {gif_path}")

        for f in image_paths:
            os.remove(f)
 
