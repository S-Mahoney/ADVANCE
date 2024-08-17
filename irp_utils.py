# %% [code]
# %% [markdown]
# # Utility Scripts for IRP

# %% [code]
# Loader imports
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr, spearmanr, skew, kurtosis, entropy
from sklearn.metrics import mean_squared_error
#from fastdtw import fastdtw
import warnings
import pywt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from statsmodels.tsa.ar_model import AutoReg

# Prediction imports
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout,BatchNormalization,Lambda,InputLayer, GaussianNoise, add, LeakyReLU
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
from keras.initializers import he_normal
from keras.layers import Layer

from tqdm.notebook import tqdm
import re

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay,f1_score, make_scorer,precision_recall_fscore_support
from sklearn.impute import SimpleImputer
import time
import random
import gc
import json
import ipywidgets as widgets
from IPython.display import display, clear_output

# %% [markdown]
# ## Loader functions

# %% [code]


def sine_wave(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def get_subcycle(feature, sub_cycle_divisor, machine_data, machine_labels, 
                 max_freq=1.0, initial_step=0.01, refinement_steps=3, 
                 max_cycle_length=1500, min_cycle_length=400, plot_results=True, 
                 user_cycle_length=None, use_individual_cycle_lengths=False):
    """
    Function to sort subcycles into arrays of size sub_cycle_divisor of nominal and anomalous data.
    Input: Chosen feature channel (int), Sub-cycle divisor size (int), Data, Data labels.
    Output: Nominal subcycles, Anomalous subcycles.
    """
    print(f"Feature {feature}")

    def estimate_cycle_length(y_data, step, already_checked_freqs):
        offset_estimate = y_data.mean()
        yf = fft(y_data - offset_estimate)
        xf = fftfreq(len(y_data), 1)
        half_len = len(yf) // 2
        yf = yf[:half_len]
        xf = xf[:half_len]
        valid_range = xf <= max_freq
        yf = yf[valid_range]
        xf = xf[valid_range]
        mask = np.isin(xf, already_checked_freqs, invert=True)
        yf = yf[mask]
        xf = xf[mask]
        if len(xf) == 0:
            return None, None
        coarse_index = np.arange(0, len(xf), max(1, int(step)))
        dominant_freq_index = coarse_index[np.argmax(2.0 / len(y_data) * np.abs(yf[coarse_index]))]
        return xf[dominant_freq_index], dominant_freq_index

    def find_cycle_length(y_data):
        already_checked_freqs = []
        dominant_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
        if dominant_freq:
            already_checked_freqs.append(dominant_freq)
        
        for _ in range(refinement_steps):
            initial_step = max(1, initial_step // 2)
            new_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
            if new_freq:
                dominant_freq = new_freq
                already_checked_freqs.append(dominant_freq)

        cycle_length = int(1 / dominant_freq) if dominant_freq else max_cycle_length

        while (cycle_length < min_cycle_length or cycle_length > max_cycle_length) and dominant_freq:
            if cycle_length < min_cycle_length:
                new_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
                if new_freq:
                    dominant_freq = new_freq
                    cycle_length = int(1 / dominant_freq)
                    already_checked_freqs.append(dominant_freq)
            elif cycle_length > max_cycle_length:
                new_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
                if new_freq:
                    dominant_freq = new_freq
                    cycle_length = int(1 / dominant_freq)
                    already_checked_freqs.append(dominant_freq)
            else:
                break
        
        return cycle_length

    x_data = np.arange(len(machine_data))
    y_data = machine_data.iloc[:, feature].to_numpy()

    if user_cycle_length:
        cycle_length = min(max(min_cycle_length, user_cycle_length), max_cycle_length)
    elif use_individual_cycle_lengths:
        cycle_length = find_cycle_length(y_data)
    else:
        already_checked_freqs = []
        dominant_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
        if dominant_freq:
            already_checked_freqs.append(dominant_freq)

        for _ in range(refinement_steps):
            initial_step = max(1, initial_step // 2)
            new_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
            if new_freq:
                dominant_freq = new_freq
                already_checked_freqs.append(dominant_freq)

        cycle_length = int(1 / dominant_freq) if dominant_freq else max_cycle_length
        print(cycle_length)
        while (cycle_length < min_cycle_length or cycle_length > max_cycle_length) and dominant_freq:
            if cycle_length < min_cycle_length:
                new_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
                if new_freq:
                    dominant_freq = new_freq
                    cycle_length = int(1 / dominant_freq)
                    already_checked_freqs.append(dominant_freq)
            elif cycle_length > max_cycle_length:
                new_freq, _ = estimate_cycle_length(y_data, initial_step, already_checked_freqs)
                if new_freq:
                    dominant_freq = new_freq
                    cycle_length = int(1 / dominant_freq)
                    already_checked_freqs.append(dominant_freq)
            else:
                break

    sub_cycle_length = max(1, int(cycle_length / sub_cycle_divisor))

    print(f"Cycle Length: {cycle_length}")
    print(f"Sub-Cycle Length: {sub_cycle_length}")

    if plot_results and cycle_length >= min_cycle_length and cycle_length <= max_cycle_length:
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label='Data')
        if not user_cycle_length and dominant_freq:
            initial_guess = [(y_data.max() - y_data.min()) / 2, 2 * np.pi * dominant_freq, 0, y_data.mean()]
            try:
                params, _ = curve_fit(sine_wave, x_data, y_data, p0=initial_guess, maxfev=2000)
                plt.plot(x_data, sine_wave(x_data, *params), color='lime', label='Fitted Sine Wave')
            except RuntimeError as e:
                print(f"Curve fitting failed: {e}")
        for i in np.arange(0, len(x_data), cycle_length):
            plt.axvline(x=i, color='black', linestyle='-', linewidth=1.5)
        for i in np.arange(0, len(x_data), sub_cycle_length):
            plt.axvline(x=i, color='black', linestyle='--', linewidth=1)
        for start, end in zip(machine_labels.index[machine_labels.iloc[:, 0] == 1], machine_labels.index[machine_labels.iloc[:, 0].shift(-1) == 1]):
            plt.axvspan(start, end, color='red', alpha=0.5, label='Anomaly' if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.title(f'Sine Wave Fitting to Feature {feature}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.show()

    nominal_sub_cycles = [[] for _ in range(sub_cycle_divisor)]
    anomalous_sub_cycles = [[] for _ in range(sub_cycle_divisor)]

    for start in range(0, len(x_data), cycle_length):
        for i in range(sub_cycle_divisor):
            sub_cycle_start = start + i * sub_cycle_length
            sub_cycle_end = sub_cycle_start + sub_cycle_length
            if sub_cycle_end <= len(x_data):
                sub_cycle_data = y_data[sub_cycle_start:sub_cycle_end]
                sub_cycle_labels = machine_labels.iloc[sub_cycle_start:sub_cycle_end, 0].to_numpy()
                if np.any(sub_cycle_labels == 1):
                    anomalous_sub_cycles[i].append(sub_cycle_data)
                else:
                    nominal_sub_cycles[i].append(sub_cycle_data)

    return nominal_sub_cycles, anomalous_sub_cycles, cycle_length

#Check that we dont enable conflicting options
def validate_cycle_length_selection(user_defined_cycle_length, use_average_cycle_length, use_most_common_cycle_length, use_individual_cycle_length):
    # Create a list of the boolean variables
    options = [user_defined_cycle_length, use_average_cycle_length, use_most_common_cycle_length, use_individual_cycle_length]
    
    # Count the number of False values
    false_count = options.count(True)
    
    # Assert that no more than one option is False
    assert false_count <= 1, "More than one option is set to False, which is not allowed."
    
def compute_wavelet_features(signal, wavelet='db1', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for coeff in coeffs:
        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(np.sum(coeff**2))  # Energy of coefficients
    return np.concatenate(features)

def is_near_constant(array, epsilon=1e-6):
    """Check if an array is near constant"""
    return np.ptp(array) < epsilon

def fourier_transform_features(data):
    fft_vals = np.fft.fft(data)
    fft_power = np.abs(fft_vals)
    return {
        'fft_mean': np.mean(fft_power),
        'fft_std': np.std(fft_power),
        'fft_max': np.max(fft_power),
    }

def rolling_features(data, window_size=5):
    features = pd.DataFrame()
    features['rolling_mean'] = data.rolling(window=window_size).mean()
    features['rolling_std'] = data.rolling(window=window_size).std()
    features['rolling_var'] = data.rolling(window=window_size).var()
    return features

def compute_mahalanobis(nom_cycle, anom_cycle):
    """Compute Mahalanobis distance between two cycles"""
    data = np.vstack([nom_cycle, anom_cycle])
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    return mahalanobis(nom_cycle, anom_cycle, inv_covmat)

def ar_model_coefficients(data, lags=5):
    model = AutoReg(data, lags=lags).fit()
    return model.params

def compute_spectral_entropy(signal):
    """Compute spectral entropy of a signal"""
    if len(signal) < 2:
        return 0
    psd = np.abs(np.fft.fft(signal))**2
    psd_sum = psd.sum()
    if psd_sum == 0:
        return 0
    psd_norm = psd / psd_sum
    return entropy(psd_norm)

def compute_similarity(cycle_1, cycle_2, feature, sub_cycle_index, comparison_type):
    # Check for near constant arrays
    cycle1_near_constant = is_near_constant(cycle_1)
    cycle2_near_constant = is_near_constant(cycle_2)
    
    # Calculate Pearson and Spearman correlations
    if cycle1_near_constant or cycle2_near_constant:
        pearson_corr = 0
        spearman_corr = 0
    else:
        try:
            pearson_corr, _ = pearsonr(cycle_1, cycle_2)
        except Exception as e:
            print(f"Pearson calculation error for feature {feature}, sub-cycle {sub_cycle_index + 1}: {e}")
            pearson_corr = 0
        
        try:
            spearman_corr, _ = spearmanr(cycle_1, cycle_2)
        except Exception as e:
            print(f"Spearman calculation error for feature {feature}, sub-cycle {sub_cycle_index + 1}: {e}")
            spearman_corr = 0
    
    # Calculate Euclidean distance
    euclidean_dist = euclidean(cycle_1, cycle_2)
    
    # Calculate Cosine similarity
    if np.all(cycle_1 == 0) or np.all(cycle_2 == 0):
        cosine_sim = 0
    else:
        cosine_sim = 1 - cosine(cycle_1, cycle_2)

    # Calculate RMSE
    rmse_val = np.sqrt(mean_squared_error(cycle_1, cycle_2))
        
#     # Calculate DTW distance
#     try:
#         dtw_dist = compute_dtw(cycle_1, cycle_2)
#     except:
#         dtw_dist = np.nan

    if cycle1_near_constant or cycle2_near_constant:
        skew_diff = 0
        kurtosis_diff = 0
    elif len(cycle_1) > 1 and len(cycle_2) > 1:
        try:
            skew_diff = np.abs(skew(cycle_1) - skew(cycle_2))
        except:
            skew_diff = 0
        try:
            kurtosis_diff = np.abs(kurtosis(cycle_1) - kurtosis(cycle_2))
        except:
            kurtosis_diff = 0
    else:
        skew_diff = 0
        kurtosis_diff = 0
        
    # Calculate Spectral Entropy
    try:
        spectral_entropy_diff = np.abs(compute_spectral_entropy(cycle_1) - compute_spectral_entropy(cycle_2))
    except Exception as e:
        print(f"Spectral Entropy calculation error for feature {feature}, sub-cycle {sub_cycle_index + 1}: {e}")
        spectral_entropy_diff = 0
    
    # Calculate additional statistics
    std_diff = np.abs(np.std(cycle_1) - np.std(cycle_2))
    min_diff = np.abs(np.min(cycle_1) - np.min(cycle_2))
    max_diff = np.abs(np.max(cycle_1) - np.max(cycle_2))
    mean_diff = np.abs(np.mean(cycle_1) - np.mean(cycle_2))
    var_diff = np.abs(np.var(cycle_1) - np.var(cycle_2))
    mad_diff = np.abs(np.median(np.abs(cycle_1 - np.median(cycle_1))) - 
                      np.median(np.abs(cycle_2 - np.median(cycle_2))))
    iqr_diff = np.abs(np.percentile(cycle_1, 75) - np.percentile(cycle_1, 25) - 
                      np.percentile(cycle_2, 75) + np.percentile(cycle_2, 25))
    mad_mean_diff = np.abs(np.mean(np.abs(cycle_1 - np.mean(cycle_1))) - 
                           np.mean(np.abs(cycle_2 - np.mean(cycle_2))))
    energy_diff = np.abs(np.sum(cycle_1**2) - np.sum(cycle_2**2))
    

    # Calculate FFT and compare the magnitudes
    if len(cycle_1) > 10 and len(cycle_2) > 10:  # Only calculate FFT if length > 10
        try:
            fft_nom = np.abs(np.fft.fft(cycle_1))
            fft_anom = np.abs(np.fft.fft(cycle_2))
            fft_diff = np.mean(np.abs(fft_nom - fft_anom))
        except Exception as e:
            print(f"FFT calculation error for feature {feature}, sub-cycle {sub_cycle_index + 1}: {e}")
            fft_diff = np.nan
    else:
        fft_diff = np.nan

#     try:
#         wavelet_cycle1 = compute_wavelet_features(cycle_1)
#         wavelet_cycle2 = compute_wavelet_features(cycle_2)
#         wavelet_diff = np.mean(np.abs(wavelet_cycle1 - wavelet_cycle2))
#     except Exception as e:
#         wavelet_diff = np.nan

    # Calculate Rolling Statistics
    rolling_1 = rolling_features(pd.Series(cycle_1)).dropna()
    rolling_2 = rolling_features(pd.Series(cycle_2)).dropna()
    
    # Calculate Fourier Transform Features
    fft_features_1 = fourier_transform_features(cycle_1)
    fft_features_2 = fourier_transform_features(cycle_2)
    
    # Calculate AR Model Coefficients
    #ar_coeffs_1 = ar_model_coefficients(cycle_1)
    #ar_coeffs_2 = ar_model_coefficients(cycle_2)

    return {
        'feature': feature,
        'sub_cycle_index': sub_cycle_index + 1,
        'comparison_type': comparison_type,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'euclidean_distance': euclidean_dist,
        'cosine_similarity': cosine_sim,
        'rmse': rmse_val,
        #'dtw_distance': dtw_dist,
        'std_diff': std_diff,
        'min_diff': min_diff,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'var_diff': var_diff,
        'mad_diff': mad_diff,
        'iqr_diff': iqr_diff,
        'mad_mean_diff': mad_mean_diff,
        'energy_diff': energy_diff,
        'fft_diff': fft_diff,
        #'wavelet_diff': wavelet_diff,
        'skew_diff': skew_diff,
        'kurtosis_diff': kurtosis_diff,
        'spectral_entropy_diff': spectral_entropy_diff,
        'rolling_mean_diff': np.abs(rolling_1['rolling_mean'].mean() - rolling_2['rolling_mean'].mean()),
        'rolling_std_diff': np.abs(rolling_1['rolling_std'].mean() - rolling_2['rolling_std'].mean()),
        'fft_mean_diff': np.abs(fft_features_1['fft_mean'] - fft_features_2['fft_mean']),
        'fft_std_diff': np.abs(fft_features_1['fft_std'] - fft_features_2['fft_std']),
        'fft_max_diff': np.abs(fft_features_1['fft_max'] - fft_features_2['fft_max']),
        #'ar_coeff_diff': np.linalg.norm(ar_coeffs_1 - ar_coeffs_2)
    }



def calculate_similarity(sub_cycles, max_nominal_comparisons, max_anomalous_comparisons,sub_cycle_divisor):
    nominal_similarities = []
    anomalous_similarities = []

    print("Calculating Similarities")

    total_nominal_comparisons = len(sub_cycles.keys()) * sub_cycle_divisor * max_nominal_comparisons
    total_anomalous_comparisons = len(sub_cycles.keys()) * sub_cycle_divisor * max_anomalous_comparisons

    with tqdm(total=total_nominal_comparisons, desc="Nominal Comparisons") as nominal_pbar, \
         tqdm(total=total_anomalous_comparisons, desc="Anomalous Comparisons") as anomalous_pbar:
        for feature in sub_cycles.keys():
            for sub_cycle_index in range(sub_cycle_divisor):  # Assuming 3 sub-cycles
                nominal_comparison_count = 0
                anomalous_comparison_count = 0

                nominal_cycles = sub_cycles[feature]['nominal'][0][sub_cycle_index]
                anomalous_cycles = sub_cycles[feature].get('anomalous', [[], [], []])[0][sub_cycle_index]

                # Compare nominal cycles
                for j in range(len(nominal_cycles)):
                    if nominal_comparison_count >= max_nominal_comparisons:
                        break
                    for k in range(j + 1, len(nominal_cycles)):
                        if nominal_comparison_count >= max_nominal_comparisons:
                            break
                        if len(nominal_cycles[j]) == len(nominal_cycles[k]):
                            similarity = compute_similarity(nominal_cycles[j], nominal_cycles[k], feature, sub_cycle_index, "nominal")
                            nominal_similarities.append(similarity)
                            nominal_comparison_count += 1
                            nominal_pbar.update(1)

                # Compare nominal and anomalous cycles if anomalous cycles exist
                for nom_cycle in nominal_cycles:
                    if anomalous_comparison_count >= max_anomalous_comparisons:
                        break
                    for anom_cycle in anomalous_cycles:
                        if anomalous_comparison_count >= max_anomalous_comparisons:
                            break
                        if len(nom_cycle) == len(anom_cycle):
                            similarity = compute_similarity(nom_cycle, anom_cycle, feature, sub_cycle_index, "anomalous")
                            anomalous_similarities.append(similarity)
                            anomalous_comparison_count += 1
                            anomalous_pbar.update(1)

    return nominal_similarities, anomalous_similarities

def plot_features(nominal_df, anomalous_df, sub_cycle_divisor):
    features = [col for col in nominal_df.columns if col not in ['feature', 'sub_cycle_index', 'comparison_type']]
    sub_cycle_indices = nominal_df['sub_cycle_index'].unique()
    
    for feature in features:
        fig, axs = plt.subplots(1, sub_cycle_divisor, figsize=(sub_cycle_divisor * 5, 5), sharey=True)
        
        for i, sub_cycle_index in enumerate(sub_cycle_indices):
            nominal_values = nominal_df[nominal_df['sub_cycle_index'] == sub_cycle_index][feature]
            anomalous_values = anomalous_df[anomalous_df['sub_cycle_index'] == sub_cycle_index][feature]
            
            x_labels = ['Nominal', 'Anomalous']
            x_ticks = [0, 1]
            
            # Plot nominal data
            axs[i].scatter([x_ticks[0]] * len(nominal_values), nominal_values, color='blue', label='Nominal' if i == 0 else "")
            
            # Plot anomalous data
            axs[i].scatter([x_ticks[1]] * len(anomalous_values), anomalous_values, color='red', label='Anomalous' if i == 0 else "")
            
            axs[i].set_xticks(x_ticks)
            axs[i].set_xticklabels(x_labels)
            axs[i].set_title(f'Sub Cycle Index: {sub_cycle_index}')
            axs[i].set_ylabel(feature)
            axs[i].grid(True)
        
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.suptitle(f'Feature: {feature}', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ## Prediction functions

# %% [code]
# Define the f1c_score function
def f1c_score(y_true, y_pred):
    def get_events(y):
        events = []
        in_event = False
        start = 0
        for i in range(len(y)):
            if y[i] == 1 and not in_event:
                in_event = True
                start = i
            elif y[i] == 0 and in_event:
                in_event = False
                events.append((start, i))
        if in_event:
            events.append((start, len(y)))
        return events

    y_true = np.array(y_true)  # Ensure y_true is a numpy array
    y_pred = np.array(y_pred)  # Ensure y_pred is a numpy array

    true_events = get_events(y_true)
    pred_events = get_events(y_pred)

    tp_events = sum(any(pred_start <= true_start <= pred_end or pred_start <= true_end <= pred_end
                        for pred_start, pred_end in pred_events) for true_start, true_end in true_events)
    fp_events = sum(not any(pred_start <= true_start <= pred_end or pred_start <= true_end <= pred_end
                            for true_start, true_end in true_events) for pred_start, pred_end in pred_events)
    fn_events = len(true_events) - tp_events

    precision_event = tp_events / (tp_events + fp_events) if tp_events + fp_events > 0 else 0
    recall_event = tp_events / (tp_events + fn_events) if tp_events + fn_events > 0 else 0
    f1c = 2 * (precision_event * recall_event) / (precision_event + recall_event) if precision_event + recall_event > 0 else 0
    return f1c

# Custom evaluation function for XGBoost
def custom_f1c_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    y_pred_binary = np.round(y_pred)  # Convert probabilities to binary predictions
    return 'f1c', f1c_score(y_true, y_pred_binary)

# Function to introduce random NaNs in the data
def introduce_nans(data, nan_fraction=0.05, random_state=42):
    np.random.seed(random_state)
    nan_mask = np.random.rand(*data.shape) < nan_fraction
    data[nan_mask] = np.nan
    return data

# Function to save the uploaded file
def handle_upload(change):
    global uploaded_file_path
    # Get the uploaded file's metadata
    uploaded_filename = list(upload_button.value.keys())[0]
    uploaded_file = upload_button.value[uploaded_filename]

    # Save the file to the Kaggle working directory
    uploaded_file_path = os.path.join('/kaggle/working', uploaded_filename)
    with open(uploaded_file_path, 'wb') as f:
        f.write(uploaded_file['content'])

    print(f"File uploaded and saved to {uploaded_file_path}")
    clear_output()
    display(upload_button)
    display(load_button)

# Function to load the dictionary from JSON
def load_from_json():
    global loaded_config, uploaded_file_path
    if uploaded_file_path:
        with open(uploaded_file_path, 'r') as json_file:
            autoencoders_dict = json.load(json_file)
        for key, value in autoencoders_dict.items():
            for func in value['functions']:
                func['regularization'] = func['regularization']
        loaded_config = autoencoders_dict
        print("Loaded configuration:", loaded_config)
    else:
        print("No file has been uploaded yet.")
        
        
class MemoryModule(Layer):
    def __init__(self, memory_size, feature_dim, **kwargs):
        super(MemoryModule, self).__init__(**kwargs)
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.memory = self.add_weight(shape=(self.memory_size, self.feature_dim),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        similarity = tf.matmul(inputs, self.memory, transpose_b=True)
        attention = tf.nn.softmax(similarity, axis=1)
        memory_read = tf.matmul(attention, self.memory)
        self.add_loss(-tf.reduce_mean(tf.reduce_sum(attention * tf.math.log(attention + 1e-8), axis=-1)))
        return memory_read, attention

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.feature_dim), (input_shape[0], self.memory_size)]

        
# Function to train individual autoencoder for a single feature
def train_autoencoder(X_train, X_val, autoencoders_dict, index):
    # Build the autoencoder model
    input_dim = X_train.shape[1]
    encoding_dim = X_train.shape[1]  # Set encoding dimension to a lower value

    input_layer = Input(shape=(input_dim,))
    x = GaussianNoise(0.1)(input_layer)

    # Encoder
    x = Dense(64, activation='relu', activity_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu', activity_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    encoder_output = Dense(encoding_dim, activation='relu', activity_regularizer=l2(0.01))(x)
    encoder_output = Dropout(0.2)(encoder_output)

    skip_input_encoder = add([encoder_output, input_layer])

    # Decoder
    x = Dense(32, activation='relu', activity_regularizer=l2(0.01))(skip_input_encoder)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', activity_regularizer=l2(0.01))(x)
    
    #skip_input_decoder = add([x, input_layer])
    x = BatchNormalization()(x)
    x = Dense(input_dim, activation='relu', activity_regularizer=l2(0.01))(x)

    autoencoder = Model(inputs=input_layer, outputs=x)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the autoencoder
    history = autoencoder.fit(X_train, X_train,
                              epochs=30,
                              batch_size=128,
                              shuffle=True,
                              validation_data=(X_val, X_val),
                              callbacks=[early_stopping],
                              verbose=0)
#################################################################################
#     input_dim = X_train.shape[1]
#     encoding_dim = 16#X_train.shape[1]#32  # Set encoding dimension to a lower value
#     memory_size = 50   # Define memory size
    
#     input_layer = Input(shape=(input_dim,))
#     x = GaussianNoise(0.2)(input_layer)

#     # Encoder
# #     x = Dense(512, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
# #     x = LeakyReLU(negative_slope=0.1)(x)
# #     x = BatchNormalization()(x)
# #     x = Dense(256, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
# #     x = LeakyReLU(negative_slope=0.1)(x)
# #     x = BatchNormalization()(x)
#     x = Dense(128, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
#     x = LeakyReLU(negative_slope=0.1)(x)
#     x = BatchNormalization()(x)
#     x = Dense(64, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
#     x = LeakyReLU(negative_slope=0.1)(x)
#     x = BatchNormalization()(x)
#     encoder_output = Dense(encoding_dim, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
#     encoder_output = LeakyReLU(negative_slope=0.1)(encoder_output)
#     encoder_output = Dropout(0.5)(encoder_output)
    
#     # Memory Module
#     memory_module = MemoryModule(memory_size, encoding_dim)
#     memory_read, attention = memory_module(encoder_output)


#     # Skip connection from input to encoder output
#     #skip_input_encoder = add([encoder_output, input_layer])
#     skip_input_encoder = add([memory_read, input_layer])

#     # Decoder
#     x = Dense(64, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(skip_input_encoder)
#     x = LeakyReLU(negative_slope=0.1)(x)
#     x = BatchNormalization()(x)
#     x = Dense(128, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
#     x = LeakyReLU(negative_slope=0.1)(x)
# #     x = Dense(256, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
# #     x = LeakyReLU(negative_slope=0.1)(x)
# #     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
# #     x = Dense(512, activity_regularizer=l2(0.01), kernel_initializer=he_normal())(x)
# #     x = LeakyReLU(negative_slope=0.1)(x)
# #     x = BatchNormalization()(x)
#     x = Dense(input_dim, activation='relu', activity_regularizer=l2(0.01))(x)

#     autoencoder = Model(inputs=input_layer, outputs=x)
#     opt = Adam(clipvalue=1.0)
#     autoencoder.compile(optimizer=opt, loss='mse')
#     #autoencoder.compile(optimizer=opt, loss=lambda y_true, y_pred: mse_loss_with_memory(y_true, y_pred, attention))

#     # Early stopping and learning rate reduction to prevent overfitting
#     early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

#     # Train the autoencoder
#     history = autoencoder.fit(X_train, X_train,
#                               epochs=30,
#                               batch_size=128,
#                               shuffle=True,
#                               validation_data=(X_val, X_val),
#                               callbacks=[early_stopping, reduce_lr],
#                               verbose=0)
#################################################################################

    # Storing the model in the dictionary
    autoencoders_dict[f'autoencoder_{index}'] = {'model': autoencoder}

    return autoencoder, X_val, history

# Corrected Event-Wise F-score
def corrected_event_wise_f_score(y_true, y_pred, beta=0.5):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    tp = ((y_pred_flat == 1) & (y_true_flat == 1)).sum()
    fp = ((y_pred_flat == 1) & (y_true_flat == 0)).sum()
    tn = ((y_pred_flat == 0) & (y_true_flat == 0)).sum()
    fn = ((y_pred_flat == 0) & (y_true_flat == 1)).sum()
    prec_corr = tp / (tp + fp) * (tn / (tn + fp + fn))
    rec = tp / (tp + fn)
    f_beta = (1 + beta**2) * (prec_corr * rec) / (beta**2 * prec_corr + rec)
    return f_beta

# Subsystem-Aware and Channel-Aware F-scores
def subsystem_aware_f_score(y_true, y_pred, subsystems, beta=0.5):
    subsystem_scores = []
    for subsystem in subsystems:
        y_true_sub = y_true[:, subsystem]
        y_pred_sub = y_pred[:, subsystem]
        f_beta = fbeta_score(y_true_sub.flatten(), y_pred_sub.flatten(), beta=beta)
        subsystem_scores.append(f_beta)
    return sum(subsystem_scores) / len(subsystem_scores)

def channel_aware_f_score(y_true, y_pred, beta=0.5):
    f_beta = fbeta_score(y_true.flatten(), y_pred.flatten(), beta=beta)
    return f_beta

# Event-Wise Alarming Precision
def event_wise_alarming_precision(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    tp = ((y_pred_flat == 1) & (y_true_flat == 1)).sum()
    tp_redundant = ((y_pred_flat == 1) & (y_true_flat == 1)).sum() - tp
    alarm_prec = tp / (tp + tp_redundant)
    return alarm_prec

# Anomaly Detection Timing Quality Curve (ADTQC)
def adtqc(y_true, y_pred, alpha=0.1, beta=0.9):
    def adtqc_function(x, alpha, beta):
        if x < -alpha:
            return 0
        elif -alpha <= x <= 0:
            return ((x + alpha) / alpha) ** np.e
        elif 0 < x < beta:
            return 1 / (1 + (x / (beta - x)) ** np.e)
        else:
            return 0
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    anomaly_starts = np.where(np.diff(y_true_flat, prepend=0) == 1)[0]
    detection_starts = np.where(np.diff(y_pred_flat, prepend=0) == 1)[0]
    adtqc_scores = []
    for detection_start in detection_starts:
        min_diff = min(anomaly_starts - detection_start, key=abs)
        adtqc_scores.append(adtqc_function(min_diff, alpha, beta))
    return np.mean(adtqc_scores)

# Modified Affiliation-Based F-score
def modified_affiliation_f_score(y_true, y_pred, beta=0.5):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    tp = ((y_pred_flat == 1) & (y_true_flat == 1)).sum()
    fp = ((y_pred_flat == 1) & (y_true_flat == 0)).sum()
    fn = ((y_pred_flat == 0) & (y_true_flat == 1)).sum()
    local_prec = tp / (tp + fp)
    local_rec = tp / (tp + fn)
    f_beta = (1 + beta**2) * (local_prec * local_rec) / (beta**2 * local_prec + local_rec)
    return f_beta

# Define the f1pa_score function
def f1pa_score(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1

# Split the nominal data into training, validation, and test sets before training
def split_nominal_data(nominal_df, test_size=0.4, val_size=0.5, random_state=42):
    
    random_state = random.randint(0, 1000)
    #print(f"random_state: {random_state}")
    X_train, X_temp = train_test_split(nominal_df, test_size=test_size, random_state=random_state)
    X_val, X_test_nominal = train_test_split(X_temp, test_size=val_size, random_state=random_state)
    return X_train, X_val, X_test_nominal


# Function to evaluate the ensemble of autoencoders
def evaluate_ensemble(autoencoders, X_test_combined, y_test, val_data_list, percentile_threshold ,global_encoder=None):
    # Combine reconstruction errors from all autoencoders
    reconstruction_errors = []
    for autoencoder, feature_data in zip(autoencoders, X_test_combined):
        #feature_data = imputer.transform(feature_data)  # Impute missing values
        test_predictions = autoencoder.predict(feature_data)
        reconstruction_error = np.mean(np.square(feature_data - test_predictions), axis=1)
        reconstruction_errors.append(reconstruction_error)
        
    # Average reconstruction errors
    avg_reconstruction_error = np.mean(np.vstack(reconstruction_errors), axis=0)

    # Set a threshold for anomaly detection
    #This section of code is a bit shitty but seems to do the job. Takes X_val from val_data_list, which is created in the next code block
    val_reconstruction_errors = [np.mean(np.square(X_val - autoencoder.predict(X_val)), axis=1) for autoencoder, X_val in zip(autoencoders, val_data_list)]
    combined_val_errors = np.mean(np.vstack(val_reconstruction_errors), axis=0)
    threshold = np.percentile(combined_val_errors, percentile_threshold)
    
    # Identify anomalies
    anomalies = avg_reconstruction_error > threshold

    # Compare predictions with actual labels
    f1c = f1c_score(y_test, anomalies.astype(int))
    f1 = f1_score(y_test, anomalies.astype(int))
    #precision, recall, f1pa = f1pa_score(y_test, anomalies.astype(int))
    precision, recall = precision_score(y_test, anomalies.astype(int)), recall_score(y_test, anomalies.astype(int))

    # Calculate additional metrics
    corrected_f1 = corrected_event_wise_f_score(y_test, anomalies)
    channel_f1 = channel_aware_f_score(y_test, anomalies)
    alarming_prec = event_wise_alarming_precision(y_test, anomalies)
    adtqc_score = adtqc(y_test, anomalies)
    affiliation_f1 = modified_affiliation_f_score(y_test, anomalies)

    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({
        'Reconstruction Error': avg_reconstruction_error,
        'Actual Label': y_test,
        'Predicted Anomaly': anomalies.astype(int),
        'Correct Prediction': (anomalies == y_test)
    })


    return results_df, f1c, f1, precision, recall, corrected_f1,channel_f1,alarming_prec,adtqc_score,affiliation_f1, threshold
    
def compute_gradient_descent(history):
    losses = history.history['val_loss']
    gradients = np.diff(losses)
    return np.min(gradients)