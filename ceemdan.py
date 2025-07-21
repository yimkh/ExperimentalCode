# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
from ssqueezepy import ssq_cwt
from multiprocessing import Pool, cpu_count
import time

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'])
        return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        raise

def preprocess_signal(signal):
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        print("Signal contains NaN or inf values. Interpolating...")
        signal = pd.Series(signal).interpolate().fillna(method='bfill').values
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(signal) == 0:
        print("Warning: Signal has zero variance. Adding small noise to enable decomposition...")
        signal = signal + np.random.normal(0, 1e-10, len(signal))
    return signal

# CEEMDAN
def ceemdan_decompose(signal_chunk):
    if len(signal_chunk) < 500:
        print(f"Warning: Signal chunk too short ({len(signal_chunk)} points). Skipping decomposition.")
        return np.zeros((1, len(signal_chunk)))
    ceemdan = CEEMDAN(trials=20, max_imf=8)
    imfs = ceemdan(signal_chunk)
    return imfs

# SST
def sst_postprocess(imfs):
    refined_imfs = []
    for imf in imfs:
        Tx, *_ = ssq_cwt(imf, wavelet='cmhat', nv=8)
        refined_imf = np.real(Tx.sum(axis=0))
        refined_imfs.append(refined_imf)
    return np.array(refined_imfs)

# IMFs
def select_imfs_by_correlation(imfs, signal, threshold=0.1):
    selected_imfs = []
    min_length = min(imfs.shape[1], len(signal))  
    signal = signal[:min_length]  
    for i, imf in enumerate(imfs):
        imf = imf[:min_length]  
        corr = np.corrcoef(imf, signal)[0, 1]
        if abs(corr) > threshold:
            selected_imfs.append(imf)
        print(f"IMF {i+1} 2.correlation: {corr:.4f}")
    return np.array(selected_imfs)

def select_imfs_by_energy(imfs, energy_threshold=0.01):
    total_energy = np.sum([np.sum(imf**2) for imf in imfs])
    selected_imfs = []
    for i, imf in enumerate(imfs):
        energy = np.sum(imf**2)
        energy_ratio = energy / total_energy
        if energy_ratio > energy_threshold:
            selected_imfs.append(imf)
        print(f"IMF {i+1} energy ratio: {energy_ratio:.4f}")
    return np.array(selected_imfs)

def plot_imfs(imfs, signal):
    num_imfs = imfs.shape[0]
    min_length = min(imfs.shape[1], len(signal))
    signal = signal[:min_length]
    imfs = imfs[:, :min_length]
    fig, axs = plt.subplots(num_imfs + 1, 1, figsize=(12, 2 * (num_imfs + 1)))

    signal_subset = signal[:1000]
    axs[0].plot(signal_subset)
    axs[0].set_title("Original Signal (PV_Active_Power, first 1000 points)")

    for i in range(num_imfs):
        imf_subset = imfs[i][:1000]
        axs[i + 1].plot(imf_subset)
        axs[i + 1].set_title(f"IMF {i + 1} (CEEMDAN + SST)")

    plt.tight_layout()
    plt.show()

def save_imfs_to_excel(imfs, file_path):
    df = pd.DataFrame(imfs.T)
    df.to_csv(file_path, index=False)
    print(f"IMFs saved to {file_path}")

def main():
    file_path = "data.CSV"
    save_path = "imfs_output.csv"
    print("Loading data...")
    start_time = time.time()
    data = load_data(file_path)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")

    signal = data['PV_Active_Power'].values
    print(f"Signal length: {len(signal)}")

    signal = preprocess_signal(signal)

    num_chunks = cpu_count()
    chunk_size = len(signal) // num_chunks
    if chunk_size < 5000:
        num_chunks = max(1, len(signal) // 5000)
        chunk_size = len(signal) // num_chunks
    signal_chunks = [signal[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
    signal_chunks = [chunk for chunk in signal_chunks if len(chunk) > 0]
    print(f"Number of chunks: {len(signal_chunks)}, chunk size: {chunk_size}")

    print("Running CEEMDAN decomposition...")
    start_time = time.time()
    if num_chunks == 1:
        print("Using single-threaded CEEMDAN decomposition...")
        imfs = ceemdan_decompose(signal)
    else:
        try:
            with Pool(processes=len(signal_chunks)) as pool:
                results = pool.map(ceemdan_decompose, signal_chunks)
            total_length = sum(len(chunk) for chunk in signal_chunks)
            imfs = np.concatenate(results, axis=1)
            if imfs.shape[1] != len(signal):
                print(f"Warning: IMFs length ({imfs.shape[1]}) does not match signal length ({len(signal)}). Adjusting...")
                if imfs.shape[1] < len(signal):
                    padding = np.zeros((imfs.shape[0], len(signal) - imfs.shape[1]))
                    imfs = np.concatenate((imfs, padding), axis=1)
                else:
                    imfs = imfs[:, :len(signal)]
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Switching to single-threaded CEEMDAN decomposition...")
            imfs = ceemdan_decompose(signal)
    print(f"IMFs length after decomposition: {imfs.shape[1]}")
    print(f"CEEMDAN decomposition completed in {time.time() - start_time:.2f} seconds")

    print("Running SST post-processing...")
    start_time = time.time()
    refined_imfs = sst_postprocess(imfs)
    print(f"Number of IMFs after decomposition: {refined_imfs.shape[0]}")
    print(f"IMFs length after SST: {refined_imfs.shape[1]}")
    print(f"SST post-processing completed in {time.time() - start_time:.2f} seconds")

    print("Selecting IMFs based on 2.correlation...")
    selected_imfs = select_imfs_by_correlation(refined_imfs, signal, threshold=0.1)
    print(f"Number of selected IMFs (2.correlation): {selected_imfs.shape[0]}")

    print("Selecting IMFs based on energy...")
    selected_imfs = select_imfs_by_energy(selected_imfs, energy_threshold=0.01)
    print(f"Number of selected IMFs (energy): {selected_imfs.shape[0]}")

    print("Saving selected IMFs...")
    start_time = time.time()
    save_imfs_to_excel(selected_imfs, save_path)
    print(f"IMFs saved in {time.time() - start_time:.2f} seconds")

    print("Plotting results...")
    plot_imfs(selected_imfs, signal)

if __name__ == "__main__":
    main()
