"""Plotting utility functions for paper-quality figures."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime


def save_figure(fig, save_path, dpi=None):
    """Save a figure without aborting when a previous Windows PDF is open.

    Windows locks PDFs opened by many viewers.  Rather than losing a completed
    experiment at its reporting stage, preserve the new figure using a unique
    timestamped filename and report where it was written.
    """
    try:
        fig.savefig(save_path, dpi=dpi)
        return save_path
    except PermissionError:
        base, extension = os.path.splitext(save_path)
        fallback = f"{base}_{datetime.now():%Y%m%d_%H%M%S}{extension}"
        fig.savefig(fallback, dpi=dpi)
        print(f"  Warning: '{save_path}' is open or locked; saved '{fallback}' instead.")
        return fallback

def setup_plot_style(config=None):
    """Set matplotlib style for publication."""
    plt.style.use('default')
    
    # Publication quality settings
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'font.family': 'serif',
        'mathtext.fontset': 'stix'
    })
    
    # Set colorblind friendly palette
    sns.set_palette("colorblind")

def plot_convergence_curves(results_dict, save_path, title):
    """Plot accuracy vs round for multiple methods."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    for i, (method_name, data) in enumerate(results_dict.items()):
        rounds = data.get('rounds', np.arange(len(data['accuracy'])))
        ax.plot(rounds, data['accuracy'], 
                label=method_name, 
                marker=markers[i % len(markers)], 
                markevery=max(1, len(rounds)//10))
        
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_bar_comparison(labels, values, save_path, title, ylabel):
    """Bar chart comparing methods."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(labels))
    width = 0.5
    
    colors = sns.color_palette("colorblind", len(labels))
    bars = ax.bar(x, values, width, color=colors)
    
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_cdf(data_dict, save_path, title, xlabel):
    """CDF plot for latency or SNR."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    linestyles = ['-', '--', '-.', ':']
    for i, (method_name, values) in enumerate(data_dict.items()):
        sorted_data = np.sort(values)
        p = 1. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
        ax.plot(sorted_data, p, 
                label=method_name, 
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.5)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_heatmap(data_2d, x_labels, y_labels, save_path, title):
    """Heatmap for 2D parameter sweeps."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(data_2d, 
                xticklabels=x_labels, 
                yticklabels=y_labels, 
                cmap='viridis', 
                annot=True, 
                fmt=".3f", 
                ax=ax)
    
    ax.set_title(title)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_mse_vs_param(param_values, mse_values_dict, save_path, xlabel, title):
    """MSE vs parameter (e.g., N or SNR) on log scale."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    for i, (method_name, mse_values) in enumerate(mse_values_dict.items()):
        ax.semilogy(param_values, mse_values, 
                   label=method_name, 
                   marker=markers[i % len(markers)],
                   linewidth=2, markersize=8)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_title(title)
    ax.grid(True, which="both", linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def plot_fairness(alpha_values, fairness_dict, save_path):
    """Fairness index vs Dirichlet alpha."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    for i, (method_name, fairness_values) in enumerate(fairness_dict.items()):
        ax.plot(alpha_values, fairness_values, 
                label=method_name, 
                marker=markers[i % len(markers)],
                linewidth=2, markersize=8)
        
    ax.set_xlabel(r'Dirichlet parameter $\alpha$ (Data Heterogeneity)')
    ax.set_ylabel('Jain\'s Fairness Index')
    ax.set_title('Fairness vs. Data Heterogeneity')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower right')
    
    # Invert x-axis if alpha is shown decreasing
    if alpha_values[0] > alpha_values[-1]:
        ax.invert_xaxis()
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
