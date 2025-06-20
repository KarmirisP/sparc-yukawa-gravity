import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent display errors
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from textwrap import wrap
import warnings

# --- Setup Logging & Warnings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Physical Constants ---
# MOND acceleration constant a0 in km^2/s^2/kpc
A0_KMS_KPC = 3700 

# ==============================================================================
#  MODEL DEFINITIONS
# ==============================================================================

def get_baryonic_velocity_sq(r_packed, ups_star):
    """Helper function to get baryonic velocity squared."""
    _, v_gas, v_disk, v_bulge, _, _ = r_packed
    return v_gas**2 + ups_star * (v_disk**2 + v_bulge**2)

# --- Geometric & MOND-based Models ---

def mond_eg_velocity(r_packed, ups_star, a0):
    """
    Calculates rotational velocity for MOND (using the simple interpolating function).
    This is the more accurate version of the MOND-EG model.
    v_obs^4 = v_bary^4 + v_bary^2 * a0 * r
    """
    radius, _, _, _, _, _ = r_packed
    v_bary_sq = get_baryonic_velocity_sq(r_packed, ups_star)
    v_obs_4 = v_bary_sq**2 + v_bary_sq * radius * a0
    return np.sqrt(np.sqrt(np.maximum(0, v_obs_4)))

def env_mond_velocity(r_packed, ups_star, a0, beta_env):
    """
    Environment-Dependent MOND (E-MOND).
    a0 is modified by the galaxy's mean surface brightness.
    """
    radius, _, _, _, sb_disk, sb_bulge = r_packed
    v_bary_sq = get_baryonic_velocity_sq(r_packed, ups_star)
    
    sb_local_disk = sb_disk[sb_disk > 0]
    sb_local_bulge = sb_bulge[sb_bulge > 0]
    if len(sb_local_disk) > 0 or len(sb_local_bulge) > 0:
        sb_mean = np.mean(np.concatenate((sb_local_disk, sb_local_bulge)))
    else:
        sb_mean = 0

    sb_local = ups_star * sb_mean
    sb_ref = 100.0
    
    a_eff = a0 * (1 + beta_env * (sb_local / sb_ref))
    
    v_obs_4 = v_bary_sq**2 + v_bary_sq * radius * a_eff
    return np.sqrt(np.sqrt(np.maximum(0, v_obs_4)))

def quartic_mond_velocity(r_packed, ups_star, a0, lambda_4):
    """
    MOND with a quartic correction term (non-linear baryonic self-interaction).
    """
    v_mond_4 = mond_eg_velocity(r_packed, ups_star, a0)**4
    v_bary_4 = get_baryonic_velocity_sq(r_packed, ups_star)**2
    
    v_total_4 = v_mond_4 + lambda_4 * v_bary_4
    return np.sqrt(np.sqrt(np.maximum(0, v_total_4)))

def oscillatory_mond_velocity(r_packed, ups_star, a0, epsilon, r_osc):
    """
    MOND combined with the oscillatory "swinging effect" from Full TF.
    """
    radius, _, _, _, _, _ = r_packed
    v_mond_sq = mond_eg_velocity(r_packed, ups_star, a0)**2
    
    if np.abs(r_osc) < 1e-6: r_osc = 1e-6
    
    modification = 1.0 + epsilon * np.sin(radius / r_osc)**2
    v_total_sq = v_mond_sq * modification
    return np.sqrt(np.maximum(0, v_total_sq))

# --- CGF & TF Models ---

def simple_cgf_model(r_packed, ups_star, m_cgf, alpha):
    """ Vanilla CGF model (Yukawa potential). """
    radius, v_gas, v_disk, v_bulge, _, _ = r_packed
    v_bary_sq = v_gas**2 + ups_star * (v_disk**2 + v_bulge**2)
    if np.abs(m_cgf) < 1e-6: return np.sqrt(np.maximum(0, v_bary_sq))
    
    lambda_cgf = 1.0 / m_cgf
    modification = 1.0 + alpha * np.exp(-radius / lambda_cgf) * (1.0 + radius / lambda_cgf)
    v_total_sq = v_bary_sq * modification
    return np.sqrt(np.maximum(0, v_total_sq))

def full_cgf_model(r_packed, ups_star, m_cgf, alpha, lambda_phi, xi):
    """ Full CGF model with a quartic potential term. """
    radius, _, _, _, _, _ = r_packed
    v_bary_sq = get_baryonic_velocity_sq(r_packed, ups_star)
    if np.abs(m_cgf) < 1e-6: return np.sqrt(np.maximum(0, v_bary_sq))

    yukawa_term = alpha * np.exp(-m_cgf * radius) * (1.0 + m_cgf * radius + xi * (m_cgf * radius)**2)
    potential_term = 1.0 + lambda_phi * np.sqrt(radius / (radius + 1.0))
    
    modification = 1.0 + yukawa_term * potential_term
    v_total_sq = v_bary_sq * modification
    return np.sqrt(np.maximum(0, v_total_sq))

def simple_tf_model(r_packed, ups_star, m_tf, epsilon):
    """ Simple Temporal Field model (phenomenologically same as Simple CGF). """
    return simple_cgf_model(r_packed, ups_star, m_tf, epsilon)

def full_tf_model(r_packed, ups_star, m_tf, epsilon, r_osc):
    """ Full Temporal Field model with an oscillatory modification term. """
    radius, _, _, _, _, _ = r_packed
    v_bary_sq = get_baryonic_velocity_sq(r_packed, ups_star)
    
    if np.abs(r_osc) < 1e-6: r_osc = 1e-6

    modification = 1.0 + epsilon * (1 - np.exp(-m_tf * radius)) * np.sin(radius / r_osc)**2
    v_total_sq = v_bary_sq * modification
    return np.sqrt(np.maximum(0, v_total_sq))

# --- Standard Model for Reference ---

def lcdm_model(r_packed, ups_star, v_halo, r_halo):
    """Standard Lambda-CDM model with an NFW dark matter halo."""
    radius, _, _, _, _, _ = r_packed
    v_bary_sq = get_baryonic_velocity_sq(r_packed, ups_star)
    if np.abs(r_halo) < 1e-6: return np.sqrt(np.maximum(0, v_bary_sq))
    
    x = radius / r_halo
    nfw_term_safe = np.log(1 + x) - x / (1 + x)
    v_nfw_sq = np.divide(v_halo**2 * nfw_term_safe, x, out=np.zeros_like(x), where=x!=0)
    v_total_sq = v_bary_sq + v_nfw_sq
    return np.sqrt(np.maximum(0, v_total_sq))

# ==============================================================================
#  ANALYSIS CORE
# ==============================================================================

def load_sparc_data(filepath):
    """Loads a single SPARC .dat file."""
    try:
        data = np.loadtxt(filepath)
        return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7]
    except Exception as e:
        logging.error(f"Could not read file {filepath}: {e}")
        return None

def calculate_aic(chi2, k):
    return chi2 + 2 * k

def analyze_galaxy(filepath):
    """Performs the full analysis for a single galaxy, fitting all models."""
    data = load_sparc_data(filepath)
    if data is None: return None
        
    radius, v_obs, err_v, v_gas, v_disk, v_bulge, sb_disk, sb_bulge = data
    n_points = len(radius)
    if n_points < 8: return None # Need enough points for more complex models

    r_packed = (radius, v_gas, v_disk, v_bulge, sb_disk, sb_bulge)
    results = {'galaxy': os.path.basename(filepath).replace('.dat', '')}

    models = {
        "MOND_EG": {"func": mond_eg_velocity, "p_names": ["ups_star", "a0"], "p0": [0.5, 3700], "bounds": ([0.1, 1000], [2.0, 8000])},
        "E_MOND": {"func": env_mond_velocity, "p_names": ["ups_star", "a0", "beta_env"], "p0": [0.5, 3700, 0.1], "bounds": ([0.1, 1000, -0.9], [2.0, 8000, 0.9])},
        "Quartic_MOND": {"func": quartic_mond_velocity, "p_names": ["ups_star", "a0", "lambda_4"], "p0": [0.5, 3700, 0.1], "bounds": ([0.1, 1000, -1.0], [2.0, 8000, 1.0])},
        "Oscillatory_MOND": {"func": oscillatory_mond_velocity, "p_names": ["ups_star", "a0", "epsilon", "r_osc"], "p0": [0.5, 3700, 0.1, 5.0], "bounds": ([0.1, 1000, 0, 0.1], [2.0, 8000, 1.0, 50.0])},
        "Simple_CGF": {"func": simple_cgf_model, "p_names": ["ups_star", "m_cgf", "alpha"], "p0": [0.5, 1.0, 1.0], "bounds": ([0.1, 0.01, 0.1], [2.0, 2.0, 50.0])},
        "Full_TF": {"func": full_tf_model, "p_names": ["ups_star", "m_tf", "epsilon", "r_osc"], "p0": [0.5, 1.0, 0.1, 5.0], "bounds": ([0.1, 0.01, 0, 0.1], [2.0, 2.0, 1.0, 50.0])},
        "LCDM": {"func": lcdm_model, "p_names": ["ups_star", "v_halo", "r_halo"], "p0": [0.5, 50, 10], "bounds": ([0.1, 10, 0.5], [10.0, 500, 100])}
    }

    for name, model in models.items():
        try:
            popt, _ = curve_fit(model["func"], r_packed, v_obs, sigma=err_v, bounds=model["bounds"], p0=model["p0"], maxfev=15000)
            v_fit = model["func"](r_packed, *popt)
            k = len(popt)
            chi2 = np.sum(((v_obs - v_fit) / err_v)**2)
            results[f'{name}_aic'] = calculate_aic(chi2, k)
            results[f'{name}_chi2_red'] = chi2 / (n_points - k) if (n_points - k) > 0 else np.inf
            for i, p_name in enumerate(model["p_names"]):
                results[f'{name}_{p_name}'] = popt[i]
        except (RuntimeError, ValueError) as e:
            results[f'{name}_aic'] = np.inf
            results[f'{name}_chi2_red'] = np.inf
            logging.warning(f"Fit failed for {name} on {results['galaxy']}: {e}")
            
    return results

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main(data_dir, output_dir, limit=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dat')])
    if limit: all_files = all_files[:limit]
    logging.info(f"Starting unified analysis of all models on {len(all_files)} galaxies...")

    all_results = [analyze_galaxy(fp) for fp in all_files]
    df_results = pd.DataFrame([res for res in all_results if res is not None])
    
    if df_results.empty:
        logging.error("No galaxies were successfully analyzed. Aborting.")
        return

    output_csv_path = os.path.join(output_dir, 'unified_model_results.csv')
    df_results.to_csv(output_csv_path, index=False)
    logging.info(f"Full results saved to {output_csv_path}")

    aic_cols = [col for col in df_results.columns if '_aic' in col]
    df_results['best_model'] = df_results[aic_cols].idxmin(axis=1).str.replace('_aic', '')
    
    model_wins = df_results['best_model'].value_counts()
    
    print("\n" + "="*60)
    print("      SPARC Analysis Summary: All Models")
    print("="*60 + "\n")
    print("--- Model Selection (AIC Wins) ---")
    print(model_wins)
    print("\n--- Mean Reduced Chi-Squared (Lower is Better) ---")
    chi2_cols = [col for col in df_results.columns if '_chi2_red' in col]
    print(df_results[chi2_cols].mean().sort_values())
    print("\n")

    plt.figure(figsize=(12, 8))
    sns.countplot(x='best_model', data=df_results, order=model_wins.index, palette='viridis')
    plt.title('Model Performance Comparison (AIC Wins)', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Number of Galaxies (Best Fit)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'unified_models_comparison.png')
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Comparison plot saved to {plot_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test all alternative and hybrid gravity theories against SPARC data.")
    parser.add_argument('data_dir', type=str, help="Directory containing SPARC.dat files.")
    parser.add_argument('--output_dir', type=str, default='unified_theory_analysis', help="Directory to save results.")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of galaxies to analyze for a quick test.")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.limit)
