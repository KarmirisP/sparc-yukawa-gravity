# sparc-yukawa-gravity
The Cosmic Gravitational Field Theory: A Comprehensive Analysis of 175 SPARC Galaxies Revealing Universal Parameters

This repository contains the Python code used in the paper "The Cosmic Gravitational Field Theory: A Comprehensive Analysis of 175 SPARC Galaxies Revealing Universal Parameters by P. Karmiris (2025).

## Overview

This code analyzes 175 galaxy rotation curves from the SPARC database, comparing:
- Yukawa-modified gravity models (CGF, TF, Basic Yukawa)  
- Standard ΛCDM with NFW dark matter halos

Key finding: The Yukawa modification scale λ ≈ 0.57 kpc is universal across all galaxy types.

## Requirements

- Python 3.8+
- NumPy, SciPy, Pandas, Matplotlib
- See requirements.txt for complete list

## Installation

```bash
git clone https://github.com/KarmirisP/sparc-yukawa-gravity.git
cd sparc-yukawa-gravity
pip install -r requirements.txt
Usage

Download SPARC data from http://astroweb.cwru.edu/SPARC/
Place .txt files in data/ directory
Run analysis:

bashpython sparc_analysis.py data/ --output_dir results/
Citation
If you use this code, please cite:
@article{Karmiris2025,
  title = {The Cosmic Gravitational Field Theory: A Comprehensive Analysis of 175 SPARC Galaxies Revealing Universal Parameters},
  author = {Karmiris, Panagiotis},
  journal = {[Preprints]},
  year = {2025},
  doi = {[DOI when published]}
}
License
MIT License - see LICENSE file
