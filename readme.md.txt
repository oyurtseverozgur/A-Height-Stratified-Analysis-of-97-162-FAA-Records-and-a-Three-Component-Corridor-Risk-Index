# UAM Wildlife Strike Risk Analysis

This repository contains the complete Python analysis pipeline and supplementary materials for the paper:

**"Wildlife Strike Risk in the Urban Air Mobility Operational Envelope: A Height-Stratified Analysis of 97,162 FAA Records and a Three-Component Corridor Risk Index"**  
*Yurtsever, O. & Küçük, H. (2026)*

## Overview

This study provides the first altitude-stratified empirical characterisation of the wildlife strike environment relevant to Urban Air Mobility (UAM) eVTOL operations, using the FAA National Wildlife Strike Database (NWSD). The analysis:

- Processes 342,075 NWSD records (1990–2025) and extracts 97,162 height-coded entries.
- Produces a height-stratified strike distribution across six altitude bands (0–50 ft to >3,500 ft).
- Computes the **UAM Corridor Risk Index (UCRI)** – a three-component composite (strike density, hazard-weighted species exposure, and seasonal peak exposure) for ten priority US UAM corridors.
- Generates all figures and tables presented in the paper, including:
  - Height-stratified strike counts and damage rates (Figure 1)
  - Phase of flight and species composition at sub-500 ft (Figure 2)
  - UCRI corridor rankings and state-level counts (Figure 3)
  - Bird size distribution and seasonal-altitude hazard matrix (Figure 4)
  - Temporal trend and empirical CDF of strike heights (Figure 5)

All results are derived entirely from real NWSD records and published Relative Hazard Scores (Ross et al., 2025); no synthetic data are used.

## Repository Structure
