"""
bioprocessnexus package

This package provides tools and modules for processing, analyzing, and managing techno economic bioprocess data.
It includes functions and classes for data management, model training, prediction, performance 
evaluation, and SHAP explanations.

Modules:
    - data_management: Functions for managing and processing data.
    - explanations: Tools for generating SHAP explanations.
    - helpers: Utility functions for various tasks.
    - hist: Functions related to histogram processing and analysis.
    - interact_hist: Interactive histogram functionalities.
    - main: The main execution script.
    - mc_subsampling: Monte Carlo subsampling utilities.
    - model_training: Tools for training predictive models.
    - optimizer: Optimization of model responses.
    - performance_eval: Functions for evaluating model performance.
    - prediction_making: Functions for making predictions with trained models.
    - scaling_performance: Utilities for data scaling and performance testing.

"""

from .main import launch_nexus
