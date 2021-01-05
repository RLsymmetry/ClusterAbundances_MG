## Code for Cluster Abundances Constraints on Modified Gravity (MG) Models

This repository stores code that calculates the constraints on MG models using σ8, the mean amplitude of the matter power spectrum over large cosmic scales, based on the forecasts of the abundances of galaxy clusters in https://arxiv.org/abs/1708.07502v2. 

The folder Manuscript_Preprint points to the latest version of the draft describing the entire project that constrains MG with both cluster abundances and galaxy clustering (the latter being the work of Dr. G. Valogiannis), which is close to submission. 

The folder Core_Code stores the main code in the calculation. `AgrowthfR_re.py` is a class with subclasses that solves the linear growth factors and the evolution of σ8 under ΛCDM as well as f(R) and nDGP modified gravity scenarios. `Fisher_MG_final.py` is a separate module that performs Fisher analyses over the model parameters using `AgrowthfR_re.py`. 
