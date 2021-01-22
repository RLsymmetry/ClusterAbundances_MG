## Code for Cluster Abundances Constraints on Modified Gravity (MG) Models

This repository stores code that calculates the constraints on MG models using σ8, the mean amplitude of the matter power spectrum over large cosmic scales. The code is part of the work of https://arxiv.org/abs/2101.08728 describint the entire project that constrains MG with both cluster abundances and galaxy clustering (the latter being the work of Dr. G. Valogiannis).

* The folder **Manuscript_Preprint** stores the manuscript before submission for display purposes. It is now replaced by the arXiv identifier link.

* The folder **Core_Codes** stores the main code in the calculation. 
  * [`AgrowthfR_re.py`](https://github.com/RLsymmetry/ClusterAbundances_MG/blob/master/Core_Codes/AgrowthfR_re.py) is a class with subclasses that solves the linear growth factors and the evolution of σ8 under ΛCDM as well as f(R) and nDGP modified gravity scenarios.  
  * [`Fisher_MG_final.py`](https://github.com/RLsymmetry/ClusterAbundances_MG/blob/master/Core_Codes/Fisher_MG_final.py) is a separate module that performs Fisher analyses over the model parameters using `AgrowthfR_re.py`. 
  * The IPython notebook [`Abundance & clustering combined.ipynb`](https://github.com/RLsymmetry/ClusterAbundances_MG/blob/master/Core_Codes/Abundance%20%26%20clustering%20combined.ipynb) displays the combination process of the constraints from cluster abundances and galaxy clustering on the MG parameters. 
  * The IPython notebook [`Growth_MG_ExampleNotebook.ipynb`](https://github.com/RLsymmetry/ClusterAbundances_MG/blob/master/Core_Codes/Growth_MG_ExampleNotebook.ipynb) demonstrates the usage of `AgrowthfR_re.py` and `Fisher_MG_final.py`.


