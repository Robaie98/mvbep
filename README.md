[<img src="https://img.shields.io/badge/PyPI-1.0.0-brightgreen">](https://pypi.org/project/mvbep/)
[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/drive/1y5Z5ieZ-RjXWEl0i1j1OuXL8fYRUOJBb?usp=sharing)
[![Documentation Status](https://readthedocs.org/projects/mvbep/badge/?version=latest)](https://mvbep.readthedocs.io/en/latest/?badge=latest)



# MVBEP 
Measurement and Verification Building Energy Prediction (MVBEP) is an open-source framework for developing data-driven models for predicting the building baseline energy consumption and estimating savings associated with retrofitting in the post-retrofit period.   

---

## Background 
Measurement and Verification (M&V) is a process of estimating the Avoided Energy Use (AEU) during the post-retrofit period. Estimating savings due to an implemented retrofit strategy depends on building a baseline against which the recorded measurements in the post-retrofit period can be compared. The figure below highlights three periods: pre-retrofit, retrofit, and post-retrofit. 

![Baseline \label{baseline}](https://github.com/Robaie98/mvbep/raw/master/docs/source/figs/baseline.png)

The first period corresponds to a building state before performing a retrofit. In this period, an analysis is conducted to determine the possible combinations for Energy Conservation Measures (ECM) that could reduce the building energy consumption and return savings that make the retrofit feasible. The retrofit period encompasses the activities of retrofitting the building and finishing the ECM installment. The last period is the period in which savings estimation is performed. The building baseline represents the energy consumption if the building was not retrofitted while the metered energy consumption represents the actual consumption of the building. Generally, the baseline should be higher than the actual consumption as the building was retrofitted to be more energy-efficient. The difference between the two in the post-retrofit period represents the savings that M&V aims to quantify. However, to generate the baseline behavior, the building must be modeled on historical data. The approach highlighted in the package aims to build that baseline by using regression models. 


---
## Methodology 
The followed methodology in this package is structured into 5 modules: initialization, transformation, development, interpretation, and quantification. The Figure below shows the flowchart of the process. Each main component is converted into a module which when they are combined, they create MVBEP. 

![methodology structure \label{methodology}](https://github.com/Robaie98/mvbep/raw/master/docs/source/figs/mvbep_struct.png)

---
## Getting Started
MVBEP is uploaded to PyPI and can be installed by pip

~~~
$ pip install mvbep
~~~

### Requirements 
The following are the requirements to run MVBEP:
```
holidays>=0.14.2
joblib>=1.1.0
numpy>=1.20.3
pandas>=1.3.4
plotly>=5.7.0
schema==0.7.5
scikit_learn>=1.0.2
shap>=0.41.0
statsmodels>=0.12.2
xgboost>=1.6.0
```

The usage of the package is described in the [package documentation](https://mvbep.readthedocs.io/en/latest/?badge=latest). The basic usage of the package is illustrated in a [Google Colab notebook](https://colab.research.google.com/drive/1y5Z5ieZ-RjXWEl0i1j1OuXL8fYRUOJBb?usp=sharing). 


---
## Future Development
The current version of the package (i.e. 1.0.0) supports multiple tasks that automate the process of building a M&V baseline. The following are enhancements to be added to the package in the future along with minor missing aspects:

- **Documentation**: The documentation of `MVBEP` describes the basic and advanced usage of the package. In addition, it describes the functions of the `MVBEP` module which combines all the necessary modules to build the baseline. The documentation of the remaining modules is still not finished. 

- **Goodness-of-Fit (GOF)**: The package uses the Coefficient of Variation of Root Mean Squared Errors (CV(RMSE)) and the Normalized Mean Bias Error (NMBE) with CV(RMSE) being the default evaluation metric to choose the best modeling approach. The GOF metric combines both CV(RMSE) and NMBE which will be introduced in the next releases. 

- **Reports Generation**: The package uses static HTML files to summarize the output of each important phase in the development of an `MVBEP` object. The next releases will utilize a single interactive file that summarizes the information of all the phases rather than generating multiple files.   

- **Testing**: The package is just tested manually on a local Jupyter notebook and in Google Colab. The package will include automated tests in the future to better describe the performance.  

---
## Repository Structure 

```
mvbep
|
:----- data
|       |
|       :----- df_pre.csv: An example of pre-retrofit data.
|       |
|       :----- df_post.csv: An example of post-retrofit data.
|
:----- docs : The documentation for MVBEP.
|       
:----- mvbep
|       |
|       :----- templates: folder containing html templates for reports generation.
|       |
|       :----- developer.py: Builds and evaluates regression models.
|       |
|       :----- transformer.py: Converts cleaned data to training and testing data.
|       |
|       :----- initializer.py: Checks the format and data requirements to develop a MVBEP model.
|       |
|       :----- interpreter.py: Ouputs interpretation of the devloped regression models.
|       |
|       :----- mvbep.py: A module that encompasses all the models into one streamlined proccess.
|       |
|       :----- writer.py: A module that writes HTML reports using templates.
|       |
|       :----- towt_utils.py: A file containing necessary functions to create TOWT dataset.
|
:----- outputs
|       |
|       :----- development_summary.html: An example of the development summary using mvbep_example notebook.
|       |
|       :----- initialization_summary.html: An example of the initialization summary using mvbep_example notebook.
|       |
|       :----- quantification_summary.html: An example of the quantification summary using mvbep_example notebook.
|       
|       
|
```

---
## License 
This package is licensed under [MIT License](LICENSE).


---
## References
- Alrobaie A, Krarti M. A Review of Data-Driven Approaches for Measurement and Verification Analysis of Building Energy Retrofits. Energies. 2022; 15(21):7824. https://doi.org/10.3390/en15217824

