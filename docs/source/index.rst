.. MVBEP documentation master file, created by
   sphinx-quickstart on Wed Nov  9 17:26:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MVBEP's documentation!
=================================

MVBEP
-----

Measurement and Verification Building Energy Prediction (MVBEP) is an
open-source framework for developing data-driven models for predicting
the building baseline energy consumption and estimating savings
associated with retrofitting in the post-retrofit period.



Background
----------

Measurement and Verification (M&V) is a process of estimating the
Avoided Energy Use (AEU) during the post-retrofit period. Estimating
savings due to an implemented retrofit strategy depends on building a
baseline against which the recorded measurements in the post-retrofit
period can be compared. The figure below highlights three periods:
pre-retrofit, retrofit, and post-retrofit.

.. figure:: ./figs/baseline.png
   :alt: Baseline :raw-latex:`\label{baseline}`

The first period corresponds to a building state before performing a
retrofit. In this period, an analysis is conducted to determine the
possible combinations for Energy Conservation Measures (ECM) that could
reduce the building energy consumption and return savings that make the
retrofit feasible. The retrofit period encompasses the activities of
retrofitting the building and finishing the ECM installment. The last
period is the period in which savings estimation is performed. The
building baseline represents the energy consumption if the building was
not retrofitted while the metered energy consumption represents the
actual consumption of the building. Generally, the baseline should be
higher than the actual consumption as the building was retrofitted to be
more energy-efficient. The difference between the two in the
post-retrofit period represents the savings that M&V aims to quantify.
However, to generate the baseline behavior, the building must be modeled
on historical data. The approach highlighted in the package aims to
build that baseline by using regression models.

Current State
-------------

The package is still under development. The following are still not
finalized - Documentation: The documentation of ``MVBEP`` is finished
which is the only thing required to use the framework. The documentation
of the remaining modules is still not finished.

-  Comprehensive Examples: The Jupyter notebook called ``mvbep_example``
   shows a simple example of the framework along with all reports as
   shown in ``docs`` file.

Requirements
------------

::

   holidays==0.14.2
   joblib==1.1.0
   numpy==1.20.3
   pandas==1.3.4
   plotly==5.7.0
   schema==0.7.5
   scikit_learn==1.1.2
   shap==0.41.0
   statsmodels==0.12.2
   xgboost==1.6.0

References
----------

-  Alrobaie A, Krarti M. A Review of Data-Driven Approaches for
   Measurement and Verification Analysis of Building Energy Retrofits.
   Energies. 2022; 15(21):7824. https://doi.org/10.3390/en15217824


.. toctree::
   :maxdepth: 4
   :caption: Documentation Content:

   usage
   examples
   mvbep
   





.. Indices and tables
.. ------------------

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
