# Pattern recognition and machine learning, project 1. Laurent Lejeune, Tatiana Fountoukidou, Guillaume de Montauzon

## Content of this archive
- run.py: Main file used to
  - Load training and testing data from original csv files
  - Generate features using k-Nearest-Neighbors
  - Run k-fold cross-validation
  - Train model and write submission csv file
- mltb.py: toolbox functions
- test_tb.py: Tests functions used during the development of linear least-squares/logistics regression
- boosting.py: Class for training and cross-validating our Adaboost implementation.
- logit_boost.py: Functions for logitboost (not retained for the report due to numerical stability issues)
- p1.py: Similar to run.py but with more experiments (PCA, logitboost, feature augmentation, etc..)