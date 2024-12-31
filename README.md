# bioactivity
This Python bioinformatics pipeline retrieved ChEMBL data, conducted molecular descriptor calculation, exploratory analysis, and machine learning classification using Random Forest with SMOTE and hyperparameter tuning to identify potential compounds targeting beta-amyloid, a hallmark of Alzheimer's disease, by predicting their bioactivity.

The pipeline starts with data retrieval from ChEMBL (1_data_retrieval.py) and data cleaning and feature engineering (2_smiles_descriptors.py). The dataset is then explored to understand its properties (3_data_exploration.py), and finally, random forest models are trained and optimized for bioactivity classification (4_bioactivity.py).

1_data_retrieval.py
This script retrieves bioactivity data from the ChEMBL database for beta-amyloid, a key target in Alzheimer's disease research. It processes and filters the retrieved data based on IC50 values, selecting compounds with high bioactivity (IC50 ≤ 100 nM).

2_smiles_descriptors.py
This script cleans and preprocesses the bioactivity data, calculates molecular descriptors for the SMILES strings of the compounds, and combines these descriptors with the original dataset. These descriptors form the feature set for downstream machine-learning models.

3_data_exploration.py
This script conducts exploratory data analysis on the calculated molecular descriptors. It visualizes the distribution of descriptors through histograms and analyzes their relationships via a correlation heatmap to understand the dataset's feature relevance.

4_bioactivity.py
This script uses the cleaned and processed dataset with molecular descriptors to train random forest models. It incorporates SMOTE to handle class imbalance and fine-tunes the model using hyperparameter optimization. The final model predicts compound bioactivity and evaluates performance using metrics like ROC-AUC and precision-recall curves.
