import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE


## Random Forest
# Set seed
df = pd.read_csv("dataset_with_descriptors.csv")
random.seed(1)

# Target variable (assuming binary bioactivity based on log_standard_value and an arbitrary threshold of 5)
y = df["log_standard_value"].apply(lambda x: 1 if x < 5 else 0)

# Feature variables
X = df[["MolecularWeight", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Baseline Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Baseline evaluation
print("\nBaseline Accuracy:", accuracy_score(y_test, y_pred))
print("Baseline Classification Report:\n", classification_report(y_test, y_pred))
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Baseline ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# 2. Improved Random Forest with class weighting
improved_model_1 = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
improved_model_1.fit(X_train, y_train)
y_pred_improved = improved_model_1.predict(X_test)

# Improved model evaluation
print("\nImproved Accuracy:", accuracy_score(y_test, y_pred_improved))
print("Improved Classification Report:\n", classification_report(y_test, y_pred_improved))
if hasattr(improved_model_1, "predict_proba"):
    y_proba_improved = improved_model_1.predict_proba(X_test)[:, 1]
    print("Improved ROC-AUC Score:", roc_auc_score(y_test, y_proba_improved))

# 3. Oversample the minority class with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Random Forest with SMOTE-resampled data
model_with_smote = RandomForestClassifier(n_estimators=100, random_state=42)
model_with_smote.fit(X_train_resampled, y_train_resampled)
y_pred_smote = model_with_smote.predict(X_test)

# SMOTE model evaluation
print("\nSMOTE Accuracy:", accuracy_score(y_test, y_pred_smote))
print("SMOTE Classification Report:\n", classification_report(y_test, y_pred_smote))
if hasattr(model_with_smote, "predict_proba"):
    y_proba_smote = model_with_smote.predict_proba(X_test)[:, 1]
    print("SMOTE ROC-AUC Score:", roc_auc_score(y_test, y_proba_smote))

# 4. Random Forest with SMOTE and class weighting
combined_model = RandomForestClassifier(class_weight="balanced", n_estimators=100, random_state=42)
combined_model.fit(X_train_resampled, y_train_resampled)
y_pred_final = combined_model.predict(X_test)

# Combined model evaluation
print("\nCombined Model Accuracy:", accuracy_score(y_test, y_pred_final))
print("Combined Model Classification Report:\n", classification_report(y_test, y_pred_final))
if hasattr(combined_model, "predict_proba"):
    x_proba_final = combined_model.predict_proba(X_test)[:, 1]
    print("Combined Model ROC-AUC Score:", roc_auc_score(y_test, x_proba_final))

## Hyperparameter Tuning
# Paramteter grid
param_grid = {
    "n_estimators": [50, 100, 200],    
    "max_depth": [None, 10, 20],   
    "min_samples_split": [2, 5, 10], 
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           scoring="roc_auc", cv=5, verbose=2, n_jobs=-1)

# Grid search on SMOTE data
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best ROC-AUC Score:", grid_search.best_score_)

# Train the SMOTE model with the best parameters
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_resampled, y_train_resampled)
y_pred_tuned = best_rf.predict(X_test)

# Tuned model evaluation
print("\nTuned Model Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Tuned Model Classification Report:\n", classification_report(y_test, y_pred_tuned))
if hasattr(best_rf, "predict_proba"):
    y_proba_tuned = best_rf.predict_proba(X_test)[:, 1]
    print("Tuned Model ROC-AUC Score:", roc_auc_score(y_test, y_proba_tuned))

# Precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_tuned)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

## Extract predicted high bioactivity compounds
# Filter X_test for predicted actives using indices
predicted_active_indices = y_pred_tuned == 1
predicted_active = X_test[predicted_active_indices].copy()

# Add relevant identifiers from the original dataset
predicted_active["molecule_chembl_id"] = df.iloc[X_test.index[predicted_active_indices]]["molecule_chembl_id"].values
predicted_active["canonical_smiles"] = df.iloc[X_test.index[predicted_active_indices]]["canonical_smiles"].values
predicted_active["log_standard_value"] = df.iloc[X_test.index[predicted_active_indices]]["log_standard_value"].values

# Save the results
predicted_active.to_csv("predicted_active_compounds.csv", index=False)

print("\nPredicted Active Compounds (High Bioactivity):")
print(predicted_active[["molecule_chembl_id", "canonical_smiles", "log_standard_value"]])

