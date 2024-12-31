import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole


## Cleaning Retrieved Data
bioactivity_df = pd.read_csv("bioactivity_results.csv")

# Drop empty columns
bioactivity_df_cleaned = bioactivity_df.dropna(axis=1, how='all')

# Drop rows with crucial values missing
bioactivity_df_cleaned = bioactivity_df.dropna(subset=["canonical_smiles", "standard_value"])

# Keep rows with standard units of nM
bioactivity_df_cleaned = bioactivity_df_cleaned[bioactivity_df_cleaned["standard_units"] == "nM"]

print(f"Original rows: {bioactivity_df.shape[0]}")
print(f"Remaining rows after cleaning: {bioactivity_df_cleaned.shape[0]}")

# Keep selected relevant columns
df = bioactivity_df_cleaned[[
    "molecule_chembl_id", "canonical_smiles", "standard_value", "standard_units", 
    "standard_type", "target_pref_name", "target_organism", "activity_comment", 
    "assay_description", "document_journal", "document_year"
]]

# df.to_csv("cleaned_bioactivity_data.csv", index=False)
print(df.head())

## Data exploration
# Histogram showing IC50 Distribution
df = df[df["standard_value"] > 0]
df["log_standard_value"] = np.log10(df["standard_value"] + 1)
plt.hist(df["log_standard_value"], bins=30, edgecolor="k")
plt.xlabel("Log10(Standard Value + 1) [IC50]")
plt.ylabel("Frequency")
plt.title("Log-Transformed Distribution of IC50 Values")
plt.show()


## Calculate Molecular Descriptors Function
morgan_generator = GetMorganGenerator(radius=2, fpSize=2048)

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptors = {
        "MolecularWeight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "Fingerprint": morgan_generator.GetFingerprint(mol).ToBitString(),
    }
    return descriptors

# Calculate descriptors for each SMILES string
descriptors_list = df["canonical_smiles"].apply(calculate_descriptors)

# Convert to df
descriptors_df = pd.DataFrame(descriptors_list.tolist())

# Combine with original dataset
result_df = pd.concat([df, descriptors_df], axis=1)
result_df.to_csv("dataset_with_descriptors.csv", index=False)


