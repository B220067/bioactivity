from chembl_webresource_client.new_client import new_client
import pandas as pd


## Retrieve data from ChEMBL
# Target search term
target_name = "beta amyloid"

# ChEMBL search
target = new_client.target
search_results = target.search(target_name)

# Display all potential matches
print(f"Found {len(search_results)} potential targets:")
for result in search_results:
    print(f"Target ID: {result['target_chembl_id']}, Name: {result['pref_name']}")

# Select the most relevant target
target_chembl_id = search_results[0]['target_chembl_id']

# Retrieve bioactivity data
bioactivity = new_client.activity.filter(target_chembl_id=target_chembl_id).filter(standard_type="IC50")

# Convert the results to a df and save
bioactivity_df = pd.DataFrame(bioactivity)
bioactivity_df.to_csv("bioactivity_results.csv", index=False)

# Display the first few rows of the bioactivity data
print(f"\n {len(bioactivity_df)} bioactivity records for target '{target_name}' ({target_chembl_id}).")
print(bioactivity_df.head())

# Filter for IC50 <= 100nM
filtered_compounds = bioactivity_df[
    (bioactivity_df["standard_value"].astype(float) <= 100)
]

print(f"\nFiltered {len(filtered_compounds)} compounds with IC50 <= 100 nM:")
print(filtered_compounds[["molecule_chembl_id", "canonical_smiles", "standard_value"]])
