from google.colab import files
import zipfile
import os
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient
import torch.nn.functional as F
import torch
from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils.structure.protein_chain import ProteinChain
from Bio.PDB import PDBParser, Superimposer
import numpy as np
import matplotlib.pyplot as plt
# Upload zip file
uploaded = files.upload()

# Get the name of the uploaded file
zip_name = list(uploaded.keys())[0]

# Extract the contents
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('.')

# List all PDB files
pdb_files = []
for root, dirs, files in os.walk('ubq_structures'):
    for file in files:
        if file.endswith('.pdb'):
            pdb_files.append(os.path.join(root, file))
print(f"Found {len(pdb_files)} PDB files.")

def process_pdb(pdb_path):
    chain = ProteinChain.from_pdb(pdb_path)
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords, plddt, residue_index = coords.cuda(), plddt.cuda(), residue_index.cuda()

    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097

    output = model.forward(
        structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
    )

    return torch.argmax(output.structure_logits, dim=-1)

# Process all PDB files
all_structure_tokens = {}
for pdb_file in pdb_files:
    print(f"Processing {os.path.basename(pdb_file)}...")
    try:
        all_structure_tokens[os.path.basename(pdb_file)] = process_pdb(pdb_file)
        print(f"Successfully processed {os.path.basename(pdb_file)}")
    except Exception as e:
        print(f"Error processing {os.path.basename(pdb_file)}: {str(e)}")

print("Processing complete.")

# Print the processed structures
print("\nProcessed structures:")
for key in all_structure_tokens.keys():
    print(key)

# accessing tokens for a specific structure
if all_structure_tokens:
    first_key = next(iter(all_structure_tokens))
    print(f"\nTokens for {first_key}:")
    print(all_structure_tokens[first_key])
    print(f"Shape: {all_structure_tokens[first_key].shape}")
else:
    print("\nNo structures were processed successfully.")

for pdb_file, tokens in all_structure_tokens.items():
    print(f"Structure: {pdb_file}")
    print(f"Tokens shape: {tokens.shape}")
    print(f"First few tokens: {tokens[0, :10]}")  # Print first 10 tokens
    print("\n")

def token_similarity(tokens1, tokens2):
    return (tokens1 == tokens2).float().mean().item() * 100

def calculate_rmsd(ref_structure, target_structure):
    parser = PDBParser()
    ref = parser.get_structure("reference", ref_structure)
    target = parser.get_structure("target", target_structure)

    ref_atoms = [atom for atom in ref.get_atoms() if atom.name == 'CA']
    target_atoms = [atom for atom in target.get_atoms() if atom.name == 'CA']

    if len(ref_atoms) != len(target_atoms):
        raise Exception("Fixed and moving atom lists differ in size")

    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, target_atoms)
    super_imposer.apply(target.get_atoms())

    return super_imposer.rms

# Use '1ubq_A.pdb' as the reference file
reference_file = '1ubq_A.pdb'
reference_path = os.path.join('ubq_structures', reference_file)

# Ensure the reference file is processed
if reference_file in all_structure_tokens:
    reference_tokens = all_structure_tokens[reference_file]
else:
    print(f"Reference file '{reference_file}' not found in processed tokens.")
    reference_tokens = None

if reference_tokens is not None:
    similarities = {}
    rmsds = {}

    for pdb_file, tokens in all_structure_tokens.items():
        if os.path.basename(pdb_file) != reference_file:
            similarities[os.path.basename(pdb_file)] = token_similarity(reference_tokens, tokens)
            try:
                rmsds[os.path.basename(pdb_file)] = calculate_rmsd(reference_path, os.path.join('ubq_structures', os.path.basename(pdb_file)))
            except Exception as e:
                print(f"Skipping RMSD calculation for {os.path.basename(pdb_file)}: {str(e)}")
                rmsds[os.path.basename(pdb_file)] = None

    # Print results
    for pdb_file in similarities.keys():
        if rmsds[pdb_file] is not None:
            print(f"{pdb_file}: Similarity = {similarities[pdb_file]:.2f}%, RMSD = {rmsds[pdb_file]:.2f} Å")
        else:
            print(f"{pdb_file}: Similarity = {similarities[pdb_file]:.2f}%, RMSD = N/A")

    # Plot
    plt.figure(figsize=(10, 6))
    valid_rmsds = {k: v for k, v in rmsds.items() if v is not None}
    valid_similarities = {k: similarities[k] for k in valid_rmsds.keys()}
    plt.scatter(list(valid_rmsds.values()), list(valid_similarities.values()))
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Token Similarity (%)')
    plt.title('ESM3 Token Similarity vs RMSD for Ubiquitin Structures')
    plt.grid(False)

    #for pdb_file in valid_similarities.keys():
        #plt.annotate(pdb_file, (valid_rmsds[pdb_file], valid_similarities[pdb_file]))

    plt.tight_layout()
    plt.show()
else:
    print("Reference structure tokens are not available, cannot calculate similarities or RMSD.")