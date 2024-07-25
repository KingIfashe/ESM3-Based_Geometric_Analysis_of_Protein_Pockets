# calculate RMSF

import mdtraj as md
import os
import matplotlib.pyplot as plt
import numpy as np

# Load all PDB files into MDTraj
pdb_files = [os.path.join('lys_structures', f) for f in os.listdir('lys_structures') if f.endswith('.pdb')]

# Load reference structure
reference = md.load('lys_structures/6lyt_A.pdb')
ref_ca_indices = reference.topology.select('name CA')

# Function to align and calculate RMSF
def process_structure(pdb_file):
    try:
        traj = md.load(pdb_file)
        traj_ca_indices = traj.topology.select('name CA')

        # Check if the number of CA atoms matches
        if len(traj_ca_indices) != len(ref_ca_indices):
            print(f"Skipping {pdb_file} due to mismatched number of CA atoms")
            return None

        # Align using only CA atoms
        aligned = traj.superpose(reference, atom_indices=traj_ca_indices, ref_atom_indices=ref_ca_indices)

        # Calculate RMSF for CA atoms
        rmsf = md.rmsf(aligned, reference, atom_indices=traj_ca_indices)

        return rmsf
    except Exception as e:
        print(f"Error processing {pdb_file}: {str(e)}")
        return None

# Process all structures
rmsfs = [process_structure(pdb) for pdb in pdb_files]
rmsfs = [rmsf for rmsf in rmsfs if rmsf is not None]

# Average RMSF across all structures
if rmsfs:
    avg_rmsf = np.mean(rmsfs, axis=0)

    # Plot RMSF
    plt.figure(figsize=(10, 6))
    plt.plot(avg_rmsf)
    plt.xlabel('Residue Index')
    plt.ylabel('RMSF (Å)')
    plt.title('Average RMSF of Lysozyme Structures')
    plt.show()

    # Highlight Residues with High RMSF
    high_rmsf_indices = np.where(avg_rmsf > np.percentile(avg_rmsf, 90))[0]  # Top 10% residues with highest RMSF
    print(f"Residues with high RMSF: {high_rmsf_indices}")
else:
    print("No valid structures were processed.")