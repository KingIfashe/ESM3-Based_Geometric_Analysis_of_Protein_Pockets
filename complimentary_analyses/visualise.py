# Analyzing High RMSD and Low Token Similarity Structures

# Steps:
# 1. Identify Divergent Structures:

# Assuming valid_similarities and valid_rmsds are dictionaries
similarity_values = list(valid_similarities.values())
rmsd_values = list(valid_rmsds.values())

high_rmsd_low_sim = [(key, sim, rmsd) for key, sim, rmsd in zip(valid_similarities.keys(), similarity_values, rmsd_values) if sim < 50 and rmsd > 1.0]

print("High RMSD but Low Token Similarity Cases:")
for case in high_rmsd_low_sim:
    print(f"Structure: {case[0]}, Token Similarity: {case[1]:.2f}%, RMSD: {case[2]:.2f} Å")

# Visualize and Compare Structures:
#  Use molecular visualization tools to inspect the divergent structures and identify regions with significant differences.

from Bio.PDB import PDBParser, Superimposer
import py3Dmol

def visualize_structure(pdb_file, ref_file):
    viewer = py3Dmol.view(width=800, height=400)
    with open(pdb_file, 'r') as file:
        pdb_content = file.read()
    with open(ref_file, 'r') as ref:
        ref_content = ref.read()
    viewer.addModel(pdb_content, 'pdb')
    viewer.addModel(ref_content, 'pdb')
    viewer.setStyle({'model': 0}, {'cartoon': {'color': 'blue'}})
    viewer.setStyle({'model': 1}, {'cartoon': {'color': 'red'}})
    viewer.zoomTo()
    return viewer

# Example usage
reference_file = os.path.join('lys_structures', '6lyt_A.pdb')  # Assuming this is the reference file
for case in high_rmsd_low_sim:
    pdb_file = os.path.join('lys_structures', case[0])
    viewer = visualize_structure(pdb_file, reference_file)
    viewer.show()