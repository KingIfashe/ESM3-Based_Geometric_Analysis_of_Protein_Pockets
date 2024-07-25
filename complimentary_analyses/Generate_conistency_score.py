# Generate conistency score 

def generate_masked_tokens(pdb_path, encoder, model, mask_center, mask_range=16):
    chain = ProteinChain.from_pdb(pdb_path)
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords, plddt, residue_index = coords.cuda(), plddt.cuda(), residue_index.cuda()

    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

    # Apply padding
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097

    mask_start = max(mask_center - mask_range, 0)
    mask_end = min(mask_center + mask_range + 1, structure_tokens.size(1))

    structure_tokens[:, mask_start:mask_end] = 0  # Masking

    output = model.forward(
        structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
    )
    return torch.argmax(output.structure_logits, dim=-1)

# Example usage for the first structure
pdb_path = pdb_files[0]
masked_tokens = generate_masked_tokens(pdb_path, encoder, model, mask_center=50)

# Compare Generated Tokens:
original_tokens = all_structure_tokens[os.path.basename(pdb_path)]
consistency_score = (masked_tokens == original_tokens).float().mean().item() * 100
print(f"Consistency Score: {consistency_score}%")