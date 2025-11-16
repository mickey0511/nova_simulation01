import sys

import os

from reaction_utils import get_smiles_from_reaction

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from rdkit import Chem
from rdkit.Chem import Descriptors

from nova_ph2.utils import (
    get_heavy_atom_count
)

def validate_molecules_sampler(
    sampler_data: dict[str, list],
    config: dict,
) -> dict[int, dict[str, list[str]]]:
    """
    Validates molecules for all random sampler (uid=0).    
    Doesn't interrupt the process if a molecule is invalid, removes it from the list instead. 
    Doesn't check allowed reactions, chemically identical, duplicates, uniqueness (handled in random_sampler.py)
    
    Args:
        uid_to_data: Dictionary mapping UIDs to their data including molecules
        config: Configuration dictionary containing validation parameters
        
    Returns:
        Dictionary mapping UIDs to their list of valid SMILES strings
    """
    
    molecules = sampler_data["molecules"]

    valid_smiles = []
    valid_names = []
    valid_keys = []
                
    for molecule in molecules:
        try:
            if molecule is None:
                continue
            
            # smiles = get_smiles(molecule)
            smiles = get_smiles_from_reaction(molecule)
            if not smiles:
                continue
            
            if get_heavy_atom_count(smiles) < config['min_heavy_atoms']:
                continue

            try:    
                mol = Chem.MolFromSmiles(smiles)
                num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                if num_rotatable_bonds < config['min_rotatable_bonds'] or num_rotatable_bonds > config['max_rotatable_bonds']:
                    continue
                
                key = Chem.MolToInchiKey(mol)
            except Exception as e:
                continue
    
            valid_smiles.append(smiles)
            valid_names.append(molecule)
            valid_keys.append(key)
        except Exception as e:
            continue
        
    return valid_names, valid_smiles, valid_keys




def find_chemically_identical(key_list: list[str]) -> dict:
    """
    Check for identical molecules in a list of SMILES strings by converting to InChIKeys.
    """
    inchikey_to_indices = {}
    
    for i, inchikey in enumerate(key_list):
        if inchikey not in inchikey_to_indices:
            inchikey_to_indices[inchikey] = [i]
        else:
            inchikey_to_indices[inchikey].append(i)
    
    duplicates = {k: v for k, v in inchikey_to_indices.items() if len(v) > 1}
    
    return duplicates