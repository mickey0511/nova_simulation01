import sys
import os
import sqlite3
import bittensor as bt
from typing import List, Tuple

from nova_ph2.combinatorial_db.reactions import (
    get_molecules as get_molecules_db,
    combine_triazole_synthons,
    perform_smarts_reaction,
    validate_and_order_reactants)

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)


REACTION_INFO_CACHE = {}
def get_reaction_info(rxn_id: int, db_path: str) -> tuple:
    if rxn_id in REACTION_INFO_CACHE:
        return REACTION_INFO_CACHE[rxn_id]

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        cursor.execute("SELECT smarts, roleA, roleB, roleC FROM reactions WHERE rxn_id = ?", (rxn_id,))
        result = cursor.fetchone()
        conn.close()

        REACTION_INFO_CACHE[rxn_id] = result
        return result
    except Exception as e:
        bt.logging.error(f"Error getting reaction info: {e}")
        return None


MOLECULES_BY_ROLE_CACHE = {}
def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
    """
    Get all molecules that have the specified role_mask.

    Args:
        role_mask: The role mask to filter by
        db_path: Path to the molecules database

    Returns:
        List of tuples (mol_id, smiles, role_mask) for molecules that match the role
    """
    if role_mask in MOLECULES_BY_ROLE_CACHE:
        return MOLECULES_BY_ROLE_CACHE[role_mask]

    try:
        abs_db_path = os.path.abspath(db_path)
        conn = sqlite3.connect(
            f"file:{abs_db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?",
            (role_mask, role_mask)
        )
        results = cursor.fetchall()
        conn.close()
        
        MOLECULES_BY_ROLE_CACHE[role_mask] = results
        update_molecules_cache(results)
        return results
    except Exception as e:
        bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
        return []
    

MOLECULES_CACHE = dict()
def update_molecules_cache(molecules: List[Tuple]):
    for m in molecules:
        if m[0] not in MOLECULES_CACHE:
            MOLECULES_CACHE[m[0]] = m
   
    
def get_molecules(mol_ids: list, db_path: str) -> list:
    """Cached get_molecules()"""
    mols = {}
    ids_db = []
    for id in mol_ids:
        if id in MOLECULES_CACHE:
            mols[id] = (MOLECULES_CACHE[id][1], MOLECULES_CACHE[id][2])
        else:
            ids_db.append(id)
            
    if len(ids_db) > 0:
        mols_db = get_molecules_db(ids_db, db_path)
        for i, id in enumerate(ids_db):
            mols[id] = mols_db[i]
           
    return [mols[id] for id in mol_ids]

#---------------------------------------------------
# Exactly same as the functions from combinatorial_db module
# Replicated here to use the newly overridden get_reaction_info() and get_molecules() functions
#---------------------------------------------------
def react_molecules(rxn_id: int, mol1_id: int, mol2_id: int, db_path: str) -> str:
    try:
        # Get reaction info and molecules
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id], db_path)
        
        if not reaction_info or not all(molecules):
            return None
            
        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2) = molecules
        
        reactant1, reactant2 = validate_and_order_reactants(smiles1, smiles2, role_mask1, role_mask2, roleA, roleB)
        if not reactant1 or not reactant2:
            return None
            
        if rxn_id == 1:  # Triazole synthesis
            return combine_triazole_synthons(reactant1, reactant2)
        else:  # SMARTS-based reactions
            return perform_smarts_reaction(reactant1, reactant2, smarts)
        
    except Exception as e:
        bt.logging.error(f"Error reacting molecules {mol1_id}, {mol2_id}: {e}")
        return None


def react_three_components(rxn_id: int, mol1_id: int, mol2_id: int, mol3_id: int, db_path: str) -> str:
    try:
        reaction_info = get_reaction_info(rxn_id, db_path)
        molecules = get_molecules([mol1_id, mol2_id, mol3_id], db_path)
        
        if not reaction_info or not all(molecules):
            return None
            
        smarts, roleA, roleB, roleC = reaction_info
        (smiles1, role_mask1), (smiles2, role_mask2), (smiles3, role_mask3) = molecules
        
        validation_result = validate_and_order_reactants(smiles1, smiles2, role_mask1, role_mask2, roleA, roleB, 
                                                        smiles3, role_mask3, roleC)
        if not all(validation_result):
            return None
        
        reactant1, reactant2, reactant3 = validation_result
        
        if rxn_id == 3:  # click_amide_cascade
            # Triazole formation
            triazole_cooh = combine_triazole_synthons(reactant1, reactant2)
            if not triazole_cooh:
                return None
            
            # Amide coupling
            amide_smarts = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
            return perform_smarts_reaction(triazole_cooh, reactant3, amide_smarts)
        
        if rxn_id == 5:  # suzuki_bromide_then_chloride (two-step cascade)
            suzuki_br_smarts = "[#6:1][Br].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"
            suzuki_cl_smarts = "[#6:1][Cl].[#6:2][B]([OH])[OH]>>[#6:1][#6:2]"

            # First couple at bromide
            intermediate = perform_smarts_reaction(reactant1, reactant2, suzuki_br_smarts)
            if not intermediate:
                return None

            # Then couple at chloride
            final_product = perform_smarts_reaction(intermediate, reactant3, suzuki_cl_smarts)
            return final_product
        
        return None
        
    except Exception as e:
        bt.logging.error(f"Error in 3-component reaction {mol1_id}, {mol2_id}, {mol3_id}: {e}")
        return None


def get_smiles_from_reaction(product_name):
    """Handle reaction format: rxn:reaction_id:mol1_id:mol2_id or rxn:reaction_id:mol1_id:mol2_id:mol3_id"""
    try:
        parts = product_name.split(":")
        if len(parts) == 4:
            _, rxn_id, mol1_id, mol2_id = parts
            rxn_id, mol1_id, mol2_id = int(rxn_id), int(mol1_id), int(mol2_id)
            
            db_path = os.path.join(os.path.dirname(__file__), "molecules.sqlite")
            return react_molecules(rxn_id, mol1_id, mol2_id, db_path)
            
        elif len(parts) == 5:
            _, rxn_id, mol1_id, mol2_id, mol3_id = parts
            rxn_id, mol1_id, mol2_id, mol3_id = int(rxn_id), int(mol1_id), int(mol2_id), int(mol3_id)
            
            db_path = os.path.join(os.path.dirname(__file__), "molecules.sqlite")
            return react_three_components(rxn_id, mol1_id, mol2_id, mol3_id, db_path)
            
        else:
            bt.logging.error(f"Invalid reaction format: {product_name}")
            return None
            
        
    except Exception as e:
        bt.logging.error(f"Error in combinatorial reaction {product_name}: {e}")
        return None 