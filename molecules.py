import os
import io
import sqlite3
import random
import math
import bittensor as bt
import pandas as pd
import numpy as np
import requests
from functools import lru_cache
from typing import List, Tuple
from datasets import load_dataset
from itertools import chain, combinations
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, ChemicalFeatures

from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from nova_ph2.utils.molecules import get_heavy_atom_count

class MoleculeUtils:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_molecules_by_role(role_mask: int, db_path: str) -> List[Tuple[int, str, int]]:
        try:
            abs_db_path = os.path.abspath(db_path)
            with sqlite3.connect(f"file:{abs_db_path}?mode=ro&immutable=1", uri=True) as conn:
                conn.execute("PRAGMA query_only = ON")
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT mol_id, smiles, role_mask FROM molecules WHERE (role_mask & ?) = ?", 
                    (role_mask, role_mask)
                )
                results = cursor.fetchall()
            return results
        except Exception as e:
            bt.logging.error(f"Error getting molecules by role {role_mask}: {e}")
            return []
    
    @staticmethod
    def num_rotatable_bonds(smiles: str) -> int:
        if not smiles:
            return 0
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return 0
            return Descriptors.NumRotatableBonds(mol)
        except Exception:
            return 0

    @staticmethod
    @lru_cache(maxsize=None)
    def mol_from_smiles_cached(smiles: str):
        if not smiles:
            return None
        try:
            return Chem.MolFromSmiles(smiles)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=None)
    def get_smiles_from_reaction_cached(name: str):
        try:
            return get_smiles_from_reaction(name)
        except Exception:
            return None
        
    @staticmethod
    @lru_cache(maxsize = None)
    def generate_inchikey(smiles: str) -> str:
        if not smiles:
            return ""
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return ""
            return Chem.MolToInchiKey(mol)
        except Exception as e:
            bt.logging.error(f"Error generating InChIKey for SMILES {smiles}: {e}")
            return ""

    @staticmethod
    def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.65) -> pd.DataFrame:
        if top_pool.empty or n_elites <= 0:
            return pd.DataFrame()

        top_candidates = top_pool.head(min(len(top_pool), n_elites * 4))
        if len(top_candidates) <= n_elites:
            return top_candidates

        max_score = top_candidates['score'].max()
        threshold = max_score * min_score_ratio
        candidates = top_candidates[top_candidates['score'] >= threshold]

        selected = []
        used_components = {'A': set(), 'B': set(), 'C': set()}

        if not candidates.empty:
            top_idx = candidates.index[0]
            top_row = candidates.iloc[0]
            selected.append(top_idx)
            parts = top_row['name'].split(":")
            if len(parts) >= 4:
                try:
                    used_components['A'].add(int(parts[2]))
                    used_components['B'].add(int(parts[3]))
                    if len(parts) > 4:
                        used_components['C'].add(int(parts[4]))
                except (ValueError, IndexError):
                    pass

        for idx, row in candidates.iterrows():
            if len(selected) >= n_elites:
                break
            if idx in selected:
                continue

            parts = row['name'].split(":")
            if len(parts) >= 4:
                try:
                    A_id = int(parts[2])
                    B_id = int(parts[3])
                    C_id = int(parts[4]) if len(parts) > 4 else None

                    is_diverse = (A_id not in used_components['A'] or
                                B_id not in used_components['B'] or
                                (C_id is not None and C_id not in used_components['C']))

                    if is_diverse or len(selected) < n_elites * 0.6:  # Increased from 0.5
                        selected.append(idx)
                        used_components['A'].add(A_id)
                        used_components['B'].add(B_id)
                        if C_id is not None:
                            used_components['C'].add(C_id)
                except (ValueError, IndexError):
                    if len(selected) < n_elites:
                        selected.append(idx)

        for idx, row in candidates.iterrows():
            if len(selected) >= n_elites:
                break
            if idx not in selected:
                selected.append(idx)

        return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)

    @staticmethod
    def select_diverse_subset(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
        smiles_list = pool["smiles"].tolist()
        for combination in combinations(smiles_list, subset_size):
            test_subset = top_95_smiles + list(combination)
            entropy = MoleculeUtils.compute_maccs_entropy(test_subset)
            if entropy >= entropy_threshold:
                bt.logging.info(f"Entropy Threshold Met: {entropy:.4f}")
                return pool[pool["smiles"].isin(combination)]

        bt.logging.warning("No combination exceeded the given entropy threshold.")
        return pd.DataFrame()

    @staticmethod
    @lru_cache(maxsize = None)
    def maccs_fp_from_smiles_cached(smiles: str):
        if not smiles:
            return None
        try:
            mol = MoleculeUtils.mol_from_smiles_cached(smiles)
            if mol is None:
                return None
            return MACCSkeys.GenMACCSKeys(mol)
        except Exception:
            return None

    @staticmethod
    def compute_maccs_entropy(smiles_list: list[str]) -> float:
        n_bits = 167 
        bit_counts = np.zeros(n_bits)
        valid_mols = 0

        for smi in smiles_list:
            fp = MoleculeUtils.maccs_fp_from_smiles_cached(smi)
            if fp is not None:
                arr = np.array(fp)
                bit_counts += arr
                valid_mols += 1

        if valid_mols == 0:
            raise ValueError("No valid molecules found.")

        probs = bit_counts / valid_mols
        entropy_per_bit = np.array([
            -p * math.log2(p) - (1 - p) * math.log2(1 - p) if 0 < p < 1 else 0
            for p in probs
        ])

        avg_entropy = np.mean(entropy_per_bit)

        return avg_entropy

    @staticmethod
    def parse_components(name: str) -> tuple[int, int, int | None]:
        parts = name.split(":")
        if len(parts) < 4:
            return None, None, None
        A = int(parts[2]); B = int(parts[3])
        C = int(parts[4]) if len(parts) > 4 else None
        return A, B, C
    
    @staticmethod
    def _heavy_atoms_dict_from_bitcounts(bitcounts: pd.DataFrame) -> dict:
        if bitcounts is None or bitcounts.empty or "heavy_atoms" not in bitcounts.columns:
            return {}
        return dict(zip(bitcounts["mol_id"], bitcounts["heavy_atoms"]))

class MoleculeManager:
    def __init__(self, config: dict, db_path: str):
        self.rxn_id = int(config["allowed_reaction"].split(":")[-1])
        self.db_path = db_path
        
        reaction_info = get_reaction_info(self.rxn_id, db_path)
        _, self.roleA, self.roleB, self.roleC = reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0
        
        self.molecules_A = MoleculeUtils.get_molecules_by_role(self.roleA, db_path)
        self.molecules_B = MoleculeUtils.get_molecules_by_role(self.roleB, db_path)
        self.molecules_C = MoleculeUtils.get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []
        
        self.moles_A_id = [mol[0] for mol in self.molecules_A]
        self.moles_B_id = [mol[0] for mol in self.molecules_B]
        self.moles_C_id = [mol[0] for mol in self.molecules_C] if self.is_three_component else None
        
        n_workers = os.cpu_count() or 1
        
        self.role_A_bitcounts = pd.DataFrame(self.molecules_A, columns=["mol_id", "smiles", "_"])[["mol_id", "smiles"]]
        self.role_A_bitcounts["heavy_atoms"] = self.role_A_bitcounts["smiles"].apply(get_heavy_atom_count)

        self.role_B_bitcounts = pd.DataFrame(self.molecules_B, columns=["mol_id", "smiles", "_"])[["mol_id", "smiles"]]
        self.role_B_bitcounts["heavy_atoms"] = self.role_B_bitcounts["smiles"].apply(get_heavy_atom_count)
        
        if self.is_three_component:
            self.role_C_bitcounts = pd.DataFrame(self.molecules_C, columns=["mol_id", "smiles", "_"])[["mol_id", "smiles"]]
            self.role_C_bitcounts["heavy_atoms"] = self.role_C_bitcounts["smiles"].apply(get_heavy_atom_count)        
        else:
            self.role_C_bitcounts = None
            
        self.dict_A = MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_A_bitcounts)
        self.dict_B = MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_B_bitcounts)
        self.dict_C = MoleculeUtils._heavy_atoms_dict_from_bitcounts(self.role_C_bitcounts) if self.role_C_bitcounts is not None else {}

    def validate_molecules(self, config: dict, data: pd.DataFrame, time_elapsed: int = 0) -> pd.DataFrame:
        condition = [20, 21, 21, 29, 43]
        if data.empty:
            return data
        def get_atom_sum(name: str) -> int:
            try:
                parts = name.split(":")
                if len(parts) < 4:
                    return 0
                A_id = int(parts[2])
                B_id = int(parts[3])
                C_id = int(parts[4]) if len(parts) > 4 else None
                total = self.dict_A.get(A_id, 0) + self.dict_B.get(B_id, 0)
                if C_id is not None:
                    total += self.dict_C.get(C_id, 0)
                return total
            except (ValueError, IndexError, KeyError):
                return 0
        data = data.copy()
        if config["min_heavy_atoms"] == 20:
            data['atom_sum'] = data["name"].map(get_atom_sum)
            data = data[data['atom_sum'] >= condition[self.rxn_id - 1]]
        
        data['smiles'] = data["name"].map(MoleculeUtils.get_smiles_from_reaction_cached)
        
        data = data[data['smiles'].notna()]
        if data.empty: return data
        data['heavy_atoms'] = data["smiles"].map(get_heavy_atom_count)
        data['bonds'] = data["smiles"].map(MoleculeUtils.num_rotatable_bonds)
        
        
        
        if time_elapsed > 1350:
            mask = (
                (data['heavy_atoms'] >= 25) &
                (data['bonds'] >= 3) &
                (data['bonds'] <= 7)
            )        
        else:
            mask = (
                (data['heavy_atoms'] >= config['min_heavy_atoms']) &
                (data['bonds'] >= config['min_rotatable_bonds']) &
                (data['bonds'] <= config['max_rotatable_bonds'])
            )
        data = data[mask].reset_index(drop=True)
        return data