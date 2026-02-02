from rdkit import Chem, DataStructs
import bittensor as bt
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from dotenv import load_dotenv
import pandas as pd
import warnings
import sqlite3
import random
import os
from functools import lru_cache
from typing import List, Tuple, Dict
load_dotenv(override=True)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction, get_reaction_info
from nova_ph2.utils.molecules import get_heavy_atom_count
from collections import defaultdict
from itertools import chain
import numpy as np
import math

# Create global Morgan fingerprint generator to avoid deprecation warnings
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@lru_cache(maxsize=1000_000)
def _get_smiles_from_reaction_cached(name: str):
    """Cache SMILES retrieval to avoid repeated database queries."""
    try:
        return get_smiles_from_reaction(name)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _mol_from_smiles_cached(smiles: str):
    """Cache molecule parsing to avoid repeated SMILES parsing."""
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


@lru_cache(maxsize=1000_000)
def _maccs_fp_from_smiles_cached(smiles: str):
    """Cache MACCS fingerprints for SMILES strings for fast Tanimoto similarity."""
    if not smiles:
        return None
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)
    except Exception:
        return None

@lru_cache(maxsize=1000_000)
def _inchikey_from_name_cached(name: str) -> str:
    """Cache InChIKey generation from molecule name to avoid repeated computation."""
    try:
        s = _get_smiles_from_reaction_cached(name)
        if not s:
            return ""
        return generate_inchikey(s)
    except Exception:
        return ""

def compute_maccs_entropy(smiles_list: list[str]) -> float:
    n_bits = 167  # RDKit uses 167 bits (index 0 is always 0)
    bit_counts = np.zeros(n_bits)
    valid_mols = 0

    for smi in smiles_list:
        fp = _maccs_fp_from_smiles_cached(smi)
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

def num_rotatable_bonds(smiles: str) -> int:
    """Get number of rotatable bonds from SMILES string."""
    if not smiles:
        return 0
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return 0
        return Descriptors.NumRotatableBonds(mol)
    except Exception:
        return 0

@lru_cache(maxsize=1000_000)
def generate_inchikey(smiles: str) -> str:
    """Generate InChIKey from SMILES string."""
    if not smiles:
        return ""
    try:
        mol = _mol_from_smiles_cached(smiles)
        if mol is None:
            return ""
        return Chem.MolToInchiKey(mol)
    except Exception as e:
        bt.logging.error(f"Error generating InChIKey for SMILES {smiles}: {e}")
        return ""


def compute_tanimoto_similarity_to_pool(
    candidate_smiles: pd.Series,
    pool_smiles: pd.Series,
) -> pd.Series:
    """
    Compute, for each candidate SMILES, the maximum MACCS Tanimoto similarity
    to any molecule in the reference pool.

    Returns a Series indexed like candidate_smiles.
    """
    if candidate_smiles.empty or pool_smiles.empty:
        # Return zeros with matching index
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    # Pre-compute fingerprints for pool molecules
    pool_fps = []
    for smi in pool_smiles.dropna().unique():
        fp = _maccs_fp_from_smiles_cached(smi)
        if fp is not None:
            pool_fps.append(fp)

    if not pool_fps:
        return pd.Series(0.0, index=candidate_smiles.index, dtype=float)

    similarities = {}
    for idx, smi in candidate_smiles.items():
        fp_cand = _maccs_fp_from_smiles_cached(smi)
        if fp_cand is None:
            similarities[idx] = 0.0
            continue
        max_sim = 0.0
        for fp_ref in pool_fps:
            try:
                sim = DataStructs.TanimotoSimilarity(fp_cand, fp_ref)
            except Exception:
                sim = 0.0
            if sim > max_sim:
                max_sim = sim
        similarities[idx] = float(max_sim)

    return pd.Series(similarities, dtype=float)

seen_cache = {}

def sample_random_valid_molecules(
    n_samples: int,
    subnet_config: dict,
    avoid_inchikeys: set[str] | None = None,
    focus_neighborhood_of: pd.DataFrame | None = None,
) -> pd.DataFrame:
    global seen_cache
    
    names = []
    for name in focus_neighborhood_of["name"]:
        try:
            parts = name.split(":")
            if len(parts) == 4:
                rxn_prefix, rxn_type, comp1_id, comp2_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)
                
                # Check if this molecule has been seen before, and adjust range accordingly
                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count

                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}"
                    names.append(new_name)
                
                # Generate neighborhood around comp2_id (keep comp1_id fixed)
                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}"
                    names.append(new_name)
                
            if len(parts) == 5:
                rxn_prefix, rxn_type, comp1_id, comp2_id, comp3_id = parts
                comp1_id = int(comp1_id)
                comp2_id = int(comp2_id)
                comp3_id = int(comp3_id)
                
                # Check if this molecule has been seen before, and adjust range accordingly
                seen_count = seen_cache.get(name, 0) + 1
                seen_cache[name] = seen_count
                # Generate neighborhood around comp1_id (keep comp2_id and comp3_id fixed)
                comp1_range = chain(range(max(1, comp1_id - seen_count * n_samples), comp1_id - (seen_count-1) * n_samples), range(max(1, comp1_id + (seen_count - 1) * n_samples), comp1_id + seen_count * n_samples + 1))
                for new_comp1 in comp1_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{new_comp1}:{comp2_id}:{comp3_id}"
                    names.append(new_name)
                
                # Generate neighborhood around comp2_id (keep comp1_id and comp3_id fixed)
                comp2_range = chain(range(max(1, comp2_id - seen_count * n_samples), comp2_id - (seen_count-1) * n_samples), range(max(1, comp2_id + (seen_count - 1) * n_samples), comp2_id + seen_count * n_samples + 1))
                for new_comp2 in comp2_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{new_comp2}:{comp3_id}"
                    names.append(new_name)
                
                # Generate neighborhood around comp3_id (keep comp1_id and comp2_id fixed)
                comp3_range = chain(range(max(1, comp3_id - seen_count * n_samples), comp3_id - (seen_count-1) * n_samples), range(max(1, comp3_id + (seen_count - 1) * n_samples), comp3_id + seen_count * n_samples + 1))
                for new_comp3 in comp3_range:
                    new_name = f"{rxn_prefix}:{rxn_type}:{comp1_id}:{comp2_id}:{new_comp3}"
                    names.append(new_name)

        except (ValueError, IndexError) as e:
            bt.logging.warning(f"Could not parse name '{name}': {e}")
            continue
    
    if not names:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = pd.DataFrame({"name": names})
    
    df = df[df["name"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = validate_molecules(df, subnet_config)
    if df.empty:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    df = df.drop_duplicates(subset=["InChIKey"], keep="first")

    if avoid_inchikeys:
        df = df[~df["InChIKey"].isin(avoid_inchikeys)]

    return df[["name", "smiles", "InChIKey"]].copy()



def validate_molecules(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Validate molecules by checking heavy atom count and rotatable bonds.
    Returns DataFrame with validated molecules and their descriptors.
    Defer InChIKey generation until after validation to avoid waste.
    """
    if data.empty:
        return data
    
    data = data.copy()
    data['smiles'] = data["name"].apply(_get_smiles_from_reaction_cached)
    
    data = data[data['smiles'].notna()]
    if data.empty:
        return data
    
    data['heavy_atoms'] = data["smiles"].apply(get_heavy_atom_count)
    data['bonds'] = data["smiles"].apply(num_rotatable_bonds)
    
    mask = (
        (data['heavy_atoms'] >= config['min_heavy_atoms']) &
        (data['bonds'] >= config['min_rotatable_bonds']) &
        (data['bonds'] <= config['max_rotatable_bonds'])
    )
    data = data[mask]
    
    if not data.empty:
        data['InChIKey'] = data["smiles"].apply(generate_inchikey)
    
    return data


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


class SynthonLibrary:
    """Manages synthon-based similarity search for component selection using Morgan fingerprints."""
    
    def __init__(self, db_path: str, rxn_id: int):
        self.db_path = db_path
        self.rxn_id = rxn_id
        self.reaction_info = get_reaction_info(rxn_id, db_path)
        
        if not self.reaction_info:
            raise ValueError(f"Could not load reaction {rxn_id}")
        
        _, self.roleA, self.roleB, self.roleC = self.reaction_info
        self.is_three_component = self.roleC is not None and self.roleC != 0
        
        # Load all components
        self.molecules_A = get_molecules_by_role(self.roleA, db_path)
        self.molecules_B = get_molecules_by_role(self.roleB, db_path)
        self.molecules_C = get_molecules_by_role(self.roleC, db_path) if self.is_three_component else []
        
        # Build fingerprint indices
        self.fps_A = self._build_fingerprint_index(self.molecules_A)
        self.fps_B = self._build_fingerprint_index(self.molecules_B)
        self.fps_C = self._build_fingerprint_index(self.molecules_C) if self.is_three_component else {}
        
        bt.logging.info(f"SynthonLibrary initialized: {len(self.fps_A)} A components, "
                       f"{len(self.fps_B)} B components" + 
                       (f", {len(self.fps_C)} C components" if self.is_three_component else ""))
    
    def _build_fingerprint_index(self, molecules: List[Tuple[int, str, int]]) -> Dict[int, object]:
        """Build fingerprint index for fast similarity search."""
        fps = {}
        for mol_id, smiles, _ in molecules:
            mol = _mol_from_smiles_cached(smiles)
            if mol:
                # Use MorganGenerator instead of deprecated method
                fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
                fps[mol_id] = fp
        return fps
    
    def find_similar_components(
        self, 
        target_smiles: str, 
        role: str = 'A',
        top_k: int = 60,  # DECREASED: Balanced exploration
        min_similarity: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Find components similar to target molecule.
        
        Args:
            target_smiles: SMILES string of target molecule
            role: 'A', 'B', or 'C' - which component pool to search
            top_k: Number of similar components to return
            min_similarity: Minimum Tanimoto similarity threshold
            
        Returns:
            List of (component_id, similarity_score) tuples
        """
        target_mol = _mol_from_smiles_cached(target_smiles)
        if not target_mol:
            return []
        
        # Use MorganGenerator instead of deprecated method
        target_fp = MORGAN_FP_GENERATOR.GetFingerprint(target_mol)
        
        # Select appropriate fingerprint index
        if role == 'A':
            fps_dict = self.fps_A
        elif role == 'B':
            fps_dict = self.fps_B
        elif role == 'C' and self.is_three_component:
            fps_dict = self.fps_C
        else:
            return []
        
        # Calculate similarities - BALANCED: check all components
        similarities = []
        for mol_id, fp in fps_dict.items():
            try:
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                if sim >= min_similarity:
                    similarities.append((mol_id, sim))
            except Exception:
                continue
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def find_similar_to_molecule_name(
        self,
        molecule_name: str,
        vary_component: str = 'both',
        top_k_per_component: int = 10,
        min_similarity: float = 0.6
    ) -> Dict[str, List[int]]:
        """
        Given a high-scoring molecule name, find similar components.
        
        Args:
            molecule_name: e.g., "rxn:1:123:456" or "rxn:3:123:456:789"
            vary_component: 'A', 'B', 'C', 'both', or 'all'
            top_k_per_component: How many similar components to find per role
            min_similarity: Minimum similarity threshold
            
        Returns:
            Dict with keys 'A', 'B', 'C' containing lists of similar component IDs
        """
        # Parse molecule name
        parts = molecule_name.split(":")
        if len(parts) < 4:
            return {}
        
        try:
            if len(parts) == 4:
                _, rxn, A_id, B_id = parts
                A_id, B_id = int(A_id), int(B_id)
                C_id = None
            else:
                _, rxn, A_id, B_id, C_id = parts
                A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
        except (ValueError, IndexError):
            return {}
        
        # Get SMILES for each component
        result = {}
        
        if vary_component in ['A', 'both', 'all']:
            A_smiles = self._get_component_smiles(A_id, 'A')
            if A_smiles:
                similar_As = self.find_similar_components(
                    A_smiles, 'A', top_k_per_component, min_similarity
                )
                result['A'] = [mol_id for mol_id, _ in similar_As if mol_id != A_id]
        
        if vary_component in ['B', 'both', 'all']:
            B_smiles = self._get_component_smiles(B_id, 'B')
            if B_smiles:
                similar_Bs = self.find_similar_components(
                    B_smiles, 'B', top_k_per_component, min_similarity
                )
                result['B'] = [mol_id for mol_id, _ in similar_Bs if mol_id != B_id]
        
        if self.is_three_component and C_id and vary_component in ['C', 'all']:
            C_smiles = self._get_component_smiles(C_id, 'C')
            if C_smiles:
                similar_Cs = self.find_similar_components(
                    C_smiles, 'C', top_k_per_component, min_similarity
                )
                result['C'] = [mol_id for mol_id, _ in similar_Cs if mol_id != C_id]
        
        return result
    
    def _get_component_smiles(self, mol_id: int, role: str) -> str:
        """Get SMILES for a component by ID and role."""
        if role == 'A':
            molecules = self.molecules_A
        elif role == 'B':
            molecules = self.molecules_B
        elif role == 'C':
            molecules = self.molecules_C
        else:
            return None
        
        for mid, smiles, _ in molecules:
            if mid == mol_id:
                return smiles
        return None
    
    def generate_similar_molecules(
        self,
        base_molecule_names: List[str],
        n_per_base: int = 5,
        min_similarity: float = 0.6
    ) -> List[str]:
        """
        Generate new molecule names by finding similar components to base molecules.
        BALANCED EXPLORATION: Moderate variations for balanced search.
        
        Args:
            base_molecule_names: List of high-scoring molecule names
            n_per_base: How many variations to generate per base molecule
            min_similarity: Minimum component similarity threshold
            
        Returns:
            List of new molecule names to try
        """
        new_molecules = []
        
        # BALANCED: When only one base molecule, moderate boost
        is_single_molecule = len(base_molecule_names) == 1
        if is_single_molecule:
            if n_per_base >= 50:
                effective_n_per_base = n_per_base  # Already good
            else:
                effective_n_per_base = n_per_base * 2  # 2x multiplier - BALANCED
        else:
            effective_n_per_base = n_per_base
        
        for base_name in base_molecule_names:
            parts = base_name.split(":")
            if len(parts) < 4:
                continue
            
            try:
                if len(parts) == 4:
                    _, rxn, A_id, B_id = parts
                    A_id, B_id = int(A_id), int(B_id)
                    
                    # Find similar components - BALANCED
                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'both', effective_n_per_base, min_similarity
                    )
                    
                    # Generate variations by replacing A
                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}")
                    
                    # Generate variations by replacing B
                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}")
                
                else:  # 3-component
                    _, rxn, A_id, B_id, C_id = parts
                    A_id, B_id, C_id = int(A_id), int(B_id), int(C_id)
                    
                    similar_comps = self.find_similar_to_molecule_name(
                        base_name, 'all', effective_n_per_base, min_similarity
                    )
                    
                    # Generate variations
                    for new_A in similar_comps.get('A', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{new_A}:{B_id}:{C_id}")
                    
                    for new_B in similar_comps.get('B', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{new_B}:{C_id}")
                    
                    for new_C in similar_comps.get('C', [])[:effective_n_per_base]:
                        new_molecules.append(f"rxn:{rxn}:{A_id}:{B_id}:{new_C}")
            
            except (ValueError, IndexError) as e:
                bt.logging.warning(f"Could not parse molecule name {base_name}: {e}")
                continue
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(new_molecules))


def generate_molecules_from_synthon_library(
    synthon_lib: SynthonLibrary,
    top_molecules: pd.DataFrame,
    n_samples: int,
    min_similarity: float = 0.6,
    n_per_base: int = 10
) -> pd.DataFrame:
    """
    Generate new molecules using synthon similarity search.
    BALANCED EXPLORATION: Balanced exploitation of top molecules.
    
    Args:
        synthon_lib: Initialized SynthonLibrary
        top_molecules: DataFrame with top-scoring molecules
        n_samples: Target number of molecules to generate
        min_similarity: Minimum component similarity
        n_per_base: Variations per base molecule
        
    Returns:
        DataFrame with new molecule names
    """
    if top_molecules.empty:
        return pd.DataFrame(columns=["name"])
    
    # BALANCED: When only 1 molecule, moderate exploitation
    if len(top_molecules) == 1:
        seed_names = top_molecules["name"].tolist()
        if n_per_base >= 50:
            effective_n_per_base = n_per_base
        else:
            effective_n_per_base = n_per_base * 2  # 2x multiplier - BALANCED
    else:
        # Multiple molecules: use appropriate number
        n_seeds = min(35, len(top_molecules))  # More seeds for exploration
        seed_names = top_molecules.head(n_seeds)["name"].tolist()
        effective_n_per_base = n_per_base
    
    # Generate similar molecules
    new_names = synthon_lib.generate_similar_molecules(
        seed_names,
        n_per_base=effective_n_per_base,
        min_similarity=min_similarity
    )
    
    if not new_names:
        return pd.DataFrame(columns=["name"])
    
    # BALANCED: Keep moderate variations
    if len(new_names) > n_samples * 2.5:  # Moderate overflow
        new_names = random.sample(new_names, int(n_samples * 1.8))  # Keep moderate
    
    return pd.DataFrame({"name": new_names})


def generate_valid_random_molecules_batch(
    rxn_id: int,
    n_samples: int,
    db_path: str,
    subnet_config: dict,
    batch_size: int = 200,
    seed: int = None,
    elite_names: list[str] | None = None,
    elite_frac: float = 0.5,
    mutation_prob: float = 0.1,
    avoid_inchikeys: set[str] | None = None,
    component_weights: dict | None = None,
) -> pd.DataFrame:
    reaction_info = get_reaction_info(rxn_id, db_path)
    if not reaction_info:
        bt.logging.error(f"Could not get reaction info for rxn_id {rxn_id}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    _, roleA, roleB, roleC = reaction_info
    is_three_component = roleC is not None and roleC != 0
    
    molecules_A = get_molecules_by_role(roleA, db_path)
    molecules_B = get_molecules_by_role(roleB, db_path)
    molecules_C = get_molecules_by_role(roleC, db_path) if is_three_component else []

    if not molecules_A or not molecules_B or (is_three_component and not molecules_C):
        bt.logging.error(f"No molecules found for roles A={roleA}, B={roleB}, C={roleC}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

    elite_As, elite_Bs, elite_Cs = set(), set(), set()
    if elite_names:
        for name in elite_names:
            A, B, C = _parse_components(name)
            if A is not None: 
                elite_As.add(A)
            if B is not None: 
                elite_Bs.add(B)
            if C is not None and is_three_component: 
                elite_Cs.add(C)

    pool_A_ids = _ids_from_pool(molecules_A)
    pool_B_ids = _ids_from_pool(molecules_B)
    pool_C_ids = _ids_from_pool(molecules_C) if is_three_component else []
    valid_dfs = []
    seen_keys = set()
    total_valid = 0
    
    while total_valid < n_samples:
        needed = n_samples - total_valid
        batch_size_actual = min(max(batch_size, 250), needed * 2)
        
        emitted_names = set()
        if elite_names:
            n_elite = max(0, min(batch_size_actual, int(batch_size_actual * elite_frac)))
            n_rand = batch_size_actual - n_elite

            elite_batch = generate_offspring_from_elites(
                rxn_id=rxn_id,
                n=n_elite,
                pool_A_ids=pool_A_ids,
                pool_B_ids=pool_B_ids,
                pool_C_ids=pool_C_ids,
                is_three_component=is_three_component,
                mutation_prob=mutation_prob,
                seed=seed,
                avoid_names=emitted_names,
                avoid_inchikeys=avoid_inchikeys,
                max_tries=10,
                elite_As=elite_As,
                elite_Bs=elite_Bs,
                elite_Cs=elite_Cs,
            )
            emitted_names.update(elite_batch)

            rand_batch = generate_molecules_from_pools(
                rxn_id, n_rand, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )
            rand_batch = [n for n in rand_batch if n and (n not in emitted_names)]
            batch_molecules = elite_batch + rand_batch

        else:
            batch_molecules = generate_molecules_from_pools(
                rxn_id, batch_size_actual, molecules_A, molecules_B, molecules_C, is_three_component, seed, component_weights
            )

        
        if not batch_molecules:
            continue
            
        batch_df = pd.DataFrame({"name": batch_molecules})
        batch_df = batch_df[batch_df["name"].notna()]  # Remove None values
        if batch_df.empty:
            continue
            
        batch_df = validate_molecules(batch_df, subnet_config)
        
        if batch_df.empty:
            continue

        batch_df = batch_df.drop_duplicates(subset=["InChIKey"], keep="first")
        
        mask = ~batch_df["InChIKey"].isin(seen_keys)
        if avoid_inchikeys:
            mask = mask & ~batch_df["InChIKey"].isin(avoid_inchikeys)
        batch_df = batch_df[mask]
        
        if batch_df.empty:
            continue
        
        seen_keys.update(batch_df["InChIKey"].values)
        valid_dfs.append(batch_df[["name", "smiles", "InChIKey"]].copy())
        total_valid += len(batch_df)
        
        if total_valid >= n_samples:
            break
        
    if not valid_dfs:
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    
    # Concatenate all DataFrames at once
    result_df = pd.concat(valid_dfs, ignore_index=True)
    return result_df.head(n_samples).copy()


def generate_molecules_from_pools(rxn_id: int, n: int, molecules_A: List[Tuple], molecules_B: List[Tuple], 
                                molecules_C: List[Tuple], is_three_component: bool, seed: int = None,
                                component_weights: dict = None) -> List[str]:
    
    rng = random.Random(seed) if seed is not None else random
    
    A_ids = [a[0] for a in molecules_A]
    B_ids = [b[0] for b in molecules_B]
    C_ids = [c[0] for c in molecules_C] if is_three_component else None
    
    # Use weighted sampling if component weights are provided
    if component_weights:
        # Build weights for each component pool
        weights_A = [component_weights.get('A', {}).get(aid, 1.0) for aid in A_ids]
        weights_B = [component_weights.get('B', {}).get(bid, 1.0) for bid in B_ids]
        weights_C = [component_weights.get('C', {}).get(cid, 1.0) for cid in C_ids] if is_three_component else None
        
        # Normalize weights
        if weights_A:
            sum_w = sum(weights_A)
            weights_A = [w / sum_w if sum_w > 0 else 1.0/len(weights_A) for w in weights_A]
        if weights_B:
            sum_w = sum(weights_B)
            weights_B = [w / sum_w if sum_w > 0 else 1.0/len(weights_B) for w in weights_B]
        if weights_C:
            sum_w = sum(weights_C)
            weights_C = [w / sum_w if sum_w > 0 else 1.0/len(weights_C) for w in weights_C]
        
        picks_A = rng.choices(A_ids, weights=weights_A, k=n) if weights_A else rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, weights=weights_B, k=n) if weights_B else rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, weights=weights_C, k=n) if weights_C else rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    else:
        # Uniform random sampling
        picks_A = rng.choices(A_ids, k=n)
        picks_B = rng.choices(B_ids, k=n)
        if is_three_component:
            picks_C = rng.choices(C_ids, k=n)
            names = [f"rxn:{rxn_id}:{a}:{b}:{c}" for a, b, c in zip(picks_A, picks_B, picks_C)]
        else:
            names = [f"rxn:{rxn_id}:{a}:{b}" for a, b in zip(picks_A, picks_B)]
    
    # Remove duplicates while preserving order
    names = list(dict.fromkeys(names))
    return names

def _parse_components(name: str) -> tuple[int, int, int | None]:
    # name format: "rxn:{rxn_id}:{A}:{B}" or "rxn:{rxn_id}:{A}:{B}:{C}"
    parts = name.split(":")
    if len(parts) < 4:
        return None, None, None
    A = int(parts[2]); B = int(parts[3])
    C = int(parts[4]) if len(parts) > 4 else None
    return A, B, C

def _ids_from_pool(pool):
    return [x[0] for x in pool]

def generate_offspring_from_elites(rxn_id: int, n: int,
                                   is_three_component: bool,
                                   pool_A_ids:list,
                                   pool_B_ids:list,
                                   pool_C_ids:list,
                                   mutation_prob: float = 0.1, seed: int | None = None,
                                   avoid_names: set[str] = None,
                                   avoid_inchikeys: set[str] = None,
                                   max_tries: int = 10,
                                   elite_As: set[int] = None,
                                   elite_Bs: set[int] = None,
                                   elite_Cs: set[int] = None) -> list[str]:
    
    rng = random.Random(seed) if seed is not None else random
    
    elite_As_list = list(elite_As) if elite_As else []
    elite_Bs_list = list(elite_Bs) if elite_Bs else []
    elite_Cs_list = list(elite_Cs) if elite_Cs else []

    out = []
    local_names = set()
    check_inchikeys = avoid_inchikeys is not None and len(avoid_inchikeys) > 0
    
    for _ in range(n):
        cand = None
        name = None
        for _try in range(max_tries):
            use_mutA = (not elite_As) or (rng.random() < mutation_prob)
            use_mutB = (not elite_Bs) or (rng.random() < mutation_prob)
            use_mutC = (not elite_Cs) or (rng.random() < mutation_prob)

            A = rng.choice(pool_A_ids) if use_mutA else rng.choice(elite_As_list)
            B = rng.choice(pool_B_ids) if use_mutB else rng.choice(elite_Bs_list)
            if is_three_component:
                C = rng.choice(pool_C_ids) if use_mutC else rng.choice(elite_Cs_list)
                name = f"rxn:{rxn_id}:{A}:{B}:{C}"
            else:
                name = f"rxn:{rxn_id}:{A}:{B}"

            # Fast checks first (set membership is O(1))
            if avoid_names and name in avoid_names:
                continue
            if name in local_names:
                continue

            if check_inchikeys:
                try:
                    key = _inchikey_from_name_cached(name)
                    if key and key in avoid_inchikeys:
                        continue
                except Exception:
                    pass

            cand = name
            break

        if cand is None:
            if name is None:
                A = rng.choice(pool_A_ids)
                B = rng.choice(pool_B_ids)
                if is_three_component:
                    C = rng.choice(pool_C_ids) if pool_C_ids else 0
                    name = f"rxn:{rxn_id}:{A}:{B}:{C}"
                else:
                    name = f"rxn:{rxn_id}:{A}:{B}"
            cand = name
        out.append(cand)
        local_names.add(cand)
        if avoid_names is not None:
            avoid_names.add(cand)
    return out

def select_diverse_elites(top_pool: pd.DataFrame, n_elites: int, min_score_ratio: float = 0.70) -> pd.DataFrame:
    """
    Select diverse elite molecules: top by score, but ensure diversity in component space.
    BALANCED: Higher threshold for balanced selection.
    """
    if top_pool.empty or n_elites <= 0:
        return pd.DataFrame()
    
    # Take MORE top candidates for better diversity selection
    top_candidates = top_pool.head(min(len(top_pool), n_elites * 3))  # DECREASED from 4
    if len(top_candidates) <= n_elites:
        return top_candidates
    
    # Score threshold: HIGHER threshold for balanced selection
    max_score = top_candidates['score'].max()
    threshold = max_score * min_score_ratio
    candidates = top_candidates[top_candidates['score'] >= threshold]
    
    # Select diverse set: prefer molecules with different components
    selected = []
    used_components = {'A': set(), 'B': set(), 'C': set()}
    
    # First, add top scorer
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
    
    # Then add diverse molecules - BALANCED diversity selection
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
                
                # Prefer molecules with new components - BALANCED
                is_diverse = (A_id not in used_components['A'] or 
                             B_id not in used_components['B'] or
                             (C_id is not None and C_id not in used_components['C']))
                
                # Balanced threshold for diversity
                if is_diverse or len(selected) < n_elites * 0.5:  # DECREASED from 0.6
                    selected.append(idx)
                    used_components['A'].add(A_id)
                    used_components['B'].add(B_id)
                    if C_id is not None:
                        used_components['C'].add(C_id)
            except (ValueError, IndexError):
                if len(selected) < n_elites:
                    selected.append(idx)
    
    # Fill remaining slots
    for idx, row in candidates.iterrows():
        if len(selected) >= n_elites:
            break
        if idx not in selected:
            selected.append(idx)
    
    return candidates.loc[selected[:n_elites]] if selected else candidates.head(n_elites)


def build_component_weights(top_pool: pd.DataFrame, rxn_id: int) -> Dict[str, Dict[int, float]]:
    """
    Build component weights based on scores of molecules containing them.
    BALANCED: Use moderate exponential weighting for balanced exploration.
    Returns dict with 'A', 'B', 'C' keys mapping to {component_id: weight}
    """
    weights = {'A': defaultdict(float), 'B': defaultdict(float), 'C': defaultdict(float)}
    counts = {'A': defaultdict(int), 'B': defaultdict(int), 'C': defaultdict(int)}
    
    if top_pool.empty:
        return weights
    
    # Extract component IDs and scores with MODERATE exponential weighting
    for idx, row in top_pool.iterrows():
        name = row['name']
        score = row['score']
        
        # MODERATE exponential weighting: balanced contribution
        # Rank-based exponential: rank 1 gets weight 2.0, rank 10 gets weight 1.1, etc.
        rank = idx + 1
        rank_weight = 2.0 * math.exp(-rank / 20.0)  # Gentler exponential decay
        weighted_score = max(0, score) * rank_weight
        
        parts = name.split(":")
        if len(parts) >= 4:
            try:
                A_id = int(parts[2])
                B_id = int(parts[3])
                weights['A'][A_id] += weighted_score
                weights['B'][B_id] += weighted_score
                counts['A'][A_id] += 1
                counts['B'][B_id] += 1
                
                if len(parts) > 4:
                    C_id = int(parts[4])
                    weights['C'][C_id] += weighted_score
                    counts['C'][C_id] += 1
            except (ValueError, IndexError):
                continue
    
    # Normalize by count and add smoothing - but preserve exponential weighting
    for role in ['A', 'B', 'C']:
        for comp_id in weights[role]:
            if counts[role][comp_id] > 0:
                # Average with exponential weighting preserved
                avg_weight = weights[role][comp_id] / counts[role][comp_id]
                # Add moderate smoothing
                weights[role][comp_id] = avg_weight + 0.10  # DECREASED smoothing
    
    return weights