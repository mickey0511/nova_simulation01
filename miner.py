import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from pathlib import Path
import nova_ph2
from itertools import combinations
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from molecules import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    compute_tanimoto_similarity_to_pool,
    sample_random_valid_molecules,
    compute_maccs_entropy,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
)

from nova_ph2.combinatorial_db.reactions import get_smiles_from_reaction

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


target_models = []
antitarget_models = []

# Create global Morgan fingerprint generator to avoid deprecation warnings
MORGAN_FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Cache for fingerprints to avoid recomputation
_fp_cache = {}
_mol_cache = {}

def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models
    target_models = []
    antitarget_models = []
    
    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)
    
    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)


def target_score_from_data(data: pd.Series):
    """Score molecules against all target models."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
            for antitarget_model in antitarget_models:
                antitarget_model.smiles_list = smiles_list
                antitarget_model.smiles_dict = target_model.smiles_dict

            scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
            target_scores.append(scores["target"])
        target_series = pd.DataFrame(target_scores).mean(axis=0)
        return target_series
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores():
    """Score molecules against all antitarget models."""
    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(antitarget_model.screen_df, 
                                            antitarget_model.model, 
                                            antitarget_model.screen_loader,
                                            os.getcwd(),
                                            save_interpret=False,
                                            ligand_dict=antitarget_model.smiles_dict, 
                                            device=antitarget_model.device,
                                            save_cluster=False,
                                            )
            scores = antitarget_model.screen_df[['predicted_binding_affinity']]
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])
        
        if not antitarget_scores:
            return pd.Series(dtype=float)
        
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)

def get_mol(smiles: str):
    """Get RDKit Mol object from SMILES, cached."""
    if smiles in _mol_cache:
        return _mol_cache[smiles]
    
    mol = Chem.MolFromSmiles(smiles)
    _mol_cache[smiles] = mol  # store None if invalid
    return mol

def get_morgan_fingerprint(smiles: str, n_bits: int = 2048):
    """Get Morgan fingerprint for a SMILES string using MorganGenerator (cached), reusing Mol objects."""
    if smiles in _fp_cache:
        return _fp_cache[smiles]

    mol = get_mol(smiles)  # <- use cached Mol
    if mol is None:
        return None

    fp = MORGAN_FP_GENERATOR.GetFingerprint(mol)
    fp_array = np.zeros(n_bits, dtype=np.uint8)
    fp_array[fp.GetOnBits()] = 1

    _fp_cache[smiles] = fp_array

    # optional: maintain cache size limit
    if len(_fp_cache) > 50000:
        keys_to_remove = list(_fp_cache.keys())[:12500]
        for key in keys_to_remove:
            del _fp_cache[key]

    return fp_array

class SurrogateModel:
    """Fast surrogate model for score prediction using Random Forest.
    Used to pre-filter candidates before expensive GPU scoring for higher throughput."""
    
    def __init__(self, max_training_samples: int = 4000):
        # Improved RF: more trees + depth for better accuracy, n_jobs=-1 for speed
        self.model = RandomForestRegressor(
            n_estimators=80, max_depth=14, min_samples_leaf=3, random_state=42,
            n_jobs=-1, max_samples=0.8
        )
        self.is_trained = False
        self.X_train = []
        self.y_train = []
        self.min_train_size = 80  # Lower to enable earlier surrogate use
        self.max_training_samples = max_training_samples
        self.last_train_iteration = 0
        self.train_interval = 2  # Train more frequently when improving
        self.enabled = True
    
    def add_training_data(self, smiles_list: list, scores: list):
        """Add training data: favor top-scorers but include some low-scorers for discrimination."""
        if not self.enabled:
            return
        
        if len(smiles_list) > 600:
            scores_array = np.array(scores)
            # Keep top 500 + sample 100 from bottom/mid for better score discrimination
            top_indices = np.argsort(scores_array)[-500:]
            mid_low = np.argsort(scores_array)[:min(200, len(scores_array)//2)]
            sample_low = list(np.random.choice(mid_low, min(100, len(mid_low)), replace=False)) if len(mid_low) > 0 else []
            keep_indices = sorted(set(list(top_indices) + sample_low))
            smiles_list = [smiles_list[i] for i in keep_indices]
            scores = [scores[i] for i in keep_indices]
        
        new_fps = []
        new_scores = []
        for smiles, score in zip(smiles_list, scores):
            fp = get_morgan_fingerprint(smiles)
            if fp is not None:
                new_fps.append(fp)
                new_scores.append(score)
        
        self.X_train.extend(new_fps)
        self.y_train.extend(new_scores)
        
        if len(self.X_train) > self.max_training_samples:
            scores_array = np.array(self.y_train)
            top_count = int(self.max_training_samples * 0.5)  # More top-scorers
            recent_count = int(self.max_training_samples * 0.5)
            top_indices = np.argsort(scores_array)[-top_count:]
            recent_indices = list(range(len(self.X_train) - recent_count, len(self.X_train)))
            keep_indices = sorted(set(list(top_indices) + recent_indices))
            self.X_train = [self.X_train[i] for i in keep_indices]
            self.y_train = [self.y_train[i] for i in keep_indices]
    
    def train(self, iteration: int = 0):
        """Train the model periodically."""
        if self.is_trained and (iteration - self.last_train_iteration) < self.train_interval:
            return
        
        if len(self.X_train) < self.min_train_size:
            self.is_trained = False
            return
        
        try:
            X = np.array(self.X_train)
            y = np.array(self.y_train)
            train_start = time.time()
            self.model.fit(X, y)
            train_time = time.time() - train_start
            self.is_trained = True
            self.last_train_iteration = iteration
            if train_time > 0.5:
                bt.logging.info(f"[SURROGATE] Trained in {train_time:.2f}s on {len(self.X_train)} samples")
        except Exception as e:
            bt.logging.warning(f"Surrogate model training failed: {e}")
            self.is_trained = False
    
    def predict(self, smiles_list: list) -> np.ndarray:
        """Predict scores for a list of SMILES."""
        if not self.is_trained:
            return np.array([0.0] * len(smiles_list))
        
        try:
            fps = []
            for smiles in smiles_list:
                fp = get_morgan_fingerprint(smiles)
                if fp is None:
                    fps.append(np.zeros(2048, dtype=np.uint8))
                else:
                    fps.append(fp)
            
            X = np.array(fps)
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            bt.logging.warning(f"Surrogate prediction failed: {e}")
            return np.array([0.0] * len(smiles_list))
    
    def filter_candidates(self, data: pd.DataFrame, n_keep: int, smiles_col: str = "smiles") -> pd.DataFrame:
        """
        Pre-filter candidates before GPU scoring. Returns top n_keep by predicted score.
        Key performance optimization: reduce GPU workload by ~2-3x.
        """
        if not self.is_trained or data.empty or len(data) <= n_keep:
            return data
        
        smiles_list = data[smiles_col].tolist()
        pred_scores = self.predict(smiles_list)
        data = data.copy()
        data["_surrogate_pred"] = pred_scores
        data = data.sort_values("_surrogate_pred", ascending=False)
        filtered = data.head(n_keep).drop(columns=["_surrogate_pred"])
        bt.logging.info(f"[SURROGATE] Filtered {len(data)} -> {n_keep} candidates (saved GPU cost)")
        return filtered.reset_index(drop=True)

def _cpu_random_candidates_with_similarity(
    n_samples: int,
    subnet_config: dict,
    top_pool_df: pd.DataFrame,
    avoid_inchikeys: set[str] | None = None,
    thresh: float = 0.8
) -> pd.DataFrame:
    try:
        random_df = sample_random_valid_molecules(
            n_samples=n_samples,
            subnet_config=subnet_config,
            avoid_inchikeys=avoid_inchikeys,
            focus_neighborhood_of=top_pool_df
        )
        if random_df.empty or top_pool_df.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

        sims = compute_tanimoto_similarity_to_pool(
            candidate_smiles=random_df["smiles"],
            pool_smiles=top_pool_df["smiles"],
        )
        random_df = random_df.copy()
        random_df["tanimoto_similarity"] = sims.reindex(random_df.index).fillna(0.0)
        random_df = random_df.sort_values(by="tanimoto_similarity", ascending=False)
        random_df_filtered = random_df[random_df["tanimoto_similarity"] >= thresh]
            
        if random_df_filtered.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
            
        random_df_filtered = random_df_filtered.reset_index(drop=True)
        return random_df_filtered[["name", "smiles", "InChIKey"]]
    except Exception as e:
        bt.logging.warning(f"[Jerry-Sur] _cpu_random_candidates_with_similarity failed: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

def select_diverse_subset(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
    smiles_list = pool["smiles"].tolist()
    for combination in combinations(smiles_list, subset_size):
        test_subset = top_95_smiles + list(combination)
        entropy = compute_maccs_entropy(test_subset)
        if entropy >= entropy_threshold:
            bt.logging.info(f"Entropy Threshold Met: {entropy:.4f}")
            return pool[pool["smiles"].isin(combination)]

    bt.logging.warning("No combination exceeded the given entropy threshold.")
    return pd.DataFrame()


def main(config: dict):
    # winner2: Tuned for score_to_beat (SurrogateModel + synthon, no exploit worker)
    base_n_samples = 650
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    iteration = 0
    
    mutation_prob = 0.25  # INCREASED: More exploration, more mutation
    elite_frac = 0.50     # DECREASED: Balanced elite focus
    
    seen_inchikeys = set()
    seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
    start = time.time()
    prev_avg_score = None
    score_improvement_rate = 0.0
    no_improvement_counter = 0
    
    synthon_lib = None
    use_synthon_search = False

    # Initialize surrogate model
    surrogate = SurrogateModel(max_training_samples=4000)
    use_surrogate = True
    
    n_samples_first_iteration = int(base_n_samples * 3.5) if config["allowed_reaction"] != "rxn:5" else base_n_samples * 2

    bt.logging.info("[winner2] Surrogate + synthon, tuned for score_to_beat")

    # Use single CPU worker for stability
    with ProcessPoolExecutor(max_workers=1) as cpu_executor:
        while time.time() - start < 1800:
            iteration += 1
            iter_start_time = time.time()
            
            # Adaptive n_samples: maintain balanced throughput
            remaining_time = 1800 - (time.time() - start)
            if remaining_time > 1500:
                n_samples = base_n_samples
            elif remaining_time > 900:
                n_samples = int(base_n_samples * 0.95)
            elif remaining_time > 600:
                n_samples = int(base_n_samples * 0.90)
            elif remaining_time > 300:
                n_samples = int(base_n_samples * 0.85)
            else:
                n_samples = int(base_n_samples * 0.80)
            
            # DECREASED: Build synthon library at iteration 2 (standard timing)
            if iteration == 2 and not top_pool.empty and synthon_lib is None:
                try:
                    bt.logging.info("[TOP2] Building synthon library at iteration 2...")
                    synthon_lib_start = time.time()
                    synthon_lib = SynthonLibrary(DB_PATH, rxn_id)
                    use_synthon_search = True
                    bt.logging.info(f"[TOP2] Synthon library ready in {time.time() - synthon_lib_start:.2f}s")
                except Exception as e:
                    bt.logging.warning(f"[TOP2] Could not build synthon library: {e}")
                    use_synthon_search = False

            if iteration >= 1 and not top_pool.empty:
                component_weights = build_component_weights(top_pool, rxn_id)
            else:
                component_weights = None
                
            # DECREASED: Moderate elite selection
            elite_df = select_diverse_elites(top_pool, min(120, len(top_pool))) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None
            
            current_max_score = top_pool['score'].max() if not top_pool.empty else None
            has_very_high_score = current_max_score is not None and current_max_score > 0.015
            
            time_elapsed = time.time() - start
            is_late_stage = time_elapsed > 1200

            if iteration == 1:
                bt.logging.info(f"[winner2] Iteration {iteration}: Initial sampling")

                data = generate_valid_random_molecules_batch(
                    rxn_id,
                    n_samples=n_samples_first_iteration,
                    db_path=DB_PATH,
                    subnet_config=config,
                    batch_size=350,
                    elite_names=None,
                    elite_frac=0.0,
                    mutation_prob=1.0,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=None,
                )
            
            elif no_improvement_counter > 5:
                bt.logging.info(f"[winner2] Iteration {iteration}: Broad exploration reset")
                data = _cpu_random_candidates_with_similarity(
                    50, config,
                    top_pool.head(100)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys, 0.0
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
                if no_improvement_counter > 7:
                    no_improvement_counter = 0
            
            elif no_improvement_counter > 4:
                bt.logging.info(f"[winner2] Iteration {iteration}: Similarity search")
                data = _cpu_random_candidates_with_similarity(
                    50, config,
                    top_pool.head(30)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys, 0.75
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
            
            elif no_improvement_counter > 2:
                bt.logging.info(f"[winner2] Iteration {iteration}: Genetic algorithm")
                data = generate_valid_random_molecules_batch(
                    rxn_id, n_samples=n_samples, db_path=DB_PATH,
                    subnet_config=config, batch_size=400,
                    elite_names=elite_names, elite_frac=elite_frac,
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=component_weights,
                )
            
            elif use_surrogate and surrogate.is_trained and iteration > 3:
                # Generate MORE candidates for filtering (3x multiplier)
                n_candidates = int(n_samples * 3.0)
                
                # TOP-1 focus when high score (score_to_beat)
                if has_very_high_score or is_late_stage:
                    n_synthon_top1 = int(n_candidates * 0.25)
                    synthon_top1_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.head(3), n_synthon_top1,
                        min_similarity=0.88,
                        n_per_base=50
                    )
                    bt.logging.info(f"[winner2] Generated {len(synthon_top1_df)} synthon TOP-1 candidates")

                    n_synthon_top2 = int(n_candidates * 0.20)
                    synthon_top2_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.iloc[5:15], n_synthon_top2,
                        min_similarity=0.60, n_per_base=100
                    )

                    n_synthon_medium = int(n_candidates * 0.20)
                    synthon_medium_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.iloc[15:35], n_synthon_medium,
                        min_similarity=0.55, n_per_base=30
                    )

                    n_synthon_broad = int(n_candidates * 0.30)
                    synthon_broad_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.iloc[35:65], n_synthon_broad,
                        min_similarity=0.45, n_per_base=20
                    )

                    n_traditional = int(n_candidates * 0.20)
                    traditional_df = generate_valid_random_molecules_batch(
                        rxn_id, n_samples=n_traditional, db_path=DB_PATH,
                        subnet_config=config, batch_size=400,
                        elite_names=elite_names, elite_frac=0.80,  # DECREASED
                        mutation_prob=0.30,  # INCREASED
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )

                    synthon_df = pd.concat([synthon_top1_df, synthon_top2_df, synthon_medium_df, synthon_broad_df, traditional_df], ignore_index=True)
                                
                else:
                    n_synthon_broad = int(n_candidates * 0.5)
                    synthon_broad_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.head(10), n_synthon_broad,
                        min_similarity=0.55, n_per_base=30
                    )
                    synthon_df = synthon_broad_df
                
                synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")
                
                # Generate traditional candidates for surrogate filtering
                n_traditional_candidates = n_candidates - len(synthon_df)
                traditional_df = generate_valid_random_molecules_batch(
                    rxn_id, n_samples=n_traditional_candidates, db_path=DB_PATH,
                    subnet_config=config, batch_size=400,
                    elite_names=elite_names, elite_frac=elite_frac,
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=component_weights,
                )
                
                # Validate synthon candidates
                if not synthon_df.empty:
                    synthon_df = validate_molecules(synthon_df, config)
                
                # Combine all candidates for surrogate filtering
                data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                   
            elif use_synthon_search and iteration >= 2 and not top_pool.empty:
                bt.logging.info(f"[Jerry-Sur] Iteration {iteration}: Multi-range synthon strategy")
                # When surrogate is trained, generate more candidates for pre-filtering
                effective_n = int(n_samples * (2.0 if (use_surrogate and surrogate.is_trained and iteration > 3) else 1.0))
                
                # DECREASED: Balanced multi-range strategy
                if has_very_high_score or is_late_stage:
                    # Balanced TOP-1: 35% allocation
                    n_synthon_top1 = int(effective_n * 0.10)  # DECREASED: 35% for TOP-1
                    synthon_top1_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.head(5), n_synthon_top1,
                        min_similarity=0.75,  # DECREASED: Moderate similarity
                        n_per_base=100  # DECREASED: Moderate variations
                    )
                    bt.logging.info(f"[Jerry-Sur] Generated {len(synthon_top1_df)} synthon TOP-1 candidates")
                    
                    n_synthon_top2 = int(effective_n * 0.20)
                    synthon_top2_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.iloc[5:15], n_synthon_top2,
                        min_similarity=0.65, n_per_base=100
                    )
                    
                    n_synthon_medium = int(effective_n * 0.20)
                    synthon_medium_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.iloc[15:30], n_synthon_medium,
                        min_similarity=0.55, n_per_base=15
                    )
                    
                    n_synthon_broad = int(effective_n * 0.30)
                    synthon_broad_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.iloc[30:60], n_synthon_broad,
                        min_similarity=0.45, n_per_base=12
                    )
                    
                    n_traditional = int(effective_n * 0.20)
                    traditional_df = generate_valid_random_molecules_batch(
                        rxn_id, n_samples=n_traditional, db_path=DB_PATH,
                        subnet_config=config, batch_size=400,
                        elite_names=elite_names, elite_frac=0.50,  # DECREASED
                        mutation_prob=0.30,  # INCREASED
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )
                    
                    synthon_df = pd.concat([synthon_top1_df, synthon_top2_df, synthon_medium_df, synthon_broad_df, traditional_df], ignore_index=True)
                
                else:
                    # Standard balanced multi-range strategy
                    n_synthon_tight = int(effective_n * 0.30)  # DECREASED: 30% for tight
                    synthon_tight_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.head(5), n_synthon_tight,
                        min_similarity=0.70,  # DECREASED
                        n_per_base=25  # DECREASED
                    )
                    
                    n_synthon_medium = int(effective_n * 0.25)
                    seed_medium = top_pool.iloc[5:25] if len(top_pool) > 25 else top_pool.iloc[3:]
                    synthon_medium_df = generate_molecules_from_synthon_library(
                        synthon_lib, seed_medium, n_synthon_medium,
                        min_similarity=0.55, n_per_base=18
                    )
                    
                    n_synthon_broad = int(effective_n * 0.25)
                    synthon_broad_df = generate_molecules_from_synthon_library(
                        synthon_lib, top_pool.head(50), n_synthon_broad,
                        min_similarity=0.45, n_per_base=15
                    )
                    
                    n_traditional = int(effective_n * 0.20)  # INCREASED: More traditional
                    traditional_df = generate_valid_random_molecules_batch(
                        rxn_id, n_samples=n_traditional, db_path=DB_PATH,
                        subnet_config=config, batch_size=400,
                        elite_names=elite_names, elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )
                    
                    synthon_df = pd.concat([synthon_tight_df, synthon_medium_df, synthon_broad_df, traditional_df], ignore_index=True)
                
                synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")
                
                if not synthon_df.empty:
                    synthon_df = validate_molecules(synthon_df, config)
                    bt.logging.info(f"[Jerry-Sur] {len(synthon_df)} synthon candidates passed validation")
                
                n_traditional_extra = max(0, effective_n - len(synthon_df))
                if n_traditional_extra > 0:
                    traditional_df = generate_valid_random_molecules_batch(
                        rxn_id, n_samples=n_traditional_extra, db_path=DB_PATH,
                        subnet_config=config, batch_size=400,
                        elite_names=elite_names, elite_frac=elite_frac,
                        mutation_prob=mutation_prob,
                        avoid_inchikeys=seen_inchikeys,
                        component_weights=component_weights,
                    )
                else:
                    traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
                
                data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                data = data.drop_duplicates(subset=["name"], keep="first")
                bt.logging.info(f"[Jerry-Sur] Combined {len(data)} candidates")
                
                synthon_df = None
                        
            else:
                bt.logging.info(f"[Jerry-Sur] Iteration {iteration}: Broad exploration reset")
                data = _cpu_random_candidates_with_similarity(
                    45, config,
                    top_pool.head(100)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys, 0.0
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])
                no_improvement_counter = 0

            gen_time = time.time() - iter_start_time
            bt.logging.info(f"[Jerry-Sur] Iteration {iteration}: {len(data)} samples in {gen_time:.2f}s")

            if data.empty:
                bt.logging.warning(f"[Jerry-Sur] Iteration {iteration}: No valid molecules produced")
                continue
            
            if not seed_df.empty:
                data = pd.concat([data, seed_df])
                data = data.drop_duplicates(subset=["InChIKey"], keep="first")

                # Filter combined pool with surrogate when we have excess candidates
                if "smiles" not in data.columns or data["smiles"].isna().any():
                    data = data.copy()
                    missing = data["smiles"].isna()
                    if missing.any():
                        data.loc[missing, "smiles"] = data.loc[missing, "name"].apply(
                            lambda n: get_smiles_from_reaction(n) if pd.notna(n) else None
                        )
                    data = data[data["smiles"].notna()].reset_index(drop=True)
                if use_surrogate and surrogate.is_trained and iteration > 3 and len(data) > n_samples:
                    data = surrogate.filter_candidates(data, n_keep=n_samples, smiles_col="smiles")
                


                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

            try:
                filtered_data = data[~data["InChIKey"].isin(seen_inchikeys)]
                if len(filtered_data) < len(data):
                    bt.logging.warning(f"[Jerry-Sur] Iteration {iteration}: {len(data) - len(filtered_data)} previously seen")

                dup_ratio = (len(data) - len(filtered_data)) / max(1, len(data))
                
                # DECREASED: More balanced adaptation
                if dup_ratio > 0.7:
                    mutation_prob = min(0.7, mutation_prob * 1.6)  # INCREASED max
                    elite_frac = max(0.20, elite_frac * 0.75)  # DECREASED min
                elif dup_ratio > 0.5:
                    mutation_prob = min(0.55, mutation_prob * 1.4)
                    elite_frac = max(0.35, elite_frac * 0.80)
                elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                    mutation_prob = max(0.20, mutation_prob * 0.96)  # INCREASED min
                    elite_frac = max(0.5, elite_frac * 1.02)  # DECREASED max

                data = filtered_data

            except Exception as e:
                bt.logging.warning(f"[Jerry-Sur] Pre-score deduplication failed: {e}")

            if data.empty:
                mutation_prob = min(0.7, mutation_prob * 2.2)  # INCREASED
                elite_frac = max(0.35, elite_frac * 0.60)  # DECREASED
                continue

            data = data.reset_index(drop=True)

            # Surrogate pre-filter: reduce GPU workload (score_to_beat)
            if use_surrogate and surrogate.is_trained and iteration > 3 and len(data) > int(n_samples * 1.2):
                data = surrogate.filter_candidates(data, n_keep=int(n_samples * 1.15), smiles_col="smiles")

            # CPU similarity search when improvement is low (score_to_beat)
            cpu_futures = []
            if not top_pool.empty and iteration > 3:
                if score_improvement_rate < 0.02 or no_improvement_counter >= 1:
                    cpu_futures.append((
                        cpu_executor.submit(_cpu_random_candidates_with_similarity, 55, config,
                                           top_pool.head(5)[["name", "smiles", "InChIKey"]], seen_inchikeys, 0.85),
                        "top5-sim0.85"
                    ))
                    cpu_futures.append((
                        cpu_executor.submit(_cpu_random_candidates_with_similarity, 45, config,
                                           top_pool.head(20)[["name", "smiles", "InChIKey"]], seen_inchikeys, 0.65),
                        "top20-sim0.65"
                    ))
            
            gpu_start_time = time.time()

            if len(data) == 0:
                continue

            data["Target"] = target_score_from_data(data["smiles"])
            data["Anti"] = antitarget_scores()
            data["score"] = data["Target"] - (config["antitarget_weight"] * data["Anti"])

            if data["score"].isna().all():
                continue
            
            gpu_time = time.time() - gpu_start_time
            bt.logging.info(f"[Jerry-Sur] GPU scoring: {gpu_time:.2f}s")
            
            # Update surrogate model
            valid_scores = data[~data["score"].isna()]
            if len(valid_scores) > 0 and surrogate.enabled:
                surrogate.add_training_data(
                    valid_scores["smiles"].tolist(),
                    valid_scores["score"].tolist()
                )
                if len(surrogate.X_train) >= surrogate.min_train_size:
                    train_start = time.time()
                    surrogate.train(iteration)
                    train_time = time.time() - train_start
                    if surrogate.is_trained and (iteration - surrogate.last_train_iteration) == 0:
                        bt.logging.info(f"[Jerry-Sur] Surrogate trained: {len(surrogate.X_train)} samples in {train_time:.2f}s")
                    elif train_time > 10.0:
                        bt.logging.warning(f"[Jerry-Sur] Surrogate training slow ({train_time:.2f}s) - disabling")
                        surrogate.enabled = False
                        use_surrogate = False
            
            if cpu_futures:
                for cpu_future, strategy_name in cpu_futures:
                    try:
                        cpu_df = cpu_future.result(timeout=0)
                        if not cpu_df.empty:
                            if seed_df.empty:
                                seed_df = cpu_df.copy()
                            else:
                                seed_df = pd.concat([seed_df, cpu_df], ignore_index=True)
                            bt.logging.info(f"[Jerry-Sur] CPU ({strategy_name}): {len(cpu_df)} candidates")
                    except (TimeoutError, Exception):
                        pass
                
                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["InChIKey"], keep="first")
            
            seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
            total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
            prev_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            if not total_data.empty:
                top_pool = pd.concat([top_pool, total_data], ignore_index=True)
                top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                top_pool = top_pool.sort_values(by="score", ascending=False)

            remaining_time = 1800 - (time.time() - start)
            if remaining_time <= 60:
                entropy = compute_maccs_entropy(top_pool.iloc[:config["num_molecules"]]['smiles'].to_list())
                if entropy > config['entropy_min_threshold']:
                    top_pool = top_pool.head(config["num_molecules"])
                    bt.logging.info(f"[Jerry-Sur] Iteration {iteration}: Sufficient entropy = {entropy:.4f}")
                else:
                    try:
                        top_95 = top_pool.iloc[:95]
                        remaining_pool = top_pool.iloc[95:]
                        additional_5 = select_diverse_subset(remaining_pool, top_95["smiles"].tolist(), 
                                                            subset_size=5, entropy_threshold=config['entropy_min_threshold'])
                        if not additional_5.empty:
                            top_pool = pd.concat([top_95, additional_5]).reset_index(drop=True)
                            entropy = compute_maccs_entropy(top_pool['smiles'].to_list())
                            bt.logging.info(f"[Jerry-Sur] Iteration {iteration}: Adjusted entropy = {entropy:.4f}")
                        else:
                            top_pool = top_pool.head(config["num_molecules"])
                    except Exception as e:
                        bt.logging.warning(f"[Jerry-Sur] Entropy handling failed: {e}")
            else:
                top_pool = top_pool.head(config["num_molecules"])
            
            current_avg_score = top_pool['score'].mean() if not top_pool.empty else None
            if current_avg_score is not None:
                if prev_avg_score is not None:
                    score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                prev_avg_score = current_avg_score

            if score_improvement_rate == 0.0:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0
            
            iter_total_time = time.time() - iter_start_time
            top_entries = {"molecules": top_pool["name"].tolist()}
            total_time = time.time() - start
            
            bt.logging.info(
                f"[Jerry-Sur] Iter {iteration} | {iter_total_time:.2f}s | Total: {total_time:.2f}s | "
                f"Avg: {top_pool['score'].mean():.4f} | Max: {top_pool['score'].max():.4f} | "
                f"Elite: {elite_frac:.2f} | Mut: {mutation_prob:.2f} | Improve: {score_improvement_rate:.4f}"
            )
            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    config = get_config()
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"{time.time() - start_time_1} seconds for model initialization")
    main(config)
