import os
import time
import json
import threading
import pandas as pd
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import nova_ph2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")
TIME_LIMIT = 1800
LIMIT_PER_REACTANT = 1000

from molecules import (
    MoleculeManager,
    MoleculeUtils
)
from tools import (
    IterationParams,
    SynthonLibrary,
    generate_valid_random_molecules,
    generate_molecules_from_synthon_library,
    cpu_random_candidates_with_similarity,
    build_component_weights
)
from models import (
    ModelManager
)
from exploit import (
    get_top_n_unexploited,
    run_exploit
)

def get_config(input_file: str = os.path.join(BASE_DIR, "input.json")):
    with open(input_file, "r") as f:
        d = json.load(f)
    return {**d.get("config", {}), **d.get("challenge", {})}


model_manager = None
molecule_manager = None

def initialize_solution(config: dict):
    global molecule_manager, model_manager
    
    molecule_manager = MoleculeManager(config = config, db_path = DB_PATH)
    model_manager = ModelManager(config)

def find_solution(config: dict, time_start: float):
    
    global molecule_manager, model_manager

    iteration = 0
    n_workers = os.cpu_count() or 1
    bt.logging.info(f"[Solution] CPU Workers: {n_workers}")
    
    params = IterationParams(config = config)
    seed_df = pd.DataFrame(columns=["name", "smiles", "tanimoto_similarity"])
    top_pool = pd.DataFrame(columns=["name", "smiles", "inchi", "score", "target", "anti"])
    _synthon_build_started = False
    _synthon_result = [None, None]  # [lib, exception]

    def _build_synthon_in_bg():
        try:
            lib = SynthonLibrary(molecule_manager=molecule_manager)
            _synthon_result[0] = lib
            bt.logging.info("[BG] Synthon library built successfully in background thread.")
        except Exception as e:
            _synthon_result[1] = e
            bt.logging.warning(f"[BG] Synthon library build failed: {e}")

    with ProcessPoolExecutor(max_workers = n_workers) as cpu_executor:
        while time.time() - time_start < TIME_LIMIT:
            iteration += 1
            iteration_start_time = time.time()
            bt.logging.info(f"[Solution] --- Starting Iteration {iteration} ---")
            
            remaining_time = TIME_LIMIT - (iteration_start_time - time_start)
            n_samples = params.get_nsamples_from_time(remaining_time)
            
            component_weights = build_component_weights(top_pool.head(config["num_molecules"]), molecule_manager.rxn_id) if not top_pool.empty else None
            elite_df = MoleculeUtils.select_diverse_elites(top_pool, min(150, len(top_pool))) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None
            
            exploited_status = False
            exploit_summary = None
            
            # Switch to exploit mode after just 1 stalled iteration (was 2)
            if params.no_improvement_counter >= 1:
                params.use_exploit_mode = True
                bt.logging.info(f"[Solution] === Switching to EXPLOIT MODE at iteration {iteration} === ")

            # Kick off synthon library build in a background thread during iteration 1's GPU work
            if iteration == 1 and not _synthon_build_started:
                _synthon_build_started = True
                _bg_thread = threading.Thread(target=_build_synthon_in_bg, daemon=True)
                _bg_thread.start()
                bt.logging.info("[Solution] Synthon library build started in background thread.")

            # Pick up synthon library from background thread when ready
            if params.synthon_lib is None and _synthon_result[0] is not None:
                params.synthon_lib = _synthon_result[0]
                params.use_synthon_search = True
                bt.logging.info("[Solution] Synthon library is now available from background thread.")
            elif params.synthon_lib is None and _synthon_result[1] is not None:
                bt.logging.warning(f"[Solution] Background synthon build failed: {_synthon_result[1]}")
                params.use_synthon_search = False

            if params.use_exploit_mode:
                bt.logging.info("[Solution] Smart molecule exploitation starts...")
                all_top_mols = top_pool.to_dict("records")
                try:
                    # Exploit top 3 unexploited molecules (was 5 but only 1 reactant processed)
                    unexploited_mols = get_top_n_unexploited(all_top_mols, params.exploited_reactants, n=3)
                    if unexploited_mols:
                        exploit_start = time.time()
                        exploit_results, exploit_summary = run_exploit(
                            manager=molecule_manager,
                            config=config,
                            top_molecules=unexploited_mols,
                            top_n=3,
                            limit_per_reactant=LIMIT_PER_REACTANT,
                            avoid_names=params.seen_molecules,
                            exploited_reactants=params.exploited_reactants,
                        )

                        exploit_time = time.time() - exploit_start
                        bt.logging.info(f"[Solution] Exploit returned {len(exploit_results)} candidates in {exploit_time:.1f}s (exploited_reactants={len(params.exploited_reactants)})")

                        if exploit_results:
                            data = pd.DataFrame(exploit_results)
                            exploited_status = True
                        else: 
                            raise Exception("No exploited molecules are found.")
                    else:
                        raise Exception("No unexploited top molecules are found.")
                except Exception as e:
                    bt.logging.warning(f"[Solution] Error in exploiting top molecules: {e}")
                    
            if iteration == 1:
                bt.logging.info(f"[Solution] Iteration 1: Generating {params.n_samples_start} molecules for the first iteration.")
                data = generate_valid_random_molecules(
                    config = config,
                    manager = molecule_manager,
                    n_samples = params.n_samples_start,
                    mutation_prob = 0,
                    elite_prob = 0,
                    executor = cpu_executor,
                    n_workers = n_workers,
                    avoid_names = params.seen_molecules,
                    elite_names = None,
                    component_weights = component_weights
                )
            elif params.use_synthon_search and iteration >= 2 and params.synthon_lib is not None and exploited_status is False:
                bt.logging.info(f"[Solution] Iteration {iteration}: Smart synthon similarity search has started.")

                if params.score_improvement_rate > 0.05:
                    sim_threshold = 0.75
                    n_per_base = 15
                    n_seeds = 20
                    synthon_ratio = 0.75
                    bt.logging.info(f"[Solution] High improvement ({params.score_improvement_rate:.4f}), tight similarity (0.75)")

                elif params.score_improvement_rate > 0.02:
                    sim_threshold = 0.70
                    n_per_base = 18
                    n_seeds = 25
                    synthon_ratio = 0.75
                    bt.logging.info(f"[Solution] Good improvement ({params.score_improvement_rate:.4f}), medium-tight similarity (0.70)")

                elif params.score_improvement_rate > 0.005:
                    sim_threshold = 0.65
                    n_per_base = 20
                    n_seeds = 30
                    synthon_ratio = 0.70
                    bt.logging.info(f"[Solution] Moderate improvement ({params.score_improvement_rate:.4f}), medium similarity (0.65)")

                else:
                    bt.logging.info(f"[Solution] Low improvement ({params.score_improvement_rate:.4f}), using PROVEN MULTI-RANGE strategy")

                    n_synthon_top1 = int(n_samples * 0.25)
                    synthon_top1_df = generate_molecules_from_synthon_library(
                        params.synthon_lib,
                        top_pool.head(1),
                        n_synthon_top1,
                        min_similarity=0.91,
                        n_per_base=50
                    )
                    bt.logging.info(f"[Solution] Generated {len(synthon_top1_df)} TOP-1 synthon candidates (sim=0.90, n_per_base=80)")

                    n_synthon_tight = int(n_samples * 0.07)
                    synthon_tight_df = generate_molecules_from_synthon_library(
                        params.synthon_lib,
                        top_pool.head(5),
                        n_synthon_tight,
                        min_similarity=0.80,
                        n_per_base=30
                    )
                    bt.logging.info(f"[Solution] Generated {len(synthon_tight_df)} TIGHT synthon candidates (sim=0.80)")

                    n_synthon_medium = int(n_samples * 0.21)
                    seed_medium = top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:]
                    synthon_medium_df = generate_molecules_from_synthon_library(
                        params.synthon_lib,
                        seed_medium,
                        n_synthon_medium,
                        min_similarity=0.55,
                        n_per_base=15
                    )
                    bt.logging.info(f"[Solution] Generated {len(synthon_medium_df)} MEDIUM synthon candidates (sim=0.55)")

                    n_synthon_broad = int(n_samples * 0.21)
                    synthon_broad_df = generate_molecules_from_synthon_library(
                        params.synthon_lib,
                        top_pool.head(100),
                        n_synthon_broad,
                        min_similarity=0.40,
                        n_per_base=20
                    )
                    bt.logging.info(f"[Solution] Generated {len(synthon_broad_df)} BROAD synthon candidates (sim=0.40)")

                    synthon_df = pd.concat([synthon_top1_df, synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)
                    synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")
                    synthon_df = synthon_df[~synthon_df["name"].isin(params.seen_molecules)]
                    
                    if not synthon_df.empty:
                        synthon_df = molecule_manager.validate_molecules(config, synthon_df, time_elapsed=iteration_start_time - time_start)
                        bt.logging.info(f"[Solution] {len(synthon_df)} multi-range synthon candidates passed validation")

                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules(
                            config = config,
                            manager = molecule_manager,
                            n_samples = n_traditional,
                            mutation_prob = params.mutation_prob,
                            elite_prob = params.elite_prob,
                            executor = cpu_executor,
                            n_workers = n_workers,
                            avoid_names = params.seen_molecules,
                            elite_names = elite_names,
                            component_weights = component_weights
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles"])

                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[Solution] Combined: {len(data)} total ({len(synthon_df)} multi-range synthon + {len(traditional_df)} GA)")

                    synthon_df = None

                if params.score_improvement_rate > 0.005:
                    n_synthon = int(n_samples * synthon_ratio)
                    synthon_gen_start = time.time()
                    synthon_df = generate_molecules_from_synthon_library(
                        params.synthon_lib,
                        top_pool.head(n_seeds),
                        n_synthon,
                        min_similarity=sim_threshold,
                        n_per_base=n_per_base
                    )
                    synthon_df = synthon_df[~synthon_df["name"].isin(params.seen_molecules)]
                    bt.logging.info(f"[Solution] Generated {len(synthon_df)} synthon candidates in {time.time() - synthon_gen_start:.2f}s")

                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules(
                            config = config,
                            manager = molecule_manager,
                            n_samples = n_traditional,
                            mutation_prob = params.mutation_prob,
                            elite_prob = params.elite_prob,
                            executor = cpu_executor,
                            n_workers = n_workers,
                            avoid_names = params.seen_molecules,
                            elite_names = elite_names,
                            component_weights = component_weights
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles"])

                    if not synthon_df.empty:
                        synthon_df = molecule_manager.validate_molecules(config, synthon_df, time_elapsed=iteration_start_time - time_start)
                        bt.logging.info(f"[Solution] {len(synthon_df)} synthon candidates passed validation")

                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[Solution] Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")

            elif params.no_improvement_counter < 3 and exploited_status is False:
                bt.logging.info(f"[Solution] Iteration {iteration}: Standard genetic algorithm")
                data = generate_valid_random_molecules(
                    config = config,
                    manager = molecule_manager,
                    n_samples = n_samples,
                    mutation_prob = params.mutation_prob,
                    elite_prob = params.elite_prob,
                    executor = cpu_executor,
                    batch_size = 400,
                    n_workers = n_workers,
                    avoid_names = params.seen_molecules,
                    elite_names = elite_names,
                    component_weights = component_weights
                )
            elif params.no_improvement_counter < 6 and exploited_status is False:
                bt.logging.info(f"[Solution] Iteration {iteration}: Exploring similar space (no_improvement={params.no_improvement_counter})")
                data = cpu_random_candidates_with_similarity(
                    molecule_manager,
                    30,
                    config,
                    top_pool.head(50)[["name", "smiles"]],
                    params.seen_molecules,
                    0.65
                )
                
                seed_df = pd.DataFrame(columns=["name", "smiles", "tanimoto_similarity"])
            
            elif exploited_status is False:
                bt.logging.info(f"[Solution] Iteration {iteration}: Broad exploration reset (no_improvement={params.no_improvement_counter})")
                data = cpu_random_candidates_with_similarity(
                    molecule_manager,
                    40,
                    config,
                    top_pool.head(100)[["name", "smiles"]],
                    params.seen_molecules,
                    0.0
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "tanimoto_similarity"])
                no_improvement_counter = 0
            
            gen_time = time.time() - iteration_start_time
            bt.logging.info(f"[Solution] Iteration {iteration}: {len(data)} Samples Generated in ~{gen_time:.2f}s (pre-score)")
            if data.empty:
                bt.logging.warning(f"[Solution] Iteration {iteration}: No valid molecules produced; continuing")
                continue
                
            if not seed_df.empty:
                data = pd.concat([data, seed_df])
                data = data.drop_duplicates(subset=["name"], keep="first")
                seed_df = pd.DataFrame(columns=["name", "smiles", "tanimoto_similarity"])
                
            try:
                filtered_data = data[~data["name"].isin(params.seen_molecules)]
                if len(filtered_data) < len(data):
                    bt.logging.warning(f"[Solution] Iteration {iteration}: {len(data) - len(filtered_data)} molecules were previously seen")
                
                dup_ratio = (len(data) - len(filtered_data)) / max(1, len(data))

                if dup_ratio > 0.7:
                    params.mutation_prob = min(0.9, params.mutation_prob * 1.5)
                    params.elite_prob = max(0.15, params.elite_prob * 0.7)
                    bt.logging.warning(f"[Solution] SEVERE duplication ({dup_ratio:.2%})! mut={params.mutation_prob:.2f}, elite={params.elite_prob:.2f}")
                elif dup_ratio > 0.5:
                    params.mutation_prob = min(0.7, params.mutation_prob * 1.3)
                    params.elite_prob = max(0.2, params.elite_prob * 0.8)
                    bt.logging.warning(f"[Solution] High duplication ({dup_ratio:.2%}), mut={params.mutation_prob:.2f}, elite={params.elite_prob:.2f}")
                elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                    params.mutation_prob = max(0.05, params.mutation_prob * 0.95)
                    params.elite_prob = min(0.85, params.elite_prob * 1.05)

                data = filtered_data

            except Exception as e:
                bt.logging.warning(f"[Solution] Pre-score deduplication failed: {e}")
            
            if data.empty:
                bt.logging.error(f"[Solution] Iteration {iteration}: ALL molecules were duplicates! Skipping scoring and continuing...")
                params.mutation_prob = min(0.95, params.mutation_prob * 2.0)
                params.elite_prob = max(0.1, params.elite_prob * 0.5)
                bt.logging.warning(f"[Miner] Emergency diversity boost: mut={params.mutation_prob:.2f}, elite={params.elite_prob:.2f}")
                continue

            data = data.reset_index(drop=True)
            
            cpu_futures = []
            if not top_pool.empty and iteration > 3:
                if params.score_improvement_rate < 0.01:
                    bt.logging.info("[Solution] CPU random candidates with similarity have been started to find.")
                    cpu_futures.append((
                        cpu_executor.submit(
                            cpu_random_candidates_with_similarity,
                            molecule_manager,
                            40,
                            config,
                            top_pool.head(5)[["name", "smiles"]],
                            params.seen_molecules,
                            0.80
                        ),
                        "tight-top5"
                    ))

                    cpu_futures.append((
                        cpu_executor.submit(
                            cpu_random_candidates_with_similarity,
                            molecule_manager,
                            30,
                            config,
                            top_pool.head(20)[["name", "smiles"]],
                            params.seen_molecules,
                            0.65
                        ),
                        "medium-top20"
                    ))
            
            if len(data) == 0:
                bt.logging.error(f"[Solution] Iteration {iteration}: No molecules to score! Continuing...")
                continue

            bt.logging.info(f"[Solution] Iteration {iteration}: Scoring {len(data)} molecules with PSICHIC...")
            
            gpu_start_time = time.time()
            data["target"] = model_manager.get_target_score_from_data(data["smiles"])
            data["anti"] = model_manager.get_antitarget_score()
            data["score"] = data["target"] - (config["antitarget_weight"] * data["anti"])
            
            gpu_time = time.time() - gpu_start_time
            
            bt.logging.info(f"[Solution] Iteration {iteration}: GPU scoring time ~{gpu_time:.2f}s")
            
            if cpu_futures:
                bt.logging.info(f"[Solution] Finalizing cpu futures to get similar random molecules")
                for cpu_future, strategy_name in cpu_futures:
                    try:
                        cpu_df = cpu_future.result(timeout=0)
                        if not cpu_df.empty:
                            if seed_df.empty:
                                seed_df = cpu_df.copy()
                            else:
                                seed_df = pd.concat([seed_df, cpu_df], ignore_index=True)
                            bt.logging.info(f"[Solution] CPU similarity ({strategy_name}) found {len(cpu_df)} candidates")
                    except TimeoutError:
                        bt.logging.warning(f"[Solution] CPU similarity ({strategy_name}) Time out")
                        pass
                    except Exception as e:
                        bt.logging.warning(f"[Solution] CPU similarity ({strategy_name}) failed: {e}")

                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["name"], keep="first")
            
            data["inchi"] = data["smiles"].map(MoleculeUtils.generate_inchikey)
            
            params.seen_molecules = params.seen_molecules | set(data["name"].to_list())
            prev_avg_score = top_pool.head(config["num_molecules"])['score'].mean() if not top_pool.empty else None
            total_data = data[["name", "smiles", "inchi", "score", "target", "anti"]]
            
            if not total_data.empty:
                if not top_pool.empty:
                    top_pool = pd.concat([top_pool, total_data], ignore_index=True)
                else:
                    top_pool = pd.concat([total_data], ignore_index = True)
                top_pool = top_pool.sort_values(by="score", ascending=False)
                top_pool = top_pool.drop_duplicates(subset=["inchi"], keep="first")
                # Larger buffer pool gives more diverse candidates for synthon/crossover
                top_pool = top_pool.head(config["num_molecules"] + 150)
            else:
                bt.logging.warning(f"[Solution] Iteration {iteration}: No valid scored data to add to pool")
            
            current_avg_score = top_pool.head(config["num_molecules"])['score'].mean() if not top_pool.empty else None
            
            if current_avg_score is not None and prev_avg_score is not None:
                params.score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
            elif current_avg_score is not None:
                params.score_improvement_rate = 1.0
                
            if params.score_improvement_rate == 0.0:
                params.no_improvement_counter += 1
            else:
                params.no_improvement_counter = 0
                
            if exploit_summary and 'exploited_reactant_ids' in exploit_summary and params.score_improvement_rate == 0.0:
                params.exploited_reactants.update(exploit_summary['exploited_reactant_ids'])
                bt.logging.info("Droping the picked reactant.......")
                
            iter_total_time = time.time() - iteration_start_time
            total_time = time.time() - time_start

            pool_avg = top_pool.head(config["num_molecules"])['score'].mean()
            pool_max = top_pool['score'].max()
            
            try:
                pool_entropy = MoleculeUtils.compute_maccs_entropy(top_pool.head(config["num_molecules"])['smiles'].tolist())
            except Exception as e:
                bt.logging.warning(f"[Solution] Error in computing maccs entropy: {e}")
                pool_entropy = 0.0

            top_entries = {"molecules": top_pool.head(config["num_molecules"])["name"].tolist()}

            mode_str = "SYNTHON"
            bt.logging.info(
                f"Iteration {iteration} | {iter_total_time:.1f}s | Total: {total_time:.0f}s | "
                f"Mode: {mode_str} | "
                f"Pool: avg={pool_avg:.4f} max={pool_max:.4f} ent={pool_entropy:.3f}"
            )
            
            if pool_entropy > config['entropy_min_threshold']:
                with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                    json.dump(top_entries, f, ensure_ascii=False, indent=2)
                bt.logging.info(f"[Solution] Top entries are saved....")
        pass
if __name__ == "__main__":
    config = get_config()
    time_start = time.time()
    
    initialize_solution(config)
    time_initialize_model = time.time() - time_start
    bt.logging.info(f"[Solution] Time to initialize the solution: {time_initialize_model}")
    
    find_solution(config, time_start)