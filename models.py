import os
import pandas as pd
import bittensor as bt
from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening

class ModelManager:
    def __init__(self, config: dict):
        self.target_models = []
        self.antitarget_models = []
    
        for seq in config["target_sequences"]:
            wrapper = PsichicWrapper()
            wrapper.initialize_model(seq)
            self.target_models.append(wrapper)
        
        for seq in config["antitarget_sequences"]:
            wrapper = PsichicWrapper()
            wrapper.initialize_model(seq)
            self.antitarget_models.append(wrapper)
    
    def get_target_score_from_data(self, data: pd.Series):
        try:
            target_scores = []
            smiles_list = data.tolist()
            for target_model in self.target_models:
                scores = target_model.score_molecules(smiles_list)
                for antitarget_model in self.antitarget_models:
                    antitarget_model.smiles_list = smiles_list
                    antitarget_model.smiles_dict = target_model.smiles_dict
                scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
                target_scores.append(scores["target"])
            target_series = pd.DataFrame(target_scores).mean(axis=0)
            return target_series
        except Exception as e:
            bt.logging.error(f"Target scoring error: {e}")
            return pd.Series(dtype=float)
    
    def get_antitarget_score(self):
        try:
            antitarget_scores = []
            for i, antitarget_model in enumerate(self.antitarget_models):
                antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
                antitarget_model.screen_df = virtual_screening(
                    antitarget_model.screen_df, 
                    antitarget_model.model, 
                    antitarget_model.screen_loader,
                    os.getcwd(),
                    save_interpret=False,
                    ligand_dict=antitarget_model.smiles_dict, 
                    device=antitarget_model.device,
                    save_cluster=False,
                )
                scores = antitarget_model.screen_df[['predicted_binding_affinity']]
                scores = scores.copy()
                scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
                antitarget_scores.append(scores[f"anti_{i}"])
            
            if not antitarget_scores:
                return pd.Series(dtype=float)

            anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
            return anti_series
        except Exception as e:
            print(f"Antitarget scoring error: {e}")
            return pd.Series(dtype=float)