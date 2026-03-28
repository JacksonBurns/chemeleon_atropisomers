from pathlib import Path
from itertools import chain

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToConcatenatedVector, MolToMorganFP, MolToRDKitPhysChem

if __name__ == "__main__":
    data_dir = Path("../cv_splits")

    # Assuming fold_0 contains ALL unique SMILES across the entire dataset
    train_df = pd.read_csv(data_dir / "repetition_0" / "fold_0" / "train.csv")
    val_df = pd.read_csv(data_dir / "repetition_0" / "fold_0" / "val.csv")
    test_df = pd.read_csv(data_dir / "repetition_0" / "fold_0" / "test.csv")

    all_smiles = list(chain(train_df["SMILES"], val_df["SMILES"], test_df["SMILES"]))
    
    feature_extractor = Pipeline(
        [
            ("auto2mol", AutoToMol()), 
            (
                "morgan_physchem",
                MolToConcatenatedVector(
                    [
                        ("morgan", MolToMorganFP(n_bits=2048, radius=2, counted=True, return_as="dense")), 
                        ("RDKitPhysChem", MolToRDKitPhysChem(standardizer=None, return_with_errors=True))
                    ]
                )
            )
        ],
        n_jobs=-1,
    )

    all_features = feature_extractor.transform(all_smiles)

    mol_embeddings = {smiles: features for smiles, features in zip(all_smiles, all_features)}

    for repetition in range(5):
        subdir = data_dir / f"repetition_{repetition}"
        for fold in range(5):
            data_subdir = subdir / f"fold_{fold}"
            
            train_df = pd.read_csv(data_subdir / "train.csv")
            val_df = pd.read_csv(data_subdir / "val.csv")
            # random forest doesn't do early stopping, so we will allow it to also train on the validation data
            train_df = pd.concat((train_df, val_df))
            test_df = pd.read_csv(data_subdir / "test.csv")

            train_embeddings = np.array([mol_embeddings[smiles] for smiles in train_df["SMILES"]])
            test_embeddings = np.array([mol_embeddings[smiles] for smiles in test_df["SMILES"]])

            train_features = np.hstack([train_embeddings, train_df['T'].values.reshape(-1, 1)])
            test_features = np.hstack([test_embeddings, test_df['T'].values.reshape(-1, 1)])

            # Impute any NaNs (from molecules that failed RDKit parsing) before hitting the RF
            imputer = SimpleImputer(strategy='median')
            train_features = imputer.fit_transform(train_features)
            test_features = imputer.transform(test_features)

            rf_model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
            rf_model.fit(train_features, train_df["ΔG(kcal/mol)"].values)

            # 5. Evaluate on Test Set
            test_preds = rf_model.predict(test_features)
                
            test_r2 = r2_score(test_df["ΔG(kcal/mol)"].values, test_preds)
            print(f"Fold {fold} Test R^2 Score: {test_r2:.4f}")
            
            # Save Predictions
            pd.DataFrame({
                'SMILES': test_df["SMILES"], 
                "true": test_df["ΔG(kcal/mol)"], 
                "pred": test_preds
            }).to_csv(data_subdir / "physchem_forest_pred.csv", index=False)
