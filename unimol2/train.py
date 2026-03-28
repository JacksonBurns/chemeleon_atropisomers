from pathlib import Path
from itertools import chain

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from unimol_tools import UniMolRepr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


class UniMolLitMLP(pl.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = self.loss_fn(predictions, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    pl.seed_everything(42)

    unimol = UniMolRepr(data_type='molecule', model_name='unimolv2', remove_hs=False)
    data_dir = Path("../cv_splits")

    train_df = pd.read_csv(data_dir /"repetition_0" / "fold_0" / "train.csv")
    val_df = pd.read_csv(data_dir / "repetition_0" / "fold_0" / "val.csv")
    test_df = pd.read_csv(data_dir / "repetition_0" / "fold_0" / "test.csv")

    all_smiles = list(chain(train_df["SMILES"], val_df["SMILES"], test_df["SMILES"]))
    unimol_embeddings = {
        smiles: rep
        for smiles, rep in
        zip(all_smiles, unimol.get_repr(all_smiles, return_tensor=True).numpy(force=True))
    }

    for repetition in range(5):
        subdir = data_dir / f"repetition_{repetition}"
        for fold in range(5):
            data_subdir = subdir / f"fold_{fold}"
        
            train_df = pd.read_csv(data_subdir / "train.csv")
            val_df = pd.read_csv(data_subdir / "val.csv")
            test_df = pd.read_csv(data_subdir / "test.csv")

            train_embeddings = np.array([unimol_embeddings[smiles] for smiles in train_df["SMILES"]])
            val_embeddings = np.array([unimol_embeddings[smiles] for smiles in val_df["SMILES"]])
            test_embeddings = np.array([unimol_embeddings[smiles] for smiles in test_df["SMILES"]])

            # Rescale temperature
            T_scaler = StandardScaler()
            train_df["T_rescaled"] = T_scaler.fit_transform(train_df[["T"]])
            val_df["T_rescaled"] = T_scaler.transform(val_df[["T"]])
            test_df["T_rescaled"] = T_scaler.transform(test_df[["T"]])

            # Construct Features
            train_features = np.hstack([train_embeddings, train_df['T_rescaled'].values.reshape(-1, 1)])
            val_features = np.hstack([val_embeddings, val_df['T_rescaled'].values.reshape(-1, 1)])
            test_features = np.hstack([test_embeddings, test_df['T_rescaled'].values.reshape(-1, 1)])

            # Convert to PyTorch DataLoaders
            train_loader = DataLoader(
                TensorDataset(
                    torch.tensor(train_features, dtype=torch.float32), 
                    torch.tensor(train_df["ΔG(kcal/mol)"].values, dtype=torch.float32).view(-1, 1)
                ), 
                batch_size=32, shuffle=True
            )
            
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(val_features, dtype=torch.float32), 
                    torch.tensor(val_df["ΔG(kcal/mol)"].values, dtype=torch.float32).view(-1, 1)
                ), 
                batch_size=32, shuffle=False
            )

            # 2. Setup Early Stopping Callback
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
            )

            # 3. Initialize PyTorch Lightning Trainer
            trainer = pl.Trainer(
                max_epochs=100,
                callbacks=[early_stop_callback],
                enable_progress_bar=True,
                logger=False # Set to True if you want to use TensorBoard
            )

            # Initialize the Model
            input_dim = train_features.shape[1]
            model = UniMolLitMLP(input_dim=input_dim, lr=1e-3)

            # 4. Train the Model!
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # 5. Evaluate on Test Set
            model.eval() # Set to evaluation mode
            with torch.no_grad():
                # Standard PyTorch inference
                X_test_tensor = torch.tensor(test_features, dtype=torch.float32)
                test_preds = model(X_test_tensor).numpy(force=True)
                
            test_r2 = r2_score(test_df["ΔG(kcal/mol)"].values, test_preds)
            print(f"Fold {fold} Test R^2 Score: {test_r2:.4f}")
            pd.DataFrame(data={'SMILES': test_df["SMILES"], "true": test_df["ΔG(kcal/mol)"], "pred": test_preds.flatten()}).to_csv(data_subdir / "unimol_pred.csv", index=False)
