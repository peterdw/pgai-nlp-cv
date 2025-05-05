import pandas as pd
from datasets import DatasetDict, Dataset
from base_trainer import BaseTrainer


class ComplaintClassifierTrainer(BaseTrainer):
    def __init__(self, model_name="bert-base-uncased", precision_mode="16bit", num_train_epochs=10):
        super().__init__(
            model_name=model_name,
            precision_mode=precision_mode,
            num_labels=2,
            num_train_epochs=num_train_epochs,
            model_dir=f"models/complaint-classifier-{model_name.replace('/', '-')}-{precision_mode}"
        )

    def load_and_prepare_data(self):

        if self.data_file is None:
            raise ValueError("Data file path must be provided.")

        df = pd.read_feather(self.data_file)
        df = df[["ProcessedTextBody", "Complaint"]].dropna()
        df = df.rename(columns={"Complaint": "label"})

        # Convert boolean to int (if needed)
        if df["label"].dtype == "bool":
            df["label"] = df["label"].astype(int)

        # Shuffle full dataset once before splitting
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        train_ratio, val_ratio, test_ratio = self.split_ratio
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1."

        n_total = len(df)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]

        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
            "test": Dataset.from_pandas(test_df),
        })

        self.print_dataset_stats(train_df, val_df, test_df)
        return dataset, train_df, val_df, test_df
