import pandas as pd
from datasets import DatasetDict, Dataset
from sklearn.preprocessing import LabelEncoder
from base_trainer import BaseTrainer


class ComplaintTypeClassifierTrainer(BaseTrainer):
    def __init__(self, model_name="bert-base-uncased", precision_mode="16bit", num_train_epochs=10):
        self.label_encoder = LabelEncoder()
        super().__init__(
            model_name=model_name,
            precision_mode=precision_mode,
            num_labels=4,
            num_train_epochs=num_train_epochs,
            model_dir=f"models/complaint-type-classifier-{model_name.replace('/', '-')}-{precision_mode}"
        )

    def load_and_prepare_data(self):
        if self.data_file is None:
            raise ValueError("Data file path must be provided.")

        df = pd.read_feather(self.data_file)

        # Filter: only valid complaints with defined and applicable types
        df = df[
            (df["Complaint"] == True) &
            (df["AGR_Type_of_Complaint__c"].notna()) &
            (df["AGR_Type_of_Complaint__c"] != "Not applicable")
            ][["ProcessedTextBody", "AGR_Type_of_Complaint__c"]].dropna()

        # Encode string labels (Quality, Logistics, etc.) into numeric values
        df["label"] = self.label_encoder.fit_transform(df["AGR_Type_of_Complaint__c"])

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

        self.print_dataset_stats(train_df, val_df, test_df, label_encoder=self.label_encoder)
        return dataset, train_df, val_df, test_df


def get_label_mapping(self):
    """Optional: Map integer back to string class"""
    return dict(zip(self.label_encoder.transform(self.label_encoder.classes_), self.label_encoder.classes_))
