from complaint_type_classifier_trainer import ComplaintTypeClassifierTrainer

if __name__ == "__main__":
    trainer = ComplaintTypeClassifierTrainer(
        model_name="distilbert-base-multilingual-cased",
        precision_mode="32bit",
        num_train_epochs=5
    )
    trainer.train()
