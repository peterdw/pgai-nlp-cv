from complaint_classifier_trainer import ComplaintClassifierTrainer

if __name__ == "__main__":

    trainer = ComplaintClassifierTrainer(
        model_name="xlm-roberta-base",
        precision_mode="32bit",
        num_train_epochs=5
    )
    trainer.train()
