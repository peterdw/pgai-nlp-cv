from complaint_classifier_trainer import ComplaintClassifierTrainer

if __name__ == "__main__":
    trainer = ComplaintClassifierTrainer(precision_mode="4bit", num_train_epochs=10)
    trainer.train()
