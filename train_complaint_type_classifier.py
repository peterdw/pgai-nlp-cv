from complaint_type_classifier_trainer import ComplaintTypeClassifierTrainer

if __name__ == "__main__":
    trainer = ComplaintTypeClassifierTrainer(precision_mode="4bit", num_train_epochs=10)
    trainer.train()

    # Optional: print label mapping
    print("Label mapping:", trainer.get_label_mapping())
