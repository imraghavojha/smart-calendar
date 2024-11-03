# Add to the train method in SchedulerTrainer class:

def train(self):
    """Train the model with progress tracking"""
    self.logger.info("Starting training...")
    
    # Calculate steps
    total_steps = len(self.train_dataset) * self.config['model']['epochs']
    self.logger.info(f"Total training steps: {total_steps}")
    
    # Prepare training arguments with progress reporting
    training_args = TrainingArguments(
        output_dir=str(self.model_dir / 'checkpoints'),
        num_train_epochs=self.config['model']['epochs'],
        per_device_train_batch_size=self.config['model']['batch_size'],
        per_device_eval_batch_size=self.config['model']['batch_size'],
        gradient_accumulation_steps=self.config['model']['gradient_accumulation_steps'],
        warmup_steps=self.config['model']['warmup_steps'],
        weight_decay=self.config['model']['weight_decay'],
        logging_dir=str(self.model_dir / 'logs'),
        logging_steps=50,  # More frequent logging
        eval_steps=200,    # More frequent evaluation
        save_steps=500,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="tensorboard",  # Add tensorboard logging
        disable_tqdm=False,      # Show progress bars
    )
    
    # Custom callback for detailed progress
    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 50 == 0:
                self.logger.info(
                    f"Step: {state.global_step}/{state.max_steps} "
                    f"Loss: {state.log_history[-1]['loss']:.4f} "
                    f"Learning rate: {state.log_history[-1]['learning_rate']:.2e}"
                )
    
    # Initialize trainer with progress callback
    trainer = Trainer(
        model=self.model,
        args=training_args,
        train_dataset=self.train_dataset,
        eval_dataset=self.val_dataset,
        tokenizer=self.tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            ProgressCallback()
        ]
    )
    
    # Train with progress tracking
    try:
        trainer.train()
        self.logger.info("Training completed successfully!")
    except Exception as e:
        self.logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise
    finally:
        # Save final model
        self.save_model()