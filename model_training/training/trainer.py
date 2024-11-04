import logging
from pathlib import Path
import json
import pandas as pd
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset
from tqdm.auto import tqdm
import psutil

class SchedulerTrainer:
    def __init__(self):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Fix paths based on actual structure
        root_dir = Path(__file__).parent.parent.parent
        self.model_dir = root_dir / 'models'
        self.config_path = root_dir / 'model_training' / 'config' / 'training_config.json'
        
        # Load config
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        # Initialize model and tokenizer
        self.logger.info("Initializing T5-Large model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large', model_max_length=256)
        self.model = T5ForConditionalGeneration.from_pretrained('t5-large')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Log memory usage
        ram = psutil.virtual_memory()
        self.logger.info(f"Available RAM: {ram.available / 1e9:.1f}GB")

    def prepare_data(self):
        """Load and prepare datasets with visible progress"""
        self.logger.info("Loading datasets...")
        
        train_path = self.model_dir / 'data' / 'train_data.json'
        val_path = self.model_dir / 'data' / 'validation_data.json'
        
        print("\nProcessing training data...")
        with open(train_path) as f:
            train_data = json.load(f)
        
        print("Processing validation data...")
        with open(val_path) as f:
            val_data = json.load(f)
            
        # Convert to pandas DataFrame
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        print(f"\nDataset loaded: {len(train_df)} training, {len(val_df)} validation examples")
        
        # Convert to HuggingFace datasets
        self.train_dataset = Dataset.from_pandas(train_df)
        self.val_dataset = Dataset.from_pandas(val_df)
        
        # Tokenize with visible progress
        print("\nTokenizing datasets (this may take a few minutes)...")
        
        self.train_dataset = self.train_dataset.map(
            self._tokenize_data,
            batched=True,
            batch_size=16,  # Smaller batch size
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing training data"
        )
        
        self.val_dataset = self.val_dataset.map(
            self._tokenize_data,
            batched=True,
            batch_size=16,
            remove_columns=self.val_dataset.column_names,
            desc="Tokenizing validation data"
        )
        
        print("\nData preparation completed! Starting training...")

    def _tokenize_data(self, examples):
        """Tokenize inputs and outputs"""
        model_inputs = self.tokenizer(
            examples['input'],
            max_length=self.config['model']['max_length'],
            padding='max_length',
            truncation=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['output'],
                max_length=self.config['model']['max_length'],
                padding='max_length',
                truncation=True
            )
            
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    def train(self):
        """Train the model with detailed progress tracking"""
        self.logger.info("Starting training...")
        
        # Calculate steps
        total_steps = len(self.train_dataset) * self.config['model']['epochs']
        self.logger.info(f"Total training steps: {total_steps}")
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / 'checkpoints'),
            num_train_epochs=self.config['model']['epochs'],
            per_device_train_batch_size=self.config['model']['batch_size'],
            per_device_eval_batch_size=self.config['model']['batch_size'],
            gradient_accumulation_steps=self.config['model']['gradient_accumulation_steps'],
            warmup_steps=self.config['model']['warmup_steps'],
            weight_decay=self.config['model']['weight_decay'],
            logging_dir=str(self.model_dir / 'logs'),
            logging_steps=1,          # Log every step
            eval_steps=10,            # Evaluate more frequently
            save_steps=10,            # Save more frequently
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="tensorboard",
            disable_tqdm=False,
            remove_unused_columns=False,  # Added for stability
            gradient_checkpointing=True,  # Added for memory efficiency
            fp16=False,                  # Disable mixed precision
            dataloader_num_workers=0      # Single worker for stability
        )
        
        # Progress callback
        class ProgressCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 1 == 0:  # Print every step
                    print(f"\nStep {state.global_step}/{state.max_steps}")
                    if state.log_history:
                        last_log = state.log_history[-1]
                        print(f"Loss: {last_log.get('loss', 'N/A'):.4f}")
                        print(f"Learning rate: {last_log.get('learning_rate', 'N/A'):.6f}")
        
        # Initialize trainer
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
        
        try:
            print("\nStarting training loop...")
            trainer.train()
            self.logger.info("Training completed successfully!")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
        finally:
            print("\nSaving model regardless of training outcome...")
            self.save_model()

            
    def save_model(self):
        """Save the trained model"""
        self.logger.info("Saving model...")
        save_dir = self.model_dir / 'final_model'
        save_dir.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        if self.config['optimization']['quantize']:
            self.quantize_model(save_dir)
            
        self.logger.info(f"Model saved to {save_dir}")

    def quantize_model(self, model_dir: Path):
        """Quantize the model to reduce size"""
        self.logger.info("Quantizing model...")
        quantized_dir = self.model_dir / 'quantized_model'
        quantized_dir.mkdir(exist_ok=True)
        
        # Quantize
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Save quantized model
        quantized_model.save_pretrained(quantized_dir)
        self.tokenizer.save_pretrained(quantized_dir)
        
        self.logger.info(f"Quantized model saved to {quantized_dir}")