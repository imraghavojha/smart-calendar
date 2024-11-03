from pathlib import Path
import json
import pandas as pd
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import logging
from typing import Dict, List

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
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        self.model = T5ForConditionalGeneration.from_pretrained('t5-large')
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

    def prepare_data(self):
        """Load and prepare datasets"""
        self.logger.info("Loading datasets...")
        
        # Use existing paths where data is already generated
        train_path = self.model_dir / 'data' / 'train_data.json'
        val_path = self.model_dir / 'data' / 'validation_data.json'
        
        # Load JSON data
        with open(train_path) as f:
            train_data = json.load(f)
        with open(val_path) as f:
            val_data = json.load(f)
            
        # Convert to pandas DataFrame
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        # Convert to HuggingFace datasets
        self.train_dataset = Dataset.from_pandas(train_df)
        self.val_dataset = Dataset.from_pandas(val_df)
        
        # Tokenize datasets
        self.logger.info("Tokenizing datasets...")
        self.train_dataset = self.train_dataset.map(
            self._tokenize_data,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        self.val_dataset = self.val_dataset.map(
            self._tokenize_data,
            batched=True,
            remove_columns=self.val_dataset.column_names
        )

    def _tokenize_data(self, examples):
        """Tokenize inputs and outputs"""
        model_inputs = self.tokenizer(
            examples['input'],
            max_length=self.config['model']['max_length'],
            padding='max_length',
            truncation=True
        )
        
        # Tokenize targets
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
        """Train the model"""
        self.logger.info("Starting training...")
        
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
            logging_steps=100,
            eval_steps=500,
            save_steps=500,
            evaluation_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save final model
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