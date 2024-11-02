# model_training/scripts/train_pipeline.py

import logging
from pathlib import Path
import sys
import torch
from datetime import datetime

# Add model_training to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from model_training.data_generation.generator import SchedulingDataGenerator
from model_training.training.trainer import SchedulerTrainer
from model_training.training.evaluator import ModelEvaluator

class TrainingPipeline:
    def __init__(self):
        self.setup_logging()
        self.root_dir = root_dir
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = root_dir / 'trained_models' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )

    def check_requirements(self):
        """Check if system meets requirements"""
        self.logger.info("Checking system requirements...")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        self.logger.info(f"CUDA available: {cuda_available}")
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"Found {gpu_count} GPU(s): {gpu_name}")
            
            # Check memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 8:
                self.logger.warning("Less than 8GB GPU memory available. Training might be slow.")
        else:
            self.logger.warning("No GPU found. Training will be slow on CPU.")

    def run_pipeline(self):
        """Run the complete training pipeline"""
        try:
            self.logger.info("Starting training pipeline...")
            
            # Check requirements
            self.check_requirements()
            
            # Generate dataset
            self.logger.info("Generating training data...")
            generator = SchedulingDataGenerator()
            generator.create_datasets()
            
            # Train model
            self.logger.info("Starting model training...")
            trainer = SchedulerTrainer()
            trainer.prepare_data()
            trainer.train()
            
            # Evaluate model
            self.logger.info("Evaluating model...")
            evaluator = ModelEvaluator(trainer.model, trainer.tokenizer)
            evaluation_results = evaluator.evaluate()
            
            # Save evaluation results
            results_dir = self.root_dir / 'trained_models' / 'evaluation'
            results_dir.mkdir(exist_ok=True)
            evaluator.save_results(evaluation_results, results_dir / 'evaluation_results.json')
            
            self.logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

def main():
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()