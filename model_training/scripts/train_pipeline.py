from pathlib import Path
import sys
import logging
import torch
from model_training.training.trainer import SchedulerTrainer
from model_training.data_generation.generator import SchedulingDataGenerator

class TrainingPipeline:
    def __init__(self):
        self.setup_logging()
        self.root_dir = Path(__file__).parent.parent.parent
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(__file__).parent.parent.parent / 'models' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training.log'),
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
        else:
            self.logger.warning("No GPU found. Training will be slow on CPU.")
            
        # Check CPU memory
        import psutil
        ram = psutil.virtual_memory()
        self.logger.info(f"Available RAM: {ram.available / 1e9:.1f}GB")

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
            
            # Initialize trainer
            self.logger.info("Starting model training...")
            trainer = SchedulerTrainer()
            
            # Prepare data
            trainer.prepare_data()
            
            # Train model
            trainer.train()
            
            self.logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise