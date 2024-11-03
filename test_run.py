from model_training.scripts.train_pipeline import TrainingPipeline
import logging

def main():
    # Set logging to show detailed progress
    logging.basicConfig(level=logging.INFO)
    
    print("Starting test run...")
    
    try:
        # Initialize and run pipeline
        pipeline = TrainingPipeline()
        
        # Check system requirements
        print("\nChecking system requirements...")
        pipeline.check_requirements()
        
        # Run pipeline
        print("\nRunning pipeline...")
        pipeline.run_pipeline()
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    main()