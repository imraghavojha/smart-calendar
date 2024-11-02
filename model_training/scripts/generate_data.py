from pathlib import Path
import json
import random
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_generation.generator import SchedulingDataGenerator

class DatasetCreator:
    def __init__(self):
        self.output_dir = Path(__file__).parent.parent.parent / 'trained_models' / 'data'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        config_path = Path(__file__).parent.parent / 'config' / 'training_config.json'
        with open(config_path) as f:
            self.config = json.load(f)

    def create_datasets(self):
        """Create and save training and validation datasets"""
        print("Initializing data generation...")
        generator = SchedulingDataGenerator()
        
        # Generate full dataset
        print(f"Generating {self.config['data']['train_examples']} examples...")
        dataset = generator.generate_dataset()
        
        # Validate data
        print("Validating generated data...")
        validated_data = self._validate_data(dataset)
        
        # Split into train/validation
        val_size = int(len(validated_data) * self.config['data']['validation_split'])
        random.shuffle(validated_data)
        train_data = validated_data[val_size:]
        val_data = validated_data[:val_size]
        
        # Save datasets
        self._save_data(train_data, 'train')
        self._save_data(val_data, 'validation')
        
        # Generate statistics
        self._generate_statistics(train_data, val_data)

    def _validate_data(self, dataset):
        """Validate and clean generated data"""
        valid_data = []
        
        for example in tqdm(dataset, desc="Validating examples"):
            if self._is_valid_example(example):
                valid_data.append(example)
        
        print(f"Kept {len(valid_data)}/{len(dataset)} valid examples")
        return valid_data

    def _is_valid_example(self, example):
        """Check if an example is valid"""
        try:
            # Check required fields
            if not example.get('input') or not example.get('output'):
                return False
            
            # Check input format
            if not all(key in example['input'] for key in ['Schedule task:', 'Duration:', 'Deadline:']):
                return False
            
            # Check output format
            if not 'Schedule from' in example['output'] or not 'Reason:' in example['output']:
                return False
            
            # Validate times
            if not self._validate_times(example['output']):
                return False
            
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def _validate_times(self, output):
        """Validate time formats in output"""
        try:
            # Extract times from output
            lines = output.split('\n')
            schedule_line = [l for l in lines if 'Schedule from' in l][0]
            times = schedule_line.replace('Schedule from ', '').split(' to ')
            
            # Try parsing times
            for time_str in times:
                datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            
            return True
        except:
            return False

    def _save_data(self, data, split_name):
        """Save dataset split"""
        # Save as JSON for easy inspection
        json_path = self.output_dir / f'{split_name}_data.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save as CSV for easy loading
        df = pd.DataFrame(data)
        csv_path = self.output_dir / f'{split_name}_data.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"Saved {split_name} data: {len(data)} examples")

    def _generate_statistics(self, train_data, val_data):
        """Generate and save dataset statistics"""
        stats = {
            'total_examples': len(train_data) + len(val_data),
            'train_examples': len(train_data),
            'validation_examples': len(val_data),
            'avg_input_length': sum(len(ex['input'].split()) for ex in train_data) / len(train_data),
            'avg_output_length': sum(len(ex['output'].split()) for ex in train_data) / len(train_data),
            'task_types': self._count_task_types(train_data),
            'time_distribution': self._analyze_time_distribution(train_data),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save statistics
        stats_path = self.output_dir / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\nDataset Statistics:")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Training examples: {stats['train_examples']}")
        print(f"Validation examples: {stats['validation_examples']}")
        print(f"Average input length: {stats['avg_input_length']:.1f} words")
        print(f"Average output length: {stats['avg_output_length']:.1f} words")

    def _count_task_types(self, data):
        """Count frequency of different task types"""
        task_types = {}
        for example in data:
            task_type = example['input'].split('\n')[0].split(':')[1].strip()
            task_types[task_type] = task_types.get(task_type, 0) + 1
        return task_types

    def _analyze_time_distribution(self, data):
        """Analyze distribution of scheduled times"""
        hours = {str(i).zfill(2): 0 for i in range(24)}
        
        for example in data:
            try:
                time = example['output'].split('\n')[0].split('from ')[1].split(' ')[1]
                hour = time.split(':')[0]
                hours[hour] += 1
            except:
                continue
                
        return hours

if __name__ == "__main__":
    print("Starting dataset creation...")
    creator = DatasetCreator()
    creator.create_datasets()
    print("Dataset creation completed!")