import json
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Dict, List
import logging

class SchedulingDataGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        config_path = Path(__file__).parent.parent / 'config'
        
        # Load configurations
        with open(config_path / 'training_config.json') as f:
            self.config = json.load(f)
        with open(config_path / 'scheduling_patterns.json') as f:
            self.patterns = json.load(f)
            
        self.output_dir = Path(__file__).parent.parent.parent / 'models' / 'data'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_datasets(self):
        """Create and save training and validation datasets"""
        self.logger.info("Generating datasets...")
        
        # Generate examples
        all_examples = self.generate_dataset(
            self.config['data']['train_examples']
        )
        
        # Split into train/validation
        split_idx = int(len(all_examples) * (1 - self.config['data']['validation_split']))
        train_data = all_examples[:split_idx]
        val_data = all_examples[split_idx:]
        
        # Save datasets
        self._save_dataset(train_data, 'train')
        self._save_dataset(val_data, 'validation')
        
        self.logger.info(f"Generated {len(train_data)} training and {len(val_data)} validation examples")

    def generate_dataset(self, num_examples: int) -> List[Dict]:
        """Generate dataset examples"""
        examples = []
        
        for _ in range(num_examples):
            # Select random task type
            task_type = random.choice(list(self.patterns['task_types'].keys()))
            task_info = self.patterns['task_types'][task_type]
            
            # Generate example
            example = self._generate_example(task_type, task_info)
            examples.append(example)
            
        return examples

    def _generate_example(self, task_type: str, task_info: Dict) -> Dict:
        """Generate single training example"""
        # Generate task details
        activity = random.choice(task_info['activities'])
        duration = self._generate_duration(task_info['durations'])
        deadline = self._generate_deadline()
        preferences = random.sample(task_info['preferences'], k=2)
        
        # Get relevant fixed schedules
        constraints = self._get_relevant_constraints()
        
        # Format input
        input_text = self._format_input(
            activity=activity,
            duration=duration,
            deadline=deadline,
            preferences=preferences,
            constraints=constraints
        )
        
        # Generate ideal schedule
        output_text = self._generate_schedule(
            task_type=task_type,
            duration=duration,
            deadline=deadline,
            constraints=constraints
        )
        
        return {
            'input': input_text,
            'output': output_text
        }

    def _generate_duration(self, duration_config: Dict) -> str:
        """Generate task duration"""
        hours = random.uniform(duration_config['min'], duration_config['max'])
        hours = round(hours * 2) / 2  # Round to nearest 30 mins
        return f"{hours} {duration_config['unit']}"

    def _generate_deadline(self) -> str:
        """Generate realistic deadline"""
        days_ahead = random.randint(1, 7)
        deadline = datetime.now() + timedelta(days=days_ahead)
        return deadline.strftime("%A %I:%M%p")

    def _get_relevant_constraints(self) -> List[Dict]:
        """Get relevant schedule constraints"""
        constraints = []
        
        # Add some fixed classes
        num_classes = random.randint(1, 3)
        classes = random.sample(self.patterns['fixed_schedules']['classes'], k=num_classes)
        constraints.extend(classes)
        
        # Add routine constraints
        constraints.extend(self.patterns['fixed_schedules']['routines'])
        
        return constraints

    def _format_input(self, activity: str, duration: str, deadline: str,
                     preferences: List[str], constraints: List[Dict]) -> str:
        """Format input text"""
        input_text = f"""Schedule task: {activity}
Duration: {duration}
Deadline: {deadline}

Preferences:
{chr(10).join('- ' + pref for pref in preferences)}

Constraints:"""

        for constraint in constraints:
            input_text += f"\n- {constraint['name']} on {constraint['days']} at {constraint['time']}"

        return input_text

    def _generate_schedule(self, task_type: str, duration: str,
                          deadline: str, constraints: List[Dict]) -> str:
        """Generate ideal schedule"""
        # Get preferred hours
        preferred = self.patterns['time_patterns']['preferred_hours'][task_type]
        
        # Generate schedule within preferred hours
        deadline_dt = datetime.strptime(deadline, "%A %I:%M%p")
        schedule_date = deadline_dt - timedelta(days=random.randint(1, 3))
        
        # Set time within preferred hours
        hour = random.randint(preferred['start'], preferred['end'])
        schedule_date = schedule_date.replace(hour=hour, minute=0)
        
        # Calculate end time
        duration_hours = float(duration.split()[0])
        end_date = schedule_date + timedelta(hours=duration_hours)
        
        # Generate reasoning
        reasons = [
            f"Scheduled during preferred hours for {task_type}",
            "Avoids conflicts with fixed schedules",
            "Provides buffer time before deadline",
            f"Optimal time for {task_type} based on preferences"
        ]

        return f"""Schedule from {schedule_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}
Reason: {random.choice(reasons)}"""

    def _save_dataset(self, data: List[Dict], split_name: str):
        """Save dataset split"""
        output_file = self.output_dir / f'{split_name}_data.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Saved {split_name} dataset to {output_file}")