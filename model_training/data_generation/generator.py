import json
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

class SchedulingDataGenerator:
    def __init__(self):
        # Load configurations
        config_path = Path(__file__).parent.parent / 'config'
        with open(config_path / 'scheduling_patterns.json') as f:
            self.patterns = json.load(f)
        with open(config_path / 'training_config.json') as f:
            self.config = json.load(f)

        self.current_date = datetime.now()

    def generate_dataset(self) -> List[Dict[str, str]]:
        """Generate complete training dataset"""
        num_examples = self.config['data']['train_examples']
        dataset = []

        for _ in range(num_examples):
            # Generate a single training example
            example = self._generate_single_example()
            dataset.append(example)

        return dataset

    def _generate_single_example(self) -> Dict[str, str]:
        """Generate a single training example"""
        # Select random task type
        task_type = random.choice(list(self.patterns['task_types'].keys()))
        task_info = self.patterns['task_types'][task_type]

        # Generate task details
        task = {
            'name': random.choice(task_info['activities']),
            'type': task_type,
            'duration': self._generate_duration(task_info['durations']),
            'deadline': self._generate_deadline(),
            'preferences': self._select_preferences(task_info['preferences']),
            'constraints': self._generate_constraints()
        }

        # Generate input and output formats
        return {
            'input': self._format_input(task),
            'output': self._generate_ideal_schedule(task)
        }

    def _generate_duration(self, duration_config: Dict) -> str:
        """Generate task duration"""
        hours = random.uniform(duration_config['min'], duration_config['max'])
        hours = round(hours * 2) / 2  # Round to nearest 30 mins
        return f"{hours} {duration_config['unit']}"

    def _generate_deadline(self) -> str:
        """Generate a realistic deadline"""
        days_ahead = random.randint(1, 7)
        deadline_date = self.current_date + timedelta(days=days_ahead)
        hour = random.randint(9, 17)  # Business hours
        deadline_date = deadline_date.replace(hour=hour, minute=0)
        return deadline_date.strftime("%A %I:%M%p")

    def _select_preferences(self, available_preferences: List[str]) -> List[str]:
        """Select relevant preferences"""
        num_preferences = random.randint(1, 3)
        return random.sample(available_preferences, k=num_preferences)

    def _generate_constraints(self) -> List[Dict[str, Any]]:
        """Generate scheduling constraints"""
        constraints = []

        # Add fixed classes
        for class_info in random.sample(self.patterns['fixed_schedules']['classes'], 
                                      k=random.randint(1, 3)):
            constraints.append({
                'type': 'fixed',
                'name': class_info['name'],
                'days': class_info['days'],
                'time': class_info['time']
            })

        # Add routines
        for routine in self.patterns['fixed_schedules']['routines']:
            constraints.append({
                'type': 'routine',
                'name': routine['name'],
                'days': routine['days'],
                'time': routine['time']
            })

        return constraints

    def _format_input(self, task: Dict) -> str:
        """Format the input for the model"""
        input_text = f"""Schedule task: {task['name']}
Duration: {task['duration']}
Deadline: {task['deadline']}

Preferences:
{chr(10).join('- ' + pref for pref in task['preferences'])}

Constraints:"""

        for constraint in task['constraints']:
            if constraint['type'] == 'fixed':
                input_text += f"\n- {constraint['name']} on {constraint['days']} at {constraint['time']}"
            else:
                input_text += f"\n- {constraint['name']} {constraint['days']} at {constraint['time']}"

        return input_text

    def _generate_ideal_schedule(self, task: Dict) -> str:
        """Generate ideal schedule based on task and constraints"""
        # Get preferred hours for task type
        preferred_hours = self.patterns['time_patterns']['preferred_hours'][task['type']]
        
        # Find a suitable day (1-3 days before deadline)
        days_before = random.randint(1, 3)
        schedule_date = datetime.strptime(task['deadline'], "%A %I:%M%p") - timedelta(days=days_before)
        
        # Set time within preferred hours
        hour = random.randint(preferred_hours['start'], preferred_hours['end'])
        schedule_date = schedule_date.replace(hour=hour, minute=0)
        
        # Parse duration
        duration_hours = float(task['duration'].split()[0])
        end_date = schedule_date + timedelta(hours=duration_hours)
        
        reasons = [
            f"Scheduled during preferred hours for {task['type']} tasks",
            f"Allows sufficient buffer time before deadline",
            f"Avoids conflicts with fixed schedules",
            f"Matches energy level preferences",
            f"Provides adequate preparation time"
        ]

        return f"""Schedule from {schedule_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}
Reason: {random.choice(reasons)}"""