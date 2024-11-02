# model_training/training/evaluator.py

import json
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging
from transformers import T5ForConditionalGeneration, T5Tokenizer

class ModelEvaluator:
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self) -> Dict:
        """Run comprehensive model evaluation"""
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'metrics': {},
            'example_predictions': [],
            'error_analysis': {}
        }
        
        # Load test cases
        test_cases = self._load_test_cases()
        
        # Run predictions
        predictions = []
        errors = []
        
        for case in test_cases:
            try:
                prediction = self.generate_schedule(case['input'])
                predictions.append({
                    'input': case['input'],
                    'expected': case['output'],
                    'predicted': prediction
                })
                
                # Check for common errors
                if self._has_timing_error(prediction):
                    errors.append(('timing_error', case['input'], prediction))
                elif self._has_format_error(prediction):
                    errors.append(('format_error', case['input'], prediction))
                    
            except Exception as e:
                errors.append(('generation_error', case['input'], str(e)))
        
        # Calculate metrics
        results['metrics'] = self._calculate_metrics(predictions)
        
        # Add example predictions
        results['example_predictions'] = predictions[:5]
        
        # Error analysis
        results['error_analysis'] = self._analyze_errors(errors)
        
        return results

    def generate_schedule(self, input_text: str) -> str:
        """Generate schedule for given input"""
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _load_test_cases(self) -> List[Dict]:
        """Load test cases for evaluation"""
        test_data_path = Path(__file__).parent.parent.parent / 'trained_models' / 'data' / 'validation_data.json'
        with open(test_data_path) as f:
            return json.load(f)

    def _calculate_metrics(self, predictions: List[Dict]) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {
            'total_predictions': len(predictions),
            'format_accuracy': 0,
            'timing_accuracy': 0,
            'reasoning_included': 0
        }
        
        for pred in predictions:
            if self._has_valid_format(pred['predicted']):
                metrics['format_accuracy'] += 1
            if self._has_valid_timing(pred['predicted']):
                metrics['timing_accuracy'] += 1
            if 'Reason:' in pred['predicted']:
                metrics['reasoning_included'] += 1
        
        # Convert to percentages
        total = len(predictions)
        metrics['format_accuracy'] = (metrics['format_accuracy'] / total) * 100
        metrics['timing_accuracy'] = (metrics['timing_accuracy'] / total) * 100
        metrics['reasoning_included'] = (metrics['reasoning_included'] / total) * 100
        
        return metrics

    def _has_valid_format(self, text: str) -> bool:
        """Check if prediction has valid format"""
        required_elements = [
            'Schedule from',
            'to',
            'Reason:'
        ]
        return all(element in text for element in required_elements)

    def _has_valid_timing(self, text: str) -> bool:
        """Check if prediction has valid timing"""
        try:
            schedule_part = text.split('Reason:')[0].strip()
            time_parts = schedule_part.replace('Schedule from ', '').split(' to ')
            
            # Try parsing both times
            for time_str in time_parts:
                datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            return True
        except:
            return False

    def _has_timing_error(self, prediction: str) -> bool:
        """Check for timing errors in prediction"""
        try:
            schedule_part = prediction.split('Reason:')[0].strip()
            start_str, end_str = schedule_part.replace('Schedule from ', '').split(' to ')
            
            start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M')
            end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M')
            
            # Check if end time is before start time
            return end_time <= start_time
        except:
            return True

    def _has_format_error(self, prediction: str) -> bool:
        """Check for format errors in prediction"""
        required_format = [
            lambda x: 'Schedule from' in x,
            lambda x: 'to' in x,
            lambda x: 'Reason:' in x,
            lambda x: len(x.split('\n')) >= 2
        ]
        
        return not all(check(prediction) for check in required_format)

    def _analyze_errors(self, errors: List[tuple]) -> Dict:
        """Analyze prediction errors"""
        error_analysis = {
            'error_types': {},
            'total_errors': len(errors),
            'error_examples': []
        }
        
        # Count error types
        for error_type, _, _ in errors:
            error_analysis['error_types'][error_type] = error_analysis['error_types'].get(error_type, 0) + 1
        
        # Add example errors
        error_analysis['error_examples'] = [
            {
                'type': error_type,
                'input': input_text,
                'error': error_detail
            }
            for error_type, input_text, error_detail in errors[:5]
        ]
        
        return error_analysis

    def save_results(self, results: Dict, output_path: Path):
        """Save evaluation results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Evaluation results saved to {output_path}")