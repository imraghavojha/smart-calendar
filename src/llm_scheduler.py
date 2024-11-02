# src/llm_scheduler.py

from datetime import datetime, timedelta
import requests
from zoneinfo import ZoneInfo
from typing import Dict, Optional

class LocalLLMScheduler:
    def __init__(self, calendar_manager, rules_manager):
        self.calendar_manager = calendar_manager
        self.rules_manager = rules_manager
        self.timezone = ZoneInfo('Asia/Kolkata')
        self.ollama_url = "http://localhost:11434/api/generate"

    def schedule_task(self, task: Dict) -> Optional[Dict]:
        """Schedule a task using local LLM for decision making"""
        # Get current constraints
        constraints = self._get_current_constraints()
        
        # Create prompt
        prompt = self._create_scheduling_prompt(task, constraints)
        
        # Get scheduling decision from local LLM
        schedule_decision = self._get_llm_decision(prompt)
        
        if schedule_decision:
            return self._implement_schedule_decision(task, schedule_decision)
        return None

    def _get_llm_decision(self, prompt: str) -> Optional[Dict]:
        """Get scheduling decision from local LLM"""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": "mistral:7b-instruct",
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._parse_llm_response(result['response'])
            else:
                print(f"Error from LLM API: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error getting LLM decision: {e}")
            return None

    def _create_scheduling_prompt(self, task: Dict, constraints: Dict) -> str:
        """Create a focused prompt for scheduling"""
        prompt = f"""As a calendar scheduling assistant, help schedule this task considering the constraints.
Task: {task['name']}
Duration: {task['duration']}
Deadline: {task['deadline']}

Current Schedule:
Fixed Events:"""

        for event in constraints['fixed_events']:
            prompt += f"\n- {event['summary']} on {event['days']}: {event['start_time']}-{event['end_time']}"

        prompt += "\n\nBlocked Times:"
        for block in constraints['blocked_times']:
            prompt += f"\n- {block['summary']} on {block['days']}: {block['start_time']}-{block['end_time']}"

        prompt += "\n\nPreferences:"
        for pref in constraints['preferences']:
            prompt += f"\n- {pref}"

        prompt += """\n\nProvide the best time slot in this exact format:
Date: YYYY-MM-DD
Start Time: HH:MM
End Time: HH:MM
Reasoning: Brief explanation

Only respond with these four lines, no other text."""

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse LLM response into structured format"""
        try:
            lines = response.strip().split('\n')
            schedule = {}
            
            for line in lines:
                if line.startswith('Date:'):
                    schedule['date'] = line.split('Date:')[1].strip()
                elif line.startswith('Start Time:'):
                    schedule['start_time'] = line.split('Start Time:')[1].strip()
                elif line.startswith('End Time:'):
                    schedule['end_time'] = line.split('End Time:')[1].strip()
            
            return schedule if len(schedule) == 3 else None
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def _implement_schedule_decision(self, task: Dict, decision: Dict) -> Dict:
        """Implement the scheduling decision by creating calendar event"""
        try:
            start_time = datetime.strptime(
                f"{decision['date']} {decision['start_time']}", 
                "%Y-%m-%d %H:%M"
            ).replace(tzinfo=self.timezone)
            
            end_time = datetime.strptime(
                f"{decision['date']} {decision['end_time']}", 
                "%Y-%m-%d %H:%M"
            ).replace(tzinfo=self.timezone)
            
            event = self.calendar_manager.add_event(
                summary=task['name'],
                start_time=start_time,
                end_time=end_time,
                description=f"Auto-scheduled task: {task['name']}\nDuration: {task['duration']}\nDeadline: {task['deadline']}"
            )
            
            return event
            
        except Exception as e:
            print(f"Error implementing schedule decision: {e}")
            return None

    def _get_current_constraints(self) -> Dict:
        """Get all current schedule constraints"""
        constraints = self.rules_manager.get_scheduling_constraints()
        
        start_date = datetime.now(self.timezone)
        end_date = start_date + timedelta(weeks=1)
        existing_events = self.calendar_manager.list_events(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "fixed_events": constraints['fixed_events'],
            "blocked_times": constraints['blocked_times'],
            "preferences": constraints['preferences'],
            "existing_events": existing_events
        }