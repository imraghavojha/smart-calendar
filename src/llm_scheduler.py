# src/llm_scheduler.py

from datetime import datetime, timedelta
import anthropic
from zoneinfo import ZoneInfo
import os
from typing import Dict, List, Optional

class LLMScheduler:
    def __init__(self, calendar_manager, rules_manager):
        self.calendar_manager = calendar_manager
        self.rules_manager = rules_manager
        self.claude = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        self.timezone = ZoneInfo('Asia/Kolkata')

    def schedule_task(self, task: Dict) -> Optional[Dict]:
        """
        Schedule a task using LLM for decision making
        
        Args:
            task: {
                "name": str,
                "duration": str,  # e.g., "3 hours"
                "deadline": str,  # e.g., "Monday 5pm"
            }
        """
        # 1. Get current schedule and constraints
        constraints = self._get_current_constraints()
        
        # 2. Create prompt for Claude
        prompt = self._create_scheduling_prompt(task, constraints)
        
        # 3. Get scheduling decision from Claude
        schedule_decision = self._get_llm_decision(prompt)
        
        # 4. Parse and validate the decision
        if schedule_decision:
            return self._implement_schedule_decision(task, schedule_decision)
        return None

    def _get_current_constraints(self) -> Dict:
        """Get all current schedule constraints"""
        # Get fixed events and rules
        constraints = self.rules_manager.get_scheduling_constraints()
        
        # Get next week's scheduled events
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

    def _create_scheduling_prompt(self, task: Dict, constraints: Dict) -> str:
        """Create a detailed prompt for Claude"""
        prompt = f"""You are a smart calendar scheduling assistant. 
Please help schedule the following task considering all constraints and preferences.

Task to Schedule:
- Name: {task['name']}
- Duration: {task['duration']}
- Deadline: {task['deadline']}

Current Schedule Constraints:

Fixed Events:"""
        
        for event in constraints['fixed_events']:
            prompt += f"\n- {event['summary']}: {self._format_days(event['days'])} {event['start_time']}-{event['end_time']}"

        prompt += "\n\nBlocked Times:"
        for block in constraints['blocked_times']:
            prompt += f"\n- {block['summary']}: {self._format_days(block['days'])} {block['start_time']}-{block['end_time']}"

        prompt += "\n\nPreferences:"
        for pref in constraints['preferences']:
            prompt += f"\n- {pref}"

        prompt += """

Please analyze these constraints and suggest the best time slot for this task.
Provide your response in the following format:

Recommended Slot:
Date: YYYY-MM-DD
Start Time: HH:MM
End Time: HH:MM

Reasoning:
1. [First reason for this choice]
2. [Second reason for this choice]
...

Alternative Slots:
1. [Alternative slot 1]
2. [Alternative slot 2]
"""
        return prompt

    def _get_llm_decision(self, prompt: str) -> Optional[Dict]:
        """Get scheduling decision from Claude"""
        try:
            response = self.claude.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse Claude's response to extract the scheduling decision
            return self._parse_llm_response(response.content)
            
        except Exception as e:
            print(f"Error getting LLM decision: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse Claude's response into a structured format"""
        try:
            # Split response into sections
            lines = response.split('\n')
            schedule = {}
            
            for line in lines:
                if line.startswith('Date:'):
                    schedule['date'] = line.split('Date:')[1].strip()
                elif line.startswith('Start Time:'):
                    schedule['start_time'] = line.split('Start Time:')[1].strip()
                elif line.startswith('End Time:'):
                    schedule['end_time'] = line.split('End Time:')[1].strip()
            
            if 'date' in schedule and 'start_time' in schedule and 'end_time' in schedule:
                return schedule
                
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
        
        return None

    def _implement_schedule_decision(self, task: Dict, decision: Dict) -> Dict:
        """Implement the scheduling decision by creating calendar event"""
        try:
            # Convert decision times to datetime objects
            start_time = datetime.strptime(
                f"{decision['date']} {decision['start_time']}", 
                "%Y-%m-%d %H:%M"
            ).replace(tzinfo=self.timezone)
            
            end_time = datetime.strptime(
                f"{decision['date']} {decision['end_time']}", 
                "%Y-%m-%d %H:%M"
            ).replace(tzinfo=self.timezone)
            
            # Create calendar event
            event = self.calendar_manager.add_event(
                summary=task['name'],
                start_time=start_time,
                end_time=end_time,
                description=f"Scheduled task: {task['name']}\nDuration: {task['duration']}\nDeadline: {task['deadline']}"
            )
            
            return event
            
        except Exception as e:
            print(f"Error implementing schedule decision: {e}")
            return None

    def _format_days(self, days: List[int]) -> str:
        """Convert day numbers to readable format"""
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return ', '.join(day_names[day] for day in days)

# Test function
def test_scheduler():
    from src.calendar_manager import CalendarManager
    from src.rules_parser import RulesManager
    
    print("Testing LLM Scheduler...")
    
    # Initialize components
    calendar = CalendarManager()
    rules = RulesManager(calendar)
    scheduler = LLMScheduler(calendar, rules)
    
    # Test task
    task = {
        "name": "Write Philosophy Essay",
        "duration": "3 hours",
        "deadline": "Monday 5pm"
    }
    
    # Try scheduling
    result = scheduler.schedule_task(task)
    
    if result:
        print(f"Task scheduled successfully: {result}")
    else:
        print("Failed to schedule task")

if __name__ == "__main__":
    test_scheduler()