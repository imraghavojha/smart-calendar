import os
from groq import Groq
from datetime import datetime, timedelta
import pytz
from typing import Dict, Optional
import logging
from .calendar_manager import CalendarManager
from .rules_parser import RulesManager

class SmartScheduler:
    def __init__(self, calendar_manager: CalendarManager, rules_manager: RulesManager):
        self.calendar_manager = calendar_manager
        self.rules_manager = rules_manager
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.logger = logging.getLogger(__name__)
        self.timezone = pytz.timezone('America/Chicago')  # Change to your timezone

    def schedule_task(self, task: Dict) -> Optional[Dict]:
        """
        Schedule a task using Groq's AI
        
        Args:
            task: {
                "name": str,
                "duration": str (e.g., "2 hours"),
                "deadline": str (e.g., "tomorrow 5pm"),
                "priority": str (optional, e.g., "high")
            }
        """
        try:
            # Get current calendar context
            context = self._get_calendar_context()
            
            # Create prompt
            prompt = self._create_scheduling_prompt(task, context)
            
            # Get AI suggestion
            completion = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an intelligent calendar assistant who helps schedule tasks optimally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Parse and validate suggestion
            schedule = self._parse_ai_response(completion.choices[0].message.content)
            
            if schedule:
                # Create calendar event
                event = self.calendar_manager.add_event(
                    summary=task['name'],
                    start_time=schedule['start_time'],
                    end_time=schedule['end_time'],
                    description=f"Auto-scheduled task\nDuration: {task['duration']}\nDeadline: {task['deadline']}\nReason: {schedule['reason']}"
                )
                return event
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error scheduling task: {str(e)}")
            return None

    def _get_calendar_context(self) -> Dict:
        """Get current calendar state and constraints"""
        # Get upcoming week's events
        now = datetime.now(self.timezone)
        week_later = now + timedelta(days=7)
        
        upcoming_events = self.calendar_manager.list_events(
            start_time=now,
            end_time=week_later
        )
        
        # Get rules and constraints
        constraints = self.rules_manager.get_scheduling_constraints()
        
        return {
            "events": upcoming_events,
            "fixed_events": constraints.get('fixed_events', []),
            "blocked_times": constraints.get('blocked_times', []),
            "preferences": constraints.get('preferences', [])
        }

    def _create_scheduling_prompt(self, task: Dict, context: Dict) -> str:
        """Create detailed prompt for Groq"""
        prompt = f"""Please help schedule this task optimally:

Task Details:
- Name: {task['name']}
- Duration: {task['duration']}
- Deadline: {task['deadline']}
{f"- Priority: {task['priority']}" if 'priority' in task else ""}

Current Calendar Context:

Fixed Events:"""
        
        for event in context['fixed_events']:
            prompt += f"\n- {event['summary']} on {event['days']} from {event['start_time']} to {event['end_time']}"

        prompt += "\n\nBlocked Times:"
        for block in context['blocked_times']:
            prompt += f"\n- {block['summary']} on {block['days']} from {block['start_time']} to {block['end_time']}"

        prompt += "\n\nUpcoming Events:"
        for event in context['events']:
            start = event.get('start', {}).get('dateTime', '')
            end = event.get('end', {}).get('dateTime', '')
            prompt += f"\n- {event['summary']} from {start} to {end}"

        prompt += "\n\nPreferences:"
        for pref in context['preferences']:
            prompt += f"\n- {pref}"

        prompt += """\n\nBased on this information, please suggest the optimal time slot for this task.
Consider:
1. Task deadline and duration
2. Fixed events and blocked times
3. User preferences
4. Energy levels throughout the day
5. Buffer time between tasks

Provide your response in exactly this format:
START_TIME: YYYY-MM-DD HH:MM
END_TIME: YYYY-MM-DD HH:MM
REASON: Brief explanation of why this time slot is optimal"""

        return prompt

    def _parse_ai_response(self, response: str) -> Optional[Dict]:
        """Parse Groq's response into schedule details"""
        try:
            lines = response.strip().split('\n')
            schedule = {}
            
            for line in lines:
                if line.startswith('START_TIME:'):
                    time_str = line.replace('START_TIME:', '').strip()
                    schedule['start_time'] = datetime.strptime(
                        time_str, 
                        '%Y-%m-%d %H:%M'
                    ).replace(tzinfo=self.timezone)
                    
                elif line.startswith('END_TIME:'):
                    time_str = line.replace('END_TIME:', '').strip()
                    schedule['end_time'] = datetime.strptime(
                        time_str, 
                        '%Y-%m-%d %H:%M'
                    ).replace(tzinfo=self.timezone)
                    
                elif line.startswith('REASON:'):
                    schedule['reason'] = line.replace('REASON:', '').strip()
            
            # Validate schedule
            if all(k in schedule for k in ['start_time', 'end_time', 'reason']):
                return schedule
            
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {str(e)}")
        
        return None

# Test the scheduler
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    calendar = CalendarManager()
    rules = RulesManager(calendar)
    scheduler = SmartScheduler(calendar, rules)
    
    # Test task
    test_task = {
        "name": "Write Project Proposal",
        "duration": "2 hours",
        "deadline": "tomorrow 5pm",
        "priority": "high"
    }
    
    # Try scheduling
    result = scheduler.schedule_task(test_task)
    
    if result:
        print(f"\nTask scheduled successfully!")
        print(f"Event: {result['summary']}")
        print(f"Time: {result['start']['dateTime']} to {result['end']['dateTime']}")
    else:
        print("\nFailed to schedule task")