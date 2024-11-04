import os
from groq import Groq
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
from pathlib import Path

class SmartScheduler:
    def __init__(self, calendar_manager, rules_manager):
        self.calendar_manager = calendar_manager
        self.rules_manager = rules_manager
        self.groq_client = Groq(api_key=os.getenv('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'))
        self.logger = logging.getLogger(__name__)

    def schedule_task(self, task_name: str, duration: str, deadline: str) -> Optional[Dict]:
        """
        Schedule a task using Groq LLM
        
        Args:
            task_name: Name of the task
            duration: Duration in format "X hours" or "X minutes"
            deadline: Deadline in natural language format
        """
        try:
            # Get current calendar state and constraints
            calendar_context = self._get_calendar_context()
            
            # Create prompt for Groq
            prompt = self._create_scheduling_prompt(
                task_name=task_name,
                duration=duration,
                deadline=deadline,
                context=calendar_context
            )
            
            # Get scheduling decision from Groq
            completion = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",  # Using Mixtral model
                messages=[
                    {"role": "system", "content": "You are an intelligent calendar assistant that helps schedule tasks optimally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent outputs
                max_tokens=1000
            )
            
            # Parse the response
            schedule = self._parse_llm_response(completion.choices[0].message.content)
            
            if schedule:
                # Create calendar event
                event = self.calendar_manager.add_event(
                    summary=task_name,
                    start_time=schedule['start_time'],
                    end_time=schedule['end_time'],
                    description=f"Auto-scheduled task\nDuration: {duration}\nDeadline: {deadline}\nReasoning: {schedule['reasoning']}"
                )
                return event
                
        except Exception as e:
            self.logger.error(f"Error scheduling task: {str(e)}")
            return None

    def _get_calendar_context(self) -> Dict:
        """Get current calendar context and constraints"""
        # Get fixed events and rules
        constraints = self.rules_manager.get_scheduling_constraints()
        
        # Get upcoming events
        start_date = datetime.now()
        end_date = start_date + timedelta(days=7)  # Look ahead 7 days
        events = self.calendar_manager.list_events(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "fixed_events": constraints['fixed_events'],
            "blocked_times": constraints['blocked_times'],
            "preferences": constraints['preferences'],
            "existing_events": events
        }

    def _create_scheduling_prompt(self, task_name: str, duration: str, deadline: str, context: Dict) -> str:
        """Create prompt for Groq LLM"""
        prompt = f"""Please help schedule this task based on the following information:

Task Details:
- Name: {task_name}
- Duration: {duration}
- Deadline: {deadline}

Current Schedule and Constraints:

Fixed Events:"""
        
        for event in context['fixed_events']:
            prompt += f"\n- {event['summary']} on {event['days']} at {event['start_time']}-{event['end_time']}"

        prompt += "\n\nBlocked Times:"
        for block in context['blocked_times']:
            prompt += f"\n- {block['summary']} on {block['days']} at {block['start_time']}-{block['end_time']}"

        prompt += "\n\nPreferences:"
        for pref in context['preferences']:
            prompt += f"\n- {pref}"

        prompt += """

Please suggest the best time slot for this task. Consider:
1. Task deadline and duration
2. Fixed events and blocked times
3. User preferences
4. Buffer time between tasks
5. Energy levels and optimal timing

Provide your response in this exact format:
START_TIME: YYYY-MM-DD HH:MM
END_TIME: YYYY-MM-DD HH:MM
REASONING: Brief explanation of why this time slot was chosen"""

        return prompt

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse Groq's response into schedule details"""
        try:
            lines = response.strip().split('\n')
            schedule = {}
            
            for line in lines:
                if line.startswith('START_TIME:'):
                    start_str = line.replace('START_TIME:', '').strip()
                    schedule['start_time'] = datetime.strptime(start_str, '%Y-%m-%d %H:%M')
                elif line.startswith('END_TIME:'):
                    end_str = line.replace('END_TIME:', '').strip()
                    schedule['end_time'] = datetime.strptime(end_str, '%Y-%m-%d %H:%M')
                elif line.startswith('REASONING:'):
                    schedule['reasoning'] = line.replace('REASONING:', '').strip()
            
            if all(k in schedule for k in ['start_time', 'end_time', 'reasoning']):
                return schedule
                
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            
        return None

# Example usage
if __name__ == "__main__":
    from calendar_manager import CalendarManager
    from rules_parser import RulesManager
    
    # Initialize managers
    calendar = CalendarManager()
    rules = RulesManager(calendar)
    
    # Initialize scheduler
    scheduler = SmartScheduler(calendar, rules)
    
    # Test scheduling
    result = scheduler.schedule_task(
        task_name="Write Essay",
        duration="3 hours",
        deadline="Monday 5pm"
    )
    
    if result:
        print(f"Task scheduled successfully: {result}")
    else:
        print("Failed to schedule task")