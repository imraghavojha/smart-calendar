import os
from groq import Groq
from datetime import datetime, timedelta
import pytz
from typing import Dict, Optional
import logging
from src.calendar_manager import CalendarManager
from src.rules_parser import RulesManager

class SmartScheduler:
    def __init__(self, calendar_manager: CalendarManager, rules_manager: RulesManager):
        self.calendar_manager = calendar_manager
        self.rules_manager = rules_manager
        self.groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.logger = logging.getLogger(__name__)
        self.timezone = pytz.timezone('Asia/Kolkata')  # Adjust to your timezone

   # In the schedule_task method, add logging:
    def schedule_task(self, task: Dict) -> Optional[Dict]:
        """Schedule a task using Groq's AI"""
        try:
            # Get current calendar context
            context = self._get_calendar_context()
            
            # Create prompt for Groq
            prompt = self._create_scheduling_prompt(task, context)
            print("\nSending prompt to Groq:")
            print(prompt)
            
            # Get AI scheduling suggestion
            completion = self.groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are an intelligent calendar assistant that helps schedule tasks optimally."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Log the response
            print("\nGroq Response:")
            print(completion.choices[0].message.content)
            
            # Parse the AI response
            schedule = self._parse_ai_response(completion.choices[0].message.content)
            
            if schedule:
                print("\nParsed Schedule:")
                print(f"Start Time: {schedule['start_time']}")
                print(f"End Time: {schedule['end_time']}")
                print(f"Reason: {schedule['reason']}")
                
                # Create calendar event
                event = self.calendar_manager.add_event(
                    summary=task['name'],
                    start_time=schedule['start_time'],
                    end_time=schedule['end_time'],
                    description=f"Auto-scheduled task\nDuration: {task['duration']}\n"
                              f"Deadline: {task['deadline']}\nReason: {schedule['reason']}"
                )
                return event
            else:
                print("\nFailed to parse schedule from response")
                
        except Exception as e:
            self.logger.error(f"Error scheduling task: {str(e)}", exc_info=True)
            return None

    def _parse_ai_response(self, response: str) -> Optional[Dict]:
        """Parse Groq's response into schedule details"""
        try:
            print("\nParsing response:")
            lines = response.strip().split('\n')
            schedule = {}
            
            for line in lines:
                print(f"Processing line: {line}")
                if line.startswith('START_TIME:'):
                    time_str = line.replace('START_TIME:', '').strip()
                    print(f"Found start time: {time_str}")
                    schedule['start_time'] = datetime.strptime(
                        time_str, 
                        '%Y-%m-%d %H:%M'
                    ).replace(tzinfo=self.timezone)
                    
                elif line.startswith('END_TIME:'):
                    time_str = line.replace('END_TIME:', '').strip()
                    print(f"Found end time: {time_str}")
                    schedule['end_time'] = datetime.strptime(
                        time_str, 
                        '%Y-%m-%d %H:%M'
                    ).replace(tzinfo=self.timezone)
                    
                elif line.startswith('REASON:'):
                    schedule['reason'] = line.replace('REASON:', '').strip()
                    print(f"Found reason: {schedule['reason']}")
            
            if all(k in schedule for k in ['start_time', 'end_time', 'reason']):
                return schedule
            else:
                print("Missing required fields in schedule")
                print(f"Found fields: {list(schedule.keys())}")
                
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {str(e)}", exc_info=True)
            return None

    def _get_calendar_context(self) -> Dict:
        """Get current calendar state and constraints"""
        # Get next week's events
        now = datetime.now(self.timezone)
        week_later = now + timedelta(days=7)
        
        upcoming_events = self.calendar_manager.list_events(
            start_date=now,
            end_date=week_later
        )
        
        # Get scheduling rules and constraints
        constraints = self.rules_manager.get_scheduling_constraints()
        
        return {
            "events": upcoming_events,
            "fixed_events": constraints.get('fixed_events', []),
            "blocked_times": constraints.get('blocked_times', []),
            "preferences": constraints.get('preferences', [])
        }

    def _create_scheduling_prompt(self, task: Dict, context: Dict) -> str:
        """Create detailed prompt for Groq"""
        prompt = f"""Help schedule this task optimally:

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

        prompt += "\n\nPreferences:"
        for pref in context['preferences']:
            prompt += f"\n- {pref}"

        prompt += """\n\nYou must provide the optimal time slot in EXACTLY this format (including the labels):
START_TIME: YYYY-MM-DD HH:MM
END_TIME: YYYY-MM-DD HH:MM
REASON: Brief explanation

For example:
START_TIME: 2024-11-06 09:00
END_TIME: 2024-11-06 11:00
REASON: Early morning slot when energy is high and no conflicts

Do not include any other text in your response."""
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
            
            if all(k in schedule for k in ['start_time', 'end_time', 'reason']):
                return schedule
                
        except Exception as e:
            self.logger.error(f"Error parsing AI response: {str(e)}")
            return None