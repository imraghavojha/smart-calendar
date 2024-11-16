# src/scheduler_test.py

import logging
from src.calendar_manager import CalendarManager
from src.rules_parser import RulesManager
from src.smart_scheduler import SmartScheduler
from datetime import datetime, timedelta
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scheduling():
    try:
        # Initialize components
        calendar = CalendarManager()
        rules = RulesManager(calendar)
        scheduler = SmartScheduler(calendar, rules)
        
        # Calculate timestamps
        tz = pytz.timezone('Asia/Kolkata')
        current_time = datetime.now(tz)
        today_noon = current_time.replace(hour=12, minute=0, second=0, microsecond=0)
        
        # Edge Case Test Tasks
        test_cases = [
    # Edge Case 7: Tight Schedule with Specific Time
    {
        'name': 'Client Meeting',
        'duration': '1h',
        'deadline': f"{current_time.strftime('%Y-%m-%d')} 14:00",  # Must happen before 2 PM
        'type': 'meeting',
        'priority': 'critical',
        'preferred_time': '11:30'  # Should try to schedule at this specific time
    },
    
    # Edge Case 8: Very Short Notice Task
    {
        'name': 'Emergency Team Sync',
        'duration': '15m',
        'deadline': current_time.strftime('%Y-%m-%d %H:%M'),  # Today's date and current time
        'type': 'meeting',
        'priority': 'urgent'
    }
]
        # Run each test case
        for i, task in enumerate(test_cases, 1):
            print(f"\n{'='*50}")
            print(f"Running Edge Case {i}:")
            print(f"Task: {task['name']}")
            print(f"Type: {task['type']}")
            print(f"Duration: {task['duration']}")
            print(f"Deadline: {task['deadline']}")
            print(f"Priority: {task['priority']}")
            print(f"{'='*50}\n")

            result = scheduler.schedule_task(task)
            
            if result:
                print(f"\nEdge Case {i} - Success!")
                print(f"Event: {result['summary']}")
                print(f"Scheduled for: {result['start'].get('dateTime')} to {result['end'].get('dateTime')}")
                print(f"Description: {result.get('description', '')}")
            else:
                print(f"\nEdge Case {i} - Failed to schedule task")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    print("Testing Smart Scheduler Edge Cases...")
    test_scheduling()