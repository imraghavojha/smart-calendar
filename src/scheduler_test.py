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
        
        # Test Cases
        test_cases = [
            # Test Case 1: Study task in morning
            {
                'name': 'Study for Database Exam',
                'duration': '2h',
                'deadline': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                'type': 'study',
                'priority': 'high'
            },
            
            # Test Case 2: Short coding task
            {
                'name': 'Debug Frontend Issue',
                'duration': '45m',
                'deadline': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'type': 'coding',
                'priority': 'medium'
            },
            
            # Test Case 3: Task with tight deadline
            {
                'name': 'Prepare Project Presentation',
                'duration': '1h30m',
                'deadline': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d') + ' 17:00',
                'type': 'presentation',
                'priority': 'high'
            },
            
            # Test Case 4: Long study session
            {
                'name': 'Research Paper Reading',
                'duration': '3h',
                'deadline': (datetime.now() + timedelta(days=4)).strftime('%Y-%m-%d'),
                'type': 'study',
                'priority': 'medium'
            }
        ]

        # Run each test case
        for i, task in enumerate(test_cases, 1):
            print(f"\n{'='*50}")
            print(f"Running Test Case {i}:")
            print(f"Task: {task['name']}")
            print(f"Type: {task['type']}")
            print(f"Duration: {task['duration']}")
            print(f"Deadline: {task['deadline']}")
            print(f"Priority: {task['priority']}")
            print(f"{'='*50}\n")

            result = scheduler.schedule_task(task)
            
            if result:
                print(f"\nTest Case {i} - Success!")
                print(f"Event: {result['summary']}")
                print(f"Scheduled for: {result['start'].get('dateTime')} to {result['end'].get('dateTime')}")
                print(f"Description: {result.get('description', '')}")
            else:
                print(f"\nTest Case {i} - Failed to schedule task")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    print("Testing Smart Scheduler...")
    test_scheduling()