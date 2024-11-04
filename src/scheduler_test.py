# Fix imports to use src prefix
from src.calendar_manager import CalendarManager
from src.rules_parser import RulesManager
from src.smart_scheduler import SmartScheduler
import logging

def test_scheduling():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        print("\nTesting Smart Scheduler...")
        
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
        
        print("\nTrying to schedule task:")
        print(f"Name: {test_task['name']}")
        print(f"Duration: {test_task['duration']}")
        print(f"Deadline: {test_task['deadline']}")
        print(f"Priority: {test_task['priority']}")
        
        # Schedule task
        result = scheduler.schedule_task(test_task)
        
        if result:
            print("\nTask scheduled successfully!")
            print(f"Event: {result.get('summary', 'No summary')}")
            print(f"Start: {result.get('start', {}).get('dateTime', 'No start time')}")
            print(f"End: {result.get('end', {}).get('dateTime', 'No end time')}")
            print(f"Description: {result.get('description', 'No description')}")
        else:
            print("\nFailed to schedule task")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_scheduling()