# src/test_calendar.py

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.calendar_manager import CalendarManager

def run_tests():
    print("\n=== Smart Calendar Testing ===\n")
    calendar = CalendarManager()
    
    # 1. Test adding events
    print("1. Adding test events...")
    now = datetime.now(ZoneInfo('Asia/Kolkata'))
    
    # Add first event (starts in 1 hour)
    event1_start = now + timedelta(hours=1)
    event1_end = event1_start + timedelta(hours=1)
    event1 = calendar.add_event(
        summary="Meeting with Team",
        start_time=event1_start,
        end_time=event1_end,
        description="Discuss project progress"
    )
    
    # Add second event (starts in 3 hours)
    event2_start = now + timedelta(hours=3)
    event2_end = event2_start + timedelta(hours=2)
    event2 = calendar.add_event(
        summary="Work on Project",
        start_time=event2_start,
        end_time=event2_end,
        description="Focus time for coding"
    )
    
    # 2. List all events
    print("\n2. Listing today's events:")
    events = calendar.list_events(
        start_date=now,
        end_date=now + timedelta(days=1)
    )
    
    for event in events:
        start = datetime.fromisoformat(event['start']['dateTime'])
        end = datetime.fromisoformat(event['end']['dateTime'])
        print(f"- {event['summary']}")
        print(f"  From: {start.strftime('%I:%M %p')}")
        print(f"  To: {end.strftime('%I:%M %p')}")
        print(f"  Description: {event.get('description', 'No description')}")
        print()
    
    # 3. Find free slots
    print("3. Finding available 30-minute slots:")
    free_slots = calendar.find_free_slots(
        duration_minutes=30,
        start_date=now,
        end_date=now + timedelta(days=1)
    )
    
    print("\nFirst 5 available slots:")
    for start, end in free_slots[:5]:
        print(f"- {start.strftime('%I:%M %p')} to {end.strftime('%I:%M %p')}")
    
    # 4. Update an event
    if event1:
        print("\n4. Updating first event...")
        updated_event = calendar.update_event(
            event_id=event1['id'],
            summary="Updated: Team Meeting",
            description="Updated description"
        )
    
    # 5. List events again to see update
    print("\n5. Listing events after update:")
    events = calendar.list_events(start_date=now, end_date=now + timedelta(days=1))
    for event in events:
        print(f"- {event['summary']}")
    
    # 6. Clean up (delete test events)
    print("\n6. Cleaning up test events...")
    if event1:
        calendar.delete_event(event1['id'])
    if event2:
        calendar.delete_event(event2['id'])
    
    print("\nTest completed!")

if __name__ == "__main__":
    run_tests()