# src/calendar_manager.py

from googleapiclient.errors import HttpError
from src.auth import GoogleCalendarAuth
import json
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class CalendarManager:
    def __init__(self):
        self.auth = GoogleCalendarAuth()
        self.service = self.auth.get_calendar_service()
        self.calendar_id = None
        self.cache_file = Path(__file__).parent.parent / 'cache' / 'calendar_cache.json'
        self.CALENDAR_NAME = "Smart Tasks"
        self.TIMEZONE = 'Asia/Kolkata'
        
        # Initialize calendar
        self.initialize_calendar()

    def initialize_calendar(self):
        """Initialize calendar - either load existing one or create new"""
        try:
            # Try to load cached calendar ID
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r') as f:
                        data = json.load(f)
                        self.calendar_id = data.get('calendar_id')
                except json.JSONDecodeError:
                    print("Cache file corrupted, will create new calendar")
                    self.calendar_id = None

            # Verify calendar still exists and is accessible
            if self.calendar_id:
                try:
                    self.service.calendars().get(calendarId=self.calendar_id).execute()
                    print(f"Successfully loaded existing calendar: {self.CALENDAR_NAME}")
                    return
                except HttpError:
                    print("Cached calendar not found, creating new one...")
                    self.calendar_id = None

            # Create new calendar if needed
            self.create_new_calendar()
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def create_new_calendar(self):
        """Create a new calendar for smart tasks"""
        calendar_body = {
            'summary': self.CALENDAR_NAME,
            'description': 'Calendar for automatically scheduled tasks',
            'timeZone': self.TIMEZONE
        }

        try:
            # Check if calendar with same name already exists
            calendar_list = self.service.calendarList().list().execute()
            for calendar in calendar_list['items']:
                if calendar['summary'] == self.CALENDAR_NAME:
                    self.calendar_id = calendar['id']
                    print(f"Found existing calendar: {self.CALENDAR_NAME}")
                    break
            
            # Create new calendar if not found
            if not self.calendar_id:
                created_calendar = self.service.calendars().insert(body=calendar_body).execute()
                self.calendar_id = created_calendar['id']
                print(f"Created new calendar: {self.CALENDAR_NAME}")

            # Cache the calendar ID
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({'calendar_id': self.calendar_id}, f)
                
        except HttpError as error:
            print(f"Error creating calendar: {error}")
            raise

    def add_event(self, summary, start_time, end_time, description=""):
        """Add an event to the calendar"""
        event_body = {
            'summary': summary,
            'description': description,
            'start': {
                'dateTime': start_time.isoformat(),
                'timeZone': self.TIMEZONE,
            },
            'end': {
                'dateTime': end_time.isoformat(),
                'timeZone': self.TIMEZONE,
            },
        }

        try:
            event = self.service.events().insert(
                calendarId=self.calendar_id,
                body=event_body
            ).execute()
            print(f"Added event: {summary}")
            return event
        except HttpError as error:
            print(f"Error adding event: {error}")
            return None

    def list_events(self, start_date=None, end_date=None, max_results=10):
        """List events in the calendar"""
        try:
            # If no dates provided, use next 7 days
            if not start_date:
                start_date = datetime.now(ZoneInfo(self.TIMEZONE))
            if not end_date:
                end_date = start_date + timedelta(days=7)

            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_date.isoformat(),
                timeMax=end_date.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            return events

        except HttpError as error:
            print(f"Error listing events: {error}")
            return []

    def update_event(self, event_id, summary=None, start_time=None, end_time=None, description=None):
        """Update an existing event"""
        try:
            # Get current event
            event = self.service.events().get(
                calendarId=self.calendar_id,
                eventId=event_id
            ).execute()

            # Update only provided fields
            if summary:
                event['summary'] = summary
            if description:
                event['description'] = description
            if start_time:
                event['start']['dateTime'] = start_time.isoformat()
            if end_time:
                event['end']['dateTime'] = end_time.isoformat()

            updated_event = self.service.events().update(
                calendarId=self.calendar_id,
                eventId=event_id,
                body=event
            ).execute()

            print(f"Updated event: {updated_event['summary']}")
            return updated_event

        except HttpError as error:
            print(f"Error updating event: {error}")
            return None

    def delete_event(self, event_id):
        """Delete an event"""
        try:
            self.service.events().delete(
                calendarId=self.calendar_id,
                eventId=event_id
            ).execute()
            print(f"Deleted event: {event_id}")
            return True
        except HttpError as error:
            print(f"Error deleting event: {error}")
            return False

    def find_free_slots(self, duration_minutes, start_date=None, end_date=None):
        """Find available time slots of specified duration"""
        try:
            if not start_date:
                start_date = datetime.now(ZoneInfo(self.TIMEZONE))
            if not end_date:
                end_date = start_date + timedelta(days=7)

            body = {
                "timeMin": start_date.isoformat(),
                "timeMax": end_date.isoformat(),
                "timeZone": self.TIMEZONE,
                "items": [{"id": self.calendar_id}]
            }

            free_busy = self.service.freebusy().query(body=body).execute()
            busy_slots = free_busy['calendars'][self.calendar_id]['busy']

            busy_periods = [
                (
                    datetime.fromisoformat(slot['start'].replace('Z', '+00:00')),
                    datetime.fromisoformat(slot['end'].replace('Z', '+00:00'))
                )
                for slot in busy_slots
            ]

            free_slots = []
            current_time = start_date

            for busy_start, busy_end in busy_periods + [(end_date, end_date)]:
                if (busy_start - current_time).total_seconds() / 60 >= duration_minutes:
                    free_slots.append((current_time, busy_start))
                current_time = busy_end

            return free_slots

        except HttpError as error:
            print(f"Error finding free slots: {error}")
            return []

if __name__ == "__main__":
    calendar = CalendarManager()
    print(f"\nCalendar ID: {calendar.calendar_id}")