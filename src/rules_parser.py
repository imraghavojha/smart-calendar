# src/rules_parser.py

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import re
from pathlib import Path
from src.calendar_manager import CalendarManager

class RulesManager:
    def __init__(self, calendar_manager):
        self.calendar_manager = calendar_manager
        self.timezone = ZoneInfo('Asia/Kolkata')
        self.rules_file = Path(__file__).parent.parent / 'config' / 'rules.txt'
        
        # Different types of rules
        self.fixed_events = []     # Regular events like classes
        self.blocked_times = []    # Times when no tasks should be scheduled
        self.preferences = []      # Preferred times for certain activities

    def get_scheduling_constraints(self):
        """Return all scheduling constraints for the LLM"""
        # First load/reload rules
        self.load_rules()
        
        # Return constraints dictionary
        return {
            'fixed_events': [
                {
                    'summary': event['summary'],
                    'days': event['days'],
                    'start_time': event['start_time'].strftime('%H:%M'),
                    'end_time': event['end_time'].strftime('%H:%M'),
                    'type': event['type']
                }
                for event in self.fixed_events
            ],
            'blocked_times': [
                {
                    'summary': block['summary'],
                    'days': block['days'],
                    'start_time': block['start_time'].strftime('%H:%M'),
                    'end_time': block['end_time'].strftime('%H:%M'),
                    'type': block['type']
                }
                for block in self.blocked_times
            ],
            'preferences': self.preferences
        }

    def load_rules(self):
        """Load and parse rules from rules.txt"""
        # Clear existing rules before loading
        self.fixed_events = []
        self.blocked_times = []
        self.preferences = []
        
        if not self.rules_file.exists():
            print(f"Creating new rules file at {self.rules_file}")
            self.rules_file.parent.mkdir(exist_ok=True)
            self.rules_file.write_text("# Add your schedule rules here\n")
            return

        print("Loading rules...")
        try:
            rules = self.rules_file.read_text().splitlines()
            
            for rule in rules:
                rule = rule.strip()
                if not rule or rule.startswith('#'):
                    continue
                    
                self.parse_rule(rule)
                
        except Exception as e:
            print(f"Error loading rules file: {e}")
            return

    def parse_rule(self, rule):
        """Parse a single rule and categorize it"""
        try:
            print(f"\nParsing rule: {rule}")
            
            # Split rule into components
            parts = rule.split('|')
            
            if len(parts) >= 3:
                rule_type = parts[0].upper()
                
                if rule_type.startswith('CLASS:'):
                    self._handle_class_schedule(rule)
                elif rule_type == 'SLEEP':
                    self._handle_sleep_schedule(rule)
                elif rule_type == 'BLOCK':
                    self._handle_blocked_time(rule)
                elif rule_type.startswith('PREFER:'):
                    self._handle_preference(rule)
                else:
                    print(f"Unknown rule type: {rule_type}")
            else:
                print(f"Invalid rule format: {rule}")
                
        except Exception as e:
            print(f"Error parsing rule '{rule}': {e}")

    def _parse_time(self, time_str):
        """Parse time string into time object"""
        time_str = time_str.lower().strip()
        
        time_patterns = [
            (r'(\d+)(?::(\d+))?\s*(am|pm)', lambda m: time(
                hour=int(m.group(1)) % 12 + (12 if m.group(3) == 'pm' else 0),
                minute=int(m.group(2)) if m.group(2) else 0
            )),
            (r'(\d+)(?::(\d+))?', lambda m: time(
                hour=int(m.group(1)),
                minute=int(m.group(2)) if m.group(2) else 0
            ))
        ]
        
        for pattern, time_builder in time_patterns:
            match = re.search(pattern, time_str)
            if match:
                return time_builder(match)
        
        raise ValueError(f"Couldn't parse time: {time_str}")

    def _parse_days(self, days_str):
        """Parse day strings into day numbers"""
        days_map = {
            'm': 0, 'mon': 0, 'monday': 0,
            't': 1, 'tue': 1, 'tuesday': 1,
            'w': 2, 'wed': 2, 'wednesday': 2,
            'th': 3, 'thu': 3, 'thursday': 3,
            'f': 4, 'fri': 4, 'friday': 4,
            'sa': 5, 'sat': 5, 'saturday': 5,
            'su': 6, 'sun': 6, 'sunday': 6
        }
        
        days_str = days_str.lower()
        
        # Special cases
        if 'weekend' in days_str:
            return [5, 6]
        if 'weekday' in days_str:
            return [0, 1, 2, 3, 4]
        if 'daily' in days_str:
            return [0, 1, 2, 3, 4, 5, 6]
        if 'mw' in days_str:
            return [0, 2]
        if 'tth' in days_str:
            return [1, 3]
        if 'tue' in days_str:
            return [1]
        if 'sunday' in days_str or 'sun' in days_str:
            return [6]
        if 'saturday' in days_str or 'sat' in days_str:
            return [5]
        
        # Handle individual days
        days = []
        for day in re.findall(r'[mtwthfsa]{1,2}|monday|tuesday|wednesday|thursday|friday|saturday|sunday', days_str):
            if day.lower() in days_map:
                days.append(days_map[day.lower()])
                
        return sorted(list(set(days)))

    def _handle_class_schedule(self, rule):
        """Handle class schedule rules"""
        try:
            parts = rule.split('|')
            if len(parts) >= 3:
                class_name = parts[0].replace('CLASS:', '')
                days_str = parts[1]
                time_str = parts[2]
                
                # Parse days
                days = self._parse_days(days_str)
                
                # Parse times
                times = time_str.split('-')
                if len(times) == 2:
                    start_time = self._parse_time(times[0])
                    end_time = self._parse_time(times[1])
                    
                    self.fixed_events.append({
                        'summary': class_name,
                        'days': days,
                        'start_time': start_time,
                        'end_time': end_time,
                        'type': 'class'
                    })
                    print(f"Added class schedule: {class_name} on days {days} from {start_time} to {end_time}")
                else:
                    print(f"Invalid time format in class rule: {time_str}")
        except Exception as e:
            print(f"Error handling class schedule: {e}")

    def _handle_sleep_schedule(self, rule):
        """Handle sleep schedule rules"""
        try:
            parts = rule.split('|')
            if len(parts) >= 3:
                time_str = parts[2]
                times = time_str.split('-')
                if len(times) == 2:
                    start_time = self._parse_time(times[0])
                    end_time = self._parse_time(times[1])
                    
                    self.blocked_times.append({
                        'summary': 'Sleep Time',
                        'days': [0, 1, 2, 3, 4, 5, 6],
                        'start_time': start_time,
                        'end_time': end_time,
                        'type': 'sleep'
                    })
                    print(f"Added sleep schedule: {start_time} to {end_time}")
                else:
                    print(f"Invalid time format in sleep rule: {time_str}")
        except Exception as e:
            print(f"Error handling sleep schedule: {e}")

    def _handle_blocked_time(self, rule):
        """Handle blocked time rules"""
        try:
            parts = rule.split('|')
            if len(parts) >= 3:
                days_str = parts[1]
                time_str = parts[2]
                
                # Parse days
                days = self._parse_days(days_str)
                
                # Parse times
                times = time_str.split('-')
                if len(times) == 2:
                    start_time = self._parse_time(times[0])
                    end_time = self._parse_time(times[1])
                    
                    self.blocked_times.append({
                        'summary': 'Blocked Time',
                        'days': days,
                        'start_time': start_time,
                        'end_time': end_time,
                        'type': 'blocked'
                    })
                    print(f"Added blocked time: days {days} from {start_time} to {end_time}")
                else:
                    print(f"Invalid time format in blocked time rule: {time_str}")
        except Exception as e:
            print(f"Error handling blocked time: {e}")

    def _handle_preference(self, rule):
        """Handle preference rules"""
        self.preferences.append(rule)
        print(f"Added preference: {rule}")

    def add_fixed_events_to_calendar(self):
        """Add all fixed events to the calendar"""
        start_date = datetime.now(self.timezone)
        end_date = start_date + timedelta(weeks=1)
        
        print("\nAdding fixed events to calendar:")
        for event in self.fixed_events + self.blocked_times:
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() in event['days']:
                    event_start = datetime.combine(
                        current_date.date(),
                        event['start_time']
                    ).replace(tzinfo=self.timezone)
                    
                    event_end = datetime.combine(
                        current_date.date(),
                        event['end_time']
                    ).replace(tzinfo=self.timezone)
                    
                    # Only add if end time is after start time
                    if event_end > event_start:
                        print(f"Adding {event['summary']} on {current_date.strftime('%A')}")
                        self.calendar_manager.add_event(
                            summary=event['summary'],
                            start_time=event_start,
                            end_time=event_end,
                            description=f"Fixed event: {event['type']}"
                        )
                
                current_date += timedelta(days=1)

if __name__ == "__main__":
    # This section is for testing the rules parser directly
    calendar = CalendarManager()
    rules = RulesManager(calendar)
    rules.load_rules()