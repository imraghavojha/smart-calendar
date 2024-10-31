# src/rules_parser.py

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import re
from pathlib import Path
from src.calendar_manager import CalendarManager

class RulesManager:
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
    def __init__(self, calendar_manager):
        self.calendar_manager = calendar_manager
        self.timezone = ZoneInfo('Asia/Kolkata')
        self.rules_file = Path(__file__).parent.parent / 'config' / 'rules.txt'
        
        # Different types of rules
        self.fixed_events = []     # Regular events like classes
        self.blocked_times = []    # Times when no tasks should be scheduled
        self.preferences = []      # Preferred times for certain activities

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
            rule = rule.lower()
            print(f"\nParsing rule: {rule}")
            
            if self._is_class_schedule(rule):
                self._handle_class_schedule(rule)
            elif self._is_sleep_schedule(rule):
                self._handle_sleep_schedule(rule)
            elif self._is_preference(rule):
                self._handle_preference(rule)
            elif self._is_blocked_day(rule):
                self._handle_blocked_day(rule)
            else:
                print(f"Unknown rule format: {rule}")
                
        except Exception as e:
            print(f"Error parsing rule '{rule}': {e}")

    def _is_class_schedule(self, rule):
        class_patterns = [r'class', r'lecture', r'lab', r'seminar']
        return any(pattern in rule for pattern in class_patterns)

    def _is_sleep_schedule(self, rule):
        return 'sleep' in rule

    def _is_preference(self, rule):
        preference_patterns = [r'prefer', r'like to', r'better in']
        return any(pattern in rule for pattern in preference_patterns)

    def _is_blocked_day(self, rule):
        return 'no tasks' in rule or 'don\'t schedule' in rule

    def _parse_time(self, time_str):
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
        
        # Special cases first
        if 'weekend' in days_str:
            return [5, 6]
        if 'weekday' in days_str:
            return [0, 1, 2, 3, 4]
        if 'mw' in days_str:
            return [0, 2]
        if 'tth' in days_str:
            return [1, 3]
        if 'tuesday' in days_str or (days_str.startswith('t ') or days_str == 't'):
            return [1]
        if 'sunday' in days_str:
            return [6]
        if 'saturday' in days_str:
            return [5]
        
        # Handle individual days
        days = []
        for day in re.findall(r'[mtwthfsa]{1,2}|monday|tuesday|wednesday|thursday|friday|saturday|sunday', days_str):
            if day in days_map:
                days.append(days_map[day])
                
        return sorted(list(set(days)))

    def _handle_class_schedule(self, rule):
        try:
            times = re.findall(r'\d+(?::\d+)?\s*(?:am|pm)?', rule)
            if len(times) >= 2:
                start_time = self._parse_time(times[0])
                end_time = self._parse_time(times[1])
                
                # Extract days
                if 'mw' in rule:
                    days = [0, 2]
                elif 'tth' in rule:
                    days = [1, 3]
                else:
                    days_match = re.search(r'(?:on\s+)?([mtwthfsa]+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)', rule)
                    if days_match:
                        days = self._parse_days(days_match.group(1))
                    else:
                        days = []
                
                if days:
                    class_name = re.search(r'(\w+)(?:\s+class|\s+lab|\s+lecture|\s+seminar)', rule)
                    summary = f"{class_name.group(1).title() if class_name else 'Class'}"
                    
                    self.fixed_events.append({
                        'summary': summary,
                        'days': days,
                        'start_time': start_time,
                        'end_time': end_time,
                        'type': 'class'
                    })
                    print(f"Added class schedule: {summary} on days {days} from {start_time} to {end_time}")
                else:
                    print(f"Could not determine days for rule: {rule}")
        
        except Exception as e:
            print(f"Error parsing class schedule: {e}")

    def _handle_sleep_schedule(self, rule):
        try:
            times = re.findall(r'\d+(?::\d+)?\s*(?:am|pm)?', rule)
            if len(times) >= 2:
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
        except Exception as e:
            print(f"Error parsing sleep schedule: {e}")

    def _handle_preference(self, rule):
        self.preferences.append(rule)
        print(f"Added preference: {rule}")

    def _handle_blocked_day(self, rule):
        rule = rule.lower()
        days = []
        
        if 'sunday' in rule:
            days = [6]
        elif 'saturday' in rule:
            days = [5]
        else:
            days_match = re.search(r'(?:on\s+)?([mtwthfsa]+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)', rule)
            if days_match:
                days = self._parse_days(days_match.group(1))
        
        # Check for specific time
        times = re.findall(r'\d+(?::\d+)?\s*(?:am|pm)?', rule)
        if times and 'after' in rule:
            start_time = self._parse_time(times[0])
            end_time = time(23, 59)
        else:
            start_time = time(0, 0)
            end_time = time(23, 59)
        
        if days:
            self.blocked_times.append({
                'summary': 'Blocked Time',
                'days': days,
                'start_time': start_time,
                'end_time': end_time,
                'type': 'blocked_day'
            })
            print(f"Added blocked time: days {days} from {start_time} to {end_time}")
        else:
            print(f"Could not determine days for blocked time rule: {rule}")

    def add_fixed_events_to_calendar(self):
        """Add all fixed events to the calendar"""
        start_date = datetime.now(self.timezone)
        end_date = start_date + timedelta(weeks=1)  # Reduced to 1 week instead of 4
        
        print("\nAdding fixed events to calendar:")
        for event in self.fixed_events + self.blocked_times:
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() in event['days']:
                    # Handle overnight events (like sleep schedule)
                    if event['type'] == 'sleep' and event['start_time'] > event['end_time']:
                        # Create two events: one for evening and one for morning
                        # Evening part (e.g., 11pm to midnight)
                        evening_start = datetime.combine(
                            current_date.date(),
                            event['start_time']
                        ).replace(tzinfo=self.timezone)
                        
                        evening_end = datetime.combine(
                            current_date.date(),
                            time(23, 59, 59)
                        ).replace(tzinfo=self.timezone)
                        
                        # Morning part (e.g., midnight to 7am)
                        morning_start = datetime.combine(
                            current_date.date() + timedelta(days=1),
                            time(0, 0)
                        ).replace(tzinfo=self.timezone)
                        
                        morning_end = datetime.combine(
                            current_date.date() + timedelta(days=1),
                            event['end_time']
                        ).replace(tzinfo=self.timezone)
                        
                        print(f"Adding {event['summary']} (overnight) on {current_date.strftime('%A')}")
                        
                        self.calendar_manager.add_event(
                            summary=event['summary'] + " (Part 1)",
                            start_time=evening_start,
                            end_time=evening_end,
                            description=f"Fixed event: {event['type']}"
                        )
                        
                        self.calendar_manager.add_event(
                            summary=event['summary'] + " (Part 2)",
                            start_time=morning_start,
                            end_time=morning_end,
                            description=f"Fixed event: {event['type']}"
                        )
                        
                    else:
                        # Normal event handling
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

def create_test_rules():
    return """# Class Schedule
Calculus class MW 4pm-9pm
Physics lab on Tuesday 2pm-5pm
Database class TTh 10am-11:30am

# Sleep Schedule
I sleep between 11pm-7am

# Preferences
I prefer to study in mornings
I prefer coding tasks in afternoon

# Blocked Times
No tasks on Sunday
Don't schedule anything on Saturday after 2pm"""

def run_tests():
    print("\n=== Testing Rules System ===\n")
    
    try:
        # Initialize calendar manager
        print("1. Initializing Calendar Manager...")
        calendar = CalendarManager()
        
        # Initialize rules manager
        print("\n2. Initializing Rules Manager...")
        rules = RulesManager(calendar)
        
        # Create and write test rules
        print("\n3. Creating test rules...")
        test_rules = create_test_rules()
        rules.rules_file.parent.mkdir(exist_ok=True)
        rules.rules_file.write_text(test_rules)
        print(f"Test rules written to: {rules.rules_file}")
        
        # Load and parse rules
        print("\n4. Loading and parsing rules...")
        rules.load_rules()
        
        # Print parsed constraints
        print("\n5. Parsed Constraints:")
        
        print("\nFixed Events:")
        for event in rules.fixed_events:
            print(f"- {event['summary']}: days {event['days']} from {event['start_time']} to {event['end_time']}")
        
        print("\nBlocked Times:")
        for block in rules.blocked_times:
            print(f"- {block['summary']}: days {block['days']} from {block['start_time']} to {block['end_time']}")
        
        print("\nPreferences:")
        for pref in rules.preferences:
            print(f"- {pref}")
        
        # Add events to calendar
        print("\n6. Adding fixed events to calendar...")
        rules.add_fixed_events_to_calendar()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise

if __name__ == "__main__":
    run_tests()