import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
from pathlib import Path

class GoogleCalendarAuth:
    def __init__(self):
        # Get project root directory (where main.py is)
        self.project_root = Path(__file__).parent.parent
        
        # Define paths relative to project root
        self.credentials_path = self.project_root / 'config' / 'credentials.json'
        self.token_path = self.project_root / 'cache' / 'token.pickle'
        
        # If modifying these scopes, delete the token.pickle file
        self.SCOPES = ['https://www.googleapis.com/auth/calendar']
        self.creds = None

    def authenticate(self):
        """Handles the complete authentication flow"""
        # Create cache directory if it doesn't exist
        self.token_path.parent.mkdir(exist_ok=True)
        
        # Load existing token if available
        if self.token_path.exists():
            with open(self.token_path, 'rb') as token:
                self.creds = pickle.load(token)

        # If no valid credentials available, let user log in
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                print("Refreshing token...")
                self.creds.refresh(Request())
            else:
                print("Getting new token...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_path), self.SCOPES)
                self.creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(self.creds, token)

        return self.creds

    def get_calendar_service(self):
        """Returns an authorized Calendar API service instance"""
        creds = self.authenticate()
        service = build('calendar', 'v3', credentials=creds)
        return service

# For testing the auth module directly
if __name__ == "__main__":
    try:
        auth = GoogleCalendarAuth()
        service = auth.get_calendar_service()
        
        print("Testing calendar access...")
        calendar_list = service.calendarList().list().execute()
        
        print("Access successful! Your calendars:")
        for calendar in calendar_list['items']:
            print(f"- {calendar['summary']}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")