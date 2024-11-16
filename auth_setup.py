from src.auth import GoogleCalendarAuth

if __name__ == "__main__":
    print("Starting Google Calendar authentication...")
    auth = GoogleCalendarAuth()
    auth.authenticate()
    print("Authentication successful! Token saved in cache/token.pickle")