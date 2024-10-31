# temporary_auth.py
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import pickle

SCOPES = ['https://www.googleapis.com/auth/calendar']
flow = InstalledAppFlow.from_client_secrets_file('config/credentials.json', SCOPES)
creds = flow.run_local_server(port=0)

with open('cache/token.pickle', 'wb') as token:
    pickle.dump(creds, token)