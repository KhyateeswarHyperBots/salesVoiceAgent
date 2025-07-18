import datetime
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/calendar.events']

def authenticate_google_calendar():
    """Authenticate with Google Calendar API"""
    creds = None
    
    # Check if token.json exists
    if os.path.exists('token.json'):
        try:
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
            print("✅ Loaded existing credentials from token.json")
        except Exception as e:
            print(f"❌ Error loading token.json: {e}")
            creds = None
    
    # If no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
            creds.refresh(Request())
                print("✅ Refreshed expired credentials")
            except Exception as e:
                print(f"❌ Error refreshing credentials: {e}")
                creds = None
        
        if not creds:
            # Check if credentials.json exists
            if not os.path.exists('credentials.json'):
                print("❌ credentials.json not found!")
                print("📝 Please download credentials.json from Google Cloud Console:")
                print("   1. Go to https://console.cloud.google.com/")
                print("   2. Create a project and enable Google Calendar API")
                print("   3. Create credentials (OAuth 2.0 Client ID)")
                print("   4. Download as credentials.json")
                return None
            
            try:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
                print("✅ Successfully authenticated with Google Calendar")
                
                # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
                print("💾 Saved credentials to token.json")
                
            except Exception as e:
                print(f"❌ Error during authentication: {e}")
                return None
    
    return creds

def test_calendar_connection():
    """Test Google Calendar API connection"""
    print("🔍 Testing Google Calendar API connection...")
    
    try:
        creds = authenticate_google_calendar()
        if not creds:
            print("❌ Authentication failed")
            return False
        
        service = build('calendar', 'v3', credentials=creds)
        
        # Test by listing calendars
        calendar_list = service.calendarList().list().execute()
        calendars = calendar_list.get('items', [])
        
        print(f"✅ Successfully connected to Google Calendar API")
        print(f"📅 Found {len(calendars)} calendars:")
        
        for calendar in calendars:
            print(f"   - {calendar['summary']} ({calendar['id']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing calendar connection: {e}")
        return False

def create_event(summary, start_time, duration_minutes=30, description="Voice agent appointment"):
    """Create a Google Calendar event"""
    try:
    creds = authenticate_google_calendar()
        if not creds:
            print("❌ Cannot create event: Authentication failed")
            return None
        
    service = build('calendar', 'v3', credentials=creds)
    
    end_time = start_time + datetime.timedelta(minutes=duration_minutes)
    event = {
        'summary': summary,
        'description': description,
        'start': {
            'dateTime': start_time.isoformat(),
                'timeZone': 'America/New_York',  # Updated timezone
        },
        'end': {
            'dateTime': end_time.isoformat(),
                'timeZone': 'America/New_York',  # Updated timezone
        },
    }

        event_result = service.events().insert(calendarId='primary', body=event).execute()
        event_link = event_result.get('htmlLink')
        
        print(f"✅ Event created successfully!")
        print(f"📅 Summary: {summary}")
        print(f"🕐 Start: {start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"⏱️ Duration: {duration_minutes} minutes")
        print(f"🔗 Link: {event_link}")
        
        return event_link
        
    except Exception as e:
        print(f"❌ Error creating calendar event: {e}")
        return None

def list_upcoming_events(max_results=10):
    """List upcoming calendar events"""
    try:
        creds = authenticate_google_calendar()
        if not creds:
            print("❌ Cannot list events: Authentication failed")
            return []
        
        service = build('calendar', 'v3', credentials=creds)
        
        # Get upcoming events
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        if not events:
            print("📅 No upcoming events found")
            return []
        
        print(f"📅 Upcoming events ({len(events)}):")
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            print(f"   - {event['summary']} ({start})")
        
        return events
        
    except Exception as e:
        print(f"❌ Error listing events: {e}")
        return []
