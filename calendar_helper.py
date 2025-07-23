import datetime
import os
import json
from typing import Optional, Dict, Any

# Google Calendar imports (optional)
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    import pickle
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    print("⚠️ Google Calendar integration not available. Install with: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")

class CalendarManager:
    """Enhanced calendar manager supporting both local and Google Calendar"""
    
    def __init__(self):
        self.calendar_type = os.getenv("CALENDAR_TYPE", "local")  # Default to local
        self.google_credentials_file = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_FILE", "credentials.json")
        self.google_token_file = os.getenv("GOOGLE_CALENDAR_TOKEN_FILE", "token.pickle")
        self.calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
        self.local_calendar_file = os.getenv("LOCAL_CALENDAR_FILE", "calendar_events.json")
        
        # Google Calendar scopes
        self.SCOPES = ['https://www.googleapis.com/auth/calendar']
        
        # Force local calendar for demo purposes
        self.calendar_type = "local"
        self.google_service = None
        print("📁 Using local calendar storage only")
    
    def _initialize_google_calendar(self):
        """Initialize Google Calendar service"""
        try:
            creds = None
            
            # Load existing token
            if os.path.exists(self.google_token_file):
                with open(self.google_token_file, 'rb') as token:
                    creds = pickle.load(token)
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.google_credentials_file):
                        print(f"❌ Google Calendar credentials file not found: {self.google_credentials_file}")
                        print("   Please download credentials.json from Google Cloud Console")
                        return None
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.google_credentials_file, self.SCOPES)
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open(self.google_token_file, 'wb') as token:
                    pickle.dump(creds, token)
            
            # Build service
            service = build('calendar', 'v3', credentials=creds)
            print("✅ Google Calendar service initialized")
            return service
            
        except Exception as e:
            print(f"❌ Error initializing Google Calendar: {e}")
            return None
    
    def create_event(self, summary: str, start_time: datetime.datetime, 
                    duration_minutes: int = 30, description: str = "", 
                    location: str = "", attendees: list = None, 
                    sales_intelligence: dict = None) -> Dict[str, Any]:
        """
        Create a calendar event (local or Google Calendar)
        
        Args:
            summary: Event title
            start_time: Start time
            duration_minutes: Duration in minutes
            description: Event description
            location: Event location
            attendees: List of attendee emails
        
        Returns:
            Dict with event details and status
        """
        try:
            # Calculate end time
            end_time = start_time + datetime.timedelta(minutes=duration_minutes)
            
            if self.calendar_type == "google" and self.google_service:
                return self._create_google_calendar_event(
                    summary, start_time, end_time, description, location, attendees, sales_intelligence
                )
            else:
                return self._create_local_calendar_event(
                    summary, start_time, end_time, description, location, attendees, sales_intelligence
                )
                
        except Exception as e:
            print(f"❌ Error creating calendar event: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Error creating event: {str(e)}"
            }
    
    def _create_google_calendar_event(self, summary: str, start_time: datetime.datetime,
                                    end_time: datetime.datetime, description: str,
                                    location: str, attendees: list) -> Dict[str, Any]:
        """Create event in Google Calendar"""
        try:
            event = {
                'summary': summary,
                'description': description,
                'location': location,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'America/New_York',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'America/New_York',
                },
            }
            
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
            
            # Add reminders
            event['reminders'] = {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},  # 1 day before
                    {'method': 'popup', 'minutes': 30},       # 30 minutes before
                ],
            }
            
            # Create the event
            event_result = self.google_service.events().insert(
                calendarId=self.calendar_id, body=event
            ).execute()
            
            # Format time for display
            time_str = start_time.strftime("%I:%M %p on %B %d, %Y")
            
            return {
                "success": True,
                "event_id": event_result.get('id'),
                "event_link": event_result.get('htmlLink'),
                "message": f"Google Calendar event '{summary}' scheduled for {time_str}",
                "details": {
                    "summary": summary,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_minutes": int((end_time - start_time).total_seconds() / 60),
                    "description": description,
                    "location": location,
                    "attendees": attendees or []
                }
            }
            
        except Exception as e:
            print(f"❌ Error creating Google Calendar event: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Error creating Google Calendar event: {str(e)}"
            }
    
    def _create_local_calendar_event(self, summary: str, start_time: datetime.datetime,
                                   end_time: datetime.datetime, description: str,
                                   location: str, attendees: list, sales_intelligence: dict = None) -> Dict[str, Any]:
        """Create event in local calendar file"""
        try:
            # Create event data
            event_data = {
                "id": f"local_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "summary": summary,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": int((end_time - start_time).total_seconds() / 60),
                "description": description,
                "location": location,
                "attendees": attendees or [],
                "created_at": datetime.datetime.now().isoformat(),
                "calendar_type": "local"
            }
            
            # Add sales intelligence data if provided
            if sales_intelligence:
                event_data["sales_intelligence"] = sales_intelligence
            
            # Load existing events
            events = []
            if os.path.exists(self.local_calendar_file):
                try:
                    with open(self.local_calendar_file, 'r') as f:
                        events = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    events = []
            
            # Add new event
            events.append(event_data)
            
            # Save updated events
            with open(self.local_calendar_file, 'w') as f:
                json.dump(events, f, indent=2)
            
            # Format time for display
            time_str = start_time.strftime("%I:%M %p on %B %d, %Y")
            
            return {
                "success": True,
                "event_id": event_data["id"],
                "event_link": f"file://{os.path.abspath(self.local_calendar_file)}",
                "message": f"Local calendar event '{summary}' scheduled for {time_str}",
                "details": event_data
            }
            
        except Exception as e:
            print(f"❌ Error creating local calendar event: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Error creating local calendar event: {str(e)}"
            }
    
    def list_events(self, days_ahead: int = 7) -> list:
        """List upcoming events"""
        try:
            if self.calendar_type == "google" and self.google_service:
                return self._list_google_calendar_events(days_ahead)
            else:
                return self._list_local_calendar_events(days_ahead)
        except Exception as e:
            print(f"❌ Error listing events: {e}")
            return []
    
    def _list_google_calendar_events(self, days_ahead: int) -> list:
        """List events from Google Calendar"""
        try:
            now = datetime.datetime.utcnow().isoformat() + 'Z'
            end_time = (datetime.datetime.utcnow() + datetime.timedelta(days=days_ahead)).isoformat() + 'Z'
            
            events_result = self.google_service.events().list(
                calendarId=self.calendar_id,
                timeMin=now,
                timeMax=end_time,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            return events_result.get('items', [])
            
        except Exception as e:
            print(f"❌ Error listing Google Calendar events: {e}")
            return []
    
    def _list_local_calendar_events(self, days_ahead: int) -> list:
        """List events from local calendar file"""
        try:
            if not os.path.exists(self.local_calendar_file):
                return []
            
            with open(self.local_calendar_file, 'r') as f:
                events = json.load(f)
            
            # Filter events within the specified days
            now = datetime.datetime.now()
            end_time = now + datetime.timedelta(days=days_ahead)
            
            upcoming_events = []
            for event in events:
                try:
                    event_start = datetime.datetime.fromisoformat(event['start_time'])
                    if now <= event_start <= end_time:
                        upcoming_events.append(event)
                except:
                    continue
            
            return upcoming_events
            
        except Exception as e:
            print(f"❌ Error listing local calendar events: {e}")
            return []

# Global calendar manager instance
calendar_manager = CalendarManager()

def create_event(summary, start_time, duration_minutes=30, description="", location="", attendees=None, sales_intelligence=None):
    """
    Create a calendar event and return a link or confirmation
    
    Args:
        summary (str): Event title/summary
        start_time (datetime): Start time of the event
        duration_minutes (int): Duration in minutes
        description (str): Event description
        location (str): Event location
        attendees (list): List of attendee emails
    
    Returns:
        str: Event link or confirmation message
    """
    result = calendar_manager.create_event(
        summary=summary,
        start_time=start_time,
        duration_minutes=duration_minutes,
        description=description,
        location=location,
        attendees=attendees,
        sales_intelligence=sales_intelligence
    )
    
    if result["success"]:
        return result["message"]
    else:
        return result["message"]

def list_upcoming_events(days_ahead=7):
    """List upcoming events"""
    return calendar_manager.list_events(days_ahead)

def get_calendar_status():
    """Get calendar system status"""
    status = {
        "calendar_type": calendar_manager.calendar_type,
        "google_calendar_available": GOOGLE_CALENDAR_AVAILABLE,
        "google_service_initialized": calendar_manager.google_service is not None,
        "local_calendar_file": calendar_manager.local_calendar_file,
        "google_credentials_file": calendar_manager.google_credentials_file
    }
    return status 