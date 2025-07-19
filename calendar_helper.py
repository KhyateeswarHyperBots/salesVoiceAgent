import datetime
import os
import json

def create_event(summary, start_time, duration_minutes=30, description=""):
    """
    Create a calendar event and return a link or confirmation
    
    Args:
        summary (str): Event title/summary
        start_time (datetime): Start time of the event
        duration_minutes (int): Duration in minutes
        description (str): Event description
    
    Returns:
        str: Event link or confirmation message
    """
    try:
        # Calculate end time
        end_time = start_time + datetime.timedelta(minutes=duration_minutes)
        
        # Create event data
        event_data = {
            "summary": summary,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
            "description": description,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Save to calendar file
        calendar_file = "calendar_events.json"
        events = []
        
        # Load existing events if file exists
        if os.path.exists(calendar_file):
            try:
                with open(calendar_file, 'r') as f:
                    events = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                events = []
        
        # Add new event
        events.append(event_data)
        
        # Save updated events
        with open(calendar_file, 'w') as f:
            json.dump(events, f, indent=2)
        
        # Format time for display
        time_str = start_time.strftime("%I:%M %p on %B %d, %Y")
        
        # Return confirmation message
        return f"Event '{summary}' scheduled for {time_str} ({duration_minutes} minutes)"
        
    except Exception as e:
        print(f"Error creating calendar event: {e}")
        return f"Error creating event: {str(e)}" 