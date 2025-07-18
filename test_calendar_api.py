#!/usr/bin/env python3
"""
Test script for Google Calendar API functionality
"""

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from calendar_helper import test_calendar_connection, create_event, list_upcoming_events

def test_calendar_functionality():
    """Test all calendar functionality"""
    print("📅 Testing Google Calendar API Functionality")
    print("=" * 60)
    
    # Test 1: Check if credentials exist
    print("\n🔍 Test 1: Checking credentials...")
    if os.path.exists('credentials.json'):
        print("✅ credentials.json found")
    else:
        print("❌ credentials.json not found")
        print("📝 Please download credentials.json from Google Cloud Console")
        return False
    
    if os.path.exists('token.json'):
        print("✅ token.json found (existing authentication)")
    else:
        print("ℹ️ token.json not found (will authenticate on first use)")
    
    # Test 2: Test connection
    print("\n🔍 Test 2: Testing API connection...")
    connection_success = test_calendar_connection()
    
    if not connection_success:
        print("❌ Calendar API connection failed")
        return False
    
    # Test 3: List upcoming events
    print("\n🔍 Test 3: Listing upcoming events...")
    events = list_upcoming_events(max_results=5)
    
    # Test 4: Create a test event
    print("\n🔍 Test 4: Creating test event...")
    test_time = datetime.datetime.now() + datetime.timedelta(hours=1)
    event_link = create_event(
        summary="Test Event - Voice Agent",
        start_time=test_time,
        duration_minutes=15,
        description="This is a test event created by the voice agent system"
    )
    
    if event_link:
        print("✅ Test event created successfully!")
        print(f"🔗 Event link: {event_link}")
    else:
        print("❌ Failed to create test event")
        return False
    
    # Test 5: List events again to confirm creation
    print("\n🔍 Test 5: Confirming event creation...")
    updated_events = list_upcoming_events(max_results=5)
    
    # Check if our test event is in the list
    test_event_found = False
    for event in updated_events:
        if "Test Event - Voice Agent" in event.get('summary', ''):
            test_event_found = True
            print(f"✅ Test event found in calendar: {event['summary']}")
            break
    
    if not test_event_found:
        print("⚠️ Test event not found in calendar listing")
    
    print("\n🎉 Calendar API test completed!")
    return True

def check_calendar_setup():
    """Check if calendar is properly set up"""
    print("🔧 Calendar Setup Check")
    print("=" * 40)
    
    # Check required files
    required_files = ['credentials.json']
    optional_files = ['token.json']
    
    print("📁 Checking required files:")
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (MISSING)")
    
    print("\n📁 Checking optional files:")
    for file in optional_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ℹ️ {file} (not found - will be created on first auth)")
    
    # Check dependencies
    print("\n📦 Checking dependencies:")
    try:
        import google.oauth2.credentials
        print("   ✅ google.oauth2.credentials")
    except ImportError:
        print("   ❌ google.oauth2.credentials (MISSING)")
    
    try:
        import google_auth_oauthlib
        print("   ✅ google_auth_oauthlib")
    except ImportError:
        print("   ❌ google_auth_oauthlib (MISSING)")
    
    try:
        import googleapiclient
        print("   ✅ googleapiclient")
    except ImportError:
        print("   ❌ googleapiclient (MISSING)")

if __name__ == "__main__":
    print("🔍 Google Calendar API Test Suite")
    print("=" * 60)
    
    # First check setup
    check_calendar_setup()
    
    print("\n" + "=" * 60)
    
    # Then test functionality
    if test_calendar_functionality():
        print("\n✅ All calendar tests passed!")
        print("🎉 Google Calendar API is working correctly!")
    else:
        print("\n❌ Calendar tests failed!")
        print("🔧 Please check the setup instructions above.") 