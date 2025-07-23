# Google Calendar Setup Guide

This guide will help you set up Google Calendar integration for the Sales Voice Agent.

## Prerequisites

1. **Google Account**: You need a Google account with access to Google Calendar
2. **Google Cloud Project**: You need to create a Google Cloud project
3. **Google Calendar API**: Enable the Google Calendar API

## Step-by-Step Setup

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter a project name (e.g., "Sales Voice Agent")
4. Click "Create"

### 2. Enable Google Calendar API

1. In your Google Cloud project, go to "APIs & Services" → "Library"
2. Search for "Google Calendar API"
3. Click on "Google Calendar API"
4. Click "Enable"

### 3. Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client IDs"
3. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: "Sales Voice Agent"
   - User support email: Your email
   - Developer contact information: Your email
   - Save and continue through the steps

4. Create OAuth 2.0 Client ID:
   - Application type: Desktop application
   - Name: "Sales Voice Agent Desktop"
   - Click "Create"

5. Download the credentials:
   - Click the download button (JSON icon)
   - Save the file as `credentials.json` in your project root directory

### 4. Configure Environment Variables

Your `config.env` file should already be configured with:

```env
CALENDAR_TYPE=google
GOOGLE_CALENDAR_CREDENTIALS_FILE=credentials.json
GOOGLE_CALENDAR_TOKEN_FILE=token.pickle
GOOGLE_CALENDAR_ID=primary
```

### 5. Install Dependencies

The required packages are already in `requirements.txt`:

```bash
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 6. First Run Authentication

When you first run the calendar functionality:

1. The system will open a browser window
2. Sign in with your Google account
3. Grant permission to access your Google Calendar
4. The authentication token will be saved to `token.pickle`

## Testing the Setup

Run the test script to verify everything works:

```bash
python test_calendar.py
```

You should see:
- Calendar Type: google
- Google Calendar Available: True
- Google Service Initialized: True

## Troubleshooting

### "Google Calendar credentials file not found"

- Make sure `credentials.json` is in your project root directory
- Verify the file name matches `GOOGLE_CALENDAR_CREDENTIALS_FILE` in config.env

### "Google Calendar not available"

- Install the required packages: `pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client`
- Check that the Google Calendar API is enabled in your Google Cloud project

### Authentication Issues

- Delete `token.pickle` if it exists
- Run the test script again to re-authenticate
- Make sure you're using the correct Google account

### Calendar Access Issues

- Verify the Google account has access to the calendar specified in `GOOGLE_CALENDAR_ID`
- Default is "primary" (your main calendar)
- You can use specific calendar IDs from Google Calendar settings

## Security Notes

- Keep `credentials.json` and `token.pickle` secure
- These files are already in `.gitignore` to prevent accidental commits
- The credentials allow access to your Google Calendar, so treat them as sensitive

## Calendar Event Features

With Google Calendar integration, your events will have:

- **Automatic Reminders**: Email 1 day before, popup 30 minutes before
- **Calendar Sync**: Events appear in your Google Calendar app
- **Attendee Support**: Can add email addresses for meeting participants
- **Location Support**: Can specify meeting locations
- **Rich Descriptions**: Full event descriptions with context

## Usage in Phone Calls

When someone says "schedule a demo" during a phone call, the system will:

1. Parse the scheduling request
2. Create a Google Calendar event
3. Send confirmation to the caller
4. Add the event to your Google Calendar with proper reminders 