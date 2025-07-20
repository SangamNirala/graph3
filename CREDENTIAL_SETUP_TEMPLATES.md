# n8n Credential Configuration Templates

This file contains the exact credential configurations needed for the AI Lead Generation workflow. Copy these configurations exactly as shown when setting up your credentials in n8n.

## 🔑 Credential Configurations

### 1. Apollo.io API Credential

**Credential Type:** HTTP Request Auth → Generic Credential

```json
{
  "name": "Apollo.io API",
  "type": "httpHeaderAuth",
  "data": {
    "headerAuth": {
      "name": "X-API-KEY",
      "value": "YOUR_APOLLO_API_KEY_HERE"
    }
  }
}
```

**Where to get Apollo.io API key:**
1. Sign up at https://apollo.io
2. Go to Settings → Integrations → API
3. Copy the API key
4. Paste in the "value" field above

---

### 2. Hunter.io API Credential

**Credential Type:** HTTP Request Auth → Generic Credential

```json
{
  "name": "Hunter.io API", 
  "type": "httpQueryAuth",
  "data": {
    "queryAuth": {
      "name": "api_key",
      "value": "YOUR_HUNTER_API_KEY_HERE"
    }
  }
}
```

**Where to get Hunter.io API key:**
1. Sign up at https://hunter.io
2. Go to Dashboard → API → API Key tab
3. Copy your API key
4. Paste in the "value" field above

---

### 3. Clearbit API Credential

**Credential Type:** HTTP Request Auth → Generic Credential  

```json
{
  "name": "Clearbit API",
  "type": "httpHeaderAuth", 
  "data": {
    "headerAuth": {
      "name": "Authorization",
      "value": "Bearer YOUR_CLEARBIT_API_KEY_HERE"
    }
  }
}
```

**Where to get Clearbit API key:**
1. Sign up at https://clearbit.com
2. Go to Dashboard → API Keys
3. Copy your Secret API Key
4. Replace "YOUR_CLEARBIT_API_KEY_HERE" with your key (keep "Bearer " prefix)

---

### 4. OpenAI API Credential

**Credential Type:** OpenAI

```json
{
  "name": "OpenAI API",
  "type": "openAiApi",
  "data": {
    "apiKey": "YOUR_OPENAI_API_KEY_HERE",
    "organization": "" // Optional: leave empty if not using organization
  }
}
```

**Where to get OpenAI API key:**
1. Sign up at https://platform.openai.com
2. Go to API Keys → Create new secret key  
3. Copy the key (starts with sk-)
4. Paste in the "apiKey" field above

**Important:** Make sure you have credits in your OpenAI account for API usage.

---

### 5. Google Sheets OAuth2 Credential

**Credential Type:** Google Sheets OAuth2 API

```json
{
  "name": "Google Sheets OAuth2",
  "type": "googleSheetsOAuth2Api",
  "data": {
    "clientId": "YOUR_GOOGLE_CLIENT_ID",
    "clientSecret": "YOUR_GOOGLE_CLIENT_SECRET", 
    "accessToken": "", // Will be auto-filled during OAuth flow
    "refreshToken": "", // Will be auto-filled during OAuth flow
    "scope": "https://www.googleapis.com/auth/spreadsheets"
  }
}
```

**How to set up Google Sheets OAuth2:**

1. **Enable Google Sheets API:**
   - Go to https://console.cloud.google.com
   - Create a new project or select existing one
   - Navigate to APIs & Services → Library
   - Search for "Google Sheets API" and enable it

2. **Create OAuth2 Credentials:**
   - Go to APIs & Services → Credentials
   - Click "Create Credentials" → OAuth 2.0 Client IDs
   - Choose "Web application"
   - Add your n8n URL as authorized redirect URI:
     ```
     https://your-n8n-instance.com/rest/oauth2-credential/callback
     ```
   - Save and copy Client ID and Client Secret

3. **Configure in n8n:**
   - Add Google Sheets OAuth2 API credential
   - Enter Client ID and Client Secret
   - Click "Connect my account" to complete OAuth flow

---

### 6. Slack OAuth2 Credential

**Credential Type:** Slack OAuth2 API

```json
{
  "name": "Slack OAuth2",
  "type": "slackOAuth2Api",
  "data": {
    "clientId": "YOUR_SLACK_CLIENT_ID",
    "clientSecret": "YOUR_SLACK_CLIENT_SECRET",
    "accessToken": "", // Will be auto-filled during OAuth flow
    "botToken": "YOUR_SLACK_BOT_TOKEN" // Starts with xoxb-
  }
}
```

**How to set up Slack OAuth2:**

1. **Create Slack App:**
   - Go to https://api.slack.com/apps
   - Click "Create New App" → "From scratch"
   - Enter app name and select workspace

2. **Configure OAuth & Permissions:**
   - Go to OAuth & Permissions in your app settings
   - Add Bot Token Scopes:
     - `chat:write`
     - `channels:read`
     - `groups:read`
     - `im:read`
     - `mpim:read`
   
3. **Install App:**
   - Click "Install to Workspace"
   - Authorize the app
   - Copy the "Bot User OAuth Token" (starts with xoxb-)

4. **Add to n8n:**
   - Create Slack OAuth2 API credential
   - Use the Bot User OAuth Token as the access token

---

## 🔧 Node-Specific Configuration Updates

After setting up credentials, you need to update these placeholder values in the workflow:

### Google Sheets Spreadsheet ID
Replace `SPREADSHEET_ID_PLACEHOLDER` in all Google Sheets nodes with your actual spreadsheet ID.

**How to find Spreadsheet ID:**
From this URL: `https://docs.google.com/spreadsheets/d/1ABC2def3GHI4jkl5MNO6pqr/edit`
The Spreadsheet ID is: `1ABC2def3GHI4jkl5MNO6pqr`

### Slack Channel ID  
Replace `SLACK_CHANNEL_ID_PLACEHOLDER` with your actual channel ID.

**How to find Channel ID:**
1. Right-click on your Slack channel
2. Select "Copy link"  
3. From URL like: `https://workspace.slack.com/archives/C1234567890`
4. Channel ID is: `C1234567890`

---

## 📋 Pre-Setup Checklist

Before importing the workflow, ensure you have:

- [ ] Apollo.io account with API access
- [ ] Hunter.io account with API credits
- [ ] Clearbit account (optional but recommended)
- [ ] OpenAI account with API credits ($5+ recommended)
- [ ] Google account with Sheets API enabled
- [ ] Slack workspace with admin permissions
- [ ] Created a Google Sheet with tabs: "High Quality Leads", "All Leads", "Outreach Messages"
- [ ] Identified target Slack channel for notifications

---

## 🚨 Common Setup Issues & Solutions

### Issue 1: "Authentication failed" errors
**Solution:** Double-check API key format and ensure no extra spaces or characters.

### Issue 2: Google Sheets permission denied
**Solution:** 
- Verify OAuth2 setup is complete
- Check that the service account has edit access to your sheet
- Ensure sheet tab names match exactly: "High Quality Leads", "All Leads", "Outreach Messages"

### Issue 3: OpenAI rate limiting
**Solution:**
- Add usage limits in OpenAI dashboard
- Implement retry logic with delays
- Consider using GPT-3.5-turbo for testing (lower cost)

### Issue 4: Apollo.io quota exceeded
**Solution:**
- Monitor your monthly credit usage
- Reduce batch size in the search parameters
- Upgrade to higher plan if needed

### Issue 5: Slack bot not posting
**Solution:**
- Verify bot has been added to the target channel
- Check bot permissions include `chat:write`
- Test with a simple message first

---

## 💡 Pro Tips for Setup

1. **Test with Small Batches First:** Start with 5-10 leads to verify everything works before scaling up.

2. **Monitor API Usage:** Set up monitoring for all APIs to avoid surprise charges.

3. **Backup Configurations:** Export your credential configurations and save them securely.

4. **Version Control:** Keep track of workflow changes and test in a staging environment first.

5. **Error Handling:** Enable error outputs on all nodes to catch and handle failures gracefully.

6. **Logging:** Enable detailed logging in n8n to troubleshoot issues more effectively.

---

*Save this file as a reference during setup and share it with team members who need to configure their own instances of the workflow.*