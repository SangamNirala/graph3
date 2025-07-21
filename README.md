# N8N HR Automation Workflow

## API Keys for Configuration

**Important:** You will need to configure these API keys in your N8N workflow after importing.

### Required API Keys:

1. **Groq API Key:**
   - Get your key from: https://console.groq.com/
   - Use the key format from Groq platform

2. **Retell Voice API Key:**
   - Get your key from: https://app.retellai.com/
   - Use the key format from Retell platform

3. **LinkedIn API Token:** (To be configured when available)
   - Get from LinkedIn Developer Portal
   - Use Bearer token format

## Configuration Instructions:

1. Import the workflow JSON file into N8N
2. Replace `GROQ_API_KEY_PLACEHOLDER` with your actual Groq API key
3. Replace `RETELL_API_KEY_PLACEHOLDER` with your actual Retell API key  
4. Configure SMTP email settings with your Gmail credentials
5. Test the workflow endpoints before activation

## Setup Notes:
- Ensure N8N has write permissions to `/tmp/` directory for local storage
- Configure Gmail SMTP with app passwords for email automation
- LinkedIn API integration requires separate setup when credentials are available

## Original Keys Reference:
The API keys were provided in the original request - please refer to the original conversation for the actual key values to use in your N8N configuration.
