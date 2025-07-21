# N8N HR Automation Workflow

## API Keys for Configuration

**Important:** You will need to configure these API keys in your N8N workflow after importing.

### Required API Keys:

1. **Groq API Key:**
   - Get your key from: https://console.groq.com/
   - Format: `gsk_[your_key_here]`

2. **Retell Voice API Key:**
   - Get your key from: https://app.retellai.com/
   - Format: `key_[your_key_here]`

3. **LinkedIn API Token:** (To be configured when available)
   - Get from LinkedIn Developer Portal
   - Format: `Bearer [your_token_here]`

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

## Original Keys (for your reference):
- Contact the developer for the actual API keys that were provided in the original request
