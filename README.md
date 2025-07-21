# N8N HR Automation Workflow

## API Keys for Configuration

**Important:** Copy these keys and configure them in your N8N workflow after importing.

### Required API Keys:

1. **Groq API Key:**
   ```
   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
   ```

2. **Retell Voice API Key:**
   ```
   key_d64cb94a0929d6bf59cf2b4ff369
   ```

3. **LinkedIn API Token:** (To be configured when available)
   ```
   LINKEDIN_ACCESS_TOKEN_PLACEHOLDER
   ```

## Configuration Instructions:

1. Import the workflow JSON file into N8N
2. Replace `GROQ_API_KEY_PLACEHOLDER` with the Groq API key above
3. Replace `RETELL_API_KEY_PLACEHOLDER` with the Retell API key above
4. Configure SMTP email settings with your Gmail credentials
5. Test the workflow endpoints before activation

## Setup Notes:
- Ensure N8N has write permissions to `/tmp/` directory for local storage
- Configure Gmail SMTP with app passwords for email automation
- LinkedIn API integration requires separate setup when credentials are available
