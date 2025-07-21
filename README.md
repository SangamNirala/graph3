# N8N HR Automation Workflow

## Overview
Complete HR automation system with job posting, resume collection, ATS scoring, interview automation, and candidate communication.

## Files
- `hr_workflow_clean.json` - N8N workflow file for import
- Contains form triggers, API integrations, local storage, and email automation

## Required API Keys

**Important:** Configure these in N8N after importing the workflow.

### API Services Needed:

1. **Groq API Key:**
   - Platform: https://console.groq.com/
   - Used for: AI resume analysis and scoring
   - Replace: `GROQ_API_KEY_PLACEHOLDER`

2. **Retell Voice API Key:**
   - Platform: https://app.retellai.com/
   - Used for: Automated phone interviews  
   - Replace: `RETELL_API_KEY_PLACEHOLDER`

3. **LinkedIn API Token:**
   - Platform: LinkedIn Developer Portal
   - Used for: Job posting automation
   - Replace: `LINKEDIN_ACCESS_TOKEN_PLACEHOLDER`

## Setup Instructions:

1. Import `hr_workflow_clean.json` into your N8N instance
2. Configure API keys in the respective HTTP Request nodes
3. Set up SMTP email credentials for Gmail
4. Configure file system permissions for `/tmp/` directory
5. Test webhook endpoints and form submissions
6. Activate workflow for production use

## Features:
- Automated job posting to LinkedIn
- Resume collection with smart limits
- AI-powered ATS scoring using Groq
- Email automation for candidate communication  
- Voice interview scheduling with Retell
- Local storage for candidate data and resumes
- Multi-stage selection process

## Architecture:
- Local JSON file storage (jobs.json, candidates.json, interviews.json)
- Webhook endpoints for resume submissions
- Cron scheduling for automated processing
- SMTP integration for bulk email campaigns
- Voice API for automated interviews
- Combined scoring system (ATS + Interview)
