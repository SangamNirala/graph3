# Sample Workflow Output & Customization Examples

This file shows exactly what data the AI Lead Generation workflow produces and provides templates for customizing it for different industries and use cases.

## 📊 Sample Workflow Output

### High Quality Leads Sheet Example:

| id | first_name | last_name | email | title | company_name | lead_score | final_score | priority_level | fit_reason |
|---|---|---|---|---|---|---|---|---|---|
| lead_001 | Sarah | Chen | sarah.chen@techcorp.ai | VP of Engineering | TechCorp AI | 9 | 8.7 | high | Perfect ICP match: VP Engineering at AI company, 500 employees, recent AI product launch |
| lead_002 | Mike | Rodriguez | m.rodriguez@dataflow.com | Head of ML | DataFlow Inc | 8 | 8.2 | high | Strong fit: Leading ML initiatives at growing fintech, actively hiring AI talent |
| lead_003 | Jennifer | Kim | jen.kim@cloudscale.io | CTO | CloudScale | 8 | 7.9 | high | Excellent match: CTO at infrastructure company, likely needs AI optimization tools |

### Personalization & Outreach Data:

| lead_name | personalization_angle | pain_points | conversation_starters | outreach_timing |
|---|---|---|---|---|
| Sarah Chen | Recent AI product launch success | Scaling ML infrastructure, Team hiring challenges | TechCorp's new AI assistant launch, ML team scaling strategies | Tuesday-Thursday, 10AM-2PM PST |
| Mike Rodriguez | FinTech ML innovation | Model deployment complexity, Regulatory compliance | DataFlow's recent Series B, ML in financial services | Monday-Friday, 9AM-5PM EST |
| Jennifer Kim | Infrastructure optimization | Cost optimization, Performance scaling | CloudScale's enterprise growth, Infrastructure AI adoption | Tuesday-Thursday, 2PM-6PM GMT |

### Sample AI-Generated Outreach Messages:

#### LinkedIn Connection Request (Sarah Chen):
```
Subject: TechCorp AI's impressive product launch

Hi Sarah,

Congrats on TechCorp AI's recent assistant launch - the technical execution looks impressive! As someone who helps engineering teams scale ML infrastructure, I'd love to connect and share some insights on what we're seeing work well for teams transitioning from prototype to production.

Would you be open to a brief chat about ML scaling strategies?

Best regards,
[Your Name]
```

#### Email Template (Mike Rodriguez):  
```
Subject: ML deployment strategies for FinTech (DataFlow Series B congrats!)

Hi Mike,

Congratulations on DataFlow's Series B! I saw the announcement and was impressed by your team's focus on ML-driven financial services innovation.

I work with FinTech ML leaders who are tackling similar challenges around model deployment and regulatory compliance. Given DataFlow's growth trajectory, I thought you might find value in a brief conversation about what we're seeing work well for teams scaling ML in highly regulated environments.

Would you have 15 minutes next week for a quick call? Happy to share some specific strategies that have worked for similar stage companies.

Best regards,
[Your Name]

P.S. I noticed your recent LinkedIn post about ML hiring - we've helped several FinTech companies build their ML teams efficiently.
```

### Slack Notification Example:
```
🎯 Lead Generation Workflow Complete!

📊 Summary:
• Total Leads Found: 47
• High Quality Leads: 12
• Average Score: 6.8/10

📈 Score Distribution:
• High (7-10): 12 leads
• Medium (4-7): 28 leads  
• Low (0-4): 7 leads

🏢 Companies Analyzed: TechCorp AI, DataFlow Inc, CloudScale, InnovateLab, AI Systems and 12 more

📋 Next Steps:
• Review high-quality leads in Google Sheets
• Check personalized outreach messages
• Begin targeted outreach campaigns

🔗 View Lead Database: https://docs.google.com/spreadsheets/d/1ABC2def3GHI4jkl5MNO6pqr

Workflow completed at Jan 27, 2025, 10:30 AM
```

---

## 🎯 Industry-Specific Customizations

### 1. SaaS/Software Companies

#### ICP Criteria Customization:
```javascript
// In "Set ICP Criteria" node, update values:
{
  "target_companies": "Stripe, Shopify, Zoom, Slack, Notion, Linear, Figma",
  "ideal_customer_profile": "CTO, VP Engineering, Head of Product at SaaS companies with $10M+ ARR, 50-500 employees, using modern tech stack",
  "search_keywords": "SaaS, software development, product engineering, API, developer tools",
  "company_size_min": 50,
  "company_size_max": 500,
  "funding_stage": "Series A, Series B, Series C"
}
```

#### Custom AI Scoring Prompt:
```
You are scoring leads for a developer productivity SaaS tool. Focus on:
- Engineering team size and growth
- Current tech stack and development practices  
- Pain points around developer velocity and code quality
- Budget authority and decision-making power
- Timeline for new tool adoption

Score 1-10 based on likelihood to purchase developer tools in next 6 months.
```

### 2. E-commerce/Retail

#### ICP Criteria Customization:
```javascript
{
  "target_companies": "Warby Parker, Casper, Glossier, Allbirds, Away, Patagonia",
  "ideal_customer_profile": "VP Marketing, Head of Growth, CMO at D2C brands with $5M+ revenue, strong digital presence",
  "search_keywords": "e-commerce, direct-to-consumer, digital marketing, customer acquisition",
  "company_size_min": 25,
  "company_size_max": 300,
  "industries": "retail, consumer goods, fashion, lifestyle"
}
```

#### Custom Pain Points Focus:
```
Focus on e-commerce specific challenges:
- Customer acquisition costs (CAC) rising
- Attribution and measurement complexity  
- Inventory management optimization
- Personalization at scale
- Mobile commerce conversion
- Post-iOS14 marketing challenges
```

### 3. Healthcare/MedTech

#### ICP Criteria Customization:
```javascript
{
  "target_companies": "Teladoc, Moderna, 23andMe, Ro, Hinge Health, Mindstrong",
  "ideal_customer_profile": "Chief Medical Officer, VP Product, Head of Engineering at digital health companies, regulatory experience preferred",
  "search_keywords": "digital health, telemedicine, medical devices, healthcare technology, HIPAA",
  "compliance_requirements": "HIPAA, FDA, SOC2",
  "company_stage": "FDA approved or in clinical trials"
}
```

### 4. FinTech/Financial Services  

#### ICP Criteria Customization:
```javascript
{
  "target_companies": "Plaid, Stripe, Square, Robinhood, Coinbase, Affirm",
  "ideal_customer_profile": "Chief Risk Officer, VP Engineering, Head of Compliance at FinTech with regulatory oversight experience",
  "search_keywords": "financial technology, payments, lending, cryptocurrency, regulatory compliance",
  "regulatory_focus": "PCI DSS, SOX, anti-money laundering, KYC",
  "funding_requirements": "Series B+, $50M+ raised"
}
```

---

## 🔧 Advanced Workflow Customizations

### 1. Multi-Source Lead Discovery

Add these additional nodes after "Apollo Lead Search":

```json
{
  "name": "LinkedIn Sales Navigator Search",
  "type": "n8n-nodes-base.httpRequest",
  "parameters": {
    "url": "https://api.linkedin.com/v2/people-search",
    "method": "GET",
    "authentication": "predefinedCredentialType",
    "nodeCredentialType": "linkedInOAuth2Api"
  }
},
{
  "name": "ZoomInfo API Search", 
  "type": "n8n-nodes-base.httpRequest",
  "parameters": {
    "url": "https://api.zoominfo.com/lookup/person",
    "method": "POST",
    "authentication": "predefinedCredentialType",
    "nodeCredentialType": "zoomInfoApi"
  }
},
{
  "name": "Merge Multi-Source Data",
  "type": "n8n-nodes-base.function",
  "parameters": {
    "functionCode": "// Merge leads from multiple sources, deduplicate by email"
  }
}
```

### 2. Advanced Lead Scoring with Company Intelligence

```javascript
// Enhanced AI scoring prompt with company intelligence
const enhancedScoringPrompt = `
Analyze this lead using these advanced criteria:

LEAD FIT ANALYSIS:
- Job title relevance (decision maker, budget authority, pain owner)
- Company stage and growth trajectory  
- Technology adoption patterns
- Recent funding, hiring, or expansion signals
- Competitive landscape positioning

TIMING INDICATORS:
- Recent company announcements or news
- Job changes or promotions in last 6 months  
- Industry events or conferences attended
- Social media activity and engagement patterns
- Company's current initiatives and priorities

ENGAGEMENT PROBABILITY:
- Response likelihood based on role and industry
- Preferred communication channels
- Best outreach timing and frequency
- Referral or warm introduction opportunities

Return enhanced JSON with timing_score (1-10), engagement_probability (1-10), and recommended_approach.
`;
```

### 3. Automated CRM Integration

Replace Google Sheets nodes with CRM-specific integrations:

```json
{
  "name": "Create Salesforce Lead",
  "type": "n8n-nodes-base.salesforce",
  "parameters": {
    "operation": "create",
    "resource": "lead",
    "columns": {
      "firstName": "={{ $json.first_name }}",
      "lastName": "={{ $json.last_name }}",
      "email": "={{ $json.email }}",
      "company": "={{ $json.company_name }}",
      "title": "={{ $json.title }}",
      "leadSource": "AI Lead Generation",
      "rating": "={{ $json.priority_level }}",
      "description": "AI Score: {{ $json.final_score }}/10\n{{ $json.fit_reason }}"
    }
  }
},
{
  "name": "Update HubSpot Contact",
  "type": "n8n-nodes-base.hubspot", 
  "parameters": {
    "operation": "create",
    "resource": "contact",
    "properties": {
      "email": "={{ $json.email }}",
      "firstname": "={{ $json.first_name }}",
      "lastname": "={{ $json.last_name }}",
      "jobtitle": "={{ $json.title }}",
      "company": "={{ $json.company_name }}",
      "ai_lead_score": "={{ $json.final_score }}",
      "lead_source": "AI Workflow"
    }
  }
}
```

### 4. Automated Email Sequences

```json
{
  "name": "SendGrid Email Sequence",
  "type": "n8n-nodes-base.sendGrid",
  "parameters": {
    "operation": "send",
    "from": "your-email@company.com",
    "to": "={{ $json.email }}",
    "subject": "={{ JSON.parse($('Generate Outreach Copy').first().json.choices[0].message.content).subject_line }}",
    "text": "={{ JSON.parse($('Generate Outreach Copy').first().json.choices[0].message.content).message_body }}",
    "trackingSettings": {
      "clickTracking": true,
      "openTracking": true
    }
  }
},
{
  "name": "Schedule Follow-up",
  "type": "n8n-nodes-base.schedule",
  "parameters": {
    "trigger": "interval",
    "intervalSize": 3,
    "unit": "days",
    "metadata": {
      "leadId": "={{ $json.id }}",
      "sequenceStep": 1
    }
  }
}
```

### 5. Lead Nurturing Automation

```json
{
  "name": "Content Recommendation Engine",
  "type": "n8n-nodes-base.openAi",
  "parameters": {
    "operation": "create",
    "resource": "chat",
    "messages": [{
      "role": "system", 
      "content": "Recommend 3 relevant blog posts, case studies, or resources for this lead based on their profile and pain points. Focus on educational content that builds trust."
    }, {
      "role": "user",
      "content": "Lead: {{ $json.first_name }} {{ $json.last_name }}, {{ $json.title }} at {{ $json.company_name }}\nPain Points: {{ $json.pain_points.join(', ') }}\nIndustry: {{ $json.company_industry }}"
    }]
  }
},
{
  "name": "Social Media Monitoring",
  "type": "n8n-nodes-base.httpRequest",
  "parameters": {
    "url": "https://api.twitter.com/2/users/by/username/{{ $json.twitter }}",
    "method": "GET",
    "authentication": "predefinedCredentialType",
    "nodeCredentialType": "twitterOAuth2Api"
  }
}
```

---

## 📈 Performance Optimization Templates

### 1. Batch Processing Configuration

```javascript
// Add to workflow settings for better performance
{
  "executionTimeout": 300, // 5 minute timeout
  "batchSize": 10, // Process 10 leads at a time
  "retryPolicy": {
    "maxRetries": 3,
    "retryDelay": 2000 // 2 second delay between retries
  },
  "rateLimiting": {
    "apolloRequestsPerMinute": 60,
    "openAiRequestsPerMinute": 30,
    "hunterRequestsPerMinute": 50
  }
}
```

### 2. Error Handling Template

```javascript
// Add to Function nodes for robust error handling
const errorHandler = (operation, data, error) => {
  const errorLog = {
    timestamp: new Date().toISOString(),
    operation: operation,
    leadId: data?.id || 'unknown',
    error: error.message,
    data: JSON.stringify(data, null, 2)
  };
  
  // Log to external service or continue with fallback
  console.error('Lead Generation Error:', errorLog);
  
  // Return data with error flag for downstream processing
  return {
    ...data,
    hasError: true,
    errorDetails: errorLog,
    needsManualReview: true
  };
};
```

### 3. Data Validation Template

```javascript
// Add validation function before AI scoring
const validateLeadData = (leadData) => {
  const requiredFields = ['first_name', 'last_name', 'company_name', 'title'];
  const validationErrors = [];
  
  requiredFields.forEach(field => {
    if (!leadData[field] || leadData[field].trim() === '') {
      validationErrors.push(`Missing ${field}`);
    }
  });
  
  // Email format validation
  if (leadData.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(leadData.email)) {
    validationErrors.push('Invalid email format');
  }
  
  return {
    ...leadData,
    isValid: validationErrors.length === 0,
    validationErrors: validationErrors,
    dataQualityScore: Math.max(0, 100 - (validationErrors.length * 20))
  };
};
```

---

## 💰 ROI Tracking & Analytics

### 1. Conversion Tracking Setup

```json
{
  "name": "Track Lead Conversions",
  "type": "n8n-nodes-base.webhook",
  "parameters": {
    "path": "lead-conversion",
    "httpMethod": "POST"
  }
},
{
  "name": "Update Conversion Data",
  "type": "n8n-nodes-base.googleSheets",
  "parameters": {
    "operation": "update",
    "sheetName": "Conversion Tracking",
    "columns": {
      "lead_id": "={{ $json.leadId }}",
      "conversion_type": "={{ $json.conversionType }}", // meeting_booked, demo_completed, trial_started, closed_won
      "conversion_date": "={{ $json.conversionDate }}",
      "deal_value": "={{ $json.dealValue }}",
      "time_to_conversion": "={{ $json.timeToConversion }}"
    }
  }
}
```

### 2. Weekly Performance Report

```json
{
  "name": "Weekly Performance Analysis",
  "type": "n8n-nodes-base.function",
  "parameters": {
    "functionCode": `
    // Calculate weekly metrics
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 7);
    
    const weeklyMetrics = {
      leadsGenerated: $('Save All Leads').all().length,
      highQualityLeads: $('Save to Google Sheets').all().length,
      averageScore: calculateAverageScore(),
      topCompanies: getTopCompanies(),
      conversionRate: calculateConversionRate(),
      roi: calculateROI(),
      recommendations: generateRecommendations()
    };
    
    return [{ json: weeklyMetrics }];
    `
  }
}
```

---

*These templates provide advanced customization options for different industries and use cases. Mix and match components based on your specific requirements and gradually add complexity as you master the basic workflow.*