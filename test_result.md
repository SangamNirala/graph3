#====================================================================================================
# START - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================

# THIS SECTION CONTAINS CRITICAL TESTING INSTRUCTIONS FOR BOTH AGENTS
# BOTH MAIN_AGENT AND TESTING_AGENT MUST PRESERVE THIS ENTIRE BLOCK

# Communication Protocol:
# If the `testing_agent` is available, main agent should delegate all testing tasks to it.
#
# You have access to a file called `test_result.md`. This file contains the complete testing state
# and history, and is the primary means of communication between main and the testing agent.
#
# Main and testing agents must follow this exact format to maintain testing data. 
# The testing data must be entered in yaml format Below is the data structure:
# 
## user_problem_statement: {problem_statement}
## backend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.py"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## frontend:
##   - task: "Task name"
##     implemented: true
##     working: true  # or false or "NA"
##     file: "file_path.js"
##     stuck_count: 0
##     priority: "high"  # or "medium" or "low"
##     needs_retesting: false
##     status_history:
##         -working: true  # or false or "NA"
##         -agent: "main"  # or "testing" or "user"
##         -comment: "Detailed comment about status"
##
## metadata:
##   created_by: "main_agent"
##   version: "1.0"
##   test_sequence: 0
##   run_ui: false
##
## test_plan:
##   current_focus:
##     - "Task name 1"
##     - "Task name 2"
##   stuck_tasks:
##     - "Task name with persistent issues"
##   test_all: false
##   test_priority: "high_first"  # or "sequential" or "stuck_first"
##
## agent_communication:
##     -agent: "main"  # or "testing" or "user"
##     -message: "Communication message between agents"

# Protocol Guidelines for Main agent
#
# 1. Update Test Result File Before Testing:
#    - Main agent must always update the `test_result.md` file before calling the testing agent
#    - Add implementation details to the status_history
#    - Set `needs_retesting` to true for tasks that need testing
#    - Update the `test_plan` section to guide testing priorities
#    - Add a message to `agent_communication` explaining what you've done
#
# 2. Incorporate User Feedback:
#    - When a user provides feedback that something is or isn't working, add this information to the relevant task's status_history
#    - Update the working status based on user feedback
#    - If a user reports an issue with a task that was marked as working, increment the stuck_count
#    - Whenever user reports issue in the app, if we have testing agent and task_result.md file so find the appropriate task for that and append in status_history of that task to contain the user concern and problem as well 
#
# 3. Track Stuck Tasks:
#    - Monitor which tasks have high stuck_count values or where you are fixing same issue again and again, analyze that when you read task_result.md
#    - For persistent issues, use websearch tool to find solutions
#    - Pay special attention to tasks in the stuck_tasks list
#    - When you fix an issue with a stuck task, don't reset the stuck_count until the testing agent confirms it's working
#
# 4. Provide Context to Testing Agent:
#    - When calling the testing agent, provide clear instructions about:
#      - Which tasks need testing (reference the test_plan)
#      - Any authentication details or configuration needed
#      - Specific test scenarios to focus on
#      - Any known issues or edge cases to verify
#
# 5. Call the testing agent with specific instructions referring to test_result.md
#
# IMPORTANT: Main agent must ALWAYS update test_result.md BEFORE calling the testing agent, as it relies on this file to understand what to test next.

#====================================================================================================
# END - Testing Protocol - DO NOT EDIT OR REMOVE THIS SECTION
#====================================================================================================



#====================================================================================================
# Testing Data - Main Agent and testing sub agent both should log testing data below this section  
#====================================================================================================

user_problem_statement: "Create a comprehensive n8n JSON workflow template for AI-Powered Lead Generation & Outreach Agent that can be imported into n8n and used immediately. The workflow should automate lead discovery from LinkedIn/Crunchbase, data enrichment, AI-powered lead scoring using GPT-4, CRM integration, and personalized outreach generation. All connections should be pre-made - users just need to add API credentials and run it."

backend:
  - task: "Create n8n workflow JSON file with complete lead generation pipeline"  
    implemented: true
    working: true
    file: "n8n-lead-generation-workflow.json"
    stuck_count: 0
    priority: "high" 
    needs_retesting: false
    status_history:
      - working: true
        agent: "main"
        comment: "Created comprehensive n8n workflow JSON with 17 interconnected nodes including Schedule Trigger, Apollo.io lead search, Hunter.io email finder, Clearbit enrichment, OpenAI GPT-4 scoring, Google Sheets storage, Slack notifications, and personalized outreach generation. All connections pre-configured."

  - task: "Create comprehensive setup documentation and guides"
    implemented: true  
    working: true
    file: "N8N_WORKFLOW_SETUP_GUIDE.md, CREDENTIAL_SETUP_TEMPLATES.md, SAMPLE_OUTPUT_AND_CUSTOMIZATIONS.md"
    stuck_count: 0
    priority: "high"
    needs_retesting: false
    status_history:
      - working: true
        agent: "main" 
        comment: "Created detailed setup guide with API credential configurations, troubleshooting, industry-specific customizations, performance optimization, and ROI tracking templates. Includes exact copy-paste credential templates and sample outputs."

frontend:
  - task: "No frontend development required for this n8n workflow template project"
    implemented: "N/A"
    working: "N/A" 
    file: "N/A"
    stuck_count: 0
    priority: "N/A"
    needs_retesting: false
    status_history:
      - working: "N/A"
        agent: "main"
        comment: "This project creates an n8n workflow template, not a web application. No frontend development needed."

metadata:
  created_by: "main_agent"
  version: "1.0"
  test_sequence: 0
  run_ui: false

test_plan:
  current_focus:
    - "n8n workflow JSON structure validation"
    - "Documentation completeness and accuracy"
  stuck_tasks: []
  test_all: false
  test_priority: "documentation_validation"

agent_communication:
  - agent: "main"
    message: "Successfully created comprehensive n8n workflow template for AI-powered lead generation with complete documentation. The workflow includes 17 interconnected nodes covering the full lead generation pipeline from discovery to personalized outreach. All API integrations are pre-configured - users only need to add their credentials. Created 4 detailed documentation files covering setup, credentials, customization examples, and sample outputs."