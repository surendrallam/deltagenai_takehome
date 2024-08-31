import pandas as pd
import openai
import os
import json
import time

# Load API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the Data Extraction Agent
def data_extraction_agent(csv_file_path):
    leads_df = pd.read_csv(csv_file_path)
    leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title', 'Industry']]
    return leads_data

# Define the Email Construction Agent
def email_construction_agent(model, lead, template):
    prompt = template.format(
        Name=lead['Name'],
        Job_Title=lead['Job Title'],
        Company=lead['Company'],
        Product_Service='[Your Product/Service]',
        Relevant_Aspect='[Relevant Aspect]',
        specific_challenge='[Specific Challenge]'
    )
    
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200
            )
            break
        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting for 20 seconds before retrying...")
            time.sleep(20)
    
    personalized_email = response.choices[0].message['content'].strip()
    return personalized_email

# Define the Quality Check Agent
def quality_check_agent(email1, email2):
    comparison_prompt = f"""
    Compare the following two emails based on the criteria: relevance, tone, and engagement potential. Determine which email is better for sending.

    Criteria:
    1. Relevance: How relevant is the email content to the recipient?
    2. Tone: Is the tone of the email appropriate and engaging?
    3. Engagement Potential: Does the email encourage the recipient to take action?

    Email 1:
    {email1}

    Email 2:
    {email2}

    Provide a detailed comparison and indicate which email is better for sending.
    """
    
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": comparison_prompt}
                ],
                max_tokens=300
            )
            break
        except openai.error.RateLimitError:
            print("Rate limit reached. Waiting for 20 seconds before retrying...")
            time.sleep(20)
    
    comparison_result = response.choices[0].message['content'].strip()
    return comparison_result

# Define the Orchestrated Agent
def orchestrated_agent(csv_file_path, output_file_path, email_template):
    # Step 1: Data Extraction
    leads_data = data_extraction_agent(csv_file_path)
    
    results = []
    
    # Step 2: Email Construction and Quality Check
    for index, lead in leads_data.iterrows():
        email_gpt35 = email_construction_agent('gpt-3.5-turbo-16k', lead, email_template)
        email_gpt35_turbo = email_construction_agent('gpt-3.5-turbo', lead, email_template)
        
        comparison_result = quality_check_agent(email_gpt35, email_gpt35_turbo)
        
        selected_email = "LLM 1" if "Email 1" in comparison_result else "LLM 2"
        
        result_entry = {
            "name": lead['Name'],
            "email": lead['Email'],
            "company": lead['Company'],
            "job_title": lead['Job Title'],
            "email_llm_1": email_gpt35,
            "email_llm_2": email_gpt35_turbo,
            "selected_email": selected_email
        }
        
        results.append(result_entry)
    
    # Step 3: Save Results
    with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {output_file_path}")

# Define the email template
cold_template = """
Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

Hi {Name},

I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

[Personalized message based on the lead's company or job title]

I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

Best regards,
[Your Name]
"""

# Run the orchestrated agent
csv_file_path = './data/sample_leads_10.csv'
output_file_path = './data/personalized_emails.json'
orchestrated_agent(csv_file_path, output_file_path, cold_template)

