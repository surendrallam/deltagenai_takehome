# import pandas as pd
# import openai

# Load API key from environment variable
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# # print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title', 'Industry']]
# print(leads_data)

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# def generate_email(model, lead):
#     prompt = cold_template.format(
#         Name=lead['Name'],
#         Job_Title=lead['Job Title'],
#         Company=lead['Company'],
#         Product_Service='[Your Product/Service]',
#         Relevant_Aspect='[Relevant Aspect]',
#         specific_challenge='[Specific Challenge]'
#     )
    
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=2000
#     )
    
#     personalized_email = response.choices[0].message['content'].strip()
#     return personalized_email


# def compare_emails(email1, email2):
#     # Define criteria for comparison
#     criteria = {
#         "relevance": "How relevant is the email content to the recipient?",
#         "tone": "Is the tone of the email appropriate and engaging?",
#         "engagement_potential": "Does the email encourage the recipient to take action?"
#     }
    
#     # Use OpenAI to evaluate the emails based on the criteria
#     comparison_prompt = f"""
#     Compare the following two emails based on the criteria: relevance, tone, and engagement potential. Determine which email is better for sending.

#     Criteria:
#     1. Relevance: How relevant is the email content to the recipient?
#     2. Tone: Is the tone of the email appropriate and engaging?
#     3. Engagement Potential: Does the email encourage the recipient to take action?

#     Email 1:
#     {email1}

#     Email 2:
#     {email2}

#     Provide a detailed comparison and indicate which email is better for sending.
#     """
    
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": comparison_prompt}
#         ],
#         max_tokens=3000
#     )
    
#     comparison_result = response.choices[0].message['content'].strip()
#     return comparison_result

# # # Generate and compare emails using both models
# # for index, lead in leads_data.iterrows():
# #     email_gpt35 = generate_email('gpt-3.5-turbo-16k', lead)
# #     email_gpt35_turbo = generate_email('gpt-3.5-turbo', lead)
    
# #     print(f"Email for {lead['Name']} using GPT-3.5:\n{email_gpt35}\n")
# #     print(f"Email for {lead['Name']} using GPT-3.5-turbo:\n{email_gpt35_turbo}\n")
    
# #     comparison_result = compare_emails(email_gpt35, email_gpt35_turbo)
# #     print(f"Comparison Result for {lead['Name']}:\n{comparison_result}\n")


# # Generate and compare emails, then store results in a JSON file
# results = []

# for index, lead in leads_data.iterrows():
#     email_gpt35 = generate_email('gpt-3.5-turbo-16k', lead)
#     email_gpt35_turbo = generate_email('gpt-3.5', lead)
    
#     comparison_result = compare_emails(email_gpt35, email_gpt35_turbo)
    
#     selected_email = "LLM 1" if "Email 1" in comparison_result else "LLM 2"
    
#     result_entry = {
#         "name": lead['Name'],
#         "email": lead['Email'],
#         "company": lead['Company'],
#         "job_title": lead['Job Title'],
#         "email_llm_1": email_gpt35,
#         "email_llm_2": email_gpt35_turbo,
#         "selected_email": selected_email
#     }
    
#     results.append(result_entry)

# # Save results to a JSON file
# output_file_path = './data/personalized_emails.json'
# with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
#     json.dump(results, jsonfile, ensure_ascii=False, indent=4)

# print(f"Results saved to {output_file_path}")

import pandas as pd
import openai
import os
import json
import time

# Load API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the CSV file
csv_file_path = './data/sample_leads_10.csv'
leads_df = pd.read_csv(csv_file_path)

# Extract relevant details
leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title', 'Industry']]

cold_template = """
Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

Hi {Name},

I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

[Personalized message based on the lead's company or job title]

I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

Best regards,
[Your Name]
"""

def generate_email(model, lead):
    prompt = cold_template.format(
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

def compare_emails(email1, email2):
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

# Generate and compare emails, then store results in a JSON file
results = []

for index, lead in leads_data.iterrows():
    email_gpt35 = generate_email('gpt-3.5-turbo-16k', lead)
    email_gpt35_turbo = generate_email('gpt-3.5-turbo', lead)
    
    comparison_result = compare_emails(email_gpt35, email_gpt35_turbo)
    
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

# Save results to a JSON file
output_file_path = './data/personalized_emails.json'
with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
    json.dump(results, jsonfile, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file_path}")
