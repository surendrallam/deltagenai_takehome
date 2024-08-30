# import pandas as pd
# from transformers import pipeline

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "gpt-3.5-turbo"
# token = "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"

# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token)
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)


# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product/Service} to Improve {Relevant Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job Title} at {Company}, and I wanted to share how our {Product/Service} can help you with {specific challenge or opportunity relevant to the lead's role}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Initialize LLMs
# llm1 = pipeline('text-generation', model='gpt-3.5-turbo')
# llm2 = pipeline('text-generation', model='gpt-4')

# # Function to generate personalized emails
# def generate_email(template, lead, llm):
#     email_content = template.format(Name=lead['Name'], Company=lead['Company'], Job_Title=lead['Job Title'], Product_Service='procurement software', Relevant_Aspect='procurement efficiency', specific_challenge='streamlining your purchasing processes and reducing costs')
#     personalized_email = llm(email_content, max_length=200)[0]['generated_text']
#     return personalized_email

# # Generate emails
# leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, llm1), axis=1)
# leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, llm2), axis=1)


# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return 'email_llm1'

# # Apply quality check
# leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)

# # Store the results in a CSV file
# output_file_path = './data/personalized_emails.csv'
# leads_data.to_csv(output_file_path, index=False)

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Generate emails
#     leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, llm1), axis=1)
#     leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, llm2), axis=1)
    
#     # Quality check
#     leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
    
#     # Store results
#     leads_data.to_csv(output_file_path, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()

# import pandas as pd
# import openai

# # Replace 'your-api-key' with your actual OpenAI API key
# openai.api_key = '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'

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Function to generate personalized emails using OpenAI API
# def generate_email(template, lead, model="gpt-3.5-turbo"):
#     email_content = template.format(
#         Name=lead['Name'], 
#         Company=lead['Company'], 
#         Job_Title=lead['Job Title'], 
#         Product_Service='procurement software', 
#         Relevant_Aspect='procurement efficiency', 
#         specific_challenge='streamlining your purchasing processes and reducing costs'
#     )
#     response = openai.Completion.create(
#         engine=model,
#         prompt=email_content,
#         max_tokens=200
#     )
#     personalized_email = response.choices[0].text.strip()
#     return personalized_email

# # Generate emails
# leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
# leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo-16k"), axis=1)

# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return 'email_llm1'

# # Apply quality check
# leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)

# # Store the results in a CSV file
# output_file_path = './data/personalized_emails.csv'
# leads_data.to_csv(output_file_path, index=False)

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Generate emails
#     leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-4"), axis=1)
    
#     # Quality check
#     leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
    
#     # Store results
#     leads_data.to_csv(output_file_path, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()

# import pandas as pd
# import openai
# import time

# # Replace 'your-api-key' with your actual OpenAI API key
# openai.api_key = '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'

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Function to generate personalized emails using OpenAI API
# def generate_email(template, lead, model="text-davinci-003"):
#     email_content = template.format(
#         Name=lead['Name'], 
#         Company=lead['Company'], 
#         Job_Title=lead['Job Title'], 
#         Product_Service='procurement software', 
#         Relevant_Aspect='procurement efficiency', 
#         specific_challenge='streamlining your purchasing processes and reducing costs'
#     )
#     try:
#         response = openai.Completion.create(
#             engine=model,
#             prompt=email_content,
#             max_tokens=200
#         )
#         personalized_email = response.choices[0].text.strip()
#         return personalized_email
#     except openai.error.RateLimitError:
#         print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
#         time.sleep(60)
#         return generate_email(template, lead, model)

# # Generate emails
# leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="text-davinci-003"), axis=1)
# leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="text-davinci-003"), axis=1)

# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return 'email_llm1'

# # Apply quality check
# leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)

# # Store the results in a CSV file
# output_file_path = './data/personalized_emails.csv'
# leads_data.to_csv(output_file_path, index=False)

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Generate emails
#     leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="text-davinci-003"), axis=1)
#     leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="text-davinci-003"), axis=1)
    
#     # Quality check
#     leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
    
#     # Store results
#     leads_data.to_csv(output_file_path, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()

# import pandas as pd
# import openai
# import time

# # Replace 'your-api-key' with your actual OpenAI API key
# openai.api_key = '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'

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Function to generate personalized emails using OpenAI API
# def generate_email(template, lead, model="gpt-3.5-turbo"):
#     email_content = template.format(
#         Name=lead['Name'], 
#         Company=lead['Company'], 
#         Job_Title=lead['Job Title'], 
#         Product_Service='procurement software', 
#         Relevant_Aspect='procurement efficiency', 
#         specific_challenge='streamlining your purchasing processes and reducing costs'
#     )
#     try:
#         response = openai.Completion.create(
#             engine=model,
#             prompt=email_content,
#             max_tokens=200
#         )
#         personalized_email = response.choices[0].text.strip()
#         return personalized_email
#     except openai.error.RateLimitError:
#         print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
#         time.sleep(60)
#         return generate_email(template, lead, model)

# # Generate emails
# leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
# leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)

# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return 'email_llm1'

# # Apply quality check
# leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)

# # Store the results in a CSV file
# output_file_path = './data/personalized_emails.csv'
# leads_data.to_csv(output_file_path, index=False)

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Generate emails
#     leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
    
#     # Quality check
#     leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
    
#     # Store results
#     leads_data.to_csv(output_file_path, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()


# import pandas as pd
# import openai
# import time

# # Replace 'your-api-key' with your actual OpenAI API key
# openai.api_key = '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'

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Function to generate personalized emails using OpenAI API with exponential backoff
# def generate_email(template, lead, model="gpt-3.5-turbo"):
#     email_content = template.format(
#         Name=lead['Name'], 
#         Company=lead['Company'], 
#         Job_Title=lead['Job Title'], 
#         Product_Service='procurement software', 
#         Relevant_Aspect='procurement efficiency', 
#         specific_challenge='streamlining your purchasing processes and reducing costs'
#     )
#     max_retries = 5
#     retry_delay = 1  # Start with a 1-second delay
#     for attempt in range(max_retries):
#         try:
#             response = openai.Completion.create(
#                 engine=model,
#                 prompt=email_content,
#                 max_tokens=200
#             )
#             personalized_email = response.choices[0].text.strip()
#             return personalized_email
#         except openai.error.RateLimitError:
#             print(f"Rate limit exceeded. Waiting for {retry_delay} seconds before retrying...")
#             time.sleep(retry_delay)
#             retry_delay *= 2  # Exponential backoff
#     raise Exception("Max retries exceeded. Please try again later.")

# # Generate emails
# leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
# leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)

# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return 'email_llm1'

# # Apply quality check
# leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)

# # Store the results in a CSV file
# output_file_path = './data/personalized_emails.csv'
# leads_data.to_csv(output_file_path, index=False)

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Generate emails
#     leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
    
#     # Quality check
#     leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
    
#     # Store results
#     leads_data.to_csv(output_file_path, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()


# import pandas as pd
# import openai
# import time

# # Replace 'your-api-key' with your actual OpenAI API key
# openai.api_key = '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'

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Function to generate personalized emails using OpenAI API with exponential backoff
# def generate_email(template, lead, model="gpt-3.5-turbo"):
#     email_content = template.format(
#         Name=lead['Name'], 
#         Company=lead['Company'], 
#         Job_Title=lead['Job Title'], 
#         Product_Service='procurement software', 
#         Relevant_Aspect='procurement efficiency', 
#         specific_challenge='streamlining your purchasing processes and reducing costs'
#     )
#     max_retries = 5
#     retry_delay = 1  # Start with a 1-second delay
#     for attempt in range(max_retries):
#         try:
#             response = openai.Completion.create(
#                 engine=model,
#                 prompt=email_content,
#                 max_tokens=200
#             )
#             personalized_email = response.choices[0].text.strip()
#             return personalized_email
#         except openai.error.RateLimitError:
#             print(f"Rate limit exceeded. Waiting for {retry_delay} seconds before retrying...")
#             time.sleep(retry_delay)
#             retry_delay *= 2  # Exponential backoff
#     raise Exception("Max retries exceeded. Please try again later.")

# # Process leads in batches
# batch_size = 2  # Adjust the batch size as needed
# for start in range(0, len(leads_data), batch_size):
#     end = start + batch_size
#     batch = leads_data.iloc[start:end]
#     batch['email_llm1'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     batch['email_llm2'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     batch['best_email'] = batch.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
#     batch.to_csv(output_file_path, mode='a', header=not start, index=False)

# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return 'email_llm1'

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Process leads in batches
#     batch_size = 2  # Adjust the batch size as needed
#     for start in range(0, len(leads_data), batch_size):
#         end = start + batch_size
#         batch = leads_data.iloc[start:end]
#         batch['email_llm1'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#         batch['email_llm2'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#         batch['best_email'] = batch.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
#         batch.to_csv(output_file_path, mode='a', header=not start, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()


# import pandas as pd
# import openai
# import time
# import os

# OPENAI_API_KEY = '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'
# # Load API key from environment variable
# openai.api_key = os.getenv('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')

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Function to generate personalized emails using OpenAI API with exponential backoff
# def generate_email(template, lead, model="gpt-3.5-turbo"):
#     email_content = template.format(
#         Name=lead['Name'], 
#         Company=lead['Company'], 
#         Job_Title=lead['Job Title'], 
#         Product_Service='procurement software', 
#         Relevant_Aspect='procurement efficiency', 
#         specific_challenge='streamlining your purchasing processes and reducing costs'
#     )
#     max_retries = 5
#     retry_delay = 1  # Start with a 1-second delay
#     for attempt in range(max_retries):
#         try:
#             response = openai.Completion.create(
#                 engine=model,
#                 prompt=email_content,
#                 max_tokens=200
#             )
#             personalized_email = response.choices[0].text.strip()
#             return personalized_email
#         except openai.error.RateLimitError:
#             print(f"Rate limit exceeded. Waiting for {retry_delay} seconds before retrying...")
#             time.sleep(retry_delay)
#             retry_delay *= 2  # Exponential backoff
#         except openai.error.OpenAIError as e:
#             print(f"An error occurred: {e}")
#             break
#     raise Exception("Max retries exceeded. Please try again later.")

# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return email1 if len(email1) > len(email2) else email2

# # Process leads in batches
# batch_size = 2  # Adjust the batch size as needed
# output_file_path = './data/output_emails.csv'
# for start in range(0, len(leads_data), batch_size):
#     end = start + batch_size
#     batch = leads_data.iloc[start:end]
#     batch['email_llm1'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     batch['email_llm2'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     batch['best_email'] = batch.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
#     batch.to_csv(output_file_path, mode='a', header=not start, index=False)

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Process leads in batches
#     batch_size = 2  # Adjust the batch size as needed
#     for start in range(0, len(leads_data), batch_size):
#         end = start + batch_size
#         batch = leads_data.iloc[start:end]
#         batch['email_llm1'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#         batch['email_llm2'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#         batch['best_email'] = batch.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
#         batch.to_csv(output_file_path, mode='a', header=not start, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()

# import pandas as pd
# import openai
# import os

# # Load API key from environment variable
# openai.api_key = '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'

# # Load the CSV file
# csv_file_path = './data/sample_leads_10.csv'
# leads_df = pd.read_csv(csv_file_path)
# print(leads_df.head())

# # Extract relevant details
# leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
# print(leads_data.head())

# cold_template = """
# Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

# Hi {Name},

# I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

# [Personalized message based on the lead's company or job title]

# I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

# Best regards,
# [Your Name]
# """

# # Function to generate personalized emails using OpenAI API with chat models
# def generate_email(template, lead, model="gpt-3.5-turbo-16k"):
#     email_content = template.format(
#         Name=lead['Name'], 
#         Company=lead['Company'], 
#         Job_Title=lead['Job Title'], 
#         Product_Service='procurement software', 
#         Relevant_Aspect='procurement efficiency', 
#         specific_challenge='streamlining your purchasing processes and reducing costs'
#     )
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": email_content}
#         ],
#         max_tokens=200
#     )
#     personalized_email = response.choices[0].message['content'].strip()
#     return personalized_email

# # Generate emails
# leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
# leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-4"), axis=1)

# # Placeholder function for quality check
# def quality_check(email1, email2):
#     # Implement your quality check logic here
#     # For simplicity, let's assume email1 is always better
#     return email1 if len(email1) > len(email2) else email2

# # Apply quality check
# leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)

# # Store the results in a CSV file
# output_file_path = './data/personalized_emails.csv'
# leads_data.to_csv(output_file_path, index=False)

# # Example of a simple orchestrator function
# def orchestrator():
#     # Load CSV
#     leads_df = pd.read_csv(csv_file_path)
#     leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
#     # Generate emails
#     leads_data['email_llm1'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo"), axis=1)
#     leads_data['email_llm2'] = leads_data.apply(lambda lead: generate_email(cold_template, lead, model="gpt-4"), axis=1)
    
#     # Quality check
#     leads_data['best_email'] = leads_data.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
    
#     # Store results
#     leads_data.to_csv(output_file_path, index=False)

# # Run the orchestrator
# orchestrator()

# # Placeholder for MORL implementation
# def morl_optimization():
#     # Implement your MORL logic here
#     pass

# # Example usage
# morl_optimization()

import pandas as pd
import openai
import os
import time

# Load API key from environment variable
openai.api_key = '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'

# Load the CSV file
csv_file_path = './data/sample_leads_10.csv'
leads_df = pd.read_csv(csv_file_path)
print(leads_df.head())

# Extract relevant details
leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
print(leads_data.head())

cold_template = """
Subject: Introducing {Product_Service} to Improve {Relevant_Aspect}

Hi {Name},

I hope this email finds you well. My name is [Your Name], and I am reaching out to you from [Your Company]. I noticed that you are the {Job_Title} at {Company}, and I wanted to share how our {Product_Service} can help you with {specific_challenge}.

[Personalized message based on the lead's company or job title]

I'd love to discuss this further and see how we can assist you. Please let me know if you're available for a quick call next week.

Best regards,
[Your Name]
"""

# Function to generate personalized emails using OpenAI API with chat models
def generate_email(template, lead, model="gpt-3.5-turbo-16k"):
    email_content = template.format(
        Name=lead['Name'], 
        Company=lead['Company'], 
        Job_Title=lead['Job Title'], 
        Product_Service='procurement software', 
        Relevant_Aspect='procurement efficiency', 
        specific_challenge='streamlining your purchasing processes and reducing costs'
    )
    max_retries = 5
    retry_delay = 1  # Start with a 1-second delay
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": email_content}
                ],
                max_tokens=5000
            )
            personalized_email = response.choices[0].message['content'].strip()
            return personalized_email
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Waiting for {retry_delay} seconds before retrying...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            break
    raise Exception("Max retries exceeded. Please try again later.")

# Placeholder function for quality check
def quality_check(email1, email2):
    # Implement your quality check logic here
    # For simplicity, let's assume email1 is always better
    return email1 if len(email1) > len(email2) else email2

# Example of a simple orchestrator function
def orchestrator():
    # Load CSV
    leads_df = pd.read_csv(csv_file_path)
    leads_data = leads_df[['Name', 'Email', 'Company', 'Job Title']]
    
    # Generate emails
    batch_size = 1  # Adjust the batch size to reduce the number of requests
    for start in range(0, len(leads_data), batch_size):
        end = start + batch_size
        batch = leads_data.iloc[start:end]
        batch['email_llm1'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo-16k"), axis=1)
        batch['email_llm2'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo-16k"), axis=1)
        batch['best_email'] = batch.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
        batch.to_csv(output_file_path, mode='a', header=not start, index=False)

# Placeholder for MORL implementation
def morl_optimization():
    # Implement your MORL logic here
    pass

# Generate emails
batch_size = 1  # Adjust the batch size to reduce the number of requests
output_file_path = './data/personalized_emails.csv'
for start in range(0, len(leads_data), batch_size):
    end = start + batch_size
    batch = leads_data.iloc[start:end]
    batch['email_llm1'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo-16k"), axis=1)
    batch['email_llm2'] = batch.apply(lambda lead: generate_email(cold_template, lead, model="gpt-3.5-turbo-16k"), axis=1)
    batch['best_email'] = batch.apply(lambda lead: quality_check(lead['email_llm1'], lead['email_llm2']), axis=1)
    batch.to_csv(output_file_path, mode='a', header=not start, index=False)


# Run the orchestrator
orchestrator()

# Example usage
morl_optimization()

