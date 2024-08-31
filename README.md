# Personalized Email Generation and Comparison

This project generates personalized emails for leads using OpenAI's GPT-3.5 models and compares the generated emails to determine the best one to send. The results are saved in a JSON file.

## Approach

1. **Load API Key**: The OpenAI API key is loaded from an environment variable.
2. **Load CSV File**: The CSV file containing lead information is loaded into a pandas DataFrame.
3. **Extract Relevant Details**: Relevant details such as Name, Email, Company, Job Title, and Industry are extracted from the DataFrame.
4. **Generate Emails**: Personalized emails are generated for each lead using two different GPT-3.5 models.
5. **Compare Emails**: The generated emails are compared based on relevance, tone, and engagement potential to determine the better email.
6. **Save Results**: The results, including the generated emails and the selected email, are saved in a JSON file.

## How to Run

1. **Set Up Environment**:
    - Ensure you have Python installed.
    - Install the required packages using the following command:
      ```bash
      pip install pandas openai
      ```

2. **Set OpenAI API Key**:
    - Set your OpenAI API key as an environment variable:
      ```bash
      export OPENAI_API_KEY='your-api-key'
      ```

3. **Prepare CSV File**:
    - Place your CSV file containing lead information in the `./data/` directory. The CSV file should have columns: `Name`, `Email`, `Company`, `Job Title`, and `Industry`.

4. **Run the Script**:
    - Execute the script to generate and compare emails:
      ```bash
      python personalised_mail_generation.py
      ```

5. **Check Results**:
    - The results will be saved in a JSON file located at `./data/personalized_emails.json`.

## Example CSV File

```csv
Name,Email,Company,Job Title,Industry
John Doe,john.doe@example.com,Example Inc.,CEO,Technology
Jane Smith,jane.smith@example.com,Sample Corp.,CTO,Finance
...
