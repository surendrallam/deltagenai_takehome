# Personalized Email Generation Workflow

## Overview

This project implements a workflow to extract data from a CSV file containing leads and generate personalized emails using two different LLMs. The outputs from both LLMs are compared to determine the best email for sending.

## Steps to Run

1. Clone the repository.
2. Install the required dependencies.
3. Place the CSV file in the specified path.
4. Run the orchestrator function to execute the workflow.
5. Review the generated emails in the output file.

## Dependencies

- pandas
- transformers

## Usage

```bash
python orchestrator.py
