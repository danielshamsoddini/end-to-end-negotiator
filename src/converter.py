import pandas as pd
import json
import re

# Load the CSV file
file_path = 'output.csv'
df = pd.read_csv(file_path)

# Function to safely load JSON data with additional cleaning
def safe_load_json(data):
    try:
        # Replace single quotes with double quotes, escape existing double quotes, and remove invalid trailing commas
        data = re.sub(r"(?<!\\)'", '"', data)
        data = re.sub(r'\\+"', '"', data)
        data = re.sub(r',\s*}', '}', data)
        data = re.sub(r',\s*]', ']', data)
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e} - Data: {data}")
        return {}

# Function to format each conversation
def format_conversation(row):
    # Extract necessary data
    chat_logs = safe_load_json(row['chat_logs'])
    participant_info = safe_load_json(row['participant_info'])
    
    # Extract input and partner_input from participant_info
    participant_input = participant_info.get('mturk_agent_1', {}).get('value2issue', {})
    partner_input = participant_info.get('mturk_agent_2', {}).get('value2issue', {})
    
    # Convert input and partner_input to required format
    input_values = f"<input> {participant_input.get('Low', 0)} {participant_input.get('Medium', 0)} {participant_input.get('High', 0)} </input>"
    partner_input_values = f"<partner_input> {partner_input.get('Low', 0)} {partner_input.get('Medium', 0)} {partner_input.get('High', 0)} </partner_input>"
    
    # Construct the dialogue string
    dialogue = ""
    if isinstance(chat_logs, list):
        for log in chat_logs:
            speaker = "THEM" if log.get('agent') == 'mturk_agent_2' else "YOU"
            dialogue += f"{speaker}: {log.get('text', '')} <eos> "
    dialogue = f"<dialogue> {dialogue.strip()} <selection> </dialogue>"
    
    # Extract output from annotations (assuming annotations have the required structure)
    annotations = safe_load_json(row['annotations'])
    output_values = ""
    if isinstance(annotations, list) and annotations:
        output_values = " ".join([f"item{idx}={value}" for idx, value in enumerate(annotations[0])])
    output_values = f"<output> {output_values} </output>"
    
    # Combine all parts into the final formatted string
    formatted_string = f"{input_values} {dialogue} {output_values} {partner_input_values}"
    return formatted_string

# Apply the function to each row and collect results
formatted_conversations = df.apply(format_conversation, axis=1)

# Save the results to a new text file
output_text_path = 'formatted_conversations.txt'
with open(output_text_path, 'w') as f:
    for conversation in formatted_conversations:
        f.write(conversation + '\n')

print(f"Formatted conversations have been saved to {output_text_path}")
