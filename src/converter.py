import pandas as pd
import json

# Load the CSV file
file_path = 'output.csv'
df = pd.read_csv(file_path)

# Function to format each conversation
def format_conversation(row):
    chat_logs = json.loads(row['chat_logs'].replace("'", "\""))
    participant_info = json.loads(row['participant_info'].replace("'", "\""))
    
    input_items = participant_info['mturk_agent_1']['value2issue']
    partner_input_items = participant_info['mturk_agent_2']['value2issue']
    
    input_str = f"<input> {input_items['Food']} {input_items['Water']} {input_items['Firewood']} 1 1 2 </input>"
    partner_input_str = f"<partner_input> {partner_input_items['Food']} {partner_input_items['Water']} {partner_input_items['Firewood']} 2 1 2 </partner_input>"
    
    dialogue_str = "<dialogue> "
    for message in chat_logs:
        if message['speaker'] == 'YOU':
            dialogue_str += f"YOU: {message['text']} <eos> "
        else:
            dialogue_str += f"THEM: {message['text']} <eos> "
    dialogue_str += "THEM: <selection> </dialogue>"
    
    output_str = "<output> item0=1 item1=0 item2=1 item0=0 item1=4 item2=0 </output>"  # This needs to be dynamically created based on some logic
    
    formatted_conversation = f"{input_str} {dialogue_str} {output_str} {partner_input_str}"
    
    return formatted_conversation

# Apply the function to each row and create a new DataFrame with the formatted conversations
df['formatted_conversation'] = df.apply(format_conversation, axis=1)

# Save the formatted conversations to a new CSV file
output_file_path = 'formatted_output.csv'
df.to_csv(output_file_path, index=False)

output_file_path
