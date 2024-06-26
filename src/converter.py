import pandas as pd

# Load the CSV file
file_path = '/mnt/data/output.csv'
df = pd.read_csv(file_path)

# Function to format each conversation
def format_conversation(row):
    # Extract necessary data
    chat_logs = row['chat_logs']
    participant_info = row['participant_info']
    
    # Extract input and partner_input from participant_info
    participant_input = participant_info['mturk_agent_1']['value2issue']
    partner_input = participant_info['mturk_agent_2']['value2issue']
    
    # Convert input and partner_input to required format
    input_values = f"<input> {participant_input['item0']} {participant_input['item1']} {participant_input['item2']} </input>"
    partner_input_values = f"<partner_input> {partner_input['item0']} {partner_input['item1']} {partner_input['item2']} </partner_input>"
    
    # Construct the dialogue string
    dialogue = ""
    for log in chat_logs:
        speaker = "THEM" if log['speaker'] == 'mturk_agent_2' else "YOU"
        dialogue += f"{speaker}: {log['text']} <eos> "
    dialogue = f"<dialogue> {dialogue.strip()} <selection> </dialogue>"
    
    # Extract output from annotations (assuming annotations have the required structure)
    annotations = row['annotations'][0]
    output_values = f"<output> {annotations} </output>"
    
    # Combine all parts into the final formatted string
    formatted_string = f"{input_values} {dialogue} {output_values} {partner_input_values}"
    return formatted_string

# Apply the function to each row and collect results
formatted_conversations = df.apply(format_conversation, axis=1)

# Save the results to a new CSV file
output_df = pd.DataFrame(formatted_conversations, columns=['formatted_conversation'])
output_df.to_csv('/mnt/data/formatted_conversations.csv', index=False)
