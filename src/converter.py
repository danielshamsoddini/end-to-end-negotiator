import pandas as pd
import json

# Load the CSV file
file_path = 'casino_data.parquet'
df = pd.read_parquet(file_path)

# Function to format each conversation

chat_logs = []
input_items = "abc"
partner_input_items = "def"
master_str = ""
for row in df.iterrows():
    #print(row[1]['chat_logs'])
    
    dialogue_str = ""
    # input_str = f"<input> {input_items['Food']} {input_items['Water']} {input_items['Firewood']} 1 1 2 </input>"
    # partner_input_str = f"<partner_input> {partner_input_items['Food']} {partner_input_items['Water']} {partner_input_items['Firewood']} 2 1 2 </partner_input>"
    for a in row[1]['chat_logs']:
        #print(a)
        id = "YOU" if a['id'] == 'mturk_agent_1' else "THEM"
        if a['text'] == "Submit-Deal" or a['text'] == "Accept-Deal":
            if "<selection>" not in dialogue_str:
                dialogue_str += f"{id}: <selection>"
        else:
            dialogue_str += f"{id}: {a['text']} <eos> "
    master_str += f"<input> 1 </input> <dialogue> {dialogue_str} </dialogue> <output> 1 </output> <partner_input> 1 </partner_input>\n"
    print(row[0])
print(chat_logs)

with open('casino_convo.txt', 'w', encoding="utf-8") as log_file:
    log_file.write(master_str)


