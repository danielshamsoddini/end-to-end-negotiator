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
val2num = {'Low': 1, 'Medium': 2, 'High': 3}
#item 0 will be firewood, item 1 will be food, item 2 will be water, alphanum order
item2index = {'Firewood': 0,  'Food': 1, 'Water': 2}
for row in df.iterrows():
    #print(row[1]['chat_logs'])
    # print(row[1])
    # print(row[1]['participant_info'])

    # break
    dialogue_str = ""
    output = ""
    for a in row[1]['chat_logs']:
        #print(a)
        id = "YOU" if a['id'] == 'mturk_agent_1' else "THEM"
        if a['text'] == "Submit-Deal" or a['text'] == "Accept-Deal":
            if a['text'] == "Submit-Deal":
                output = f"item0={} item1={} item2={} item0={} item1{} item2={}"
            if "<selection>" not in dialogue_str:
                dialogue_str += f"{id}: <selection>"
        elif a['text'] == 'Walk-Away':
            output = "<disagree> " * 6
            print(output)
        else:
            dialogue_str += f"{id}: {a['text']} <eos> "

    #each item has a total quantity of 3
    input_arr = {row[1]["participant_info"]['mturk_agent_1']["value2issue"][k]:v for k,v in val2num.items()}
    partner_arr = {row[1]["participant_info"]['mturk_agent_2']["value2issue"][k]:v for k,v in val2num.items()}
    # print(row[1]["participant_info"]['mturk_agent_1']["value2issue"])
    # print(row[1]["participant_info"]['mturk_agent_2']["value2issue"])
    # print(input_arr)
    # print(partner_arr)
    master_str += f"<input> 3 {input_arr["Firewood"]} 3 {input_arr["Food"]} 3 {input_arr["Water"]} </input> <dialogue> {dialogue_str} </dialogue> <output> {output}</output> <partner_input> 3 {partner_arr["Firewood"]} 3 {partner_arr["Food"]} 3 {partner_arr["Water"]} </partner_input>\n"
    print(row[0])

with open('data/casino/casino_convo.txt', 'w', encoding="utf-8") as log_file:
    log_file.write(master_str)

with open('casino_convo.txt', 'w', encoding="utf-8") as log_file:
    log_file.write(master_str)


