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
                print(a['task_data'])
                agent1out = a['task_data']['issue2youget'] if id == "YOU" else a['task_data']['issue2theyget']
                agent2out =  a['task_data']['issue2youget'] if id == "THEM" else a['task_data']['issue2theyget']
                output = f"item0={agent1out["Firewood"]} item1={agent1out["Food"]} item2={agent1out["Water"]} item0={agent2out["Firewood"]} item1={agent2out["Food"]} item2={agent2out["Water"]} "
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

#split the data into train, val, test
master_str = master_str.split("\n")
train = master_str[:int(len(master_str)*0.8)]
val = master_str[int(len(master_str)*0.8):int(len(master_str)*0.9)]
test = master_str[int(len(master_str)*0.9):]

with open('data/casino/train.txt', 'w', encoding="utf-8") as log_file:
    log_file.write("\n".join(train))

with open('data/casino/val.txt', 'w', encoding="utf-8") as log_file:
    log_file.write("\n".join(val))

with open('data/casino/test.txt', 'w', encoding="utf-8") as log_file:
    log_file.write("\n".join(test))


with open('data/casino/casino_convo.txt', 'w', encoding="utf-8") as log_file:
    log_file.write("\n".join(master_str))

with open('casino_convo.txt', 'w', encoding="utf-8") as log_file:
    log_file.write("\n".join(master_str))


