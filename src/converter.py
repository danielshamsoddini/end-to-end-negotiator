import pandas as pd
import json
from unidecode import unidecode
import string
import random
import sys
random.seed(int(sys.argv[1]))
def add_space_around_punctuation(text):
    punctuation = string.punctuation.replace("'", "").replace(":", "")
    for char in punctuation:
        text = text.replace(char, f" {char} ")
    return ' '.join(text.split())


# Load the CSV file
file_path = 'casino_data.parquet'
df = pd.read_parquet(file_path)

# Function to format each conversation

chat_logs = []
input_items = "abc"
partner_input_items = "def"
master_str = []
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
            dialogue_str += f"{id}: {add_space_around_punctuation(unidecode(a['text']).lower())} <eos> "

    #each item has a total quantity of 3
    input_arr = {row[1]["participant_info"]['mturk_agent_1']["value2issue"][k]:v for k,v in val2num.items()}
    partner_arr = {row[1]["participant_info"]['mturk_agent_2']["value2issue"][k]:v for k,v in val2num.items()}
    # print(row[1]["participant_info"]['mturk_agent_1']["value2issue"])
    # print(row[1]["participant_info"]['mturk_agent_2']["value2issue"])
    # print(input_arr)
    # print(partner_arr)
    master_str.append(f"<input> 3 {input_arr["Firewood"]} 3 {input_arr["Food"]} 3 {input_arr["Water"]} </input> <dialogue> {dialogue_str} </dialogue> <output> {output}</output> <partner_input> 3 {partner_arr["Firewood"]} 3 {partner_arr["Food"]} 3 {partner_arr["Water"]} </partner_input>")
    print(row[0])

#split the data into train, val, test
random.shuffle(master_str)
dup_coeff = 2
train = master_str[:int(len(master_str)*0.8)] * dup_coeff
val = master_str[int(len(master_str)*0.8):int(len(master_str)*0.9)] * dup_coeff
test = master_str[int(len(master_str)*0.9):] * dup_coeff
xyz = "understand we do have to think about water being by far the most important thing in life since it bascially is right 🙂 and well how"
print(xyz)
print(unidecode(xyz))
for a in range(1):
    with open('data/casino/train.txt', 'w') as log_file:
        for s in train:
            log_file.write(s + "\n")

    with open('data/casino/val.txt', 'w') as log_file:
        for s in val:
            log_file.write(s + "\n")

    with open('data/casino/test.txt', 'w') as log_file:
        for s in test:
            log_file.write(s + "\n")


with open('data/casino/casino_convo.txt', 'w') as log_file:
    log_file.write("\n".join(master_str))

with open('casino_convo.txt', 'w') as log_file:
    log_file.write("\n".join(master_str))


