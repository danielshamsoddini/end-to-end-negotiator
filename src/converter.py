import pandas as pd
import json
from unidecode import unidecode
import string
import random
import sys
random.seed(int(sys.argv[1]))
stopwords = set([
    'the', 'and', 'to', 'of', 'a', 'in', 'that', 'is', 'was', 'he', 'it', 'with', 'as', 'his', 'on', 'be', 
    'at', 'by', 'this', 'had', 'not', 'are', 'but', 'from', 'or', 'have', 'an', 'they', 'which', 'one', 
    'were', 'her', 'all', 'she', 'there', 'would', 'their', 'him', 'been', 'has', 'when', 'who', 'will', 'more', 
    'no', 'if', 'out', 'so', 'said', 'what', 'up', 'its', 'about', 'into', 'than', 'can', 'only', 'other', 'new', 
    'some', 'could', 'time', 'these', 'may', 'first', 'then', 'do', 'my', 'now', 'such', 'like', 'over', 
    'me', 'even', 'made', 'after', 'also', 'did', 'many', 'before', 'must', 'through', 'back', 'years', 
    'where', 'much', 'way', 'well', 'down', 'should', 'because', 'each', 'people'
])

def add_space_around_punctuation(text):
    punctuation = string.punctuation.replace("'", "").replace(":", "")
    for char in punctuation:
        text = text.replace(char, f" {char} ")
    return ' '.join(text.split())

def stopword_remover(input):
    words = input.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)
# Load the CSV file
file_path = 'casino_data.parquet'
df = pd.read_parquet(file_path)

# Function to format each conversation

chat_logs = []
input_items = "abc"
partner_input_items = "def"
master_str = []
val2num = {'Low': 3, 'Medium': 6, 'High': 9}
#item 0 will be firewood, item 1 will be food, item 2 will be water, alphanum order
item2index = {'Firewood': 0,  'Food': 1, 'Water': 2}
for row in df.iterrows():
    #print(row[1]['chat_logs'])
    # print(row[1])
    # print(row[1]['participant_info'])

    # break
    dialogue_str = ""
    invert_dialogue_str = ""
    output = ""
    for a in row[1]['chat_logs']:
        #print(a)
        id = "YOU" if a['id'] == 'mturk_agent_1' else "THEM"
        invert_id = "THEM" if a['id'] == 'mturk_agent_1' else "YOU"
        if a['text'] == "Submit-Deal" or a['text'] == "Accept-Deal":
            if a['text'] == "Submit-Deal":
                print(a['task_data'])
                agent1out = a['task_data']['issue2youget'] if id == "YOU" else a['task_data']['issue2theyget']
                agent2out =  a['task_data']['issue2youget'] if id == "THEM" else a['task_data']['issue2theyget']
                output = f"item0={agent1out["Firewood"]} item1={agent1out["Food"]} item2={agent1out["Water"]} item0={agent2out["Firewood"]} item1={agent2out["Food"]} item2={agent2out["Water"]} "
                invert_output = f"item0={agent2out["Firewood"]} item1={agent2out["Food"]} item2={agent2out["Water"]} item0={agent1out["Firewood"]} item1={agent1out["Food"]} item2={agent1out["Water"]} "
            if "<selection>" not in dialogue_str:
                dialogue_str += f"{id}: <selection>"
                invert_dialogue_str += f"{invert_id}: <selection>"
        elif a['text'] == 'Walk-Away':
            output = "<no_agreement> " * 6
            print(output)
        else:
            dialogue_str += f"{id}: {add_space_around_punctuation(unidecode(a['text']).lower())} <eos> "
            invert_dialogue_str += f"{invert_id}: {stopword_remover(add_space_around_punctuation(unidecode(a['text']).lower()))} <eos> "

    #each item has a total quantity of 3
    input_arr = {row[1]["participant_info"]['mturk_agent_1']["value2issue"][k]:v for k,v in val2num.items()}
    partner_arr = {row[1]["participant_info"]['mturk_agent_2']["value2issue"][k]:v for k,v in val2num.items()}
    # print(row[1]["participant_info"]['mturk_agent_1']["value2issue"])
    # print(row[1]["participant_info"]['mturk_agent_2']["value2issue"])
    # print(input_arr)
    # print(partner_arr)
    item_quant = [3 for i in range(3)]
    currline = f"<input> {item_quant[0]} {input_arr["Firewood"]} {item_quant[1]} {input_arr["Food"]} {item_quant[2]} {input_arr["Water"]} </input> <dialogue> {dialogue_str} </dialogue> <output> {output}</output> <partner_input> 3 {partner_arr["Firewood"]} 3 {partner_arr["Food"]} 3 {partner_arr["Water"]} </partner_input>"
    currline_invert = f"<input> {item_quant[0]} {partner_arr["Firewood"]} {item_quant[1]} {partner_arr["Food"]} {item_quant[2]} {partner_arr["Water"]} </input> <dialogue> {invert_dialogue_str} </dialogue> <output> {invert_output}</output> <partner_input> 3 {input_arr["Firewood"]} 3 {input_arr["Food"]} 3 {input_arr["Water"]} </partner_input>"
    master_str.append((currline,currline_invert))
    print(row[0])

#split the data into train, val, test
random.shuffle(master_str)
dup_coeff = 1
train = master_str[:int(len(master_str)*0.8)] * dup_coeff
val = master_str[int(len(master_str)*0.8):int(len(master_str)*0.9)] * dup_coeff
test = master_str[int(len(master_str)*0.9):] * dup_coeff
xyz = "understand we do have to think about water being by far the most important thing in life since it bascially is right 🙂 and well how"
print(xyz)
print(stopword_remover(unidecode(xyz)))
for a in range(1):
    with open('data/casino/train.txt', 'w') as log_file:
        for s in train:
            a,b = s
            log_file.write(a + "\n")
            log_file.write(b + "\n")

    with open('data/casino/val.txt', 'w') as log_file:
        for s in val:
            a,b = s
            log_file.write(a + "\n")
            log_file.write(b + "\n")

    with open('data/casino/test.txt', 'w') as log_file:
        for s in test:
            a,b = s
            log_file.write(a + "\n")
            log_file.write(b + "\n")


with open('data/casino/casino_convo.txt', 'w') as log_file:
    for s in master_str:
        a,b = s
        log_file.write(a + "\n")
        log_file.write(b + "\n")

with open('casino_convo.txt', 'w') as log_file:
    for i, s in enumerate(master_str):
        a, b = s
        log_file.write(a + "\n")
        if i == len(master_str) - 1:
            log_file.write(b)
        else:
            log_file.write(b + "\n")


