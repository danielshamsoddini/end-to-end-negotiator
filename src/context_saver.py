from utils import DNDContextGenerator

abc = DNDContextGenerator("data/casino/test.txt").iter()

with open('data/casino/selfplay.txt', 'w') as log_file:
    for ctx in abc:
        a,b = ctx
        print(a,b)
        log_file.write(' '.join(a) + "\n")
        log_file.write(' '.join(b) + "\n")