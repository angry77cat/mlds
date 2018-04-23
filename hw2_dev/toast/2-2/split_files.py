# since 'clr_conversation.txt' is too large to load in one time,
# I decide to split it to 56523 files first, each file contains
# exactly one conversation session
id = 0
with open('data/clr_conversation.txt', 'r') as f:
    split_file = open('data/clr/%d.txt' % id, 'w+')
    for line in f:
        line = line.strip('\n')
        if line != '+++$+++':
            split_file.write(line + '\n')
        else:
            split_file.close()
            id += 1
            split_file = open('data/clr/%d.txt' % id, 'w+')
            print('split files: %3d' % id, end='\r')
