import os

if __name__ == '__main__':
    path = 'data/filtered'
    filtered_list = open('filtered_list.txt', 'w')
    for file in os.listdir(path):
        filtered_list.write(file + '\n')