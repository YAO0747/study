import csv
import random


def load_data(fileName):
    rows = []
    max_skill_num = 0
    max_num_problems = 0
    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # row可能是学生id或做题序列或答对01序列
            rows.append(row)
    print("filename: " + fileName + "the number of rows is " + str(len(rows)))

    index = 0
    tuple_rows = []
    while(index < len(rows)-1):
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index+1]))
        if(tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
        if(problems_num <= 2):
            index += 3
        else:
            if problems_num > max_num_problems:
                max_num_problems = problems_num
            tup = (rows[index], rows[index+1], rows[index+2])
            # tup:[题目个数, 题目序列, 答对情况]
            tuple_rows.append(tup)
            index += 3
    # shuffle the tuple
    random.shuffle(tuple_rows)
    # tuple_rows的每一行是tup:[[题目个数], [题目序列], [答对情况]], max_num_problems最长题目序列, max_skill_num是知识点(题目)个数
    return tuple_rows, max_num_problems, max_skill_num+1
