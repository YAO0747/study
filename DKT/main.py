from typing import List
import torch
import torch.nn as nn
import argparse
import numpy as np
from data import load_data
from deep_knowledge_tracing_model import DeepKnowledgeTracing

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser(description='Deep Knowledge tracing model')
parser.add_argument('-epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer')
parser.add_argument('-l2_lambda', type=float, default=0.3, help='Lambda for l2 loss')
parser.add_argument('-learning_rate', type=float, default=0.1, help='Learning rate')
parser.add_argument('-max_grad_norm', type=float, default=20, help='Clip gradients to this norm')
parser.add_argument('-keep_prob', type=float, default=0.6, help='Keep probability for dropout')
parser.add_argument('-hidden_layer_num', type=int, default=1, help='The number of hidden layers')
parser.add_argument('-hidden_size', type=int, default=200, help='The number of hidden nodes')
parser.add_argument('-evaluation_interval', type=int, default=5, help='Evalutaion and print result every x epochs')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('-epochs', type=int, default=150, help='Number of epochs to train')
parser.add_argument('-allow_soft_placement', type=bool, default=True, help='Allow device soft device placement')
parser.add_argument('-log_device_placement', type=bool, default=False, help='Log placement ofops on devices')
parser.add_argument('-train_data_path', type=str, default='data/0910_b_train.csv', help='Path to the training dataset')
parser.add_argument('-test_data_path', type=str, default='data/0910_b_test.csv',help='Path to the testing dataset')
args = parser.parse_args()
print(args)


def add_gradient_noise(t, stddev=1e-3):
    m = torch.zeros(t.size())
    stddev = torch.full(t.size(), stddev)
    gn = torch.normal(mean=m, std=stddev)
    return torch.add(t, gn)


# 用于初始化隐层
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# m:DeepKnowledgeTracing, optimizer:优化器, num_skills:124, num_steps: 全局最长的做题序列而不是某个批次最长
def run_epoch(m, optimizer, students, batch_size, num_steps, num_skills, training=True, epoch=1):
    # 1.初始化
    total_loss = 0
    input_size = num_skills * 2
    index = 0
    actual_labels = []
    pred_labels = []
    hidden = m.init_hidden(num_steps)
    count = 0
    batch_num = len(students) // batch_size

    # 2. 获取每个batch训练的数据集
    # x: 批量喂入的数据集
    # target_id: 数据集的题号(喂入一个批量的数据集, 模型输出是所有知识点的掌握情况, 需要计算准确率的是本批次的题号)
    # target_correctness: 对应target_id的对错
    while(index+ batch_size < len(students)):
        x = np.zeros((num_steps, batch_size))
        target_id: List[int] = []
        target_correctness = []
        for i in range(batch_size):
            # student: [[题目个数], [题目序列], [答对情况]]
            student = students[index+i]
            problem_ids = student[1]
            correctness = student[2]
            # 答题序列的前n-1个作为模型的输入
            for j in range(len(problem_ids)-1):
                problem_id = int(problem_ids[j])
                # 答对就是124+题号, 答错就是题号, 方便后面转化成one_hot
                if(int(correctness[j]) == 0):
                    label_index = problem_id
                else:
                    label_index = problem_id + num_skills
                x[j, i] = label_index
                # 需要预测的是答题序列的后n-1个(t时刻需要预测t+1时刻)
                target_id.append(j*batch_size*num_skills+i*num_skills+int(problem_ids[j+1]))
                target_correctness.append(int(correctness[j+1]))
                actual_labels.append(int(correctness[j+1]))

        index += batch_size
        count += 1
        target_id = torch.tensor(target_id, dtype=torch.int64)
        target_correctness = torch.tensor(target_correctness, dtype=torch.float)

        # input_data:(batch_size, num_steps, input_size), 此时batch_size = 32, input_size = 248
        x = torch.tensor(x, dtype=torch.int64)
        x = torch.unsqueeze(x, 2)
        input_data = torch.FloatTensor(num_steps, batch_size, input_size)
        input_data.zero_()
        # scatter_用于生成one_hot向量，这里会将所有答题序列统一为num_steps
        input_data.scatter_(2, x, 1)

        # 训练
        if training:
            # 初始化隐层，相当于hidden = m.init_hidden(batch_size)
            hidden = repackage_hidden(hidden)
            # 把模型中参数的梯度设为0
            optimizer.zero_grad()
            # 前向计算, output:(num_steps, batch_size, num_skills)
            output, hidden = m(input_data, hidden)

            # 将输出层转化为一维张量
            output = output.contiguous().view(-1)
            # tf.gather用一个一维的索引数组，将张量中对应索引的向量提取出来
            logits = torch.gather(output, 0, target_id)
            # 预测, preds是0~1的数组
            preds = torch.sigmoid(logits)
            for p in preds:
                pred_labels.append(p.item())

            # 计算误差，相当于nn.functional.binary_cross_entropy_with_logits()
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, target_correctness)

            # 反向传播
            loss.backward()
            # 梯度截断，防止在RNNs或者LSTMs中梯度爆炸的问题
            torch.nn.utils.clip_grad_norm_(m.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
        # 测试
        else:
            with torch.no_grad():
                # 前向计算
                m.eval()
                output, hidden = m(input_data, hidden)
                output = output.contiguous().view(-1)
                logits = torch.gather(output, 0, target_id)
                preds = torch.sigmoid(logits)
                for p in preds:
                    pred_labels.append(p.item())

                # 计算误差，但不进行反向传播
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, target_correctness)
                total_loss += loss.item()
                hidden = repackage_hidden(hidden)

        # 打印误差等信息
        rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
        fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("Epoch: {},  Batch {}/{} AUC: {}".format(epoch, count, batch_num, auc))

        r2 = r2_score(actual_labels, pred_labels)

    return rmse, auc, r2


def main():
    train_data_path = args.train_data_path
    test_data_path  = args.test_data_path
    batch_size = args.batch_size
    train_students, train_max_num_problems, train_max_skill_num = load_data(train_data_path)
    num_steps = train_max_num_problems
    num_skills = train_max_skill_num
    num_layers = 1
    test_students, test_max_num_problems, test_max_skill_num = load_data(test_data_path)
    model = DeepKnowledgeTracing('LSTM', num_skills*2, args.hidden_size, num_skills, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.epsilon)
    for i in range(args.epochs):
        rmse, auc, r2 = run_epoch(model, optimizer,  train_students, batch_size, num_steps, num_skills, epoch=i)
        print(rmse, auc, r2)
        # Testing
        if ((i + 1) % args.evaluation_interval == 0):
            rmse, auc, r2 = run_epoch(model, optimizer, test_students, batch_size, num_steps, num_skills, training=False)
            print('Testing')
            print(rmse, auc, r2)


if __name__ == '__main__':
    main()
