import random
import time
import os
import datetime

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 配置config
class TrainConfig(object):
    epochs = 10
    decay_rate = 0.92
    learning_rate = 0.01
    evaluate_every = 100
    checkpoint_every = 100
    max_grad_norm = 3.0


class ModelConfig(object):
    hidden_layers = [200]
    dropout_keep_prob = 0.6


class Config(object):
    batch_size = 32
    num_skills = 124
    input_size = num_skills * 2

    trainConfig = TrainConfig()
    modelConfig = ModelConfig()
    

# 实例化config
config = Config()

# 生成数据
class DataGenerator(object):
    # 导入的seqs是train_seqs，或者是test_seqs
    def __init__(self, fileName, config):
        self.fileName = fileName
        self.train_seqs = []
        self.test_seqs = []
        self.infer_seqs = []
        self.batch_size = config.batch_size
        self.pos = 0
        self.end = False
        self.num_skills = config.num_skills
        self.skills_to_int = {}  # 知识点到索引的映射
        self.int_to_skills = {}  # 索引到知识点的映射

    def read_file(self):
        # 从文件中读取数据，返回读取出来的数据和知识点个数
        # 保存每个学生的做题信息 {学生id: [[知识点id，答题结果], [知识点id，答题结果], ...]}，用一个二元列表来表示一个学生的答题信息
        seqs_by_student = {}
        skills = []  # 统计知识点的数量，之后输入的向量长度就是两倍的知识点数量
        count = 0
        with open(self.fileName, 'r') as f:
            for line in f:
                fields = line.strip().split(" ")  # 一个列表，[学生id，知识点id，答题结果]
                student, skill, is_correct = int(fields[0]), int(fields[1]), int(fields[2])
                skills.append(skill)  # skill实际上是用该题所属知识点来表示的
                seqs_by_student[student] = seqs_by_student.get(student, []) + [[skill, is_correct]]  # 保存每个学生的做题信息
        return seqs_by_student, list(set(skills))

    def gen_dict(self, unique_skills):
        sorted_skills = sorted(unique_skills)
        skills_to_int = {}
        int_to_skills = {}
        for i in range(len(sorted_skills)):
            skills_to_int[sorted_skills[i]] = i
            int_to_skills[i] = sorted_skills[i]

        self.skills_to_int = skills_to_int
        self.int_to_skills = int_to_skills

    def split_dataset(self, seqs_by_student, sample_rate=0.2, random_seed=1):
        # 将数据分割成测试集和训练集
        sorted_keys = sorted(seqs_by_student.keys())  # 得到排好序的学生id的列表

        random.seed(random_seed)
        # 随机抽取学生id，将这部分学生作为测试集
        test_keys = set(random.sample(sorted_keys, int(len(sorted_keys) * sample_rate)))

        # 此时是一个三层的列表来表示的，最外层的列表中的每一个列表表示一个学生的做题信息
        test_seqs = [seqs_by_student[k] for k in seqs_by_student if k in test_keys]
        train_seqs = [seqs_by_student[k] for k in seqs_by_student if k not in test_keys]
        return train_seqs, test_seqs


    def gen_attr(self, is_infer=False):
        if is_infer:
            seqs_by_students, skills = self.read_file()
            self.infer_seqs = seqs_by_students
        else:
            seqs_by_students, skills = self.read_file()
            train_seqs, test_seqs = self.split_dataset(seqs_by_students)
            self.train_seqs = train_seqs
            self.test_seqs = test_seqs

        self.gen_dict(skills)  # 生成知识点到索引的映射字典

    def pad_sequences(self, sequences, maxlen=None, value=0.):
        # 按每个batch中最长的序列进行补全, 传入的sequences是二层列表
        # 统计一个batch中每个序列的长度，其实等于seqs_len
        lengths = [len(s) for s in sequences]
        # 统计下该batch中序列的数量
        nb_samples = len(sequences)
        # 如果没有传入maxlen参数就自动获取最大的序列长度
        if maxlen is None:
            maxlen = np.max(lengths)
        # 构建x矩阵
        x = (np.ones((nb_samples, maxlen)) * value).astype(np.int32)

        # 遍历batch，去除每一个序列
        for idx, s in enumerate(sequences):
            trunc = np.asarray(s, dtype=np.int32)
            x[idx, :len(trunc)] = trunc

        return x

    def num_to_one_hot(self, num, dim):
        # 将题目转换成one-hot的形式， 其中dim=num_skills * 2，前半段表示错误，后半段表示正确
        base = np.zeros(dim)
        if num >= 0:
            base[num] += 1
        return base

    # seqs:(人数(bactch_size), 最长做题序列， 回答结果), 例如A的做题序列是[ [11, 0], [12, 1], [10, 0] ]
    # 1.转化成长度相同的做题序列 --> 2.
    # 生成输入数据和输出数据，输入数据是每条序列的前n-1个元素，输出数据是每条序列的后n-1个元素
    def format_data(self, seqs):
        # 统计一个batch_size中每条序列的长度，在这里不对序列固定长度，通过条用tf.nn.dynamic_rnn让序列长度可以不固定
        seq_len = np.array(list(map(lambda seq: len(seq) - 1, seqs)))
        max_len = max(seq_len)  # 获得一个batch_size中最大的长度

        # x_sequences: (人数, 某人前n-1的题目序列)，例如A的做题序列会被转换为：[11, 12+124, 10]
        x_sequences = np.array([[(self.skills_to_int[j[0]] + self.num_skills * j[1]) for j in i[:-1]] for i in seqs])
        # x: (人数, 最长做题序列)：将x_sequences用-1进行补全，补全后的长度为当前batch的最大序列长度
        x = self.pad_sequences(x_sequences, maxlen=max_len, value=-1)

        # 构建输入值input_x: (人数, 前n-1做题序列的one_hot表示)
        # 例如A的做题序列：[11, 12+124, 10] --> [[0,0,..,1,...,0], [0,..0,,1,...,0], [0,..0,,1,...,0]]
        input_x = np.array([[self.num_to_one_hot(j, self.num_skills * 2) for j in i] for i in x])

        # target_id:(人数, 后n-1最长做题序列idx)：不包含是否答对的信息,只表示做了哪道题的idx
        # 获取目标的序列（即期望输出的序列）target_id_seqs:(人数, 某个人的后n-1个题目序列)
        # 做法是：遍历seqs，取每条序列的后len(i)-1个元素作为此人的target_id
        target_id_seqs = np.array([[self.skills_to_int[j[0]] for j in i[1:]] for i in seqs])
        target_id = self.pad_sequences(target_id_seqs, maxlen=max_len, value=0)

        # target_correctness:(人数, 后n-1最长做题序列)：表示某人做题是否正确
        target_correctness_seqs = np.array([[j[1] for j in i[1:]] for i in seqs])
        target_correctness = self.pad_sequences(target_correctness_seqs, maxlen=max_len, value=0)

        return dict(input_x=input_x, target_id=target_id, target_correctness=target_correctness,
                    seq_len=seq_len, max_len=max_len)

    def next_batch(self, seqs):
        # 接收一个序列，生成batch
        length = len(seqs)
        num_batchs = length // self.batch_size
        start = 0
        for i in range(num_batchs):
            batch_seqs = seqs[start: start + self.batch_size]
            start += self.batch_size
            params = self.format_data(batch_seqs)
            yield params

            
# 构建模型
class TensorFlowDKT(object):
    def __init__(self, config):
        # 导入配置好的参数
        self.hiddens = hiddens = config.modelConfig.hidden_layers  # 200个隐层节点
        self.num_skills = num_skills = config.num_skills
        self.input_size = input_size = config.input_size
        self.batch_size = batch_size = config.batch_size
        self.keep_prob_value = config.modelConfig.dropout_keep_prob

        # 定义需要喂给模型的参数
        self.max_steps = tf.placeholder(tf.int32, name="max_steps")  # 当前batch中最大序列长度
        # input_data: (32, None, 248)：None表示的是序列的长度, 即max_len/num_steps/最长做题序列
        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name="input_x")

        self.sequence_len = tf.placeholder(tf.int32, [batch_size], name="sequence_len")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # dropout keep prob

        self.target_id = tf.placeholder(tf.int32, [batch_size, None], name="target_id")
        self.target_correctness = tf.placeholder(tf.float32, [batch_size, None], name="target_correctness")
        self.flat_target_correctness = None

        # 构建lstm模型结构self.hidden_cell，包含hiddens(200)个节点
        hidden_layers = []
        for idx, hidden_size in enumerate(hiddens):
            lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
            hidden_layer = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_layer, output_keep_prob=self.keep_prob)
            hidden_layers.append(hidden_layer)
        self.hidden_cell = tf.nn.rnn_cell.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

        # 采用动态rnn，动态输入序列的长度
        outputs, self.current_state = tf.nn.dynamic_rnn(cell=self.hidden_cell,
                                                        inputs=self.input_data,
                                                        sequence_length=self.sequence_len,
                                                        dtype=tf.float32)

        # 隐层到输出层的权重系数(最后隐层的神经元数量，知识点数(num_skills))
        output_w = tf.get_variable("W", [hiddens[-1], num_skills])
        output_b = tf.get_variable("b", [num_skills])

        # output: (batch_size * max_steps, 最后隐层的神经元数量)
        self.output = tf.reshape(outputs, [batch_size * self.max_steps, hiddens[-1]])
        # 输出层logits:(batch_size * max_steps, num_skills),猜测是做完第step道题目之后，每个学生对每个知识点的掌握情况
        self.logits = tf.matmul(self.output, output_w) + output_b
        # 转化为(batch_size, max_steps, num_skills)
        self.mat_logits = tf.reshape(self.logits, [batch_size, self.max_steps, num_skills])

        # 对每个batch中每个序列中的每个时间点的输出中的每个值进行sigmoid计算，这里的值表示对某个知识点的掌握情况，
        self.pred_all = tf.sigmoid(self.mat_logits, name="pred_all")

        # self.target_correctness是做题序列目标结果，即0或1的序列，由用户进行输入
        flat_target_correctness = tf.reshape(self.target_correctness, [-1])
        # flat_target_correctness是target_correctness的一维表示
        self.flat_target_correctness = flat_target_correctness
        flat_base_target_index = tf.range(batch_size * self.max_steps) * num_skills
        flat_base_target_id = tf.reshape(self.target_id, [-1])
        # 目标的序列flat_target_id： 长度是batch_size * num_steps
        flat_target_id = flat_base_target_id + flat_base_target_index

        # flat_logits是模型预测的输出，其长度为batch_size * num_steps * num_skills
        flat_logits = tf.reshape(self.logits, [-1])
        # tf.gather用一个一维的索引数组，将张量中对应索引的向量提取出来
        flat_target_logits = tf.gather(flat_logits, flat_target_id)

        # 对切片后的数据进行sigmoid转换
        self.pred = tf.sigmoid(tf.reshape(flat_target_logits, [batch_size, self.max_steps]), name="pred")
        # 将sigmoid后的值表示为0或1
        self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.float32, name="binary_pred")

        # 定义损失函数
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_target_correctness, logits=flat_target_logits))


# 训练模型
def mean(item):
    return sum(item) / len(item)


# 输出准确率时需要用到
def gen_metrics(sequence_len, binary_pred, pred, target_correctness):
    binary_preds = []
    preds = []
    target_correctnesses = []
    for seq_idx, seq_len in enumerate(sequence_len):
        binary_preds.append(binary_pred[seq_idx, :seq_len])
        preds.append(pred[seq_idx, :seq_len])
        target_correctnesses.append(target_correctness[seq_idx, :seq_len])

    new_binary_pred = np.concatenate(binary_preds)
    new_pred = np.concatenate(preds)
    new_target_correctness = np.concatenate(target_correctnesses)

    auc = roc_auc_score(new_target_correctness, new_pred)
    accuracy = accuracy_score(new_target_correctness, new_binary_pred)
    precision = precision_score(new_target_correctness, new_binary_pred)
    recall = recall_score(new_target_correctness, new_binary_pred)
    return auc, accuracy, precision, recall


class DKTEngine(object):

    def __init__(self):
        self.config = Config()
        self.train_dkt = None
        self.test_dkt = None
        self.sess = None
        self.global_step = 0        # 表示当前是全局的第几步

    def add_gradient_noise(self, grad, stddev=1e-3, name=None):
        """
        Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
        """
        with tf.op_scope([grad, stddev], name, "add_gradient_noise") as name:
            grad = tf.convert_to_tensor(grad, name="grad")
            gn = tf.random_normal(tf.shape(grad), stddev=stddev)
            return tf.add(grad, gn, name=name)

    def train_step(self, params, train_op, train_summary_op, train_summary_writer):
        """
        A single training step
        """
        dkt = self.train_dkt
        sess = self.sess
        global_step = self.global_step

        feed_dict = {dkt.input_data: params['input_x'],
                     dkt.target_id: params['target_id'],
                     dkt.target_correctness: params['target_correctness'],
                     dkt.max_steps: params['max_len'],
                     dkt.sequence_len: params['seq_len'],
                     dkt.keep_prob: self.config.modelConfig.dropout_keep_prob}

        # sess.run表示开始训练，train_op是最后的节点，训练sess.run意味着训练整个神经网络
        # loss等可以放在下面，也可以不放，不影响训练的结果，这里放的原因是希望在后面输出它们
        _, step, summaries, loss, binary_pred, pred, target_correctness = sess.run(
            [train_op, global_step, train_summary_op, dkt.loss, dkt.binary_pred, dkt.pred, dkt.target_correctness],
            feed_dict)

        auc, accuracy, precision, recall = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)

        time_str = datetime.datetime.now().isoformat()
        print("train: {}: step {}, loss {}, acc {}, auc: {}, precision: {}, recall: {}".format(time_str, step, loss, accuracy, 
                                                                                               auc, precision, recall))
        train_summary_writer.add_summary(summaries, step)

    # 测试数据集的相关操作
    def dev_step(self, params, dev_summary_op, writer=None):
        """
        Evaluates model on a dev set
        """
        dkt = self.test_dkt
        sess = self.sess
        global_step = self.global_step

        feed_dict = {dkt.input_data: params['input_x'],
                     dkt.target_id: params['target_id'],
                     dkt.target_correctness: params['target_correctness'],
                     dkt.max_steps: params['max_len'],
                     dkt.sequence_len: params['seq_len'],
                     dkt.keep_prob: 1.0}
        step, summaries, loss, pred, binary_pred, target_correctness = sess.run(
            [global_step, dev_summary_op, dkt.loss, dkt.pred, dkt.binary_pred, dkt.target_correctness],
            feed_dict)

        auc, accuracy, precision, recall = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)

        if writer:
            writer.add_summary(summaries, step)

        return loss, accuracy, auc, precision, recall

    def run_epoch(self, fileName):
        # 实例化配置参数对象
        config = Config()

        # 实例化数据生成对象
        dataGen = DataGenerator(fileName, config)
        dataGen.gen_attr()  # 生成训练集和测试集

        # 下列两个数组的形式是：[ [[知识点id，答题结果], [知识点id，答题结果], ...], [[],[],[],...], [[],[],[]],... ]
        # 例如train_seqs有3384个元组，每个元组是某个学生的做题序列：[[知识点id，答题结果], [知识点id，答题结果], ...]
        train_seqs = dataGen.train_seqs     # length: 3384
        test_seqs = dataGen.test_seqs       # length: 843

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        self.sess = sess

        with sess.as_default():
            # 实例化dkt模型对象
            with tf.name_scope("train"):
                with tf.variable_scope("dkt", reuse=None):
                    # train_dkt: 一个TensorFlowDKT模型
                    train_dkt = TensorFlowDKT(config)

            with tf.name_scope("test"):
                with tf.variable_scope("dkt", reuse=True):
                    test_dkt = TensorFlowDKT(config)

            self.train_dkt = train_dkt  # 一个TensorFlowDKT模型
            self.test_dkt = test_dkt    # 一个TensorFlowDKT模型

            global_step = tf.Variable(0, name="global_step", trainable=False)
            self.global_step = global_step  # <tf.Variable 'global_step:0' shape=( ) dtype=int32_ref>

            # 定义一个优化器
            optimizer = tf.train.AdamOptimizer(config.trainConfig.learning_rate)
            grads_and_vars = optimizer.compute_gradients(train_dkt.loss)    # 误差是train_dkt.loss

            # 对梯度进行截断，并且加上梯度噪音
            grads_and_vars = [(tf.clip_by_norm(g, config.trainConfig.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]

            # 定义图中最后的节点
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            # 保存各种变量或结果的值，保存到文件中
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("writing to {}".format(out_dir))

            # 训练时的 Summaries
            train_loss_summary = tf.summary.scalar("loss", train_dkt.loss)
            train_summary_op = tf.summary.merge([train_loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # 测试时的 summaries
            test_loss_summary = tf.summary.scalar("loss", test_dkt.loss)
            dev_summary_op = tf.summary.merge([test_loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables())

            sess.run(tf.global_variables_initializer())

            print("初始化完毕，开始训练")
            for i in range(config.trainConfig.epochs):
                np.random.shuffle(train_seqs)
                for params in dataGen.next_batch(train_seqs):
                    # 批次获得训练集，训练模型
                    # params是一个map，包含的元素有(举个例子：)
                    # 1) input_x --> shape:(32, 1109, 248), 32个学生，做题序列长度为1109，每个序列包含124个知识点，用one-hot表示
                    # 2) target_id --> shape:(32, 1109), 32个学生，表示学生做题的序号，后面用0来填充：[idx1, idx2 ,...,0,0,0]
                    # 3) target_correctness --> 表示target_id是否作对
                    # 4) seq_len: shape:(32)，32个学生的做题序列长度
                    # 5) max_len: shape:(), seq_len的最大值:1109
                    # 开始训练：输入params，
                    self.train_step(params, train_op, train_summary_op, train_summary_writer)

                    current_step = tf.train.global_step(sess, global_step)

                    # 对结果进行记录
                    if current_step % config.trainConfig.evaluate_every == 0:
                        print("\nEvaluation:")
                        # 获得测试数据

                        losses = []
                        accuracys = []
                        aucs = []
                        precisions = []
                        recalls = []
                        for params in dataGen.next_batch(test_seqs):
                            loss, accuracy, auc, precision, recall = self.dev_step(params, dev_summary_op, writer=None)
                            losses.append(loss)
                            accuracys.append(accuracy)
                            aucs.append(auc)
                            precisions.append(precision)
                            recalls.append(recall)

                        time_str = datetime.datetime.now().isoformat()
                        print("dev: {}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".
                              format(time_str, current_step, mean(losses), mean(accuracys), mean(aucs), mean(precisions), mean(recalls)))

                    if current_step % config.trainConfig.checkpoint_every == 0:
                        path = saver.save(sess, "model/my-model", global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    fileName = "./data/assistments.txt"
    dktEngine = DKTEngine()
    dktEngine.run_epoch(fileName)
