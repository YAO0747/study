import torch.nn as nn


class DeepKnowledgeTracing(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_skills, nlayers, dropout=0.6, tie_weights=False):
        super(DeepKnowledgeTracing, self).__init__()

        # 选择网络类型, 生成RNN的隐层
        if rnn_type == 'LSTM':
            # input_size: 输入数据的特征维数, hidden_size: LSTM中隐层的维度, nlayers: 网络层数
            self.rnn = nn.LSTM(input_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        elif rnn_type == 'RNN_TANH':
            self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity='tanh', dropout=dropout)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, nlayers, nonlinearity='relu', dropout=dropout)

        # nn.Linear是一个全连接层，hidden_size是输入层维数，num_skills是输出层维数
        # 因此，decoder是隐层(self.rnn)到输出层的网络
        self.decoder = nn.Linear(hidden_size, num_skills)

        # 初始化权重及其他
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.05
        # 隐层到输出层的网络的权重
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # 前向计算, 网络结构是：input --> hidden(self.rnn) --> decoder(输出层)
    # https://pytorch.org/docs/stable/generated/torch.nn.RNN.html?highlight=rnn#torch.nn.RNN
    # 根据官网,torch.nn.RNN接收的参数input形状是[时间步数, 批量大小, 特征维数], hidden: 旧的隐藏层的状态
    def forward(self, input, hidden):
        # output: 隐藏层在各个时间步上计算并输出的隐藏状态, 形状是[时间步数, 批量大小, 隐层维数]
        output, hidden = self.rnn(input, hidden)
        # decoded: 形状是[时间步数, 批量大小, num_skills]
        decoded = self.decoder(output.contiguous().view(output.size(0) * output.size(1), output.size(2)))
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
