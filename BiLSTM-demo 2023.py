# -*- coding: utf-8 -*-
"""#https://blog.csdn.net/CSTGYinZong/article/details/121462095
#pip install torchtext==0.6.0
#需要下载en_core_web_sm-3.0.0.tar.gz，pip install 存放位置/en_core_web_sm-3.0.0.tar.gz
#pip install spacy==3.0.0，注意：spacy的版本要与en_core_web_sm版本要相同！
#            from spacy.lang.en import English
#            spacy = English()
#downloading aclImdb_v1.tar.gz和glove.6B.zip
#下载的数据集会在project里面的.data\imdb\文件夹下
"""
import torch
from torch import nn, optim
#from torchtext.legacy import data, datasets
from torchtext import data, datasets
from torchtext.vocab import Vectors
import numpy as np
import time
import pandas as pd
# 根据当前环境选择是否调用GPU进行训练
print('GPU:', torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

#1.文本数据准备
#TEXT = data.Field(tokenize='spacy',fix_length=200)#用spacy()处理文本，句子长度为200
TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm',
                  fix_length=200)
LABEL = data.LabelField(dtype=torch.float)
#加载数据
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print('len of train data:', len(train_data))#输出训练集大小
print('len of test data:', len(test_data))#输出测试集大小
print('示例文本',train_data.examples[15].text)
print('示例标签',train_data.examples[15].label)
#从测试集中选取部分做验证集
import random
#train_data, valid_data = train_data.split(random_state = random.seed(1234))
test_data, valid_data = test_data.split(random_state = random.seed(1234))
#使用预先训练好的词嵌入来构建词汇表
#vec=Vectors("glove.6B.50d","./vector_cache/")
vec_dim=50
TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.50d")
LABEL.build_vocab(train_data)
print('词汇量大小总量:',len(TEXT.vocab))#词汇量大小总量是max_size+2,因为还会多[pad] [unk]
print('词向量维度:',vec_dim)
batchsz = 32 #批量大小
# 创建数据迭代器
train_iterator, valid_iterator,test_iterator = data.BucketIterator.splits(
    (train_data, valid_data,test_data),
    batch_size = batchsz,
    device=device)
print('训练batch',len(train_iterator)/batchsz, '  验证batch',len(valid_iterator)/batchsz,' 测试batch',len(test_iterator)/batchsz)

#2.搭建循环网络
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        # embedding嵌入层（词向量）
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN变体——双向LSTM
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,# 层数
                            bidirectional=True,#是否双向
                           dropout=0.5)#随机去除神经元
        # 线性连接层
        self.fc = nn.Linear(hidden_dim*2, 1)# 因为前向传播+后向传播有两个hidden sate,且合并在一起,所以乘以2
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedding = self.dropout(self.embedding(x))#text 的形状 [sent len, batch size]
        # embedded 的形状 [sent len, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedding)#output的形状[sent len, batch size, hid dim * num directions]
        # hidden 的形状 [num layers * num directions, batch size, hid dim]
        # cell 的形状 [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        #hidden 的形状 [batch size, hid dim * num directions]
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out

#初始化网络
rnn = RNN(len(TEXT.vocab), vec_dim, 64)
print(rnn)
#输出模型参数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(rnn):,} trainable parameters')

pretrained_embedding = TEXT.vocab.vectors
# 检查词向量形状 [vocab size, embedding dim]，vocab：还包括‘<unk>’,‘<pad>’
print('pretrained_embedding:', pretrained_embedding.shape)# 检查词向量形状 [vocab size, embedding dim]
#将导入的词向量作为embedding.weight的初始值
# 用预训练的embedding词向量替换原始模型初始化的权重参数
rnn.embedding.weight.data.copy_(pretrained_embedding)
#将[pad]和[unk]的词向量初始化为0。
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
rnn.embedding.weight.data[UNK_IDX] = torch.zeros(vec_dim)
rnn.embedding.weight.data[PAD_IDX] = torch.zeros(vec_dim)
print('embedding layer inited.')

#定义优化器
optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
criteon = nn.BCEWithLogitsLoss().to(device)#二分类交叉熵损失BCELoss和sigmoid融合
rnn=rnn.to(device)

def binary_acc(preds, y):#计算二分类正确率
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(rnn, iterator, optimizer, criteon):#定义一个训练函数，用来训练模型
    avg_acc = []
    epoch_loss = 0
    epoch_acc = 0
    rnn.train()#设置模型为训练模式
    for i, batch in enumerate(iterator):
        # [seq, b] => [b, 1] => [b]
        pred = rnn(batch.text).squeeze(1)#预测的标签
        loss = criteon(pred, batch.label)# 计算损失函数值
        acc = binary_acc(pred, batch.label).item()#计算二分类正确率
        avg_acc.append(acc)
        optimizer.zero_grad()#将梯度初始化为零
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
        #if i%10 == 0:
            #print(i, "train-loss=",loss,"train-acc=",acc)
    avg_acc = np.array(avg_acc).mean()
    #print('train: avg acc=', avg_acc)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
def eval(rnn, iterator, criteon):#定义一个测试函数
    avg_acc = []
    epoch_loss = 0
    epoch_acc = 0
    rnn.eval() #设置模型为评估模式
    with torch.no_grad():
        for batch in iterator:
            # [b, 1] => [b]
            pred = rnn(batch.text).squeeze(1)            #
            loss = criteon(pred, batch.label)
            acc = binary_acc(pred, batch.label).item()
            avg_acc.append(acc)
            epoch_loss += loss.item()
            epoch_acc += acc
    avg_acc = np.array(avg_acc).mean()
    #print('>>test: avg_acc=', avg_acc)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

best_valid_loss = float('inf')
num_epochs=10
train_loss_avg = []
train_acc_avg = []
val_loss_avg = []
val_acc_avg = []
for epoch in range(num_epochs):
    #print("epoch:", epoch)
    start_time = time.time()
    train_loss, train_acc =train(rnn, train_iterator, optimizer, criteon)
    valid_loss, valid_acc = eval(rnn, valid_iterator, criteon)
    end_time = time.time()
    print(f'Epoch: {epoch:02} |Epoch Time: {end_time-start_time}')
    # 保留最好的训练结果的那个模型参数，之后加载这个进行预测
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(rnn.state_dict(), 'tut2-model.pt')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    train_loss_avg.append(train_loss)
    val_loss_avg.append(valid_loss)
    train_acc_avg.append(train_acc)
    val_acc_avg.append(valid_acc)

# 组成数据表格train_process输出
train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                       "train_loss_avg": train_loss_avg,
                                       "val_loss_avg": val_loss_avg,
                                       "train_acc_avg": train_acc_avg,
                                       "val_acc_avg": val_acc_avg})
#https://zhuanlan.zhihu.com/p/410587234
#可视化模型训练过程中
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_process.epoch,train_process.train_loss_avg,"ro-",label="Train loss")
plt.plot(train_process.epoch,train_process.val_loss_avg,"bs-",label="Val loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(train_process.epoch,train_process.train_acc_avg,"ro-",label="Train acc")
plt.plot(train_process.epoch,train_process.val_acc_avg,"bs-",label="Val acc")
plt.xlabel("epoch")
plt.ylabel("acc")
plt.legend()
plt.show()

print('__________最终测试结果_____________')
rnn.load_state_dict(torch.load('tut2-model.pt'))
test_loss, test_acc = eval(rnn, test_iterator, criteon)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

print('__________模型验证_____________')#模型验证
import spacy
nlp = spacy.load('en_core_web_sm')

def predict_sentiment(model, sentence):
    rnn.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    #prediction = torch.sigmoid(rnn(tensor, length_tensor))
    prediction = torch.sigmoid(rnn(tensor))
    return prediction.item()
#负面评论的例子：
print("This film is terrible:  ", predict_sentiment(rnn, "This film is terrible"))
#正面评论的例子：
print("This film is great:  ", predict_sentiment(rnn, "This film is great"))
