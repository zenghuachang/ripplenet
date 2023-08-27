import numpy as np
import torch

from model import RippleNet


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)   # 相当于模型的初始化
    if args.use_cuda:     # 使用cude
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )       # filter(lambda p: p.requires_grad, model.parameters())中   lambda p: p.requires_grad就是以p为参数的满足p.requires_grad的true的条件的函数
            # 而参数p赋值的元素从列表model.parameters()里取，所以只取param.requires_grad = True（模型参数的可导性是true的元素），就过滤掉为false的元素

    for step in range(args.n_epoch):    # 迭代训练的次数，n_epoch=10
        # training
        np.random.shuffle(train_data)   # 将训练数据随机打乱
        start = 0
        while start < train_data.shape[0]:    # 当start小于训练数据的行数，每一次start=start+batch_size，train_data.shape[0] = 25398
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))    # return_dict = {dict:5}
            loss = return_dict["loss"]      # 取return_dict的loss     loss = base_loss + kge_loss + l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size    # batch_size = 1024
            if show_loss:       # 是否打印损失值的信息
                print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))

        # evaluation
        train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
        eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)

        print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))


def get_feed_dict(args, model, data, ripple_set, start, end):
    items = torch.LongTensor(data[start:end, 1])        # items={Tensor:(1024,)}
    labels = torch.LongTensor(data[start:end, 2])       # labels={Tensor:(1024,)}
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):     # n_hop=2
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))    # memories_h = {list:2}  0或1 = {Tensor:(1024,32)}
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))    # memories_r = {list:2}  0或1 = {Tensor:(1024,32)}
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))    # memories_t = {list:2}  0或1 = {Tensor:(1024,32)}
    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))
    return items, labels, memories_h, memories_r,memories_t     # 返回的这些都是带cuda的


def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()    # 模型的eval
    while start < data.shape[0]:    # 所选择(训练，评估，测试)数据的行数
        auc, acc = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        # get_feed_dict(args, model, data, ripple_set, start, start + batch_size)  return items, labels, memories_h, memories_r, memories_t  # 返回的这些都是带cuda的
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size         # start = start + batch_size
    model.train()   # 模型的train
    return float(np.mean(auc_list)), float(np.mean(acc_list))
