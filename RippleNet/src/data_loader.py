import collections
import os
import numpy as np

def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)     # user_history_dict作为输入
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)    # 这里的kg和user_history_dict都应该关系到输入
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set

def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '../data/' + args.dataset + '/ratings_final'  # 评分文件,ratings_final.txt是在preprocess.py内convert_rating()函数写入的
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')  # rating_np：ndarray{42346,3} 用户-物品-评分
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np)


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]  # 找出评分文件里的数量(个数) 42346

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)  # ndarray:{8469,}评价的索引(行)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False) # ndarray:{8469,}测试的索引(行)
    train_indices = list(left - set(test_indices))  # ndarray:{25408,}训练的索引(行)
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:     # 遍历训练数据的索引(行)
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)    # user_history_dict:{dict:1864}，将rating为1的item添加到相应的user里

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]  # train_indices:{list:25398}，就是在rating为1的基础上找的索引(行)
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]    # eval_indices:{list:8455}，就是在rating为1的基础上找的索引(行)
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]    # test_indices:{list:8457}，就是在rating为1的基础上找的索引(行)
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]   # train_data:{ndarray:(25398,3)}，就是在rating为1的基础上找的，作为训练数据
    eval_data = rating_np[eval_indices]     # eval_data:{ndarray:(8455,3)}，就是在rating为1的基础上找的，作为评价数据
    test_data = rating_np[test_indices]     # test_data:{ndarray:(8457,3)}，就是在rating为1的基础上找的，作为测试数据

    return train_data, eval_data, test_data, user_history_dict  # user_history_dict:{dict:1864},训练数据里，用户交互过(rating为1)的所有item


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'  # 知识图谱文件,kg_final.txt是在preprocess内convert_kg()函数写入的
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')   # kg_np = {ndarray:(15518,3)}   知识图谱的三元组
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))   # n_entity:9366，找出实体的数量(包括头实体和尾实体)
    n_relation = len(set(kg_np[:, 1]))  # n_entity:60，找出关系的数量

    kg = construct_kg(kg_np)    # kg = {defaultdict:3846} defaultdict(<class 'list'>) 例如=2086: [(3846, 0)] 根据kg_np构造知识图谱

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:  # 这里的kg_np为{ndarray:(15518,3)}，其中的头实体有可能是重复的
        kg[head].append((tail, relation))   # 以头实体为准,将其所关联的尾实体和关系加进来
      #  kg[tail].append((head,relation))   # train auc acc和eval auc acc都有所提高，test auc acc有所下降
    return kg


def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:  # 遍历交互过item的用户
        for h in range(args.n_hop): # 几跳，循环几次
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:      # 如果是第0跳，也就是刚开始时
                tails_of_last_hop = user_history_dict[user]     # tails_of_last_hop为list类型,将该用户交互过的item作为头实体
            else:
                tails_of_last_hop = ripple_set[user][-1][2]     # tails_of_last_hop为list类型,将该用户最后一次交互过的尾实体作为下一跳的头实体

            for entity in tails_of_last_hop:    # 遍历该用户交互过的item或头实体
                for tail_and_relation in kg[entity]:   # 遍历知识图谱上该头实体的尾实体和关系
                    memories_h.append(entity)    # 将该头实体添加进列表
                    memories_r.append(tail_and_relation[1])     # 将该关系添加进列表
                    memories_t.append(tail_and_relation[0])     # 将该尾实体添加进列表

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:    # 如果头实体的列表为空
                ripple_set[user].append(ripple_set[user][-1])     # 填充
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory   # 如果头实体的数量小于32
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)    # 根据头实体数量的长度，取32个索引，允许重复
                memories_h = [memories_h[i] for i in indices]   # 遍历索引，抽取头实体
                memories_r = [memories_r[i] for i in indices]   # 遍历索引，抽取关系
                memories_t = [memories_t[i] for i in indices]   # 遍历索引，抽取尾实体
                ripple_set[user].append((memories_h, memories_r, memories_t))   # 将头实体，关系，尾实体添加到该用户内

    return ripple_set   # defaultdict:1864 (<class 'list'>)     用户多少跳的三元组集合
