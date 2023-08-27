import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'news': 'ratings.txt', 'music': 'user_artists.dat'})
SEP = dict({'movie': '::', 'book': ';', 'news': '\t', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0, 'music': 0})


def read_item_index_to_entity_id_file():
    # file = '../data/' + DATASET + '/item_index2entity_id_rehashed.txt'  # 当数据集是book或movie时
    file = '../data/' + DATASET + '/item_index2entity_id.txt'     # 当数据集是music时

    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i      # item_index_old2new = {dict:3846}
        entity_id2index[satori_id] = i      # entity_id2index = {dict:3846}
        i += 1


def convert_rating():
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())     # item_set = {set:3846}
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])    # array = {list:3}

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]       # array[1]是artistID
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]     # 根据item_index_old，在item_index_old2new找到对应的值

        user_index_old = int(array[0])     # array[0]是userID

        rating = float(array[2])    # array[2]是weight
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)    # user_pos_ratings = {dict:(1872)}
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)    # user_neg_ratings = {dict:0}

    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():      # 遍历对正样本的user和它的item
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]     # 得到这个user_index

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set     # 暂时还没确定的，包含有正样本和负样本
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):   # 随机抽取和正样本数量相同的，并且不重复
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)     # entity_cnt: 3846
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if DATASET == 'movie':
        files.append(open('../data/' + DATASET + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('../data/' + DATASET + '/kg_part2_rehashed.txt', encoding='utf-8'))
    elif DATASET == 'book':
        files.append(open('../data/' + DATASET + '/kg_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('../data/' + DATASET + '/kg.txt', encoding='utf-8'))  # 当数据集是music时，打开对应数据集的kg.txt

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:     # entity_id2index = {dict:3846}
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]        # 得到的head是entity_id2index的value值

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]        # 得到的tail是entity_id2index的value值

            if relation_old not in relation_id2index:       # relation_id2index = {dict:}
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))   # 将头实体、关系、尾实体写入

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')
