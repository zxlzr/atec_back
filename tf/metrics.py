import numpy as np
from collections import defaultdict

def round3(x):
    return round(x, 3)

def round4(x):
    return round(x, 4)


def cal_prob(logits):
    b = np.exp(logits)
    s = np.sum(b, axis=1)
    prob = b[:, 1] / s
    prob = prob.tolist()
    return map(round3, prob)


def cal_avg_precision(pred_list, real_list):
    ap = []
    len_real_list = len(real_list)
    hit_index, cur_index = 0.0, 0
    for p in pred_list:
        cur_index += 1
        if p in real_list:
            hit_index += 1
            ap.append(hit_index / cur_index)
        else:
            ap.append(0)
    return np.sum(ap) / len_real_list


def cal_contains(pred_list, real_list):
    for r in real_list:
        if r in pred_list:
            return 1
    return 0


def process_predict_result(prediction_path, first_n=3):
    all_scenes_path = "./data/all_scene.csv"
    test_goods_path = "./data/test.csv"
    all_scenes = []
    f = open(all_scenes_path)
    for line in f:
        scene_word = line.strip()
        all_scenes.append(scene_word)

    id2title = {}
    real_scene = defaultdict(list)
    f = open(test_goods_path)
    for line in f:
        parts = line.strip().split("\t")
        item_id, title, scene = parts[:3]
        id2title[item_id] = title
        real_scene[item_id].append(scene)

    avg_precisions = []
    avg_contains = []
    processed_path = "./prediction/result.csv"
    with open(prediction_path) as f, open(processed_path, "w") as g:
        for line in f:
            parts = line.strip().split(",")
            item_id = parts[0]
            probs = map(float, parts[1:])
            pairs = zip(probs, all_scenes)
            pairs = sorted(pairs, key=lambda x: x[0], reverse=True)[:first_n]

            new_line = item_id + ": " + id2title[item_id] + "; \t"
            for p in pairs:
                new_line += p[1] + ", "
            new_line += "\t|\t"
            for s in real_scene[item_id]:
                new_line += s + ", "
            g.write(new_line + "\n")

            pred_scene = map(lambda x: x[1], pairs)
            avg_precisions.append(cal_avg_precision(pred_scene, real_scene[item_id]))
            avg_contains.append(cal_contains(pred_scene, real_scene[item_id]))
    print "mean_avg_precision: ", np.mean(avg_precisions)
    print "hit@{}: ".format(first_n), np.mean(avg_contains)
