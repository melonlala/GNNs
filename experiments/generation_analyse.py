import os
import json
import numpy as np
# from GDesigner.utils.globals import Time
import time
import matplotlib.pyplot as plt 

data_dir = './simulations/eval/full_humaneval/'
# data_dir = 'simulations/eval/filter_results/'
data_dir_name    = os.path.basename(os.path.normpath(data_dir))
task_file = 'datasets/humaneval/humaneval-py.jsonl'
current_time = time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime())
save_dir = f'datasets/humaneval/{data_dir_name}_{current_time}'
os.makedirs(save_dir, exist_ok=True)
pic_path = f'{save_dir}/rank_matrix.png'
save_chosen_file = f'{save_dir}/chosen_tasks.jsonl'
save_false_file = f'{save_dir}/all_false_tasks.jsonl'
save_true_file = f'{save_dir}/all_true_tasks.jsonl'
data = []
label = []
question_name = []
overlap = {}
all_true = 0
all_false = 0
true_and_false = 0
true_and_false_samples = []
true_and_false_question = []
true_samples = []
false_samples = []
true_question = []
false_question = []
num_workflows = 0
for file in os.listdir(data_dir):
    if file.endswith('.json'):
        file_path = os.path.join(data_dir, file)
        num_workflows += 1
        with open(file_path, 'r') as f:
            data += json.load(f)    

print("Number of workflows: ", num_workflows)

for item in data:
    label.append(item['Solved'])
    question_name.append( item['Question']['name'])
    if item['Question']['name'] in question_name and item["Question"]['name'] not in overlap.keys():
        overlap[item['Question']['name']] = [item['Solved']]
    elif item['Question']['name'] in question_name and item["Question"]['name'] in overlap.keys():
        overlap[item['Question']['name']].append(item['Solved'])
num_questions = len(overlap)
print("Number of questions: ", num_questions)

# for item in overlap.items():
#     print(item[0], ":", len(item[1]))
# mat = np.zeros((len(overlap), 10))

mat = np.zeros((num_questions, num_workflows))
print(mat.shape)
for i, item in enumerate(overlap.items()):
    try:
        mat[i] = item[1]
    except:
        print(item[0], ":", len(item[1]))
    if len(item[1]) != num_workflows:
        # padd with 0.5
        mat[i][len(item[1]):] = 0.5
        print(item[0], ":", sum(item[1]))
    if sum(item[1]) > num_workflows*0.9:
        all_true += 1
        true_samples.append(sum(item[1])/10)
        true_question.append(item[0])
    elif sum(item[1]) < num_workflows*0.1:
        all_false += 1
        false_samples.append(sum(item[1])/10)
        false_question.append(item[0])
    else:
        true_and_false += 1
        true_and_false_samples.append(sum(item[1])/10)
        true_and_false_question.append(item[0])
with open(task_file, 'r') as f1:
    with open(save_chosen_file, 'a') as f2:
        with open(save_false_file, 'a') as f3:
            with open(save_true_file, 'a') as f4:
                for line in f1:
                    item = json.loads(line)
                    if item['name'] in true_and_false_question:
                        f2.write(line)
                    elif item['name'] in false_question:
                        f3.write(line)
                    elif item['name'] in true_question:
                        f4.write(line)
# mat = np.zeros((num_workflows, num_questions))
# print(mat.shape)
# for i, item in enumerate(overlap.items()):
#     for j, value in enumerate(item[1]):
#         mat[j][i] = value


fig, ax = plt.subplots(figsize=(20,10))  # 设置图片尺寸
im = ax.imshow(mat, cmap='YlOrRd')

# 使用 question_name 作为 x 轴标签
plt.xlabel('Workflow ID')
plt.xticks(np.arange(num_workflows))  # 设置 x 轴为问题名称并旋转标签

# 将 y 轴标签设置为 workflow id
plt.ylabel("Question ID")
plt.yticks(np.arange(num_questions))

# plt.colorbar(im)  # 可选：添加颜色条
plt.tight_layout()  # 可选：自动调整布局以避免标签重叠
plt.savefig(pic_path)
rank = np.linalg.matrix_rank(mat)
print("Rank: ", rank)
# import pdb; pdb.set_trace()

print("All True: ", all_true)
print("All False: ", all_false)
print("True and False: ", true_and_false)
# import pdb; pdb.set_trace()