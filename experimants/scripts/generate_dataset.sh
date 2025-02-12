# humaneval after 183
# nohup python experiments/scripts/generate_workflow.py --dataset HumanEval --initial_round 216 --validation_rounds 3 --random_rate 0.3 &>humaneval.log &
# python experiments/scripts/extract_workflow.py --dataset HumanEval
# collect 213-235

# fix openaibug humaneval before 183 
# nohup python /home/ubuntu/DATA2/yuchenhou/GNN/MetaGPT/experiments/scripts/generate_label.py \
# --workflow_dir experiments/HumanEval/unfinished_workflows_latest \
# --labels_dir experiments/HumanEval/unfinished_workflows_latest_results &>humaneval_unfinished.log
# done

# MMLU after 59 2400 tasks
# nohup python experiments/scripts/generate_workflow.py --dataset MMLU --initial_round 59 --validation_rounds 3 --random_rate 0.3 &>mmlu.log &
# 993083


# hotpotqa_validate.jsonl is filterd 300 tasks, now have no saved graph
# nohup python experiments/scripts/generate_workflow.py --dataset HotpotQA --initial_round 45 --validation_rounds 3 --random_rate 0.3 &>hotpotqa.log &

# python experiments/scripts/extract_workflow.py --dataset HotpotQA

# nohup python experiments/scripts/generate_label.py \
# --dataset HotpotQA \
# --workflow_dir experiments/HotpotQA/unfinished_workflows_latest \
# --labels_dir experiments/HotpotQA/unfinished_workflows_latest_results \
# --validation_rounds 3 &>hotpotqa_unfinished.log &



# math_validate.jsonl is full tasks
nohup python experiments/scripts/generate_workflow.py --dataset MATH --initial_round 137 --validation_rounds 3 --random_rate 0.3  &
158546

nohup python experiments/scripts/generate_label.py \
--dataset MATH \
--workflow_dir experiments/GSM8K/workflows_latest \
--labels_dir experiments/GSM8K/_workflow_results \
--validation_rounds 3 &>math_gsm8k_labels.log &

207621


# mbpp_validate.jsonl is full tasks
# nohup python experiments/scripts/generate_workflow.py --dataset MBPP --initial_round 1 --validation_rounds 3 --random_rate 0.3 &>mbpp.log
# got 17
nohup python experiments/scripts/generate_workflow.py --dataset MBPP  --initial_round 42 --validation_rounds 3 --random_rate 0.3 > /dev/null 2>&1  &
43451

nohup python experiments/scripts/generate_label.py \
--dataset MBPP \
--workflow_dir experiments/MBPP/unfinished_workflows_latest \
--labels_dir experiments/MBPP/unfinished_workflows_latest_results \
--validation_rounds 3 &
206147

# gsm8k 604 tasks
# nohup python experiments/scripts/generate_workflow.py --dataset GSM8K --initial_round 20 \
#  --validation_rounds 3 --random_rate 0.3 \
#  --http_proxy "http://127.0.0.1:7866" &>gsm8k.log &
# 816938
# python experiments/scripts/extract_workflow.py --dataset GSM8K


# nohup python experiments/scripts/generate_label.py \
# --dataset GSM8K \
# --workflow_dir experiments/GSM8K/workflows_from_math \
# --labels_dir experiments/GSM8K/unfinished_workflows_latest_results \
# --validation_rounds 3 &>gsm8k_unfinished.log &
# 847505

# python experiments/scripts/extract_workflow.py --dataset GSM8K \
# --dir_path experiments/GSM8K/workflows_from_math


ps aux | grep "python experiments/scripts/generate_workflow.py" | grep -v grep | awk '{print $2}' | xargs kill