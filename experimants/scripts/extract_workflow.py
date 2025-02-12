

import ast
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
import shutil
import importlib

def parse_args():
    parser = argparse.ArgumentParser(description='Extract workflow from Python code')
    parser.add_argument('--dir_path', type=str, default=None, help='specific directory path')
    parser.add_argument('--dataset', type=str, default='HumanEval', help='dataset name')
    parser.add_argument('--original_dir', type=str, default='metagpt/ext/aflow/scripts/optimized', help='original directory')
    parser.add_argument('--output_base_dir', type=str, default='experiments', help='output directory')

    return parser.parse_args()
    

def root_based_layout(graph, root_node="problem"):
    """
    从指定的根节点出发布局有向图
    """
    # 确保图中有根节点
    if root_node not in graph:
        raise ValueError(f"Root node '{root_node}' not found in the graph.")

    # 使用 BFS 确定节点层次
    levels = nx.single_source_shortest_path_length(graph, root_node)
    max_level = max(levels.values())

    # 分层布局
    pos = {}
    width_per_level = 1.0 / (max_level + 1)
    level_nodes = {level: [] for level in range(max_level + 1)}

    for node, level in levels.items():
        level_nodes[level].append(node)

    for level, nodes in level_nodes.items():
        y_pos = 1.0 - level * width_per_level
        for i, node in enumerate(nodes):
            x_pos = (i + 1) / (len(nodes) + 1)
            pos[node] = (x_pos, y_pos)

    return pos

def visualize_graph(graph, output_file=None):
    """使用 NetworkX 可视化调用图"""
    plt.figure(figsize=(12, 8))
    # pos = root_based_layout(graph)
    pos = nx.spring_layout(graph, k=0.5, iterations=100)
    node_colors = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "list":
            node_colors.append("lightgreen")
        elif data.get("type") == "call":
            node_colors.append("lightblue")
        elif data.get("type") == "joined":
            node_colors.append("lightcoral")
        elif data.get("type") == "root":
            node_colors.append("red")
        else: 
            node_colors.append("lightgray")  
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=2000,
        font_size=10,
        font_weight="bold",
        arrowsize=20,
    )

    # 标注边属性
    edge_labels_dict = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        instruction = data.get("instruction", "")
        if instruction:
            edge_labels_dict[(u, v)].append(instruction)
    
    # Combine labels for multiple edges
    combined_edge_labels = {k: ", ".join(v) for k, v in edge_labels_dict.items()}
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=combined_edge_labels,
        font_size=8
    )
    if output_file:
        plt.savefig(output_file, format="png")
        print(f"Graph saved as {output_file}")
    else:
        plt.show()

def copy_if_not_exists(src, dst):
    """
    Copies a file from src to dst only if dst does not already exist.
    """
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

if __name__ == "__main__":
    args = parse_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = args.dataset
    if args.dir_path is not None:
        dir_path = args.dir_path
    else:
        dir_path = os.path.join(args.original_dir, dataset, f"workflows")
    output_dir = os.path.join(args.output_base_dir, dataset, "history", f"workflows_{current_time}")
    unfinished_dir = os.path.join(args.output_base_dir, dataset, "history", f"unfinished_workflows_{current_time}")
    latest_dir = os.path.join(args.output_base_dir, args.dataset, 'workflows_latest')
    latest_unfinished_dir = os.path.join(args.output_base_dir, args.dataset, 'unfinished_workflows_latest')
    latest_unfinished_results_dir = os.path.join(args.output_base_dir, args.dataset, 'unfinished_workflows_latest_results')
    os.makedirs(output_dir)
    os.makedirs(unfinished_dir)
    shutil.rmtree(latest_dir, ignore_errors=True)
    shutil.rmtree(latest_unfinished_dir, ignore_errors=True)
    if not os.path.exists(latest_dir):
        os.makedirs(latest_dir)
    if not os.path.exists(latest_unfinished_dir):
        os.makedirs(latest_unfinished_dir)
    count = 0
    for sub_dir in os.listdir(dir_path):
        # if not sub_dir == "round_20":
        #     continue
        if not sub_dir.startswith("round_"):
            continue
        flag = True
        if not os.path.isdir(os.path.join(dir_path, sub_dir)):
            print(f"Skipping non-directory {sub_dir}")
            flag = False
        filepath = os.path.join(dir_path, sub_dir, 'graph.py')
        if not os.path.exists(filepath):
            print(f"Skipping non-exist file {filepath}")
            flag = False
        

        for file in os.listdir(os.path.join(dir_path, sub_dir)):
            if file.startswith("log") or file.startswith("experience"):
                continue
            if file.endswith(".json") and (file.startswith("0.") or file.startswith("1.")):
                score = float(file.split("_")[0])
                if score != 0.0:
                    flag = True
                    
                    break

            flag = False
            
        if flag:
            with open(os.path.join(dir_path, sub_dir, file), "r") as f:
                        records = json.load(f)
            # count specific strings numbers in records
            item_count = 0
            for record in records:
                if "<html>\r\n<head><title>429 Too Many Requests<" in record['prediction'] or "Timeout" in record['prediction']:
                    item_count += 1
            if item_count > 0.3 * len(records) and not os.path.exists(os.path.join(latest_unfinished_results_dir, sub_dir)):
                #copy to unfinished dir
                shutil.copytree(os.path.join(dir_path, sub_dir), os.path.join(unfinished_dir, sub_dir))
                # os.system(f"cp -r {os.path.join(unfinished_dir, sub_dir)} {latest_unfinished_dir}/")
                # copy but skip existing files
                shutil.copytree(os.path.join(dir_path, sub_dir), 
                                os.path.join(latest_unfinished_dir, sub_dir),
                                copy_function=copy_if_not_exists,
                                dirs_exist_ok=True,)
                
            else:                                
                # copy 
                if item_count <= 0.3 * len(records):
                    shutil.copytree(os.path.join(dir_path, sub_dir), os.path.join(output_dir, sub_dir))
                else:
                    # replace the score_**.json with unfinished_results
                    shutil.copytree(os.path.join(latest_unfinished_results_dir, sub_dir), os.path.join(output_dir, sub_dir))
                    # copy but skip score_**.json files
                    for file in os.listdir(os.path.join(dir_path, sub_dir)):
                        if file.endswith(".json"):
                            if file.startswith("log") or file.startswith("experience"):
                                shutil.copy2(os.path.join(dir_path, sub_dir, file), os.path.join(output_dir, sub_dir))
                        elif os.path.isfile(os.path.join(dir_path, sub_dir, file)):
                            shutil.copy2(os.path.join(dir_path, sub_dir, file), os.path.join(output_dir, sub_dir))
                    
            
                # extract workflow
                module_name = f"utils.extract_{dataset}_workflow"
                try:
                    module = importlib.import_module(module_name)
                    CallGraphParser = getattr(module, "CallGraphParser")
                except:
                    print(f"Failed to import {module_name}")
                parser = CallGraphParser(filepath)
                parser.parse(filepath)

                # 提取图数据
                nodes, edges = parser.extract_graph_data()
                workflow_save_path = os.path.join(output_dir, sub_dir, 'workflow.json')
                with open(workflow_save_path, "w") as f:
                    json.dump({"nodes": nodes, "edges": edges}, f,ensure_ascii=False, indent=4)
                    f.write("\n")
            

                # 可视化调用图
                fig_path = os.path.join(output_dir, sub_dir, 'call_graph_with_attributes.png')
                visualize_graph(parser.graph, output_file=fig_path)
                shutil.copytree(os.path.join(output_dir, sub_dir), 
                                    os.path.join(latest_dir, sub_dir),
                                    copy_function=copy_if_not_exists,
                                    dirs_exist_ok=True,)
                count += 1
                
            
    print(f"Extracted {count} workflows")
    print(f"left {len(os.listdir(unfinished_dir))} unfinished workflows")

        
       
