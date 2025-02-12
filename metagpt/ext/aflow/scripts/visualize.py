

import ast
import networkx as nx
import matplotlib.pyplot as plt
import os

def extract_variable_names(node):
    """
    递归提取表达式中的变量名，包括列表、字典和嵌套结构
    """
    variable_names = []

    if isinstance(node, ast.Name):
        # 单个变量名
        variable_names.append(node.id)

    elif isinstance(node, ast.Subscript):
        # 下标访问，例如 initial_solution['response']
        if isinstance(node.value, ast.Name):
            variable_names.append(node.value.id)  # 提取下标前的变量名
        # 继续递归解析下标中的其他部分（例如 keys 或 indices）
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                variable_names.extend(extract_variable_names(value))

    elif isinstance(node, ast.List):
        # 列表解析
        for element in node.elts:
            variable_names.extend(extract_variable_names(element))

    elif isinstance(node, ast.Dict):
        # 字典解析
        for value in node.values:
            variable_names.extend(extract_variable_names(value))

    elif isinstance(node, (ast.BinOp, ast.Call, ast.Attribute, ast.JoinedStr, ast.FormattedValue)):
        # 操作符、调用、属性访问、字符串拼接等，递归子节点
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    variable_names.extend(extract_variable_names(item))
            elif isinstance(value, ast.AST):
                variable_names.extend(extract_variable_names(value))
    elif isinstance(node, ast.JoinedStr):
        # 处理格式化字符串 f"{var}"
        for value in node.values:
            if isinstance(value, ast.FormattedValue):
                variable_names.extend(extract_variable_names(value.value))
    return variable_names


class CallGraphParser(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()  # 有向图
        self.current_function = None  # 当前函数上下文
        self.node_counter = {}  # 记录不同调用的计数
        self.variable_sources = {}  # 记录变量的来源（哪个节点生成）
        self.instance_mapping = {}  # 记录实例化的对象及其类来源

    def visit_FunctionDef(self, node):
        """处理函数定义"""
        if node.name == "__init__":
            self.current_function = "__init__"
            self.generic_visit(node) 
            return

        if node.name == "__call__":
            self.current_function = node.name
            # 找到problem参数
            for arg in node.args.args:
                if arg.arg == "problem":
                    self.graph.add_node("problem", type="root")
            self.generic_visit(node)  

    def visit_AsyncFunctionDef(self, node):
        """处理异步函数定义"""
        self.visit_FunctionDef(node)

    def visit_Assign(self, node):
        """
        解析赋值语句：记录变量与函数调用关系
        """ 
        # 左侧变量名
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            output_var = node.targets[0].id
        else:
            self.generic_visit(node)
            return
        
        if isinstance(node.value, ast.Call):
            # 处理类实例化，如 self.custom = operator.Custom(self.llm)
            if self.current_function == "__init__":
                self._handle_instance_assignment(output_var, node.value)

        # 右侧是否是调用
        if isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call):
            self._handle_call(output_var, node.value.value)
        elif isinstance(node.value, ast.Call):
            self._handle_call(output_var, node.value)
        
        elif isinstance(node.value, ast.BinOp) or isinstance(node.value, ast.JoinedStr):
            # 处理拼接变量
            inputs = extract_variable_names(node.value)
            for input_var in inputs:
                if input_var in self.variable_sources:
                    self.graph.add_edge(self.variable_sources[input_var], output_var, type="combine")
                else:
                    self.graph.add_edge("problem", output_var, input_var=input_var)

    def _handle_instance_assignment(self, target_var, call_node):
        """
        处理对象实例化，记录实例与类的映射关系
        """
        if isinstance(call_node.func, ast.Attribute):
            # 处理如 operator.Custom
            class_name = f"{call_node.func.value.id}.{call_node.func.attr}"
        elif isinstance(call_node.func, ast.Name):
            # 处理直接调用的类名
            class_name = call_node.func.id
        else:
            return

        self.instance_mapping[target_var] = class_name


    def _handle_call(self, output_var, call_node):
        """
        处理调用节点，提取调用的输入和函数名
        """
        # 提取被调用方法名
        if isinstance(call_node.func, ast.Attribute):
            callee = call_node.func.attr
            if isinstance(call_node.func.value, ast.Name):
                instance_name = call_node.func.value.id
                if instance_name in self.instance_mapping:
                    # 用完整类名替换节点名
                    callee = f"{self.instance_mapping[instance_name]}.{callee}"
                else:
                    callee = f"{instance_name}.{callee}"

            callee = f"{callee}_{self._get_node_count(callee)}"

        elif isinstance(call_node.func, ast.Name):
            callee = call_node.func.id
        else:
            print(f"Unsupported call node: {ast.dump(call_node)}")
            return  # 无法解析

        inputs = []
        for arg in call_node.keywords:  # 遍历函数的关键字参数
            inputs.extend(extract_variable_names(arg.value))  # 提取变量名

        # 添加调用节点
        self.graph.add_node(callee, type="call")
        # if self.current_function:
        #     self.graph.add_edge(self.current_function, callee, type="calls")

        # 添加输入变量的边
        for input_var in inputs:
            if input_var in self.variable_sources:
                self.graph.add_edge(self.variable_sources[input_var], callee, input_var=input_var)
            else:
                self.graph.add_edge("problem", callee, input_var=input_var, output_var=output_var)

        # 更新输出变量来源
        self.variable_sources[output_var] = callee

    def _get_node_count(self, callee):
        """
        获取节点计数器，用于区分不同的 self.custom 调用
        """
        if callee not in self.node_counter:
            self.node_counter[callee] = 0
        self.node_counter[callee] += 1
        return self.node_counter[callee]

    def parse(self, filename):
        """解析文件并生成图"""
        with open(filename, "r", encoding="utf-8") as file:
            try:
                tree = ast.parse(file.read())
                self.visit(tree)
            except SyntaxError as e:
                print(f"Syntax error in {filename}: {e}")


    def extract_graph_data(self):
        """提取节点和边数据"""
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))
        return nodes, edges
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
    pos = root_based_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=2000,
        font_size=10,
        font_weight="bold",
        arrowsize=20,
    )

    # 标注边属性
    edge_labels = nx.get_edge_attributes(graph, "output_var")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    if output_file:
        plt.savefig(output_file, format="png")
        print(f"Graph saved as {output_file}")
    else:
        plt.show()



dir_path = "./metagpt/ext/aflow/scripts/optimized/HotpotQA/workflows"
for sub_dir in os.listdir(dir_path):
    if not os.path.isdir(os.path.join(dir_path, sub_dir)):
        print(f"Skipping non-directory {sub_dir}")
        continue
    filepath = os.path.join(dir_path, sub_dir, 'graph.py')
    save_path = os.path.join(dir_path, sub_dir, 'call_graph_with_attributes.png')
    parser = CallGraphParser()

    parser.parse(filepath)

    # 提取图数据
    nodes, edges = parser.extract_graph_data()
    # print("Nodes:", nodes)
    # print("Edges:", edges)

    # 可视化调用图
    visualize_graph(parser.graph, output_file=save_path)
