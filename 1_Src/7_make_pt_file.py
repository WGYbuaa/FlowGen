# 制作pt文件，用于模型训练
from utils import read_uc_from_stand_json,print_location,check_all_under_same_method,mix_items,unify_ncet_format
from imports import torch,re,ARGC_20,PUB_GROUPING_UC_20_1, string,chain,Data,MODEL,Counter,GROUPING_UC_20

def local_id_to_global_id(uc_list):
    id = 0
    for uc in uc_list:
        uc['global id'] = id
        id += 1
    return uc_list

# 定义一个函数来分割字符串
def split_string(s, delimiters):
    # 先将字符串中的分隔符替换为统一的分隔符（这里选择英文逗号）
    for delimiter in delimiters:
        s = s.replace(delimiter, ',')
    # 然后使用统一的分隔符来分割字符串
    return s.split(',')

def is_pure_punctuation(s):
    # 去除字符串首尾的空白字符
    s = s.strip()
    # 检查字符串是否为空或者是否只包含标点符号
    return s != "" and all(char in string.punctuation for char in s)




def check_agc(uc_list):
    # 统计所有node数目 act/obj/key
    if isinstance(uc_list[1]["AF act"][0], list): # 如果是嵌套列表，即node已经按照step分好了。
        node_bf_act, node_bf_obj, node_key,node_af_act, node_af_obj = [], [],[],[], []
        for uc in uc_list:
            # 1、收集 Basic Flow 的 act 、 obj node
            for step_act, step_obj in zip(uc['BF act'], uc['BF obj']):
                for act, obj in zip(step_act, step_obj):
                    if act not in node_bf_act:
                        node_bf_act.append(act)
                    if obj not in node_bf_obj:
                        node_bf_obj.append(obj)
            # 2、收集 Alternative Flow 的 act 、 obj node
            if 'AF act' in uc and len(uc['AF act']) != 0:  # 有分支流的情况下
                for step_af_act, step_af_obj in zip(uc['AF act'], uc['AF obj']):
                    for act_af,obj_af in zip(step_af_act,step_af_obj):
                        if isinstance(act_af,str) and isinstance(obj_af,str): # 如果是一级列表,这里就是act/obj的字符串了。ncet 不用断句，本身就是一级列表。
                            if act_af not in node_af_act:  # 选择修改ncet数据集的格式，所以应该不会进入这个分支了
                                node_af_act.append(act_af)
                            if obj_af not in node_af_obj:
                                node_af_obj.append(obj_af)
                        else:
                            for act_seg,obj_seg in zip(act_af,obj_af): # 断句后的
                                if act_seg not in node_af_act:
                                    node_af_act.append(act_seg)
                                if obj_seg not in node_af_obj:
                                    node_af_obj.append(obj_seg)

            # 3、收集各类 key node
            if 'key_act' in uc:
                if isinstance(uc['key_act'], str):
                    uc['key_act'] = splite_2_list_en(uc['key_act'])  # 从str to list
                for item in uc['key_act']:
                    if item not in node_key and item != "None":
                        node_key.append(item)
            
            if 'key_obj' in uc:
                if isinstance(uc['key_obj'], str):
                    uc['key_obj'] = splite_2_list_en(uc['key_obj'])
                for item in uc['key_obj']:
                    if item not in node_key and item != "None":
                        node_key.append(item)

            if 'key_name' in uc:
                if isinstance(uc['key_name'], str):
                    uc['key_name'] = splite_2_list_en(uc['key_name'])
                for item in uc['key_name']:
                    if item not in node_key:
                        node_key.append(item)

            if 'key_path' in uc:
                if isinstance(uc['key_path'], str):
                    uc['key_path'] = splite_2_list_en(uc['key_path'])
                for item in uc['key_path']:
                    if item not in node_key:
                        node_key.append(item)

            if 'keyword' in uc:
                for item in uc['keyword']:
                    if item not in node_key:
                        node_key.append(item)

    else:
        print_location() # 定位

    node_total = len(node_bf_act) + len(node_bf_obj) + len(node_key) + len(node_af_act) + len(node_af_obj)

    return {'node_total': node_total, 'node_bf_act': len(node_bf_act), 
    'node_bf_obj': len(node_bf_obj), 'node_key': len(node_key),
    'node_af_act': len(node_af_act), 'node_af_obj': len(node_af_obj)}

def group_uc_fixed(use_case_list, group_uc):
    use_case_list_new = []
    for index_list in group_uc:
        sub_uc_list = []
        for index in index_list:
            sub_uc_list.append(use_case_list[index])
        use_case_list_new.append(sub_uc_list)
    return use_case_list_new

def get_edges_dict_and_node_to_UCText_only_rgat(use_case_list):
    edges_dict_list = []
    node_to_UCText_list = []
    for uc_list in use_case_list:
        edge_dict, node_to_UCText = create_edges_af(uc_list)
        edges_dict_list.append(edge_dict)
        node_to_UCText_list.append(node_to_UCText)

    return edges_dict_list, node_to_UCText_list

# 拉平嵌套列表
def flatten_list(nested_list, flat_list=None):
    if flat_list is None:
        flat_list = []

    for item in nested_list:
        if isinstance(item, list):
            # 如果item是列表，则递归调用flatten_list
            flatten_list(item, flat_list)
        else:
            # 如果item不是列表，则将其添加到flat_list中
            flat_list.append(item)

    return flat_list

# 参考实验室电脑 utils_2nd.py/create_edges_llm_only_rgat()
def create_edges_af(uc_list):
    edges_dict = {}
    node_to_UCText = {}  # 用于存放keyword、act、obj节点与UCText的对应关系
    for uc in uc_list:
        # 1. Make "keyword-keyword" edge(考虑了与前后node均相连)
        edges_dict = make_key_key_edge(uc, edges_dict)

        # 2. Add "keyword-act" edges, "keyword-obj" edges
        edges_dict = make_key_act_obj(uc, edges_dict, 'BF act', 'BF obj')
        edges_dict = make_key_act_obj(uc, edges_dict, 'AF act', 'AF obj')

        # 3. Add "act-act" edges (只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = make_common_edges(uc['BF act'], edges_dict, "BF act")
        if 'AF act' in uc :
            edges_dict = make_common_edges(uc['AF act'], edges_dict, "AF act")

        # 4. Add "obj-obj" edges(只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = make_common_edges(uc['BF obj'], edges_dict, 'BF obj')
        if 'AF obj' in uc :
            edges_dict = make_common_edges(uc['AF obj'], edges_dict, 'AF obj')

        # 5. Add "act-obj" edge
        edges_dict = make_act_obj_edges_miss(uc, edges_dict,'BF act','BF obj' )
        edges_dict = make_act_obj_edges_miss(uc, edges_dict,'AF act','AF obj')


        # 6. create (node to uctext)'s map (其实是bf、af node to key node)
        node_to_UCText = create_node_uctext(node_to_UCText, uc)



    return edges_dict, node_to_UCText

# 制作act-act 或者 obj-obj 的边（PUB DATASET用这个,其他也可能用到）
def make_common_edges(node_list, edges_dict, label):
    node_list = flatten_list(node_list)
    for node in node_list:
        if (node, label) not in edges_dict.keys():
            edges_dict[(node, label)] = []

    # 只链接node后一个act/obj即可，不管前一个(与keyword不同，感觉应该考虑一下是否应该考虑前一个node)
    for i in range(len(node_list)):
        if i < len(node_list) - 1:
            # 添加后一个元素到当前元素的相邻列表中
            edges_dict[(node_list[i], label)].append((node_list[i + 1], label))

    return edges_dict



def make_key_key_edge(uc, edges_dict):
    # only rgat方法中，keyword就是全部链接起来即可，不分act/obj。
    # 一个keyword node链接前后的keyword node即可。
    keyword_list = []
    for key in ["key_name", "key_path",'key_act','key_obj']:
        if key in uc.keys():
            keyword_list = list(chain(keyword_list, uc[key]))

    # 1、创建source node(dict 的 key)
    for item in keyword_list:
        if (item, 'keyword') not in edges_dict:
            edges_dict[(item, 'keyword')] = []
    # 2、遍历列表，为每个node找到相邻的node
    for i in range(len(keyword_list)):
        # if i > 0:  # 因为后面变成无向边了（有向边反转），所以不需要添加前一个元素到当前元素了
        #     # 添加前一个元素到当前元素的相邻列表中
        #     edges_dict[(keyword_list[i], 'keyword')].append((keyword_list[i - 1], 'keyword'))
        if i < len(keyword_list) - 1:
            # 添加后一个元素到当前元素的相邻列表中
            edges_dict[(keyword_list[i], 'keyword')].append((keyword_list[i + 1], 'keyword'))

    return edges_dict

# 制作所有"keyword-act" edges, "keyword-obj" edges。其中act/obj 没有按照step 存放
def make_key_act_obj(uc, edges_dict, act_key, obj_key):
    # 先不考虑去重，先把边都加上（不去重也可以帮助后续数）
    keyword_list = []
    for key in ["key_name", "key_path",'key_act','key_obj']:
        if key in uc.keys():
            keyword_list = list(chain(keyword_list, uc[key]))

    for keyword in keyword_list:
        if act_key in uc.keys():
            act_list = flatten_list(uc[act_key])
            obj_list = flatten_list(uc[obj_key])

            for act in act_list:
                edges_dict[(keyword, 'keyword')].append((act, act_key))
            for obj in obj_list:
                edges_dict[(keyword, 'keyword')].append((obj, obj_key))

    return edges_dict

# act/obj已经按照step存放，存在有的step没有提取出node的情况
def make_act_obj_edges_miss(uc, edges_dict, act_key, obj_key):
    # 根据 A.全连接（同一个step中所有act与obj全连接）; B.结合（一个step中所有act认作一个act，所有obj认作一个）; C.形式化断句后，一step只有一对act-obj
    if act_key in uc :
        edges_dict = act_obj_one_on_one(edges_dict, uc[act_key], uc[obj_key],act_key, obj_key)  # c方法，一对一

    return edges_dict

def act_obj_one_on_one(edges_dict, act_list, obj_list,act_key, obj_key):
    for (sub_act, sub_obj) in zip(act_list, obj_list):
        for (act, obj) in zip(sub_act, sub_obj):
            if isinstance(act,list): # AF多一层
                for act_seg,obj_seg in zip(act,obj):
                    if (act_seg, act_key) not in edges_dict.keys():
                        edges_dict[(act_seg, act_key)] = []  # 没有的话创建一个
                    edges_dict[(act_seg, act_key)].append((obj_seg, obj_key))

            elif isinstance(act, str): # BF
                if (act, act_key) not in edges_dict.keys():
                    edges_dict[(act, act_key)] = []  # 没有的话创建一个
                edges_dict[(act, act_key)].append((obj, obj_key))

    return edges_dict

def create_node_uctext(node_to_UCText, uc):
    keyword_list = []
    for key in ["key_name", "key_path",'key_act','key_obj']:
        if key in uc.keys():
            keyword_list = list(chain(keyword_list, uc[key]))

    for keyword in keyword_list:
        if f"{keyword}_{'keyword'}" not in node_to_UCText.keys():
            node_to_UCText[f"{keyword}_{'keyword'}"] = []
        node_to_UCText[f"{keyword}_{'keyword'}"].append(uc['global id'])

    for act_list in uc['BF act']:
        for act in act_list:
            if f"{act}_{'BF act'}" not in node_to_UCText.keys():
                node_to_UCText[f"{act}_{'BF act'}"] = []
            node_to_UCText[f"{act}_{'BF act'}"].append(uc['global id'])

    for obj_list in uc['BF obj']:
        for obj in obj_list:
            if f"{obj}_{'BF obj'}" not in node_to_UCText.keys():
                node_to_UCText[f"{obj}_{'BF obj'}"] = []
            node_to_UCText[f"{obj}_{'BF obj'}"].append(uc['global id'])

    
    if 'AF act' in uc :
        for act_list in uc['AF act']:
            for act in act_list:
                for act_seg in act:
                    if f"{act_seg}_{'AF act'}" not in node_to_UCText.keys():
                        node_to_UCText[f"{act_seg}_{'AF act'}"] = []
                    node_to_UCText[f"{act_seg}_{'AF act'}"].append(uc['global id'])

    if 'AF obj' in uc :
        for obj_list in uc['AF obj']:
            for obj in obj_list:
                for obj_seg in obj:
                    if f"{obj_seg}_{'AF obj'}" not in node_to_UCText.keys():
                        node_to_UCText[f"{obj_seg}_{'AF obj'}"] = []
                    node_to_UCText[f"{obj_seg}_{'AF obj'}"].append(uc['global id'])

    return node_to_UCText

# 统计节点数量（入参为dict）, 根据子图统计
def count_node_data_dict_sub(edges_dict_list):
    node_total, node_max_sub_data = 0, 0
    act_bf_node_num, obj_bf_node_num, key_node_num,act_af_node_num, obj_af_node_num = [], [], [],[],[]  # 不同子图会包含相同的点，所以需要去重
    for edges_list in edges_dict_list:
        # print(f'edge_list index: {edges_dict_list.index(edges_list)}')
        # 1. 计算每个子图中各种点的个数 (这里去重的edge数没意义（edges_deduplicate），因为只是每个子图的去重，整体还是会有重复)
        act_bf, obj_bf, key, act_af, obj_af, edges_num = count_node_sub_data(edges_list)

        # 2. 记录所有子图中最多node(act+obj+key)数目
        if (len(act_bf) + len(obj_bf) + len(key) + len(act_af) + len(obj_af)) > node_max_sub_data:
            node_max_sub_data = (len(act_bf) + len(obj_bf) + len(key) + len(act_af) + len(obj_af))

        # 3. 统计各个类别的node总数
        act_bf_node_num.extend(set(act_bf) - set(act_bf_node_num))
        obj_bf_node_num.extend(set(obj_bf) - set(obj_bf_node_num))
        key_node_num.extend(set(key) - set(key_node_num))
        act_af_node_num.extend(set(act_af) - set(act_af_node_num))
        obj_af_node_num.extend(set(obj_af) - set(obj_af_node_num))

    # 6. 累加所有node(act+obj+key)数目
    node_total = len(act_bf_node_num) + len(obj_bf_node_num) + len(key_node_num) + len(act_af_node_num) + len(obj_af_node_num)

    print(
        f'node总数: {node_total}, bf act总数: {len(act_bf_node_num)}, bf obj: {len(obj_bf_node_num)},'
        f' key总数: {len(key_node_num)}, 子图中最多node数: {node_max_sub_data},'
        f'act af 总数: {len(act_af_node_num)}, obj af : {len(obj_af_node_num)}')

    return {'node_total': node_total, 'node_bf_act': len(act_bf_node_num), 'node_bf_obj': len(obj_bf_node_num),
            'node_key': len(key_node_num),'node_af_act': len(act_af_node_num), 'node_af_obj': len(obj_af_node_num),
            'node_max_sub': node_max_sub_data}

def count_node_sub_data(edges_list):
    act_bf, obj_bf, key, act_af, obj_af = [], [], [],[],[]  # 去重后的
    edges_num = 0
    edges = []  # 去重前后的边的数目
    for source_node, target_nodes in edges_list.items():
        # source node
        if source_node[1] == "keyword" and source_node not in key:
            key.append(source_node)
        elif source_node[1] == "BF act" and source_node not in act_bf:
            act_bf.append(source_node)
        elif source_node[1] == "BF obj" and source_node not in obj_bf:
            obj_bf.append(source_node)
        elif source_node[1] == "AF act" and source_node not in act_af:
            act_af.append(source_node)
        elif source_node[1] == "AF obj" and source_node not in obj_af:
            obj_af.append(source_node)
        elif source_node[1] != "keyword" and source_node[1] != "BF act" and source_node[1] != "BF obj" and source_node[1] != "AF act" and source_node[1] != "AF obj":
            print(f'ERROR!! source node type error!')
            print_location()

        # target node
        for target in target_nodes:
            if target[1] == "keyword" and target not in key:
                key.append(target)
            elif target[1] == "BF act" and target not in act_bf:
                act_bf.append(target)
            elif target[1] == "BF obj" and target not in obj_bf:
                obj_bf.append(target)
            elif target[1] == "AF act" and target not in act_af:
                act_af.append(target)
            elif target[1] == "AF obj" and target not in obj_af:
                obj_af.append(target)
            elif target[1] != "keyword" and target[1] != "BF act" and target[1] != "BF obj" and target[1] != "AF act" and target[1] != "AF obj":
                print(f'ERROR!! source node type error!')
                print_location()

            # count edges
            edges.append((source_node, target))  # 与edges_num对比验证


        edges_num += len(target_nodes)

    if len(edges) != edges_num:
        print(f'ERROR!! edge num error')

    return act_bf, obj_bf, key, act_af, obj_af, edges_num

def find_diff_dict(dict1, dict2):
    diff = {}
    # 找出dict1中与dict2不同的项
    for k, v in dict1.items():
        if k not in dict2 or dict1[k] != dict2[k]:
            diff[k] = v

    # 找出dict2中与dict1不同的项（反向差异）
    for k, v in dict2.items():
        if k not in dict1 or dict1[k] != dict2[k]:
            # 如果键已经在diff中且值来自dict1，则不覆盖（保持dict1的值）
            if k not in diff:
                diff[k] = v
    return diff

# 生成数据集,修改自_6_multi_Data_makeDataset 的 generate_dataset_1()
# 这个为多数据集版本，且包含x, edge_index, edge_type, edge_attr、y、train_mask
def generate_dataset_4turbo(edges_dict, node_to_UCText, para_dict,uc_bf_af_map):
    # 1、获得边的重复度，之后可用作权重
    for key in edges_dict.keys():
        edges_dict[key] = [(item, count) for item, count in Counter(edges_dict[key]).items()]

    # 2、节点到编号的映射(编号是当前子图中节点的编号，从0开始)
    node_to_id = {}
    id_to_node = {}

    # 遍历边数据以建立节点到编号的映射和收集节点类型
    next_id = 0
    for source_node, edges in edges_dict.items():
        source_node_str = f"{source_node[0]}_{source_node[1]}"  # 节点名称+类型作为唯一标识，因为可能用名称相同，但是类型不同的节点。如果仅用名称为标识，则可能出错
        if source_node_str not in node_to_id:
            node_to_id[source_node_str] = next_id
            id_to_node[next_id] = source_node_str
            next_id += 1
        for target_node, times in edges:
            target_node_str = f"{target_node[0]}_{target_node[1]}"
            if target_node_str not in node_to_id:
                node_to_id[target_node_str] = next_id
                id_to_node[next_id] = target_node_str
                next_id += 1

    # 3、创建节点特征embedding
    node_embeddings = []
    for node_id in id_to_node:
        node_str = id_to_node[node_id]
        # 分割节点名称和类型.是按照id顺序存储的，所以能和下面edge_index对得上
        node_name, node_type = node_str.rsplit('_', 1)
        # 获取嵌入
        embedding = MODEL.encode([node_name])[0]
        embedding = torch.from_numpy(embedding)  # 将这个 NumPy 数组转换为 PyTorch 张量
        node_embeddings.append(embedding)

    # 将嵌入转换为PyTorch张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.stack(node_embeddings, dim=0).to(device, dtype=torch.float)

    # 4、创建边数据
    edge_index = []
    edge_attr = []  # 边权重
    edge_type = []  # 用于存储边的类型

    for source_node, edges in edges_dict.items():
        source_id = node_to_id[f"{source_node[0]}_{source_node[1]}"]
        for target_node, weight in edges:
            target_id = node_to_id[f"{target_node[0]}_{target_node[1]}"]
            edge_index.append([source_id, target_id])
            edge_attr.append(weight)

            # 确定边的类型
            if source_node[1] == 'BF act' and target_node[1] == 'BF act':
                edge_type.append(0)  # 假设0表示BF act-BF act边
            elif source_node[1] == 'BF act' and target_node[1] == 'BF obj':
                edge_type.append(1)  # 假设1表示BF act-BF obj边
            elif source_node[1] == 'BF obj' and target_node[1] == 'BF obj':
                edge_type.append(2)  # 假设2表示BF obj-BF obj边
            elif source_node[1] == 'keyword' and target_node[1] == 'BF act':
                edge_type.append(3)  # 假设3表示 keyword-BF act 边
            elif source_node[1] == 'keyword' and target_node[1] == 'BF obj':
                edge_type.append(4)  # 假设4表示 keyword-BF obj 边
            elif source_node[1] == 'keyword' and target_node[1] == 'keyword':
                edge_type.append(5)  # 假设5表示 keyword-keyword 边
            elif source_node[1] == 'AF act' and target_node[1] == 'AF act':
                edge_type.append(6)  # 假设6表示 AF act- AF act边
            elif source_node[1] == 'AF act' and target_node[1] == 'AF obj':
                edge_type.append(7)  # 假设7表示 AF act- AF obj边
            elif source_node[1] == 'AF obj' and target_node[1] == 'AF obj':
                edge_type.append(8)  # 假设8表示 AF obj- AF obj边
            elif source_node[1] == 'keyword' and target_node[1] == 'AF act':
                edge_type.append(9)  # 假设9表示 keyword- AF act 边
            elif source_node[1] == 'keyword' and target_node[1] == 'AF obj':
                edge_type.append(10)  # 假设10表示 keyword- AF obj 边
            elif source_node[1] == 'BF act' and target_node[1] == 'AF act':
                edge_type.append(11)  # 假设11表示 BF act- AF act 边
            elif source_node[1] == 'BF obj' and target_node[1] == 'AF obj':
                edge_type.append(12)  # 假设12表示 BF obj- AF obj 边
            else:
                print(source_node[1], target_node[1])
                print(f'edge_type Error!!!')

    # 将edge_index和edge_type转换为PyTorch张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index_reverse = edge_index[[1, 0], :]  # 将edge的index反转，等于是将原本有向边变为无向边，因为两个方向的edge都有了
    edge_index = torch.cat((edge_index, edge_index_reverse), dim=1)  # 然后直接拼接在后面

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 权重是出现次数，所以使用int
    edge_attr = edge_attr.view(-1, 1)  # 转换为二维，形状为 [num_edges, 1]
    edge_attr = torch.cat((edge_attr, edge_attr), dim=0)  # 有向边变成无向边，权重直接复制一份放在后面即可

    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_type = torch.cat((edge_type, edge_type), dim=0)  # 有向边变成无向边，类型也是直接复制一份放在后面即可

    # 5、生成非预定义参数（不能用y的原因是Data对象要求y必须是[节点数]的向量，形状不能变化）
    # 5.1、生成标签 y，是一个列表，顺序对应keyword，存放每个keyword关联的act/obj的id
    y = [] # 顺序按照node_to_id中keyword的顺序
    keyword_id = []  # 记录keyword节点的id
    for key_node in node_to_id:
        if 'keyword' in key_node or 'BF' in key_node: # 20250915前:if 'keyword' in key_node:
            keyword_id.append(node_to_id[key_node])
            related_node = []
            node_name, node_type = key_node.rsplit('_', 1)
            for (tar_node_str, target_node_type), time in edges_dict[(node_name, node_type)]:
                if target_node_type != 'keyword':# if target_node_type != 'keyword' and 'BF' not in target_node_type: 如果是这样的话，keyword关联的bf node，以及bf之间的edge就没有了
                    tar_node = f"{tar_node_str}_{target_node_type}"
                    related_node.append(node_to_id[tar_node])
            if len(related_node) == 0:  # line497-498,20250915 加  # 专门针对没有关联节点的keyword/bfnode，补充-1，不然会被认为是错误节点
                related_node = [-2]  # 如果没有关联的节点，则补充-1
                # print(keyword_id)
            y.append(related_node)
        else:
            y.append([-1])  # AF act/obj 节点位置补充 -1

    # 确定最大的列表长度为 节点总数,需要所有列表长度一致
    max_length = para_dict['max_node_subdata']  # 所有子图最多节点数原为1257
    # 创建一个填充后的二维列表，填充'-99'表示无效数据，补齐列表
    padded_y = [sublist + [-99] * (max_length - len(sublist)) for sublist in y]
    # 将填充后的列表转换为张量
    y_tensor = torch.tensor(padded_y, dtype=torch.long)
    padded_y_1 = y_tensor.tolist()

    # # 5.2、生成训练掩码train_mask, 用整张子图（Data对象）来训练，因为抽百分比的数据的话，数据集会不平衡（各种类型的点数量不一致）
    # # 计算训练节点的数量
    # train_mask = torch.zeros(len(id_to_node), dtype=torch.bool)
    # train_mask[keyword_id] = True

    # 6、转变node_to_UCText为非预定义参数
    node_to_UCText_new = {}  # 创建一个新的用于存放
    for node in node_to_UCText.keys():
        node_id = node_to_id[node]
        node_to_UCText_new[node_id] = node_to_UCText[node]  # node_to_UCText_new中item是 节点id：uctext id
        node_to_UCText_new[node_id] = list(set(node_to_UCText_new[node_id]))  # 列表去重

    sort_key = sorted(node_to_UCText_new.keys())  # 按照节点id排序
    padded_UC = [[item + para_dict['uctext_start'] for item in node_to_UCText_new[key]] for key in
                 sort_key]  # 按照节点id排序 并从总节点id (原来为)3241后开始计算UCText的id

    # 填充无效数据'-99'至最长维度，这里的最长长度max_length变成了众多子图中最多的UC数目
    max_length = para_dict['max_length']  # 原为1744
    # 遍历result中的每个小列表
    for i in range(len(padded_UC)):
        # 计算当前小列表需要的填充长度
        padding_length = max_length - len(padded_UC[i])
        # 如果需要填充，则用-99填充到长度为max_length
        if padding_length > 0:
            padded_UC[i] += [-99] * padding_length

    for sublist in padded_UC:  # 检查长度是否补齐
        if len(sublist) != max_length:
            print(f"子列表 {sublist} 的长度不等于最长{max_length}，其id为：{padded_UC.index(sublist)}")
    node_to_UCText_new = torch.tensor(padded_UC, dtype=torch.long)

    # 7、将数据添加到Data对象中,包含x、edge_index、edge_attr和edge_attr
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
    data.keyword_relation = y_tensor
    data.to_UCText = node_to_UCText_new  # 表示 每个node存在于哪个UCText里，节点id：UCText_id

    # （2025年9月20日）8、获取 uc - bf node - af node
    uc_bf_af_map = get_uc_bf_af(uc_bf_af_map,node_to_id,para_dict['uctext_start'])

    return data,uc_bf_af_map

def get_uc_bf_af(uc_bf_af_map_list,node_to_id,uc_id_start):
    new_uc_bf_af_map_list = []

    for uc_bf_af_map in uc_bf_af_map_list:  # 传入的是整个sub data的所有uc
        new_uc_bf_af_map = {}  # 将所有的点由name变为id，uc的id变为global id。
        # 1. 处理 uc id 以及 keyword id
        new_uc_id = int(next(iter(uc_bf_af_map)).split(':')[1]) + uc_id_start  # 获得第一个key，即为原来的uc_id
        new_uc_bf_af_map[new_uc_id] = [] # 存放keyword的id
        for keyword in uc_bf_af_map[next(iter(uc_bf_af_map))]:
            new_uc_bf_af_map[new_uc_id].append(node_to_id[keyword])
        
        # 2. 处理 bf node
        for key in ['BF act','BF obj']:
            new_uc_bf_af_map[key] = [] # 存放keyword的id
            for bf in uc_bf_af_map[key]:
                lst1 = []
                for bf_sge in bf:
                    lst1.append(node_to_id[bf_sge])
                new_uc_bf_af_map[key].append(lst1)

        # 3. 处理af node
        for key in ['AF act','AF obj']:
            new_uc_bf_af_map[key] = [] # 存放keyword的id
            for bf_index in uc_bf_af_map[key]:  # af 是按照bf的index存放的
                if len(bf_index)<0:
                    new_uc_bf_af_map[key].append([])  # 该index上的bf上没有af
                else:
                    lst3 = []
                    for af in bf_index:
                        lst2 = []
                        for af_step in af:  # 二层列表
                            lst4 = []
                            for af_sge in af_step:
                                lst4.append(node_to_id[af_sge])
                            lst2.append(lst4)
                        lst3.append(lst2)
                new_uc_bf_af_map[key].append(lst3)
        
        new_uc_bf_af_map_list.append(new_uc_bf_af_map)

    return new_uc_bf_af_map_list


            
        


if __name__ == '__main__':
    task_name = 'make_pt_ncet'
    print(f"*** task_name: {task_name} !!!!! ***")

    if task_name =="make_pt": # 模仿BFGen的代码：_8_/task6
        in_file_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json'
        in_file_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json'

        pt_save_path_ernie = 'E:/Trae_project/AFGen_pt_file/0_after_SIP/ERNIE_pub_20.pt'
        pt_save_path_gpt = 'E:/Trae_project/AFGen_pt_file/0_after_SIP/gpt_pub_20.pt'

        for in_file_path, pt_save_path in zip([in_file_path_gpt],[pt_save_path_gpt]):

            # 检测文件是同一个method的
            if not check_all_under_same_method("gpt",[in_file_path, pt_save_path]): # 这次跑gpt的，更换string即可判断出入路径是否统一
                break

            # 1. 读取文件
            uc_list = read_uc_from_stand_json(in_file_path)

            # 2. 将local id 更新为全局 id
            uc_list = local_id_to_global_id(uc_list)

            # 3. 生成全局edge数目和 node 数目
            argc1 = check_agc(uc_list)

            # 4. 进行分组（使用固定分组/随机分组）
            uc_list = group_uc_fixed(uc_list, PUB_GROUPING_UC_20_1) # ncet数据集是这个：GROUPING_UC_20

            # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
            edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_only_rgat(uc_list)

            # 5.1 统计：所有子图中最多的节点数（act+obj+key）
            argc2 = count_node_data_dict_sub(edges_dict_list)
            # 5.2 判断两次统计是否一致. 除了'node_max_sub'和’edge_num_total‘是后面加的，其他都应该一样.
            print(f'全局统计和分图统计不一致的项为：{find_diff_dict(argc1, argc2)}') # 目前只显示“node_max_sub"

            # 6、生成数据集
            dataset = []
            # uctext_start 为uctext节点的起始点（全部数据中act+obj+key节点总数）；max_node_subdata 所有子图最多节点数（act+obj+key）; max_length 所有子图中包含最多uc的个数
            para_dict = {"uctext_start": argc2['node_total'], "max_node_subdata": argc2['node_max_sub'],
                        "max_length": ARGC_20['max_uc_in_sub']}
            for i in range(len(edges_dict_list)):
                subgraph_data = generate_dataset_4turbo(edges_dict_list[i], node_to_UCText_list[i], para_dict)
                dataset.append(subgraph_data)
            
            # 7、将data保存到文件
            torch.save(dataset, pt_save_path)


    elif task_name == 'make_pt_ncet': #模仿上面"make_pt"任务，因为ncet有一些格式不同，所以再写一个。
        af_node_ground_truth_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        bf_node_ernie = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json'
        af_node_ground_truth_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'
        bf_node_gpt = '../0_Data/4_alt_flow_data/0_raw_data/GPT_NCET_integrated.json'

        ncet_save_path_ernie = 'E:/Trae_project/AFGen_pt_file/1_wo_SIP/ERNIE_ncet_20.pt'
        ncet_save_path_gpt = 'E:/Trae_project/AFGen_pt_file/1_wo_SIP/gpt_ncet_20.pt'

        for in_file_path, pt_save_path, bf_path in zip([af_node_ground_truth_gpt],[ncet_save_path_gpt],[bf_node_gpt]):

            # 检测文件是同一个method的
            if not check_all_under_same_method("gpt",[in_file_path, pt_save_path,bf_path]): # 这次跑gpt的，更换string即可判断出入路径是否统一
                break

            # 1. 读取文件
            uc_list = read_uc_from_stand_json(in_file_path)

            # 2. 将local id 更新为全局 id
            uc_list = local_id_to_global_id(uc_list)

            # 2.5. 需要将af node 与 BFGen中的数据（bf node，各种key node）融合
            uc_list = mix_items(uc_list,read_uc_from_stand_json(bf_path),['key_act','key_obj','key_name','key_path','BF act','BF obj'])

            # 2.6. 将ncet数据集中的格式往pub数据集靠拢，主要是列表嵌套层数
            uc_list = unify_ncet_format(uc_list)

            # 3. 生成全局edge数目和 node 数目
            argc1 = check_agc(uc_list)

            # 4. 进行分组（使用固定分组/随机分组）
            uc_list = group_uc_fixed(uc_list, GROUPING_UC_20)

            # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
            edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_only_rgat(uc_list)

            # 5.1 统计：所有子图中最多的节点数（act+obj+key）
            argc2 = count_node_data_dict_sub(edges_dict_list)
            # 5.2 判断两次统计是否一致. 除了'node_max_sub'和’edge_num_total‘是后面加的，其他都应该一样.
            print(f'全局统计和分图统计不一致的项为：{find_diff_dict(argc1, argc2)}') # 目前只显示“node_max_sub",就对了。

            # 6、生成数据集
            dataset = []
            # uctext_start 为uctext节点的起始点（全部数据中act+obj+key节点总数）；max_node_subdata 所有子图最多节点数（act+obj+key）; max_length 所有子图中包含最多uc的个数
            para_dict = {"uctext_start": argc2['node_total'], "max_node_subdata": argc2['node_max_sub'],
                        "max_length": ARGC_20['max_uc_in_sub']}
            for i in range(len(edges_dict_list)):
                subgraph_data = generate_dataset_4turbo(edges_dict_list[i], node_to_UCText_list[i], para_dict)
                dataset.append(subgraph_data)
            
            # 7、将data保存到文件
            torch.save(dataset, pt_save_path)
    
    print(f"*** task_name: {task_name} Finish! *** ")
