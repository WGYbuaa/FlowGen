# 之前的pt是没有加入branching point 的版本，目前需要加入branching point。

from imports import datetime,PUB_GROUPING_UC_20_1,importlib,ARGC_20,torch,REPLACEMENT_MAP,re,string,chain,GROUPING_UC_20
spec = importlib.util.spec_from_file_location("module_name", "7_make_pt_file.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from utils import read_uc_from_stand_json,check_all_under_same_method,print_location,list_depth_recursive,get_3_list,write_uc_to_stand_json

# 定义一个函数来分割字符串
def split_string(s, delimiters):
    # 先将字符串中的分隔符替换为统一的分隔符（这里选择英文逗号）
    for delimiter in delimiters:
        s = s.replace(delimiter, ',')
    # 然后使用统一的分隔符来分割字符串
    return s.split(',')


def splite_2_list_en(input_str):
    input = input_str.translate(str.maketrans(REPLACEMENT_MAP))  # 将中文符号转换为英文符号
    input = re.sub(r'\band\b|\bor\b', ',', input)  # and和or 替换成逗号

    delimiters = ['，', ',', '。']  # 分割str的符号
    clean_list = []
    if isinstance(input, str):
        input = split_string(input, delimiters)

    update = []
    for item in input:
        if ":" in item:  # （如果有的话）先根据冒号，删除冒号前llm的描述语句
            item = item.split(':')[1]
        split = split_string(item, delimiters)
        update.extend(split)

    for item in update:
        if item != "" and not item.isdigit() and not is_pure_punctuation(item):
            item = item.strip()  # 去除两端空格
            item = re.sub(r'^[\W_]+|[\W_]+$', '', item)  # 去除两端特殊符号
            clean_list.append(item.lower())  # 统一大小写
    return clean_list

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


def get_edges_dict_and_node_to_UCText_only_rgat_withbp(use_case_list,uc_list_bp):
    edges_dict_list = []
    node_to_UCText_list = []
    uc_bf_af_map_list = []
    for uc_list,uc_bp in zip(use_case_list,uc_list_bp):
        edge_dict, node_to_UCText,uc_bf_af_map = create_edges_af_with_bp(uc_list,uc_bp)
        edges_dict_list.append(edge_dict)
        node_to_UCText_list.append(node_to_UCText)
        uc_bf_af_map_list.append(uc_bf_af_map)

    return edges_dict_list, node_to_UCText_list, uc_bf_af_map_list


def create_node_uctext_withbp(node_to_UCText, uc):
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

def make_bf_af_edges(bf_node_list, af_node_list,edges_dict,bp_list,label_bf,label_af):
    bp_list_cpy = bp_list.copy()
    if len(bp_list_cpy) != len(af_node_list):
        print(f"*** Warning!!! len(bp_list) != len(af_node_list) ***")
    if any(id > len(bf_node_list) for id in bp_list_cpy):
        print(f"*** Warning!!! 有分支点的index大于Basic Flow的step数 ***")
    for i in range(len(bp_list_cpy)):
        for bfnode in bf_node_list[bp_list_cpy[i]]: # 找到对应的bf
            for af in af_node_list[bp_list_cpy.index(bp_list_cpy[i])]: # 找到对应的af.af是二级列表
                for afnode in af:
                    edges_dict[(bfnode, label_bf)].append((afnode, label_af))
        bp_list_cpy[i] = -99  # 防止重复使用同一个bp_index


    return edges_dict

def get_branching_point_info(uc):
    bp_list = []
    # for dicts in uc['Alt. Flow']:  
    for dicts in uc['Exc. Flow']:  
        id_n = list(dicts.keys())[0]
        branching_index = id_n.split('_')[0]
        bp_list.append(int(branching_index))

    return bp_list

def create_edges_af_with_bp(uc_list,uc_list_bp):
    edges_dict = {}
    node_to_UCText = {}  # 用于存放keyword、act、obj节点与UCText的对应关系
    uc_bf_af_map = []  # 用于存放每个uc中 uc - keyword - bf node - af node 的对应关系
    for uc,uc_bp in zip(uc_list,uc_list_bp):
        # 0. get branching point info for uc_bp
        bp_list = []
        if len(uc['Exc. Flow'])>0:
        # if len(uc['Alt. Flow'])>0:
            bp_list = get_branching_point_info(uc_bp)

        # 1. Make "keyword-keyword" edge(考虑了与前后node均相连)
        edges_dict = module.make_key_key_edge(uc, edges_dict)

        # 2. Add "keyword-act" edges, "keyword-obj" edges
        edges_dict = module.make_key_act_obj(uc, edges_dict, 'BF act', 'BF obj')
        edges_dict = module.make_key_act_obj(uc, edges_dict, 'AF act', 'AF obj')

        # 3. Add "act-act" edges (只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = module.make_common_edges(uc['BF act'], edges_dict, "BF act")
        if 'AF act' in uc :
            edges_dict = module.make_common_edges(uc['AF act'], edges_dict, "AF act")

        # 4. Add "obj-obj" edges(只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = module.make_common_edges(uc['BF obj'], edges_dict, 'BF obj')
        if 'AF obj' in uc :
            edges_dict = module.make_common_edges(uc['AF obj'], edges_dict, 'AF obj')

        # 5. Add "act-obj" edge
        edges_dict = module.make_act_obj_edges_miss(uc, edges_dict,'BF act','BF obj' )
        edges_dict = module.make_act_obj_edges_miss(uc, edges_dict,'AF act','AF obj')

        # 6. Add "basic flow - alternative flow" edges
        # if len(uc['Alt. Flow'])>0:
        if len(uc['Exc. Flow'])>0:
            edges_dict = make_bf_af_edges(uc['BF act'], uc['AF act'],edges_dict,bp_list,'BF act','AF act')
            edges_dict = make_bf_af_edges(uc['BF obj'], uc['AF obj'],edges_dict,bp_list,'BF obj','AF obj')


        # 7. create (node to uctext)'s map (其实是bf、af node to key node)
        node_to_UCText = create_node_uctext_withbp(node_to_UCText, uc)

        # 8. 每个用例一个字典，内容为 {"uc_id:1":[key_id,key_id],"bf_act_id:2":[af_act_id],"bf_obj_id:2":[af_obj_id]...} 其中af列表长度即为step个数
        uc_bf_af_map.append(create_uc_bf_af_map(uc,bp_list))

    return edges_dict, node_to_UCText,uc_bf_af_map


def create_uc_bf_af_map(uc,bp_list):
    uc_golbal_id,keyword_list = "uc_id:" + str(uc['global id']),[]
    uc_bf_af_map = {uc_golbal_id:[],
                    'BF act':[],
                    'BF obj':[],
                    'AF act':[],
                    'AF obj':[]}
    
    for key in ["key_name", "key_path",'key_act','key_obj']:
        if key in uc.keys():
            keyword_list += uc[key]
    uc_bf_af_map[uc_golbal_id] = [s + '_keyword' for s in  keyword_list]
  
    uc_bf_af_map["BF act"] = [[s + '_BF act' for s in inner_list] for inner_list in uc['BF act']]
    uc_bf_af_map["BF obj"] = [[s + '_BF obj' for s in inner_list] for inner_list in uc['BF obj']]


    uc_bf_af_map['AF act'], uc_bf_af_map['AF obj'] = [[] for _ in range(len(uc['Basic flow']))],[[] for _ in range(len(uc['Basic flow']))]   # 初始化都为空
    if 'AF act' in uc and len(uc['AF act'])>0:
        for i in range(len(uc['AF act'])):        
            map_to_bf = bp_list[i]  # 该af对应的bf的index ## 因为有的bp出来多个af
            uc_bf_af_map['AF act'][map_to_bf].append([[s + '_AF act' for s in inner_list] for inner_list in uc['AF act'][i]])
            uc_bf_af_map['AF obj'][map_to_bf].append([[s + '_AF obj' for s in inner_list] for inner_list in uc['AF obj'][i]])



    return uc_bf_af_map

def get_edges_from_dict(edges_dict_list):
    bf_af_act_edges,bf_af_obj_edge = [],[]
    for edges_dict in edges_dict_list: # 多个子图
        for key1 in edges_dict:
            if key1[1] == 'BF act':
                for edge in edges_dict[key1]:
                    if edge[1] == 'AF act':
                        bf_af_act_edges.append((key1,edge))
            
            elif key1[1] == 'BF obj':
                for edge in edges_dict[key1]:
                    if edge[1] == 'AF obj':
                        bf_af_obj_edge.append((key1,edge))
        
    return len(bf_af_act_edges),len(bf_af_obj_edge)

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

def get_edges_from_list(uc_list_sub,uc_list_bp_sub):
    bf_af_act_edges,bf_af_obj_edge = 0,0
    for uc_list,uc_list_bp in zip(uc_list_sub,uc_list_bp_sub):
        for uc,uc_bp in zip(uc_list,uc_list_bp):
            # if len(uc['Alt. Flow'])>0:
            if len(uc['Exc. Flow'])>0:
                bp_list = get_branching_point_info(uc_bp)
                for i in range(len(bp_list)):
                    bf_af_act_edges += len(uc['BF act'][bp_list[i]]) * len(flatten_list(uc['AF act'][i]))
                    bf_af_obj_edge += len(uc['BF obj'][bp_list[i]]) * len(flatten_list(uc['AF obj'][i]))

    return bf_af_act_edges,bf_af_obj_edge

            

def check_af_bf_edges(edges_dict_list,uc_list,uc_list_bp):
    # 1. 获得edges_dict_list中生成的所有bf-af edges的数目
    bf_af_act_edges,bf_af_obj_edge = get_edges_from_dict(edges_dict_list)
    # 2. 获得uc_list_bp中所有bf-af edges的数目
    bf_af_act_edges_bp,bf_af_obj_edge_bp = get_edges_from_list(uc_list,uc_list_bp)
    # 3. 进行对比
    if bf_af_act_edges != bf_af_act_edges_bp:
        print(f"*** Warning!!! bf-af act edges 不一致: {bf_af_act_edges} != {bf_af_act_edges_bp} ***")
    if bf_af_obj_edge != bf_af_obj_edge_bp:
        print(f"*** Warning!!! bf-af obj edges 不一致: {bf_af_obj_edge} != {bf_af_obj_edge_bp} ***")
    print(f'数目对的上')

def mix_items(uc_list_ernie,uc_alt_node_list_ernie,list1):  # 第三个参数是混合字典时，需要保留的item
    for uc,uc_ref in zip(uc_list_ernie, uc_alt_node_list_ernie):
        if uc['id'] != uc_ref['id']:
            print(f"Error: uc id {uc['id']} 不等于 uc_ref id {uc_ref['id']}")
        # 只保留uc_ref中的item
        uc_ref_filtered = {k: v for k, v in uc_ref.items() if k in list1}
        
        # 将uc_ref_filtered的内容添加到uc中
        uc.update(uc_ref_filtered)
    
    
    return uc_list_ernie

def unify_ncet_format(uc_list):
    for uc in uc_list:

        # 1、 修改af act/obj的格式
        if 'AF act' in uc:
            if list_depth_recursive(uc['AF act']) < 3 and list_depth_recursive(uc['AF act']) == 2:  #PUB数据集是3层，ncet是2层。往pub数据集靠拢，这样可以不用修改后面的代码。
                uc['AF act'] = get_3_list(uc['AF act'])
                uc['AF obj'] = get_3_list(uc['AF obj'])
            else:
                print(f"ERROR!! AF act depth error!")

        # 2、 修改key node的格式
        key_list = ['key_name', 'key_path', 'key_act', 'key_obj']
        for key in key_list:
            if key in uc and len(uc[key]) == 1 and isinstance(uc[key][0],str) and ',' in uc[key][0]:
                uc[key] = splite_2_list_en(uc[key][0])
        
    return uc_list

def check_step_node_num(uc_list):
    for uc in uc_list:
        if len(uc['BF act']) != len(uc['BF obj']):
            print(f"*** Warning!!! UC ID {uc['id']} BF act/obj step num not equal ***")
        if len(uc['Basic flow']) != len(uc['BF act']):
            print(f"*** Warning!!! UC ID {uc['id']} Basic flow/BF act step num not equal ***")
        if 'AF act' in uc:
            if len(uc['Exc. Flow']) != len(uc['AF act']):
                print(f"*** Warning!!! UC ID {uc['id']} Exc. flow/AF act step num not equal ***")
            if len(uc['AF act']) != len(uc['AF obj']):
                print(f"*** Warning!!! UC ID {uc['id']} AF act/obj step num not equal ***")

            for af,afnode in zip(uc['Exc. Flow'],uc['AF obj']):
                if len(af) != len(afnode):
                    print(f"*** Warning!!! UC ID {uc['id']} AF act/obj node num not equal ***")

def check_af_in_all_sub_graph(uc_list):
    list1 = []
    for i in range(len(uc_list)):
        count1 = 0
        for uc in uc_list[i]:
            if len(uc["Exc. Flow"])>0:
                count1 += 1
        if count1 == 0:
            list1.append(i)

    if len(list1) ==0:
        print('所有 sub graph 都有分支流。')
    else:
        print(f'{list1}')
    print(f'***')



if __name__ == '__main__':
    task_name = 'make_pt_ncet'
    print(f'*** task_name: {task_name} , Starting time: {datetime.now()}  !!!!! ***')

    if task_name =='make_pt_pub':
        branching_point_path = '../0_Data/5_branching_point/1_gpt_added_bp/pub_with_gpt_bp.json'  # 只需要其中标注的分支点信息,其中node是ernie提取的，node这里不重要。

        in_file_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json'
        in_file_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json'


        pt_save_path_ernie = 'E:/Trae_project/AFGen_pt_file/3_with_branching_point/ERNIE_pub_withbp_20.pt'
        pt_save_path_gpt = 'E:/Trae_project/AFGen_pt_file/3_with_branching_point/gpt_pub_withbp_11.pt'

        map_path_gpt = "../0_Data/5_branching_point/3_uc_bf_af_map/gpt_pub_11.json" # gpt_pub_11.json只包含11个uc,全部uc是：gpt_pub.json
        map_path_ernie = "../0_Data/5_branching_point/3_uc_bf_af_map/Ernie_pub.json"

        for in_file_path, pt_save_path,map_path in zip([in_file_path_gpt],[pt_save_path_gpt],[map_path_gpt]):

            # 检测文件是同一个method的
            if not check_all_under_same_method("gpt",[in_file_path, pt_save_path,map_path]): # 这次跑gpt的，更换string即可判断出入路径是否统一
                break

            # 1. 读取文件
            uc_list = read_uc_from_stand_json(in_file_path)

            # 2. 将local id 更新为全局 id
            uc_list = module.local_id_to_global_id(uc_list)
            uc_list_bp = module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path))

            # # #2025年9月24日
            # uc_list_new,uc_list_bp_new = [],[]
            # for uc,uc_bp in zip(uc_list,uc_list_bp):
            #     if uc['global id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10]:
            #         uc_list_new.append(uc)
            #         uc_list_bp_new.append(uc_bp)
            # uc_list,uc_list_bp = uc_list_new,uc_list_bp_new

            # 3. 生成全局edge数目和 node 数目
            argc1 = check_agc(uc_list)

            # 4. 进行分组（使用固定分组/随机分组） 
            # uc_list = module.group_uc_fixed(uc_list, [[0, 1, 2, 3, 4, 5, 6, 7, 8],[9],[10]])  #2025年9月24日
            # uc_list_bp = module.group_uc_fixed(uc_list_bp, [[0, 1, 2, 3, 4, 5, 6, 7, 8],[9],[10]]) #2025年9月24日
            uc_list = module.group_uc_fixed(uc_list, PUB_GROUPING_UC_20_1) # ncet数据集是这个：GROUPING_UC_20
            uc_list_bp = module.group_uc_fixed(uc_list_bp, PUB_GROUPING_UC_20_1) # ncet数据集是这个：GROUPING_UC_20

            # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
            edges_dict_list, node_to_UCText_list,uc_bf_af_map_list = get_edges_dict_and_node_to_UCText_only_rgat_withbp(uc_list,uc_list_bp)

            # 5.0 检查是否全部bf-af都edge了
            check_af_bf_edges(edges_dict_list,uc_list,uc_list_bp)

            # 5.1 统计：所有子图中最多的节点数（act+obj+key）
            argc2 = module.count_node_data_dict_sub(edges_dict_list)
            # 5.2 判断两次统计是否一致. 除了'node_max_sub'和’edge_num_total‘是后面加的，其他都应该一样.
            print(f'全局统计和分图统计不一致的项为：{module.find_diff_dict(argc1, argc2)}') # 目前只显示“node_max_sub"

            # 6、生成数据集
            dataset,uc_bf_af_map = [],[]
            # uctext_start 为uctext节点的起始点（全部数据中act+obj+key节点总数）；max_node_subdata 所有子图最多节点数（act+obj+key）; max_length 所有子图中包含最多uc的个数
            para_dict = {"uctext_start": argc2['node_total'], "max_node_subdata": argc2['node_max_sub'],
                        "max_length": ARGC_20['max_uc_in_sub']}
            for i in range(len(edges_dict_list)):
                subgraph_data,sub_uc_bf_af_map = module.generate_dataset_4turbo(edges_dict_list[i], node_to_UCText_list[i], para_dict, uc_bf_af_map_list[i])
                dataset.append(subgraph_data)
                uc_bf_af_map.append(sub_uc_bf_af_map)

            
            # 7、将data保存到文件
            write_uc_to_stand_json(map_path,uc_bf_af_map)
            torch.save(dataset, pt_save_path)
        
    elif task_name == 'make_pt_ncet':
        branching_point_path = '../0_Data/5_branching_point/2_ncet_bp/NCET_with_bp.json'  # 只需要其中标注的分支点信息

        in_file_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        in_file_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'

        bf_node_ernie = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json'
        bf_node_gpt = '../0_Data/4_alt_flow_data/0_raw_data/GPT_NCET_integrated.json'

        pt_save_path_ernie = 'E:/Trae_project/AFGen_pt_file/3_with_branching_point/ERNIE_ncet_withbp_20.pt'
        pt_save_path_gpt = 'E:/Trae_project/AFGen_pt_file/3_with_branching_point/gpt_ncet_withbp_20.pt'

        map_path_gpt = "../0_Data/5_branching_point/3_uc_bf_af_map/gpt_ncet_map.json" 
        map_path_ernie = "../0_Data/5_branching_point/3_uc_bf_af_map/ERNIE_ncet_map.json" 


        for in_file_path, pt_save_path,bf_path,map_path in zip([in_file_path_ernie],[pt_save_path_ernie],[bf_node_ernie],[map_path_ernie]):

            # 检测文件是同一个method的
            if not check_all_under_same_method("ernie",[in_file_path, pt_save_path,bf_path,map_path]): # 更换"gpt"即可判断出入路径是否统一
                break

            # 1. 读取文件
            uc_list = read_uc_from_stand_json(in_file_path)

            # 2. 将local id 更新为全局 id
            uc_list = module.local_id_to_global_id(uc_list)
            uc_list_bp = module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path))

            # 2.5. 需要将af node 与 BFGen中的数据（bf node，各种key node）融合
            uc_list = mix_items(uc_list,read_uc_from_stand_json(bf_path),['Basic flow','key_act','key_obj','key_name','key_path','BF act','BF obj'])
            # 2.6. 将ncet数据集中的格式往pub数据集靠拢，主要是列表嵌套层数
            uc_list = unify_ncet_format(uc_list)
            # 2.7. 检查bfstep和node的数目；以及afstep和node的数目是否对应
            check_step_node_num(uc_list)

            # 3. 生成全局edge数目和 node 数目
            argc1 = check_agc(uc_list)

            # 4. 进行分组（使用固定分组/随机分组）
            uc_list = module.group_uc_fixed(uc_list, GROUPING_UC_20)
            check_af_in_all_sub_graph(uc_list)
            uc_list_bp = module.group_uc_fixed(uc_list_bp, GROUPING_UC_20)

            # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
            edges_dict_list, node_to_UCText_list,uc_bf_af_map_list = get_edges_dict_and_node_to_UCText_only_rgat_withbp(uc_list,uc_list_bp)

            # 5.0 检查是否全部bf-af都edge了
            check_af_bf_edges(edges_dict_list,uc_list,uc_list_bp)

            # 5.1 统计：所有子图中最多的节点数（act+obj+key）
            argc2 = module.count_node_data_dict_sub(edges_dict_list)
            # 5.2 判断两次统计是否一致. 除了'node_max_sub'和’edge_num_total‘是后面加的，其他都应该一样.
            print(f'全局统计和分图统计不一致的项为：{module.find_diff_dict(argc1, argc2)}') # 目前只显示“node_max_sub"

            # 6、生成数据集
            dataset,uc_bf_af_map = [],[]
            # uctext_start 为uctext节点的起始点（全部数据中act+obj+key节点总数）；max_node_subdata 所有子图最多节点数（act+obj+key）; max_length 所有子图中包含最多uc的个数
            para_dict = {"uctext_start": argc2['node_total'], "max_node_subdata": argc2['node_max_sub'],
                        "max_length": ARGC_20['max_uc_in_sub']}
            for i in range(len(edges_dict_list)):
                subgraph_data,sub_uc_bf_af_map = module.generate_dataset_4turbo(edges_dict_list[i], node_to_UCText_list[i], para_dict, uc_bf_af_map_list[i])
                dataset.append(subgraph_data)
                uc_bf_af_map.append(sub_uc_bf_af_map)
            
            # 7、将data保存到文件
            write_uc_to_stand_json(map_path,uc_bf_af_map)
            torch.save(dataset, pt_save_path)


    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')