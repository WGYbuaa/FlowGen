from utils import read_uc_from_stand_json,read_uc_from_json,current_function,is_pure_punctuation,contains_only_digits_symbols_spaces,write_uc_to_stand_json,print_location,is_pure_punctuation
from imports import StanfordCoreNLP,re,ARGC_20,torch,GROUPING_UC_20,MODEL,REPLACEMENT_MAP,chain,Counter,Data,PUB_GROUPING_UC_20_1,datetime,importlib
spec = importlib.util.spec_from_file_location("module_name", "7_make_pt_file_with_bp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def find_different(argc1,argc2,list1):  # 用于检查两次统计的差异
    bf_obj = []
    for key1 in list1:
        for (node,label) in argc2[key1]:
            bf_obj.append(node)
        diff_list1 = list(set(bf_obj) - set(argc1[key1]))
        print(f'{key1} 在分图统计中多出的node数目: {len(diff_list1)}, 具体为: {diff_list1}')

def find_node(uc_list,node,key1):
    for uc in uc_list:
        if key1 in uc:
            for af in uc[key1]:
                if node in af:
                    print(f"uc global id: {uc['global id']}")


def formate_node_eng_list(node_list):
    update1 = []
    if isinstance(node_list, list):
        for item1 in node_list:
            if item1 != "" and not item1.isdigit() and not is_pure_punctuation(item1):
                item1 = item1.strip()  # 去除两端空格
                item1 = re.sub(r'^[\W_]+|[\W_]+$', '', item1)  # 去除两端特殊符号
                item1 = item1 #.lower()  # 统一大小写
                update1.append(item1)
    elif isinstance(node_list, str):
        if node_list != "" and not node_list.isdigit() and not is_pure_punctuation(node_list):
            print(f'{node_list} 类型不是list')
            item2 = node_list.strip()  # 去除两端空格
            item2 = re.sub(r'^[\W_]+|[\W_]+$', '', item2)  # 去除两端特殊符号
            update1 = item2 #.lower()  # 统一大小写
    else:
        print(f'{node_list} 类型不是list')
        return None
    return update1

def format_node_eng(input1):
    update1 = []
    if isinstance(input1, list):
        for item1 in input1:
            if item1 != "" and not item1.isdigit() and not is_pure_punctuation(item1):
                item1 = item1.strip()  # 去除两端空格
                item1 = re.sub(r'^[\W_]+|[\W_]+$', '', item1)  # 去除两端特殊符号
                item1 = item1.lower()  # 统一大小写
                update1.append(item1)
    elif isinstance(input1, str):
        if input1 != "" and not input1.isdigit() and not is_pure_punctuation(input1):
            item2 = input1.strip()  # 去除两端空格
            item2 = re.sub(r'^[\W_]+|[\W_]+$', '', item2)  # 去除两端特殊符号
            update1 = item2.lower()  # 统一大小写
    else:
        print(f'类型 出错, {input1}')

    if update1 == []:
        return None
    return update1

# 英文数据集pub dataset专用
def extract_from_ucText_eng(use_case_list):
    # 这里不标注lang=zh，默认英文
    nlp = StanfordCoreNLP(r'../0_Data/stanford-corenlp-4.0.0', memory='8g')

    for uc in use_case_list:  # 数据中原本的key都是llm提取的，这里该用stf方法重新提取一次
        uc['key_name'], uc['key_path'], uc['key_act'], uc['key_obj'] = [], [], [], []

        # 0、将uc name中驼峰命名法分开,提取其中的动词和名词。
        uc_name = re.sub(r'(?=[A-Z])', ' ', uc["Name"])
        uc_name = uc_name.split()
        for sent in uc_name:
            dict = nlp.pos_tag(sent)
            for str1, tag in dict:
                # 收集其中所有的名词和动词
                if tag.startswith(('V', 'N')) and tag != 'NT' and tag != 'VA':  # NT是时间名词, VA是表语形容词
                    uc['key_name'].append(format_node_eng(str1))  # 都变为小写
        if len(uc['key_name']) == 0:  # 如果name中没有动、名词，则将整个name加入。
            uc['key_name'].append(format_node_eng(uc["Name"]))

        # 1、将 dataset 也作为key加入其中，相当于 uc path。
        uc_dataset = uc['dataset']
        uc['key_path'].append(format_node_eng(uc_dataset))

        # 2、将uctext依据句点和换行符进行断句。
        if 'Brief Description' not in uc or uc['Brief Description'] == 'None':
            continue
        uctext = None
        if "Brief Description" in uc and isinstance(uc['Brief Description'], str):
            if "i.e." in uc['Brief Description']:
                uc['Brief Description'] = uc['Brief Description'].replace("i.e.", "ie ")
            if "e.g." in uc['Brief Description']:
                uc['Brief Description'] = uc['Brief Description'].replace("e.g.", "eg ")
            uctext = re.split(r'[.\n]', uc['Brief Description'])
            uctext = [s for s in uctext if s.strip()]  # 去除空格行

        # 1、提取 uctext 中的关键词,存放在 uc['keyword']
        if not uctext and "Brief Description" in uc and uc['Brief Description'] != 'None':
            uctext = uc['Brief Description']
        for sent in uctext:
            dict = nlp.pos_tag(sent)
            for str2, tag in dict:
                # 收集其中所有的名词和动词
                if tag.startswith(('V')) and tag != 'NT' and tag != 'VA':  # NT是时间名词, VA是表语形容词
                    if format_node_eng(str2):
                        uc['key_act'].append(format_node_eng(str2))
                elif tag.startswith(('N')) and tag != 'NT' and tag != 'VA':  # NT是时间名词, VA是表语形容词
                    if format_node_eng(str2):
                        uc['key_obj'].append(format_node_eng(str2))


    nlp.close()
    return use_case_list


def re_extract_act_obj_eng(use_case_list,list1):
    nlp = StanfordCoreNLP(r'../0_Data/stanford-corenlp-4.0.0', memory='8g')
    for uc in use_case_list:
        uc[list1[0]], uc[list1[1]],act_list,obj_list = [], [],[], []
        for step in uc[list1[2]]:
            if contains_only_digits_symbols_spaces(step):  # 如果句子中没有字母，只有数字、符号和空格等
                continue
            va, action, object = None, [], []  # 清零上一个(VA（副词）)
            # 先从依存关系寻找dobj关系，如果有直接宾语dobj，则直接用
            tokens = nlp.word_tokenize(step)
            dependency = nlp.dependency_parse(step)
            for item in dependency:
                if item[0] == 'dobj':
                    action.append(tokens[item[1] - 1])
                    object.append([tokens[item[2] - 1]]) # 这里似乎多了一个中括号，不过因为从没有跑进这个分支，所以不影响
                    break

            # 如果没有dobj，则从动词中寻找
            if len(action) == 0:
                tags = nlp.pos_tag(step)
                for str, tag in tags:
                    if tag.startswith(('V')) and tag != 'VA':  # VA是表语形容词
                        action.append(str)  # NCET因为是rtcm，一句话只有一个动作。pub数据集不是严格rtcm，一句话中会有多个动作，所以全部收集。
                    elif tag.startswith(('N')) and tag != 'NT':  # NT是时间名词
                        object.append(str)
                    elif tag == 'VA':
                        va = str  # 记录va，在没有dobj的情况下使用va

            if len(action) != 0 and len(object) == 0 and va:
                object.append(va)

            if len(action) == 0:  # 如果还是有不存在act的情况，则取一个除obj外的词
                for str, tag in tags:
                    if object and str not in object and not is_pure_punctuation(str):
                        action.append(str)
                        break
            if len(object) == 0:
                if len(action) == 0:
                    action.append(max(step.split(), key=len))
                object.append(action[0])  # 很多语句就是没有obj，例如：人工返回
            if len(action) == 0:  # 很多句子中act被误认为是obj，例如：'User types Master Password'。找一个句子中一个没有大写的项。
                action = [item for item in object if item.islower()]

            # 如果还是有确实act/obj的
            if len(action) == 0 or len(object) == 0:
                if len(action) == 0:
                    action = object
                elif len(object) == 0:
                    object = action
                else:
                    print("test step is not have action or obj:", step)

            if len(action) == 0 or len(object) == 0:  # 最终
                print("test step is not have action or obj:", step)

            aaa = formate_node_eng_list(action)
            if aaa:
                act_list.append(aaa[-1])

            ooo = formate_node_eng_list(object)
            if ooo:
                obj_list.append(ooo[-1])


            if len(act_list)<1 and len(obj_list)>0: # 如果最后还是没有，则选一个
                act_list = obj_list
            uc[list1[0]].append(act_list)
            uc[list1[1]].append(obj_list)
            act_list,obj_list = [],[]

    nlp.close()
    return use_case_list


def local_id_to_global_id(uc_list):
    id = 0
    for uc in uc_list:
        uc['global id'] = id
        id += 1
    return uc_list


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
            if len(uc['AF act']) != 0:  # 有分支流的情况下
                for step_af_act, step_af_obj in zip(uc['AF act'], uc['AF obj']):
                    for act_seg,obj_seg in zip(step_af_act,step_af_obj):  # baseline only rgat 没有断句，所以少一层列表
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



def check_agc_cn(uc_list):
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
            if len(uc['AF act']) != 0:  # 有分支流的情况下
                for step_af_act, step_af_obj in zip(uc['AF act'], uc['AF obj']):
                    for act_seg,obj_seg in zip(step_af_act,step_af_obj):  # baseline only rgat 没有断句，所以少一层列表
                        if act_seg not in node_af_act:
                            node_af_act.append(act_seg)
                        if obj_seg not in node_af_obj:
                            node_af_obj.append(obj_seg)

            # 3、收集各类 key node
            if 'key_act' in uc:
                if isinstance(uc['key_act'], str):
                    uc['key_act'] = splite_2_list_en(uc['key_act'])  # 从str to list
                for item in uc['key_act']:
                    if item not in node_key:
                        node_key.append(item)
            
            if 'key_obj' in uc:
                if isinstance(uc['key_obj'], str):
                    uc['key_obj'] = splite_2_list_en(uc['key_obj'])
                for item in uc['key_obj']:
                    if item not in node_key:
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
    'node_af_act': len(node_af_act), 'node_af_obj': len(node_af_obj)},{'node_bf_act': node_bf_act, 'node_bf_obj': node_bf_obj}

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

def group_uc_fixed(use_case_list, group_uc):
    use_case_list_new = []
    for index_list in group_uc:
        sub_uc_list = []
        for index in index_list:
            sub_uc_list.append(use_case_list[index])
        use_case_list_new.append(sub_uc_list)
    return use_case_list_new

def get_edges_dict_and_node_to_UCText_only_rgat(use_case_list,uc_list_bp):
    edges_dict_list = []
    node_to_UCText_list = []
    for uc_list,uc_bp in zip(use_case_list,uc_list_bp):
        edge_dict, node_to_UCText = create_edges_af(uc_list,uc_bp)
        edges_dict_list.append(edge_dict)
        node_to_UCText_list.append(node_to_UCText)

    return edges_dict_list, node_to_UCText_list

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
        f'node总数: {node_total}, act bf 总数: {len(act_bf_node_num)}, obj bf : {len(obj_bf_node_num)},'
        f' key总数: {len(key_node_num)}, 子图中最多node数: {node_max_sub_data},'
        f'act af 总数: {len(act_af_node_num)}, obj af : {len(obj_af_node_num)}')

    return {'node_total': node_total, 'node_bf_act': len(act_bf_node_num), 'node_bf_obj': len(obj_bf_node_num),
            'node_key': len(key_node_num),'node_af_act': len(act_af_node_num), 'node_af_obj': len(obj_af_node_num),
            'node_max_sub': node_max_sub_data}

def make_bf_af_edges_baseline(bf_node_list, af_node_list,edges_dict,bp_list,label_bf,label_af):
    bp_list_cpy = bp_list.copy()
    if len(bp_list_cpy) != len(af_node_list):
        print(f"*** Warning!!! len(bp_list) != len(af_node_list) ***")
    if any(id > len(bf_node_list) for id in bp_list_cpy):
        print(f"*** Warning!!! 有分支点的index大于Basic Flow的step数 ***")
    for i in range(len(bp_list_cpy)):
        for bfnode in bf_node_list[bp_list_cpy[i]]: # 找到对应的bf
            for afnode in af_node_list[bp_list_cpy.index(bp_list_cpy[i])]: # 找到对应的afnode
                edges_dict[(bfnode, label_bf)].append((afnode, label_af))
        bp_list_cpy[i] = -99  # 防止重复使用同一个bp_index


    return edges_dict


# 参考实验室电脑 utils_2nd.py/create_edges_llm_only_rgat()
def create_edges_af(uc_list,uc_list_bp):
    edges_dict = {}
    node_to_UCText = {}  # 用于存放keyword、act、obj节点与UCText的对应关系
    for uc,uc_bp in zip(uc_list,uc_list_bp):
        # 0. get branching point info for uc_bp
        if len(uc['Alt. Flow'])>0:
            bp_list = module.get_branching_point_info(uc_bp)

        # 1. Make "keyword-keyword" edge(考虑了与前后node均相连)
        edges_dict = make_key_key_edge(uc, edges_dict)

        # 2. Add "keyword-act" edges, "keyword-obj" edges
        edges_dict = make_key_act_obj(uc, edges_dict, 'BF act', 'BF obj')
        edges_dict = make_key_act_obj(uc, edges_dict, 'AF act', 'AF obj')

        # 3. Add "act-act" edges (只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = make_common_edges(uc['BF act'], edges_dict, "BF act")
        edges_dict = make_common_edges(uc['AF act'], edges_dict, "AF act")

        # 4. Add "obj-obj" edges(只考虑了与后面node相连，暂不考虑和前面node相连)
        edges_dict = make_common_edges(uc['BF obj'], edges_dict, 'BF obj')
        edges_dict = make_common_edges(uc['AF obj'], edges_dict, 'AF obj')

        # 5. Add "act-obj" edge
        edges_dict = make_act_obj_edges_miss(uc, edges_dict,'BF act','BF obj' )
        edges_dict = make_act_obj_edges_miss(uc, edges_dict,'AF act','AF obj')

        # 6. Add "basic flow - alternative flow" edges
        if len(uc['Alt. Flow'])>0:
            edges_dict = make_bf_af_edges_baseline(uc['BF act'], uc['AF act'],edges_dict,bp_list,'BF act','AF act')
            edges_dict = make_bf_af_edges_baseline(uc['BF obj'], uc['AF obj'],edges_dict,bp_list,'BF obj','AF obj')


        # 6. create (node to uctext)'s map (其实是bf、af node to key node)
        node_to_UCText = create_node_uctext(node_to_UCText, uc)



    return edges_dict, node_to_UCText


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
    
    for act_list in uc['AF act']:
        for act_seg in act_list:
            if f"{act_seg}_{'AF act'}" not in node_to_UCText.keys():  # baseline only rgat 的方法少一层列表
                node_to_UCText[f"{act_seg}_{'AF act'}"] = []
            node_to_UCText[f"{act_seg}_{'AF act'}"].append(uc['global id'])

    for obj_list in uc['AF obj']:
        for obj_seg in obj_list:
            if f"{obj_seg}_{'AF obj'}" not in node_to_UCText.keys(): # baseline only rgat 的方法少一层列表
                node_to_UCText[f"{obj_seg}_{'AF obj'}"] = []
            node_to_UCText[f"{obj_seg}_{'AF obj'}"].append(uc['global id'])

    return node_to_UCText

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
def generate_dataset_4turbo(edges_dict, node_to_UCText, para_dict):
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
        # 分割节点名称和类型
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
    y = []  # 顺序按照node_to_id中keyword的顺序
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
            if len(related_node) == 0:  # line778-779,20250915 加  # 专门针对没有关联节点的keyword/bfnode，补充-1，不然会被认为是错误节点
                related_node = [-1]  # 如果没有关联的节点，则补充-1
            y.append(related_node)
        else:
            y.append([-1])  # 除了keyword之外的节点位置补充-1

    # 确定最大的列表长度为 节点总数,需要所有列表长度一致
    max_length = para_dict['max_node_subdata']  # 所有子图最多节点数原为1257
    # 创建一个填充后的二维列表，填充'-99'表示无效数据，补齐列表
    padded_y = [sublist + [-99] * (max_length - len(sublist)) for sublist in y]
    # 将填充后的列表转换为张量
    y_tensor = torch.tensor(padded_y, dtype=torch.long)

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

    return data

def re_extract_act_obj_eng_af(use_case_list,list1):
    nlp = StanfordCoreNLP(r'../0_Data/stanford-corenlp-4.0.0', memory='8g')
    for uc in use_case_list:
        uc[list1[0]], uc[list1[1]],act_list,obj_list = [], [],[], []
        if list1[2] not in uc or len(uc[list1[2]]) < 1:
            continue
        
        for step_list in uc[list1[2]]:  # af是嵌套数组
            for step in step_list:
                if contains_only_digits_symbols_spaces(step):  # 如果句子中没有字母，只有数字、符号和空格等
                    continue
                va, action, object = None, [], []  # 清零上一个(VA（副词）)
                # 先从依存关系寻找dobj关系，如果有直接宾语dobj，则直接用
                tokens = nlp.word_tokenize(step)
                dependency = nlp.dependency_parse(step)
                for item in dependency:
                    if item[0] == 'dobj':
                        action.append(tokens[item[1] - 1])
                        object.append([tokens[item[2] - 1]])  # 这里似乎多了一个中括号，不过因为没跑进过这个分支，所以不影响
                        break

                # 如果没有dobj，则从动词中寻找
                if len(action) == 0:
                    tags = nlp.pos_tag(step)
                    for str, tag in tags:
                        if tag.startswith(('V')) and tag != 'VA':  # VA是表语形容词
                            action.append(str)  # NCET因为是rtcm，一句话只有一个动作。pub数据集不是严格rtcm，一句话中会有多个动作，所以全部收集。
                        elif tag.startswith(('N')) and tag != 'NT':  # NT是时间名词
                            object.append(str)
                        elif tag == 'VA':
                            va = str  # 记录va，在没有dobj的情况下使用va

                if len(action) != 0 and len(object) == 0 and va:
                    object.append(va)

                if len(action) == 0:  # 如果还是有不存在act的情况，则取一个除obj外的词
                    for str, tag in tags:
                        if object and str not in object and not is_pure_punctuation(str):
                            action.append(str)
                            break
                if len(object) == 0:
                    if len(action) == 0:
                        action.append(max(step.split(), key=len))
                    object.append(action[0])  # 很多语句就是没有obj，例如：人工返回
                if len(action) == 0:  # 很多句子中act被误认为是obj，例如：'User types Master Password'。找一个句子中一个没有大写的项。
                    action = [item for item in object if item.islower()]

                # 如果还是有确实act/obj的
                if len(action) == 0 or len(object) == 0:
                    if len(action) == 0:
                        action = object
                    elif len(object) == 0:
                        object = action
                    else:
                        print("test step is not have action or obj:", step)

                if len(action) == 0 or len(object) == 0:  # 最终
                    print("test step is not have action or obj:", step)

                aaa = formate_node_eng_list(action)
                if aaa:
                    act_list.append(aaa[-1])

                ooo = formate_node_eng_list(object)
                if ooo:
                    obj_list.append(ooo[-1])

            uc[list1[0]].append(act_list)
            uc[list1[1]].append(obj_list)
            act_list,obj_list = [],[]

    nlp.close()
    return use_case_list

# 提取key path不动，只提取key act/obj
def extract_from_ucText_cn(uc_list):
    nlp = StanfordCoreNLP(r'../0_Data/stanford-corenlp-4.0.0', lang='zh', memory='8g')
    for uc in uc_list:
        del uc['key_obj']
        uc['key_act'] = [] # 不需要区分act/obj
        dict = nlp.pos_tag(uc['Name'])
        for str, tag in dict:
            # NT是时间名词, VA是表语形容词
            if tag.startswith(('V', 'N')) and tag != 'NT' and tag != 'VA': # NT是时间名词, VA是表语形容词
                uc['key_act'].append(str)  # 不需要区分act/obj
        
    nlp.close()
    return uc_list

def re_extract_act_obj_cn(uc_list,list1):
    nlp = StanfordCoreNLP(r'../0_Data/stanford-corenlp-4.0.0', lang='zh', memory='8g')
    for uc in uc_list:
        uc[list1[0]], uc[list1[1]],act_list,obj_list = [], [],[], []
        for step in uc[list1[2]]:
            if contains_only_digits_symbols_spaces(step):
                continue
            va, action, object = None, [], []  # 清零上一个(VA（副词）)
            # 先从依存关系寻找dobj关系，如果有直接宾语dobj，则直接用
            tokens = nlp.word_tokenize(step)
            dependency = nlp.dependency_parse(step)
            for item in dependency:
                if item[0] == 'dobj':
                    action.append(tokens[item[1] - 1])
                    object.append(tokens[item[2] - 1])
                    # break
            if len(action)>1:
                action = [item for item in action if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的
            if len(object)>1:
                object = [item for item in object if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的


            # 如果没有dobj，则从动词中寻找
            if len(action) == 0:
                tags = nlp.pos_tag(step)
                for str, tag in tags:
                    if tag.startswith(('V')) and tag != 'VA':  # VA是表语形容词
                        action.append(str)  # NCET因为是rtcm，一句话只有一个动作。pub数据集不是严格rtcm，一句话中会有多个动作，所以全部收集。
                    elif tag.startswith(('N')) and tag != 'NT':  # NT是时间名词
                        object.append(str)
                    elif tag == 'VA':
                        va = str  # 记录va，在没有dobj的情况下使用va

            if len(action) != 0 and len(object) == 0 and va:
                object.append(va)

            if len(action) == 0:  # 如果还是有不存在act的情况，则取一个除obj外的词
                for str, tag in tags:
                    if object and str not in object and not is_pure_punctuation(str):
                        action.append(str)
                        break
            if len(object) == 0:
                if len(action) == 0:
                    action.append(max(step.split(), key=len))
                object.append(action[0])  # 很多语句就是没有obj，例如：人工返回
            if len(action) == 0:  # 很多句子中act被误认为是obj，例如：'User types Master Password'。找一个句子中一个没有大写的项。
                action = [item for item in object if item.islower()]

            # 如果还是有确实act/obj的
            if len(action) == 0 or len(object) == 0:
                if len(action) == 0:
                    action = object
                elif len(object) == 0:
                    object = action
                else:
                    print("test step is not have action or obj:", step)

            if len(action) == 0 or len(object) == 0:  # 最终
                print("test step is not have action or obj:", step)

            aaa = formate_node_eng_list(action)
            if len(aaa)>1:
                aaa = [item for item in aaa if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的
            if aaa:
                act_list.append(aaa[-1])

            ooo = formate_node_eng_list(object)
            if len(ooo)>1:
                ooo = [item for item in ooo if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的
            if ooo:
                obj_list.append(ooo[-1])


            if len(act_list)<1:
                if len(obj_list)>0: # 如果最后还是没有，则选一个
                    act_list.append(obj_list[0])
                else:
                    act_list.append(step)
            if len(obj_list)<1: # 如果最后还是没有，则选一个
                if len(act_list)>0:
                    obj_list.append(act_list[0])
                else:
                    obj_list.append(step)
            uc[list1[0]].append(act_list)
            uc[list1[1]].append(obj_list)
            act_list,obj_list = [],[]

    nlp.close()
    return uc_list

def re_extract_act_obj_cn_af(use_case_list,list1):
    nlp = StanfordCoreNLP(r'../0_Data/stanford-corenlp-4.0.0', lang='zh', memory='8g')

    for uc in use_case_list:
        uc[list1[0]], uc[list1[1]],act_list,obj_list = [], [],[], []
        if list1[2] not in uc or len(uc[list1[2]]) < 1:
            continue
        
        for step_list in uc[list1[2]]:  # af是嵌套数组
            for step in step_list:
                if contains_only_digits_symbols_spaces(step):  # 如果句子中没有字母，只有数字、符号和空格等
                    continue
                va, action, object = None, [], []  # 清零上一个(VA（副词）)
                # 先从依存关系寻找dobj关系，如果有直接宾语dobj，则直接用
                tokens = nlp.word_tokenize(step)
                dependency = nlp.dependency_parse(step)
                for item in dependency:
                    if item[0] == 'dobj':
                        action.append(tokens[item[1] - 1])
                        object.append(tokens[item[2] - 1])
                        # break

                if len(action)>1:
                    action = [item for item in action if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的
                if len(object)>1:
                    object = [item for item in object if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的



                # 如果没有dobj，则从动词中寻找
                if len(action) == 0:
                    tags = nlp.pos_tag(step)
                    for str, tag in tags:
                        if tag.startswith(('V')) and tag != 'VA':  # VA是表语形容词
                            action.append(str)  # NCET因为是rtcm，一句话只有一个动作。pub数据集不是严格rtcm，一句话中会有多个动作，所以全部收集。
                        elif tag.startswith(('N')) and tag != 'NT':  # NT是时间名词
                            object.append(str)
                        elif tag == 'VA':
                            va = str  # 记录va，在没有dobj的情况下使用va

                if len(action) != 0 and len(object) == 0 and va:
                    object.append(va)

                if len(action) == 0:  # 如果还是有不存在act的情况，则取一个除obj外的词
                    for str, tag in tags:
                        if object and str not in object and not is_pure_punctuation(str):
                            action.append(str)
                            break
                if len(object) == 0:
                    if len(action) == 0:
                        action.append(max(step.split(), key=len))
                    object.append(action[0])  # 很多语句就是没有obj，例如：人工返回
                if len(action) == 0:  # 很多句子中act被误认为是obj，例如：'User types Master Password'。找一个句子中一个没有大写的项。
                    action = [item for item in object if item.islower()]

                # 如果还是有确实act/obj的
                if len(action) == 0 or len(object) == 0:
                    if len(action) == 0:
                        action = object
                    elif len(object) == 0:
                        object = action
                    else:
                        print("test step is not have action or obj:", step)

                if len(action) == 0 or len(object) == 0:  # 最终
                    print("test step is not have action or obj:", step)

                aaa = formate_node_eng_list(action)
                if len(aaa)>1:
                    aaa = [item for item in aaa if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的
                if aaa:
                    act_list.append(aaa[-1])

                ooo = formate_node_eng_list(object)
                if len(ooo)>1:
                    ooo = [item for item in ooo if not re.match(r'^\d+$|^[a-zA-Z]+$', item)]  # 如果有多个，且有纯字母数字的，则去掉纯字母数字的
                if ooo:
                    obj_list.append(ooo[-1])

            uc[list1[0]].append(act_list)
            uc[list1[1]].append(obj_list)
            act_list,obj_list = [],[]

    nlp.close()
    return use_case_list

if __name__ == '__main__':

    task_name = 'make_pt_pub'
    print(f'*** task_name: {task_name}, Starting time: {datetime.now()}  !!!!! ***')

    # 使用Stanford方法提取bf node和af node
    if task_name =='extract_node_only_rgat_pub': # 仿制： _8_/task5
        bf_path = "E:/bertTest/20240421/ControlledExper/4_dataset_pred_node/ERNIE_4_Turbo_8k/with_tp/2nd_round/step_seg/Ernie_pub_seg.json"
        af_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json" # 只是用这些数据中的desc和af，所以跟ERNIE/gpt无关。
        out_path = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/Standard_pub_alt_node.json'
        
        uc_list = read_uc_from_stand_json(af_path)
        bf_list = read_uc_from_json(bf_path)
        for uc,uc_bf in zip(uc_list,bf_list):
            if uc['id'] != uc_bf['index'] or uc['dataset'] != uc_bf['dataset'] or uc['Name'] != uc_bf['ucName']:
                print(f'Error！ {current_function()}')
            for key in ['BF act','BF obj','key_name', 'key_act','key_obj','Basic flow','Brief Description']: # 删除其他方法提取的node.这个文件中的'Basic flow'已经用LLM断句，所以要退回之前的。
                if key in uc:
                    uc[key] = []
            uc['Basic flow'] = uc_bf['steps']
            if 'uctext' in uc_bf:
                uc['Brief Description'] = uc_bf['uctext']

        # 3.1、用Stanford提取key node（uc name\uc text)。  # 参考 _8_/task5
        uc_list = extract_from_ucText_eng(uc_list)
        # 3.2、用当前方法提取uc step中的act/obj。第二个参数为涉及到的key名称
        uc_list = re_extract_act_obj_eng(uc_list,['BF act','BF obj','Basic flow'])
        uc_list = re_extract_act_obj_eng_af(uc_list,['AF act','AF obj','Alt. Flow'])

        write_uc_to_stand_json(out_path,uc_list)

    elif task_name == 'make_pt_pub':   # 用到的函数全部来自于 7_make_pt_file.py
        branching_point_path = '../0_Data/5_branching_point/1_gpt_added_bp/pub_with_gpt_bp.json'  # 只需要其中标注的分支点信息

        in_path = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/Stanford_pub_alt_node.json'
        pt_save_path = 'E:/Trae_project/AFGen_pt_file/1_wo_SIP/baseline_only_rgat_pub_withbp_20.pt'
        
        # 1. 读取文件
        uc_list = read_uc_from_stand_json(in_path)

        # 2. 将local id 更新为全局 id
        uc_list = local_id_to_global_id(uc_list)
        uc_list_bp = local_id_to_global_id(read_uc_from_stand_json(branching_point_path))


        # 3. 生成全局edge数目和 node 数目
        argc1 = check_agc(uc_list)

        # 4. 进行分组（使用固定分组/随机分组）
        uc_list = group_uc_fixed(uc_list, PUB_GROUPING_UC_20_1) # ncet数据集是这个：GROUPING_UC_20
        uc_list_bp = group_uc_fixed(uc_list_bp, PUB_GROUPING_UC_20_1) # ncet数据集是这个：GROUPING_UC_20

        # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
        edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_only_rgat(uc_list,uc_list_bp)

        # 5.1 统计：所有子图中最多的节点数（act+obj+key）
        argc2 = count_node_data_dict_sub(edges_dict_list)

        # find_different(argc1,argc2)  # 可删

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

    elif task_name == 'extract_node_only_rgat_ncet': 
        af_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json" # 只是用这些数据中的desc和af，所以跟ERNIE/gpt无关。
        out_path = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/Stanford_NCET_alt_node.json'

        uc_list = read_uc_from_stand_json(af_path)

        # 3.1、用Stanford提取key node（key path不动，只重新提取key_act 和 key_obj)。
        uc_list = extract_from_ucText_cn(uc_list)
        # 3.2、用当前方法提取uc step中的act/obj。第二个参数为涉及到的key名称
        uc_list = re_extract_act_obj_cn(uc_list,['BF act','BF obj','Basic flow'])
        uc_list = re_extract_act_obj_cn_af(uc_list,['AF act','AF obj','Exc. Flow'])

        for uc in uc_list:
            if len(uc['BF act'])<1 or len(uc['BF obj'])<1 or len(uc['key_act'])<1:
                if "Exc. Flow" in uc and len(uc['AF act'])<1 or len(uc['AF obj'])<1:
                    print(f"Error uc id:{uc['global id']}, BF act:{uc['BF act']}, BF obj:{uc['BF obj']}, key_act:{uc['key_act']}, AF act:{uc['AF act']}, AF obj:{uc['AF obj']}")

        write_uc_to_stand_json(out_path,uc_list)  # 明天来对比一下输入和输出文件的差别，是不是只有几个act obj有差别？


    elif task_name == 'make_pt_ncet': # 模仿make_pt_pub
        in_path = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/Standard_NCET_alt_node.json'
        pt_save_path = 'E:/Trae_project/AFGen_pt_file/1_wo_SIP/baseline_only_rgat_ncet_20.pt'

        # 1. 读取文件
        uc_list = read_uc_from_stand_json(in_path)

        # 2. 将local id 更新为全局 id
        uc_list = local_id_to_global_id(uc_list)

        # 3. 生成全局edge数目和 node 数目
        argc1,diff_dict1 = check_agc_cn(uc_list)

        # 4. 进行分组（使用固定分组/随机分组）
        uc_list = group_uc_fixed(uc_list, GROUPING_UC_20)

        # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
        edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_only_rgat(uc_list)

        # 5.1 统计：所有子图中最多的节点数（act+obj+key）
        argc2,diff_dict2 = count_node_data_dict_sub(edges_dict_list)

        # 用于检查两次统计的差异
        # find_different(diff_dict1,diff_dict2,['node_bf_act','node_bf_obj'])  # 可删

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

    print(f'*** task_name: {task_name} Finish! Finish time: {datetime.now()} *** ')