from imports import json,inspect,Path,string,re,REPLACEMENT_MAP

# 读取用dict保存的uc。BFGen的按行写的json文件
def read_uc_from_json(file_path):
    use_case_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # line = line.replace("'", '"')  # 有时候数据集中会有多余的单引号或者双引号
            try:
                uc = json.loads(line)
                use_case_list.append(uc)
            except json.JSONDecodeError as e:
                print(f"错误信息: {e},  Line: {inspect.currentframe().f_lineno}, json读取失败")
    return use_case_list

def output_uc_to_json(use_case_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(use_case_list, json_file, ensure_ascii=False, indent=4)
    
    print(f"成功 {len(use_case_list)} 个use case并保存到 {file_path}")

def read_uc_from_stand_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        uc_list_alt = json.load(f)
    return uc_list_alt

def write_uc_to_stand_json(json_file_path, use_case_list):
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(use_case_list, json_file, ensure_ascii=False, indent=4)


def print_location():
    """高效获取调用者信息"""
    # 获取当前帧的上一帧（调用者帧）
    caller_frame = inspect.currentframe().f_back
    # 直接提取行号和函数名
    line_no = caller_frame.f_lineno
    func_name = caller_frame.f_code.co_name
    # 打印结果
    print(f"Print语句位置 → 行号: {line_no}, 函数名: '{func_name}'")
    # 释放帧引用避免内存泄漏
    del caller_frame

def get_json_paths(directory):
    dir_path = Path(directory)
    # 递归匹配所有.json文件（含子目录）
    return [str(file) for file in dir_path.rglob('*.json')]

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


def metric_auc_llm_match(uc): # 这种情况不包括负例也有命中的情况，即af中只存在正例，而pred中的负例都是预测错误的情况，不存在负例预测对的情况。
    ground_truth = uc["AF act"] + uc["AF obj"]

    # 该问题下公式可化为： auc = (|tp|*0.5) / |ground true|
    auc = (0.5 * len(uc["tp"])) / len(ground_truth) if len(ground_truth) > 0 else 0

    return auc

def metric_precision_strict_llm_match(uc):
    # p = tp/(tp+fp)  精度 = 正确预测的数量 / 预测的总数量 ； 查准率

    pred_list = uc["pred_af_act"] + uc["pred_af_obj"]
    p_value = len(uc["tp"]) / (len(pred_list)) if (len(pred_list)) > 0 else 0

    return p_value

def metric_recall_llm_match(uc):
    # r = tp/(tp+fn)  召回率 = 正确预测的数量 / 所有正确的总数量 ；查全率
    actu_list = uc["AF act"] + uc["AF obj"]

    recall_value = len(uc["tp"]) / (len(actu_list)) if len(actu_list) > 0 else 0

    return recall_value

def metric_F1(uc):
    # fi = 2PR/(P+R)

    p_value = uc["p"]
    recall_value = uc["r"]

    if (p_value + recall_value) != 0:
        f1_value = (2 * p_value * recall_value) / (p_value + recall_value)
    else:
        f1_value = 0

    return f1_value


def mix_items(uc_list_ernie,uc_alt_node_list_ernie,list1):  # 第三个参数是混合字典时，需要保留的item
    for uc,uc_ref in zip(uc_list_ernie, uc_alt_node_list_ernie):
        if uc['id'] != uc_ref['id']:
            print(f"Error: uc id {uc['id']} 不等于 uc_ref id {uc_ref['id']}")
        # 只保留uc_ref中的item
        uc_ref_filtered = {k: v for k, v in uc_ref.items() if k in list1}
        
        # 将uc_ref_filtered的内容添加到uc中
        uc.update(uc_ref_filtered)
    
    
    return uc_list_ernie

def count_node(uc_list, label):
    count = 0
    for uc in uc_list:
        if label in uc:
            count += len(uc[label])
    return count
    
def clean_and_convert(text):
    # 1. 删除所有换行符
    text = text.replace('\r', '').replace('\n', '')
    
    # 2. 中文标点转英文
    punct_map = {"，": ",", "。": ".", "；": ";", "：": ":", "？": "?", "！": "!", 
                 "“": "\"", "”": "\"", "‘": "'", "’": "'", "（": "(", "）": ")"}
    trans_table = str.maketrans(punct_map)
    return text.translate(trans_table)

def clean_string(s):
    # 定义要删除的字符：数字 + 标点符号
    chars_to_remove = string.digits + string.punctuation
    return s.strip(chars_to_remove)

# 输出列表的深度，即几级列表
def list_depth_recursive(lst):
    if not isinstance(lst, list):  # 非列表元素深度为0
        return 0
    if not lst:  # 空列表深度为1
        return 1
    return 1 + max((list_depth_recursive(item) for item in lst), default=0)


def current_function():
    # 获取当前栈帧 → 提取函数名
    func_name = inspect.currentframe().f_code.co_name  
    return func_name


def convert_string_to_list(string):
    string = clean_and_convert(string)  # 清理字符串中的所有换行符，以及转变中文符号为英文符号
    lists = re.findall(r'\[(.*?)\]', string)
    result,result_1 = [],[]
    for item in lists:
        # 按逗号分割并去除首尾空格
        result.extend([x.strip() for x in item.split(',') if x.strip()])

    if isinstance(result, list):
        for item in result:
            item = item.strip().strip("'\"`[](){}<>")
            result_1.append(item)
    else:
        print(f"Error: Expected a list but got {type(result)}")
    return result_1


def find_list_in_string(uc_list):
    for uc in uc_list:
        if 'pred_af' in uc:
            if 'pred_af_act' in uc and 'pred_af_obj' in uc and isinstance(uc['pred_af_act'], str) and isinstance(uc['pred_af_obj'], str):
                # 将字符串转换为列表
                uc['pred_af_act'] = convert_string_to_list(uc['pred_af_act'])
                uc['pred_af_obj'] = convert_string_to_list(uc['pred_af_obj'])
            
            else:
                print(f"uc id: {uc['id']} has no af_act or af_obj")
            
    return uc_list

def delete_error(list1, list2):
    # 1、删除空项、空格、或者仅包含符号的项
    symbols_to_check = set('!@#$%^&*()"')  # 定义一个符号集合，用于检查元素中是否包含这些符号

    # 使用列表推导式来过滤列表
    list1 = [item for item in list1 if not (
            not item  # 检查是否为空字符串
            or any(symbol in item for symbol in symbols_to_check)  # 检查是否包含任何指定符号
    )]

    # 2、删除error node
    filtered_list1 = []

    # 遍历list1中的每个元素
    for item in list1:
        # 检查item中是否包含list2中的任意字符串
        contains_substring = any(sub_str in item for sub_str in list2)
        # 如果不包含，则将其添加到过滤后的列表中
        if not contains_substring:
            filtered_list1.append(item)
    return filtered_list1


# 判断是否都是同一方法下的文件.例如文件名下是否都包含ernie，即为ernie baseline方法的文件。
def check_all_under_same_method(str1,list1):
    list1 = list(map(str.lower, list1))
    str1 = str1.lower()
    if not all(str1 in s for s in list1):
        print(f'{str1} 不在任意路径中，选错了文件！！！')  # 检测文件是同一个baseline的

        for path in list1:
            if str1 not in path:
                print(f'{str1} 不在路径 {path} 中！！！')  # 检测文件是同一个baseline的


        return False
    return True

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

def delete_error(list1, list2):
    # 1、删除空项、空格、或者仅包含符号的项
    symbols_to_check = set('!@#$%^&*()"')  # 定义一个符号集合，用于检查元素中是否包含这些符号

    # 使用列表推导式来过滤列表
    list1 = [item for item in list1 if not (
            not item  # 检查是否为空字符串
            or any(symbol in item for symbol in symbols_to_check)  # 检查是否包含任何指定符号
    )]

    # 2、删除error node
    filtered_list1 = []

    # 遍历list1中的每个元素
    for item in list1:
        # 检查item中是否包含list2中的任意字符串
        contains_substring = any(sub_str in item for sub_str in list2)
        # 如果不包含，则将其添加到过滤后的列表中
        if not contains_substring:
            filtered_list1.append(item)
    return filtered_list1


# 来自 BFGen项目中 实验室电脑 PE_based_public_dataset/task13、task14
def eval_Ernie_pred_ncet(use_case_list):
    for uc in use_case_list:
        if "tp_act" in uc and "tp_obj" in uc:
            uc["tp"] = uc["tp_act"] + uc["tp_obj"]
            uc['AF act'] = flatten_list(uc['AF act'])
            uc['AF obj'] = flatten_list(uc['AF obj'])

            # 2.1、AUC
            current_uc = metric_auc_llm_match(uc)
            uc["auc"] = current_uc

            # 2.2、Precision
            p = metric_precision_strict_llm_match(uc)
            uc["p"] = p

            # 2.3、Recall
            r = metric_recall_llm_match(uc)
            uc["r"] = r

            # 2.4、F1 Score
            f1 = metric_F1(uc)
            uc["f1"] = f1

    return use_case_list


def calculate_all_dataset(use_case_list, read_path):
    p, r, f1, auc = [], [], [], []
    for uc in use_case_list:
        if "p" in uc:
            p.append(uc["p"])
            r.append(uc["r"])
            f1.append(uc["f1"])
            auc.append(uc["auc"])

    print(f" len of metrics: {len(p)}, {len(r)}, {len(f1)}, {len(auc)}")

    with open(read_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps("Precision mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(p) / len(p), ensure_ascii=False) + '\n')
        f.write(json.dumps("Recall mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(r) / len(r), ensure_ascii=False) + '\n')
        f.write(json.dumps("F1 Score mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(f1) / len(f1), ensure_ascii=False) + '\n')
        f.write(json.dumps("AUC mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(auc) / len(auc), ensure_ascii=False) + '\n')


def calculate_all_dataset_eval(use_case_list, read_path,eval_list):
    eval_list = flatten_list(eval_list)
    use_case_list = local_id_to_global_id(use_case_list)  # 将uc_list中的id转换为全局id

    p, r, f1, auc = [], [], [], []
    for uc in use_case_list:
        if uc['global id'] in eval_list:
            if "p" in uc:
                p.append(uc["p"])
                r.append(uc["r"])
                f1.append(uc["f1"])
                auc.append(uc["auc"])

    print(f" len of metrics: {len(p)}, {len(r)}, {len(f1)}, {len(auc)}")

    with open(read_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps("Precision mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(p) / len(p), ensure_ascii=False) + '\n')
        f.write(json.dumps("Recall mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(r) / len(r), ensure_ascii=False) + '\n')
        f.write(json.dumps("F1 Score mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(f1) / len(f1), ensure_ascii=False) + '\n')
        f.write(json.dumps("AUC mean: ", ensure_ascii=False) + '\n')
        f.write(json.dumps(sum(auc) / len(auc), ensure_ascii=False) + '\n')



def contains_only_digits_symbols_spaces(s):
    for char in s:
        if char.isalpha():
            return False
        # 这里假设空格和符号都不是字母也不是数字
    return True

def local_id_to_global_id(uc_list):
    id = 0
    for uc in uc_list:
        uc['global id'] = id
        id += 1
    return uc_list


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


def get_3_list(list1): # 专用于unify_ncet_format()函数
    list2 = []
    for item in list1:
        list3 = []
        for seg in item:
            list3.append([seg])
        list2.append(list3)

    return list2

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


def get_test_sub_graph(uc_list,test_sub_graph):
    new_uc_list = []
    for sub_graph in test_sub_graph:
        new_sub = []
        for global_id in sub_graph:
            if uc_list[global_id]['global id'] != global_id:
                print(f'global id 对不上！{global_id}')
            new_sub.append(uc_list[global_id])
        new_uc_list.append(new_sub)
    
    return new_uc_list