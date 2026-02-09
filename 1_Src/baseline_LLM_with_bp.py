from imports import datetime,PUB_GROUPING_UC_20_1,GROUPING_UC_20,importlib,GPT_4o,ERNIE_4_turbo,json,requests,qianfan,re,Counter
from utils import read_uc_from_stand_json,read_uc_from_json,write_uc_to_stand_json,flatten_list,get_test_sub_graph
spec = importlib.util.spec_from_file_location("module_name", "7_make_pt_file.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def get_pred_bp_af_by_llm(uc_list,out_path,model_name,uc_ref_list):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for sub_graph,sub_graph_ref in zip(uc_list,uc_ref_list):
            json_file.write('[\n')  # 手动添加逗号和换行
            for uc,uc_ref in zip(sub_graph,sub_graph_ref):
                print(f"*** uc global id: {uc['global id']} ***")
                if uc['dataset'] != uc_ref['dataset'] and uc['id'] != uc_ref['id']:
                    print(f"Dataset or ID mismatch: {uc['dataset']} vs {uc_ref['dataset']}, {uc['id']} vs {uc_ref['id']}")
                    break
                    
                # 使用LLM：判断是否有分支点、有的话生成分支流
                if model_name == ERNIE_4_turbo:
                    uc_ref['pred_af'] = pred_bp_af_ernie(uc_ref,model_name)
                elif model_name == GPT_4o:
                    uc_ref['pred_af'] = pred_bp_af_gpt(uc_ref,model_name)
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
                        
                json.dump(uc_ref, json_file, ensure_ascii=False, indent=4)
                json_file.write(',\n')  # 手动添加逗号和换行

            json_file.write(']\n')  # 手动添加逗号和换行

    return uc_ref_list


def pred_bp_af_gpt(uc,model_name):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"


    sys_pmt = "Given a use case, decide whether it contains branch flows." \
    "If there are no branch flows, output exactly the single word: false.Do not add any other words, symbols, or explanations." \
    "If there are branch flows, output them strictly in the following Python dictionary format:" \
    "{step_index: [[Branch flow 1], [Branch flow 2], ...],...}"\
    "where step_index is the index (starting from 0) of the sublist in basic flow where the branch occurs, "\
    "and the value is a list of branch flows that you generate which originate from that step. "\
    "Do not output anything else outside this format."
    
    if 'Brief Description' in uc:
        user_pmt = "Use case description:'" + str(uc['Brief Description'])
    
    user_pmt += ".Use case basic flow:'" + str(uc['Basic flow'])

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }

    # 定义请求数据
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": sys_pmt
            },
            {
                "role": "user",
                "content": user_pmt
            }
        ]
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 处理响应
    if response.status_code == 200:
        # 如果请求成功，解析JSON响应
        result = response.json()
        exist = result["choices"][0]["message"]["content"].strip()  # 去除两端空格
        return exist
    else:
        print(f"Error: {response.status_code}, {response.text}")
        exist = pred_bp_af_gpt(uc,model_name)
        return exist

def pred_bp_af_ernie(uc,model_name):
    chat_comp = qianfan.ChatCompletion()

    # ERNIE 模型没有 sys_pmt
    if 'Brief Description' in uc:
        user_pmt = "Use case description:'" + str(uc['Brief Description'])
    
    user_pmt += ".Use case basic flow:'" + str(uc['Basic flow']) + "'.Given a use case, decide whether it contains branch flows." \
    "If there are no branch flows, output exactly the single word: false.Do not add any other words, symbols, or explanations." \
    "If there are branch flows, output them strictly in the following Python dictionary format:" \
    "{step_index: [[Branch flow 1], [Branch flow 2], ...],...}"\
    "where step_index is the index (starting from 0) of the sublist in basic flow where the branch occurs, "\
    "and the value is a list of branch flows that you must generate yourself, describing branch flows related to that step."\
    "Do not output anything else outside this format."

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"].strip()  # 去除两端空格
    if '{' in exist and '}' in exist:
        exist = re.search(r'\{[^}]*\}', exist).group()
        return exist
    elif 'false' in exist.lower() and '{' not in exist:
        return 'false'

def get_pred_af_given_bp_by_llm_pub(uc_ref_list,out_path,model_name,uc_bp_info):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for sub_graph,bp_sub in zip(uc_ref_list,uc_bp_info):
            json_file.write('[\n')  # 手动添加逗号和换行
            for uc_ref,bp_uc in zip(sub_graph,bp_sub):
                print(f"*** uc global id: {uc_ref['global id']} ***")
                if bp_uc['dataset'] != uc_ref['dataset'] or bp_uc['id'] != uc_ref['id'] or len(bp_uc['Basic flow'])!=len(uc_ref['Basic flow']):
                    print(f"Dataset or ID mismatch: {bp_uc['golbal id']} vs {uc_ref['golbal id']}")
                    break
                    
                uc_ref['pred_af'] = []
                for af in bp_uc['Alt. Flow']:
                    # 获得分支点
                    bp_local = list(af)[0].split('_')[0]
                                
                    # 使用LLM：在给定分支点的情况下，生成分支流
                    if model_name == ERNIE_4_turbo:
                        uc_ref['pred_af'].append({str(bp_local):pred_af_given_bp_ernie(uc_ref,model_name,bp_local)})
                    elif model_name == GPT_4o:
                        uc_ref['pred_af'].append({str(bp_local):pred_af_given_bp_gpt(uc_ref,model_name,bp_local)})
                    else:
                        print(f"Model {model_name} not recognized. Skipping prediction.")
                        
                json.dump(uc_ref, json_file, ensure_ascii=False, indent=4)
                json_file.write(',\n')  # 手动添加逗号和换行

            json_file.write('],\n')  # 手动添加逗号和换行

    return uc_ref_list

def pred_af_given_bp_gpt(uc,model_name,bp_local):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"

    sys_pmt = "Given a use case and a branch point," \
    "where the branch point is the index (starting from 0) of the sublist in the basic flow where branch flows occur." \
    "Generate branch flows for this branch point and output them strictly in the following Python List format:" \
    "[[Branch flow 1], [Branch flow 2], ...]"\
    ".Each inner list contains one branch flow that you generate yourself, describing an alternative or exceptional situation that may occur at that basic flow sublist."\
    "Do not output anything else outside this format."
    
    if 'Brief Description' in uc:
        user_pmt = "Use case description:'" + str(uc['Brief Description'])
    
    user_pmt += ".Use case basic flow:'" + str(uc['Basic flow'])
    user_pmt += '.Branch point:'+ str(bp_local)

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }

    # 定义请求数据
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": sys_pmt
            },
            {
                "role": "user",
                "content": user_pmt
            }
        ]
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 处理响应
    if response.status_code == 200:
        # 如果请求成功，解析JSON响应
        result = response.json()
        exist = result["choices"][0]["message"]["content"].strip()  # 去除两端空格
        return exist
    else:
        print(f"Error: {response.status_code}, {response.text}")
        exist = pred_bp_af_gpt(uc,model_name)
        return exist

def pred_af_given_bp_ernie(uc,model_name,bp_local):
    chat_comp = qianfan.ChatCompletion()# ERNIE 模型没有 sys_pmt
    if 'Brief Description' in uc:
        user_pmt = "Use case description:" + str(uc['Brief Description'])
    
    user_pmt += ".Use case basic flow:" + str(uc['Basic flow']) + ".Branch point:"+ str(bp_local) +".Given a use case and a branch point," \
    "where the branch point is the index (starting from 0) of the sublist in the basic flow where branch flows occur." \
    "Generate branch flows for this branch point and output them strictly in the following Python List format:" \
    "[[Branch flow 1], [Branch flow 2], ...]"\
    ".Each inner list contains one branch flow that you generate yourself, describing an alternative or exceptional situation that may occur at that basic flow sublist."\
    "Do not output anything else outside this format."

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"].strip()  # 去除两端空格
    if '[' in exist and ']' in exist:
        return exist
    elif '[' not in exist:
        exist = pred_af_given_bp_ernie(uc,model_name,bp_local)
        return exist


def pred_af_node_gpt(af_list,GPT_4o,label):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"

    if label == 'act':
        sys_pmt = "Extract all the verbs from str1, output as a **List**."
    elif label == 'obj':    
        sys_pmt = "Extract all the nouns from str1, output as a **List**."

    user_pmt = "str1='" + str(af_list)

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }

    # 定义请求数据
    data = {
        "model": GPT_4o,
        "messages": [
            {
                "role": "system",
                "content": sys_pmt
            },
            {
                "role": "user",
                "content": user_pmt
            }
        ]
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 处理响应
    if response.status_code == 200:
        # 如果请求成功，解析JSON响应
        result = response.json()
        exist = result["choices"][0]["message"]["content"].strip()  # 去除两端空格
        return exist
    else:
        print(f"Error: {response.status_code}, {response.text}")
        exist = pred_af_node_gpt(af_list,GPT_4o,label)
        return exist
    

def has_duplicate_keys(dict_list):
    all_keys = []
    for d in dict_list:
        all_keys.extend(d.keys())
    return len(all_keys) != len(set(all_keys))

def merge_dicts_safe(dict_list):
    if has_duplicate_keys(dict_list):
        return None  # 或者抛出异常，根据您的错误处理偏好
    merged_dict = {}
    for d in dict_list:
        merged_dict.update(d)
    return merged_dict

def find_tp_gpt(node_list,model_name,af_list):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"


    sys_pmt = "Check which words from the given list appear in the paragraph, either exactly or with similar meaning." \
    "Output only those words in a Python list format:" \
    "['word1', 'word2', ...]"\
    "Include duplicates if they appear in the input list and are found in the paragraph."\
    "Do not remove duplicates, add new words, or output anything else outside the list."
    
    user_pmt = "Word list:" + str(node_list)
    user_pmt += "Paragraph:'" + str(af_list)

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key
    }

    # 定义请求数据
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": sys_pmt
            },
            {
                "role": "user",
                "content": user_pmt
            }
        ]
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 处理响应
    if response.status_code == 200:
        # 如果请求成功，解析JSON响应
        result = response.json()
        exist = result["choices"][0]["message"]["content"].strip()  # 去除两端空格
        return exist
    else:
        print(f"Error: {response.status_code}, {response.text}")
        exist = pred_bp_af_gpt(uc,model_name)
        return exist
    

def pairwise_auc(y_true, y_score): #存在负值命中的情况，即uc没有bp，baseline也判断没有bp的情况，所以与metric_auc_llm_match函数不同
    """
    计算基于 pairwise 比较定义的 AUC（无外部库版）
    y_true: list[int]  —— ground truth 标签 (0/1)
    y_score: list[float] —— 模型预测分数或概率（可以是0/1）
    """
    assert len(y_true) == len(y_score), "ground truth 和预测长度必须一致"

    # 取出正样本和负样本的分数
    pos_scores = [s for y, s in zip(y_true, y_score) if y == 1]
    neg_scores = [s for y, s in zip(y_true, y_score) if y == 0]

    n_pos, n_neg = len(pos_scores), len(neg_scores)

    # 如果没有正或没有负，AUC无法定义
    if n_pos == 0 or n_neg == 0:
        return None  # 或返回0.5作为工程定义

    # 逐对比较
    total = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                total += 1
            elif p == n:
                total += 0.5
            # 若 p < n，加 0（无需写）

    auc = total / (n_pos * n_neg)
    return auc

def precision_recall_f1(y_true, y_pred):
    """
    计算 Precision / Recall / F1
    y_true: list[int] - ground truth 标签 (0/1)
    y_pred: list[int] - 模型预测标签 (0/1)
    """
    assert len(y_true) == len(y_pred), "ground truth 和预测长度必须一致"

    TP = FP = FN = TN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            TP += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:
            FN += 1
        elif t == 0 and p == 0:
            TN += 1

    # 避免除零
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def get_pred_and_gt(uc,uc_gt):
    if len(uc['Basic flow']) != len(uc_gt['Basic flow']):
        print(f"Error: Basic flow length mismatch for UC {uc['id']}.")
    pred_,ground_truth= [0] * len(uc_gt['Basic flow']),[0] * len(uc_gt['Basic flow'])
    if isinstance(uc['pred_af'],dict):
        for bp_local in uc['pred_af'].keys():
            if int(bp_local) < len(pred_):
                pred_[int(bp_local)] = 1
            else:
                print(f"Error: bp_local {bp_local} out of range for Basic flow length {len(uc_gt['Basic flow'])}.")
    elif uc['pred_af'] != 'false':
        print(f"Error: uc['pred_af'] is neither 'false' nor dict.")


    if len(uc_gt['Alt. Flow'])>0:  # =0的情况则全部为负例
        for items in uc_gt['Alt. Flow']:
            for bp in items.keys():
                bp_local = bp.split('_')[0]  # 获得分支点
                if int(bp_local) < len(ground_truth):
                    ground_truth[int(bp_local)] = 1
                else:
                    print(f"Error: bp_local {bp_local} out of range for Basic flow length {len(uc_gt['Basic flow'])}.")

    return pred_,ground_truth
    
def get_right_bp(pred,gt):
    indices = []
    for i, (a, b) in enumerate(zip(pred,gt)):
        if a == 1 and b == 1:
            indices.append(i)
    return indices

def return_pred_lst(lst1,lst2):
    # 计算两个列表中每个元素的出现次数
    count_lst1 = Counter(lst1)
    count_lst2 = Counter(lst2)

    # 计算需要补充的元素及其数量：lst1计数 - lst2计数，并对差值取正值（忽略负值）
    supplement_items = count_lst1 - count_lst2
    # 由于 Counter 的减法会忽略负值，所以 supplement_items 中只包含需要补充的元素和数量

    # 根据次数差，生成需要补充的元素列表
    list_to_append = []
    for item, count in supplement_items.items():
        list_to_append.extend([item] * count) # 将元素重复 count 次加入补充列表

    # 将补充列表追加到 lst2 的末尾
    lst2.extend(list_to_append)
    return lst2

def metrics_cond_af(right_bp,uc,uc_gt,af_node_uc):
    cond_af = {}
    pred_act_dicts = merge_dicts_safe(uc['pred_af_act'])
    pred_obj_dicts = merge_dicts_safe(uc['pred_af_obj'])
    af_dicts = merge_dicts_safe(uc_gt['Alt. Flow'])

    for bp in right_bp:
        pred_num,p,r,f1,auc, = 0,0,0,0,0,
        tp = len(uc['pred_af_act_tp'][str(bp)])+len(uc['pred_af_obj_tp'][str(bp)])
        
        pred_num += len(return_pred_lst(pred_act_dicts[str(bp)],uc['pred_af_act_tp'][str(bp)]))
        pred_num += len(return_pred_lst(pred_obj_dicts[str(bp)],uc['pred_af_obj_tp'][str(bp)]))

        for key1 in af_dicts:
            if str(bp) in key1:
                id = list(af_dicts.keys()).index(key1)
                gt_act = flatten_list(af_node_uc['AF act'][id])
                gt_obj = flatten_list(af_node_uc['AF obj'][id])

        auc = (0.5 * tp) / len(gt_act + gt_obj) if len(gt_act + gt_obj) > 0 else 0
        p = tp / pred_num if pred_num > 0 else 0
        r = tp / (len(gt_act + gt_obj)) if len(gt_act + gt_obj) > 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0

        cond_af[str(bp)] = {'tp':tp,'pred_num':pred_num,'gt_num':len(gt_act + gt_obj),'p':p,'r':r,'f1':f1,'auc':auc}

    uc['cond_af'] = cond_af
    return uc






if __name__ == '__main__':  # 这个.py文件都是pub数据集的内容。ncet数据集的内容在baseline_LLM_with_bp_ncet.py中。
    task_name = 'count_pred_af_node'
    print(f'*** task_name: {task_name} , Starting time: {datetime.now()}  !!!!! ***')
    
    if task_name =='get_pred_bp_af_by_llm_pub': # 让大模型判断有无分支点、分支点位置，并生成对应分支流
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json" #branching_point_path与这个的bf是一样的
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_bf_af.json"
        out_path_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_bf_af.json"

        # 用BFGen项目中预测时用的数据集，刚刚好。但是其实这个实验中没有用这个uc_list
        uc_list = read_uc_from_json("E:/GitHub/ASSAM/data/2_dataset_origin_node/Ernie-4-Turbo/Ernie_pub_ground_truth.json")
        uc_list = module.local_id_to_global_id(uc_list)  # 更新global id     

        for ref_path, out_path,model_name in [[Ernie_integ_path,out_path_Ernie,ERNIE_4_turbo],[GPT_integ_path,out_path_gpt,GPT_4o]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("ernie",[ref_path, out_path,model_name]): 
                break
            
            uc_ref_list = read_uc_from_stand_json(ref_path)
            uc_ref_list = module.local_id_to_global_id(uc_ref_list)  # 更新global id     

            uc_list = get_test_sub_graph(uc_list,PUB_GROUPING_UC_20_1[-3:]) # 只需要test所涉及的子图的数据即可
            uc_ref_list= get_test_sub_graph(uc_ref_list,PUB_GROUPING_UC_20_1[-3:])
            
            uc_list = get_pred_bp_af_by_llm(uc_list,out_path,model_name,uc_ref_list)

    elif task_name == 'get_pred_af_given_bp_by_llm_pub':  # 给定分支点，让llm输出分支流。
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json" #branching_point_path与这个的bf是一样的
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_af_given_bp.json"
        out_path_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_af_given_bp.json"

        branching_point_path = '../0_Data/5_branching_point/1_gpt_added_bp/pub_with_gpt_bp.json'  # 只需要其中标注的分支点信息,其中node是ernie提取的，不过不重要。

        for ref_path, out_path,model_name in [[GPT_integ_path,out_path_gpt,GPT_4o],[Ernie_integ_path,out_path_Ernie,ERNIE_4_turbo]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("gpt",[ref_path, out_path,model_name]): 
                break
            
            uc_bp_info = module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path))
            uc_bp_info= get_test_sub_graph(uc_bp_info,PUB_GROUPING_UC_20_1[-3:])# 只需要test所涉及的子图的数据即可

            uc_ref_list = read_uc_from_stand_json(ref_path)
            uc_ref_list = module.local_id_to_global_id(uc_ref_list)  # 更新global id     
            uc_ref_list= get_test_sub_graph(uc_ref_list,PUB_GROUPING_UC_20_1[-3:])# 只需要test所涉及的子图的数据即可
            
            uc_ref_list = get_pred_af_given_bp_by_llm_pub(uc_ref_list,out_path,model_name,uc_bp_info)

    elif task_name == "get_pred_af_node_given_bp": # 模仿 baseline_LLM_pub.py 中的 get_pred_af_node 任务。
        in_path_given_bp_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_af_given_bp.json"
        in_path_given_bp_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_af_given_bp.json"

        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_af_given_bp_extra_node.json"
        out_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_af_given_bp_extra_node.json"

        for in_path, out_path in [[in_path_given_bp_Ernie,out_path_Ernie],[in_path_given_bp_gpt,out_path_gpt]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("ernie",[in_path, out_path]): 
                break

            uc_list = read_uc_from_stand_json(in_path)
            for sub in uc_list:
                for uc in sub:
                    if len(uc['pred_af'])>0:
                        uc['pred_af_act'],uc['pred_af_obj'] = [],[]
                        for items in uc['pred_af']:
                            for bp_local,af_list in items.items():
                                uc['pred_af_act'].append({str(bp_local):pred_af_node_gpt(af_list,GPT_4o,'act')})
                                uc['pred_af_obj'].append({str(bp_local):pred_af_node_gpt(af_list,GPT_4o,'obj')})
            
            write_uc_to_stand_json(out_path,uc_list)

    elif task_name == 'find_tp_in_llm_af_given_bp': # 从af_given_bp的node中，找到tp。
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/pub/GPT_pub_pred_af_given_bp_extra_node.json"
        in_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/pub/Ernie_pub_pred_af_given_bp_extra_node.json"

        afnode_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json'
        afnode_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json'

        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_af_given_bp_extra_node_withtp.json"
        out_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_af_given_bp_extra_node_withtp.json"

        for in_path, out_path, afnode_path in [[in_path_Ernie,out_path_Ernie,afnode_ernie],[in_path_gpt,out_path_gpt,afnode_gpt]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("ernie",[in_path, out_path]): 
                break

            uc_list = read_uc_from_stand_json(in_path)
            uc_af_list = get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(afnode_path)),PUB_GROUPING_UC_20_1[-3:])
            for sub,sub_gt in zip(uc_list,uc_af_list):
                for uc,uc_gt in zip(sub,sub_gt):
                    if len(uc['pred_af'])>0:
                        if uc['global id'] != uc_gt['global id'] or len(uc['dataset']) != len(uc_gt['dataset']):
                            print(f'uc -- uc_gt do not match！！')
                            break
                        
                        # 先将uc中的pred_af_act、obj变为字典。但是需要判断字典中的key是否有重复的
                        merge_dicts_act = merge_dicts_safe(uc['pred_af_act'])
                        merge_dicts_obj = merge_dicts_safe(uc['pred_af_obj'])
                        if not merge_dicts_act or not merge_dicts_obj:
                            print("合并失败：存在重复键")
                       
                        uc['pred_af_act_tp'] ,  uc['pred_af_obj_tp'] = {},{}  
                        for items in uc_gt['Alt. Flow']:  # 选择标准答案循环
                            for bp_local,af_list in items.items():  # 在给定bp情况下，bp是正确的，只需要判断后续节点即可。
                                bp = bp_local.split('_')[0]
                                if bp in merge_dicts_act.keys(): # 但是本文件中没有标注好的bp：af，只能从branching_point_path中获得
                                    uc['pred_af_act_tp'][str(bp)]= find_tp_gpt(merge_dicts_act[bp],GPT_4o,af_list)
                                    uc['pred_af_obj_tp'][str(bp)]= find_tp_gpt(merge_dicts_obj[bp],GPT_4o,af_list)
            
            write_uc_to_stand_json(out_path,uc_list)
  

    elif task_name == "get_pred_af_node_wo_bp": # 讨论后不需要这个。#模仿 baseline_LLM_pub.py 中的 get_pred_af_node 任务。
        in_path_Ernie_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_bf_af.json"
        in_path_gpt_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_bf_af.json"

        out_path_Ernie_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_bf_af_extra_node.json"
        out_path_gpt_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_bf_af_extra_node.json"
        
        for in_path, out_path in [[in_path_gpt_wo_bp,out_path_gpt_wo_bp],[in_path_Ernie_wo_bp,out_path_Ernie_wo_bp],]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("gpt",[in_path, out_path]): 
                break

            uc_list = read_uc_from_stand_json(in_path)
            for sub in uc_list:
                for uc in sub:
                    if len(uc['pred_af'])>0 and uc['pred_af'] != 'false':
                        uc['pred_af_act'],uc['pred_af_obj'] = [],[]
                        for bp_local,af_list in uc['pred_af'].items():
                            uc['pred_af_act'].append({str(bp_local):pred_af_node_gpt(af_list,GPT_4o,'act')})
                            uc['pred_af_obj'].append({str(bp_local):pred_af_node_gpt(af_list,GPT_4o,'obj')})
            
            write_uc_to_stand_json(out_path,uc_list)

    elif task_name == 'find_tp_in_llm_af_wo_bp': # 讨论后不需要这个。# 从af_given_bp的node中，找到tp。
        branching_point_path = '../0_Data/5_branching_point/1_gpt_added_bp/pub_with_gpt_bp.json'  # 只需要其中标注的分支点信息,其中node是ernie提取的，不过不重要。

        in_path_Ernie_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_bf_af_extra_node.json"
        in_path_gpt_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_bf_af_extra_node.json"

        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/GPT_pub_pred_bf_af_extra_node_withtp.json"
        out_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/Ernie_pub_pred_bf_af_extra_node_withtp.json"

        for in_path, out_path in [[in_path_gpt_wo_bp,out_path_gpt],[in_path_Ernie_wo_bp,out_path_Ernie]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("gpt",[in_path, out_path]): 
                break

            uc_list = read_uc_from_stand_json(in_path)
            uc_af_list = get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path)),PUB_GROUPING_UC_20_1[-3:])
            for sub,sub_gt in zip(uc_list,uc_af_list):
                for uc,uc_gt in zip(sub,sub_gt):
                    if len(uc['pred_af'])>0 and uc['pred_af']!= 'false' and len(uc_gt['Alt. Flow'])>0:
                        if uc['global id'] != uc_gt['global id'] or len(uc['dataset']) != len(uc_gt['dataset']):
                            print(f'uc -- uc_gt do not match！！')
                            break
                        
                        # 先将uc中的pred_af_act、obj变为字典。但是需要判断字典中的key是否有重复的
                        merge_dicts_act = merge_dicts_safe(uc['pred_af_act'])
                        merge_dicts_obj = merge_dicts_safe(uc['pred_af_obj'])
                        if not merge_dicts_act or not merge_dicts_obj:
                            print("合并失败：存在重复键")
                       
                        uc['pred_af_act_tp'] ,  uc['pred_af_obj_tp'] = {},{}
            #             for items in uc_gt['Alt. Flow']:  # 选择标准答案循环
            #                 for bp_local,af_list in items.items():  # 在给定bp情况下，bp是正确的，只需要判断后续节点即可。
            #                     bp = bp_local.split('_')[0]
            #                     if bp in merge_dicts_act.keys(): # 但是本文件中没有标注好的bp：af，只能从branching_point_path中获得
            #                         # 错了 要给定af node。 uc['pred_af_act_tp'][str(bp)]= find_tp_gpt(merge_dicts_act[bp],GPT_4o,af_list)
            #                         # uc['pred_af_obj_tp'][str(bp)]= find_tp_gpt(merge_dicts_obj[bp],GPT_4o,af_list)
            
            # write_uc_to_stand_json(out_path,uc_list)


    elif task_name == "metrics_wo_bp":  # 没有给定bp的情况下，需要判定llm生成bp的准确率，包括macro和micro。
        branching_point_path = '../0_Data/5_branching_point/1_gpt_added_bp/pub_with_gpt_bp.json'  # 只需要其中标注的分支点信息,其中node是ernie提取的，不过不重要。

        in_path_gpt_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/pub/GPT_pub_pred_bf_af_extra_node_withtp.json"
        in_path_Ernie_wo_bp =  "../0_Data/baseline_data/baseline_LLM_with_bp/pub/Ernie_pub_pred_bf_af_extra_node_withtp.json"    

        out_path_gpt_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/pub/GPT_pub_pred_bf_af_metric_macro.json"
        out_path_Ernie_wo_bp =  "../0_Data/baseline_data/baseline_LLM_with_bp/pub/Ernie_pub_pred_bf_af_metric_macro.json"

        for in_path,out_path in [[in_path_Ernie_wo_bp,out_path_Ernie_wo_bp],[in_path_gpt_wo_bp,out_path_gpt_wo_bp]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            lable = "ernie"
            if not module.check_all_under_same_method(lable,[in_path, out_path]): 
                break
            uc_af_list = get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path)),PUB_GROUPING_UC_20_1[-3:])

            pred_all,ground_truth_all,macro_list_p,macro_list_r,macro_list_f1 = [],[],[],[],[]
            uc_list = read_uc_from_stand_json(in_path)
            for sub,sub_gt in zip(uc_list,uc_af_list):
                for uc,uc_gt in zip(sub,sub_gt):
                    if len(uc['pred_af'])>0 and uc['pred_af']!= 'false' and len(uc_gt['Alt. Flow'])>0:
                        if uc['global id'] != uc_gt['global id'] or len(uc['dataset']) != len(uc_gt['dataset']):
                            print(f'uc -- uc_gt do not match！！')
                            break
                    uc = {k: v for k, v in uc.items() if k not in ["Brief Description","Basic flow","BF act","BF obj","key_name","key_act","key_obj"]}
                   
                    # 1.micro平均：用全部uc的全部bf中的step作为候选集，其中的分支点为正例，其中的非分支点是负例。最后整个数据集给出一组数据。
                    pred_,ground_truth= get_pred_and_gt(uc,uc_gt)          
                    pred_all += pred_
                    ground_truth_all += ground_truth
            
                    # 2.macro平均：对每个uc单独计算AF的指标，然后对所有uc的指标取平均。
                    if 1 in pred_ and 1 in ground_truth:  # precision和recall都有意义的情况
                        right_bp = get_right_bp(pred_,ground_truth)
                        tp = len(right_bp)
                        uc['macro_p'] = tp / pred_.count(1) if pred_.count(1) > 0 else print("Error: pred_ count is zero when calculating macro precision.")
                        uc['macro_r'] = tp / ground_truth.count(1) if ground_truth.count(1) > 0 else print("Error: ground_truth count is zero when calculating macro recall.")
                        uc['macro_f1'] = (2 * uc['macro_p'] * uc['macro_r']) / (uc['macro_p'] + uc['macro_r']) if (uc['macro_p'] + uc['macro_r']) > 0 else 0
                        macro_list_p.append(uc['macro_p'])
                        macro_list_r.append(uc['macro_r'])
                        macro_list_f1.append(uc['macro_f1'])

            bp_p,bp_r,bp_f1= precision_recall_f1(ground_truth_all,pred_all)
            print(f'{lable} Micro: bp_p: {bp_p}, bp_r: {bp_r}, bp_f1: {bp_f1}')

            print(f'{lable} Macro: bp_p: {sum(macro_list_p)/len(macro_list_p)}, bp_r: {sum(macro_list_r)/len(macro_list_r)}, bp_f1: {sum(macro_list_f1)/len(macro_list_f1)}')
            # write_uc_to_stand_json(out_path,uc_list)



    elif task_name == "metrics_given_bp": # 给定bp的情况下，需要判定llm生成af的准确率。给定 gold BP 时的 AF 效果（Oracle_AF）
        in_path_gpt_given_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/pub/GPT_pub_pred_af_given_bp_extra_node_withtp.json"
        in_path_Ernie_given_bp =  "../0_Data/baseline_data/baseline_LLM_with_bp/pub/Ernie_pub_pred_af_given_bp_extra_node_withtp.json"

        afnode_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json'
        afnode_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json'

        out_path_gpt_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/pub/GPT_pub_pred_af_given_bp_metric.json"
        out_path_Ernie_wo_bp =  "../0_Data/baseline_data/baseline_LLM_with_bp/pub/Ernie_pub_pred_af_given_bp_metric.json"

        for in_path,out_path,af_path in [[in_path_gpt_given_bp,out_path_gpt_wo_bp,afnode_gpt],[in_path_Ernie_given_bp,out_path_Ernie_wo_bp,afnode_ernie]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("gpt",[in_path, out_path,af_path]): 
                break
            uc_list = read_uc_from_stand_json(in_path)
            uc_af_list = get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(af_path)),PUB_GROUPING_UC_20_1[-3:])
            p_list,r_list,f1_list,auc_list = [],[],[],[]
            for sub,sub_gt in zip(uc_list,uc_af_list):
                for uc,uc_gt in zip(sub,sub_gt):
                    # 4、仅需要判定给定bp的情况下。评估在给定bp的情况下，af的准确率。
                    if len(uc['pred_af']) == 0 :    
                        continue
                    tp,fp,fn,uc['p'],uc['r'],uc['f1'], uc['auc'] = 0,0,0,[],[],[],[]
                    if len(uc['pred_af_act_tp']) != len(uc_gt['AF act']) or len(uc['pred_af_act']) != len(uc_gt['AF obj']):
                        print(f'uc -- uc_gt do not match！！')
                        break
                    for bp_tp,pred_dict_act,pred_dict_obj,af_act,af_obj,pred_af in zip(uc['pred_af_act_tp'].keys(),uc['pred_af_act'],uc['pred_af_obj'],uc_gt['AF act'],uc_gt['AF act'],uc['pred_af']):
                        if bp_tp != next(iter(pred_dict_act)) or bp_tp != next(iter(pred_dict_obj)):
                            print(f'uc -- uc_gt do not match！！')
                            break
                        tp = min(len(uc['pred_af_act_tp'][bp_tp]) + len(uc['pred_af_obj_tp'][bp_tp]),len(flatten_list(af_act))+len(flatten_list(af_obj)))
                        # fp = min(len(next(iter(pred_dict_act.values()))) + len(next(iter(pred_dict_obj.values()))) - tp,len(flatten_list(next(iter(pred_af.values()))))*2-tp)
                        fp = len(next(iter(pred_dict_act.values()))) + len(next(iter(pred_dict_obj.values()))) - tp
                        fn = len(flatten_list(af_act))+len(flatten_list(af_obj))-tp
                        p = tp/(tp+fp) if (tp+fp)>0 else 0
                        r = tp/(tp+fn) if (tp+fn)>0 else 0
                        uc['p'].append(tp/(tp+fp) if (tp+fp)>0 else 0)
                        uc['r'].append(tp/(tp+fn) if (tp+fn)>0 else 0)
                        uc['f1'].append((2*p*r)/(p+r) if (p+r)>0 else 0)
                        uc['auc'].append(0.5*tp/(len(flatten_list(af_act))+len(flatten_list(af_obj))) if (len(flatten_list(af_act))+len(flatten_list(af_obj)))>0 else 0)

                        if p>0:  # auc=0的值无异议，排除该样例. p=0的那个auc也等于0，所以用p=0来排除。
                            p_list.append(p)
                            r_list.append(r)
                            f1_list.append((2*p*r)/(p+r) if (p+r)>0 else 0)
                            auc_list.append(0.5*tp/(len(flatten_list(af_act))+len(flatten_list(af_obj))) if (len(flatten_list(af_act))+len(flatten_list(af_obj)))>0 else 0)
            print(f'p: {sum(p_list)/len(p_list)}, r: {sum(r_list)/len(r_list)}, f1: {sum(f1_list)/len(f1_list)}, auc: {sum(auc_list)/len(auc_list)}')
            write_uc_to_stand_json(out_path,uc_list)


    elif task_name == 'count_pred_af_node': # 统计每个uc中llm预测的af node数量,即论文中用到的数据：大模型方法生成的af平均多少个step。
        in_path_gpt_given_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/pub/GPT_pub_pred_af_given_bp_extra_node_withtp.json"
        in_path_Ernie_given_bp =  "../0_Data/baseline_data/baseline_LLM_with_bp/pub/Ernie_pub_pred_af_given_bp_extra_node_withtp.json"

        afnode_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json'
        afnode_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json'


        for in_path,af_path in [[in_path_gpt_given_bp,afnode_gpt],[in_path_Ernie_given_bp,afnode_ernie]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("gpt",[in_path,af_path]): 
                break
            uc_list = read_uc_from_stand_json(in_path)
            uc_af_list = get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(af_path)),PUB_GROUPING_UC_20_1[-3:])

            num_act_pred,num_obj_pred,num_act_gt,num_obj_gt = [],[],[],[]
            for sub,sub_gt in zip(uc_list,uc_af_list):
                for uc,uc_gt in zip(sub,sub_gt):
                    if len(uc['pred_af']) == 0 :    
                        continue
                                        
                    if len(uc['pred_af_act_tp']) != len(uc_gt['AF act']) or len(uc['pred_af_act']) != len(uc_gt['AF obj']):
                        print(f'uc -- uc_gt do not match！！')
                        break
                    
                    for bp_tp,pred_dict_act,pred_dict_obj,af_act,af_obj,pred_af in zip(uc['pred_af_act_tp'].keys(),uc['pred_af_act'],uc['pred_af_obj'],uc_gt['AF act'],uc_gt['AF act'],uc['pred_af']):
                        if bp_tp != next(iter(pred_dict_act)) or bp_tp != next(iter(pred_dict_obj)):
                            print(f'uc -- uc_gt do not match！！')
                            break
                        num_act_pred.append(len(next(iter(pred_dict_act.values()))))
                        num_obj_pred.append(len(next(iter(pred_dict_obj.values()))))
                        num_act_gt.append(len(flatten_list(af_act)))
                        num_obj_gt.append(len(flatten_list(af_obj)))
                    
                    if len(num_act_pred) != len(num_obj_pred) or len(num_act_pred) != len(num_act_gt) or len(num_act_pred) != len(num_obj_gt):
                        print("Error: Length mismatch among predicted and ground truth action/object counts.")
                    
            print(f'For {in_path}, Average predicted AF action nodes: {sum(num_act_pred)/len(num_act_pred) if len(num_act_pred)>0 else 0}, Average predicted AF object nodes: {sum(num_obj_pred)/len(num_obj_pred) if len(num_obj_pred)>0 else 0}')
            print(f'For {in_path}, Average ground truth AF action nodes: {sum(num_act_gt)/len(num_act_gt) if len(num_act_gt)>0 else 0}, Average ground truth AF object nodes: {sum(num_obj_gt)/len(num_obj_gt) if len(num_obj_gt)>0 else 0}')
                        


    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')