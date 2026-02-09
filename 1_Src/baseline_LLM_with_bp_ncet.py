from imports import datetime,ERNIE_4_turbo,GPT_4o,importlib,GROUPING_UC_20,json,requests,qianfan,re,time
spec = importlib.util.spec_from_file_location("module_name", "7_make_pt_file.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from utils import read_uc_from_stand_json,get_test_sub_graph,clean_string,write_uc_to_stand_json


def pred_bp_af_ernie_cn(uc,model_name):
    chat_comp = qianfan.ChatCompletion()

    # ERNIE 模型没有 sys_pmt
    user_pmt = "用例描述:'" + clean_string(uc['Name']) + "." + clean_string(str(uc['Brief Description']))
    
    user_pmt += ".基本流:'" + str(uc['Basic flow']) + "'.给定上述用例，判断其是否包含分支流。" \
    "若无分支流，只输出以下单词: false.不要输出任何其他文字、符号或解释。" \
    "如果存在分支流，请严格按照以下Python字典格式输出：" \
    "{step_index: [[分支流1], [分支流2], ...],...}"\
    "其中step_index是发生分支流的基本流步骤索引（从0开始），"\
    "值为由你生成的分支流文本列表。"\
    "不要输出任何其他内容或格式之外的信息。"

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"].strip()  # 去除两端空格
    if '{' in exist and '}' in exist:
        exist = re.search(r'\{([\s\S]*?)\}(?=[^}]*$)', exist).group()
        return exist
    elif 'false' in exist.lower() and '{' not in exist:
        return 'false'

def pred_bp_af_gpt_cn(uc,model_name):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"


    sys_pmt = "给定一个用例，判断其是否包含分支流。" \
    "若无分支流，只输出以下单词: false.不要输出任何其他文字、符号或解释。" \
    "如果存在分支流，请严格按照以下Python字典格式输出：" \
    "{step_index: [[分支流1], [分支流2], ...],...}"\
    "其中step_index是发生分支流的基本流步骤索引（从0开始），"\
    "值为由你生成的分支流文本列表。"\
    "不要输出任何其他内容或格式之外的信息。"
    
    user_pmt = "用例描述:'" + clean_string(uc['Name']) + "." + clean_string(uc['Brief Description'])
    
    user_pmt += "。基本流:'" + str(uc['Basic flow'])

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

    # 加入重试机制
    max_retries = 10
    for attempt in range(max_retries):
        try:
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
                time.sleep(2)  # 等待2秒后重试
                exist = pred_bp_af_gpt_cn(uc,model_name)
                return exist
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")

    # 最终失败后递归调用或返回None
    print("多次重试后仍失败，递归调用 pred_af_gpt_cn")
    exist = pred_bp_af_gpt_cn(uc,model_name)
    return exist


def get_pred_bp_af_by_llm_cn(out_path,model_name,uc_ref_list):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for sub_graph in uc_ref_list:
            json_file.write('[\n')  # 手动添加逗号和换行
            for uc_ref in sub_graph:
                print(f"*** uc global id: {uc_ref['global id']} ***")
                                    
                # 使用LLM：判断是否有分支点、有的话生成分支流
                if model_name == ERNIE_4_turbo:
                    uc_ref['pred_af'] = pred_bp_af_ernie_cn(uc_ref,model_name)
                elif model_name == GPT_4o:
                    uc_ref['pred_af'] = pred_bp_af_gpt_cn(uc_ref,model_name)
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
                        
                json.dump(uc_ref, json_file, ensure_ascii=False, indent=4)
                json_file.write(',\n')  # 手动添加逗号和换行

            json_file.write(']\n')  # 手动添加逗号和换行

    return uc_ref_list

def pred_af_given_bp_ernie_cn(uc,model_name,bp_local):
    chat_comp = qianfan.ChatCompletion() # ERNIE 模型没有 sys_pmt
    user_pmt = "用例描述:" + clean_string(uc['Name']) + "." + str(uc['Brief Description']).strip('[]')
    
    user_pmt += ".基本流:" + str(uc['Basic flow']) + ".分支点:"+ str(bp_local) +".给定一个用例和分支点," \
    "分支点是基本流中发生分支流的步骤索引（从0开始计数）。" \
    "为该分支点生成分支流，并严格按照以下Python列表格式输出：" \
    "[[分支流1], [分支流2], ...]"\
    "。每个内层列表包含一条由你生成的分支流,不要输出任何其他内容或超出此格式的文字。"

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
        print(f'ernie api 调用错误！！！重新调用 pred_af_given_bp_ernie_cn')
        exist = pred_af_given_bp_ernie_cn(uc,model_name,bp_local)
        return exist

def pred_af_given_bp_gpt_cn(uc,model_name,bp_local):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"

    sys_pmt = "给定一个用例和分支点" \
    ",分支点是基本流中发生分支流的步骤索引（从0开始计数）。" \
    "为该分支点生成分支流，并严格按照以下Python列表格式输出：" \
    "[[分支流1], [分支流2], ...]"\
    "。每个内层列表包含一条由你生成的分支流,不要输出任何其他内容或超出此格式的文字。"
    

    user_pmt = "用例描述:" + clean_string(uc['Name']) + "." + clean_string(uc['Brief Description'])
    
    user_pmt += ".基本流:" + str(uc['Basic flow'])
    user_pmt += '.分支点:'+ str(bp_local)

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
    # 加入重试机制
    max_retries = 10
    for attempt in range(max_retries):
        try:
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
                time.sleep(2)  # 等待2秒后重试
                exist = pred_af_given_bp_gpt_cn(uc,model_name,bp_local)
                return exist    
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")

    # 最终失败后递归调用或返回None
    print("多次重试后仍失败，递归调用 pred_af_gpt_cn")
    exist = pred_af_given_bp_gpt_cn(uc,model_name,bp_local)
    return exist

def get_pred_af_given_bp_by_llm_ncet(uc_ref_list,out_path,model_name,uc_bp_info):  
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for sub_graph,bp_sub in zip(uc_ref_list,uc_bp_info):
            json_file.write('[\n')  # 手动添加逗号和换行
            for uc_ref,bp_uc in zip(sub_graph,bp_sub):
                print(f"*** uc global id: {uc_ref['global id']} ***")
                if bp_uc['dataset'] != uc_ref['dataset'] or bp_uc['id'] != uc_ref['id']:
                    print(f"Dataset or ID mismatch: {bp_uc['global id']} vs {uc_ref['global id']}")
                    break
                    
                uc_ref['pred_af'] = []
                for af in bp_uc['Exc. Flow']:
                    # 获得分支点
                    bp_local = list(af)[0].split('_')[0]
                                
                    # 使用LLM：在给定分支点的情况下，生成分支流
                    if model_name == ERNIE_4_turbo:
                        uc_ref['pred_af'].append({str(bp_local):pred_af_given_bp_ernie_cn(uc_ref,model_name,bp_local)})
                    elif model_name == GPT_4o:
                        uc_ref['pred_af'].append({str(bp_local):pred_af_given_bp_gpt_cn(uc_ref,model_name,bp_local)})
                    else:
                        print(f"Model {model_name} not recognized. Skipping prediction.")
                        
                json.dump(uc_ref, json_file, ensure_ascii=False, indent=4)
                json_file.write(',\n')  # 手动添加逗号和换行

            json_file.write('],\n')  # 手动添加逗号和换行

    return uc_ref_list



def pred_af_node_gpt_cn(af_list,GPT_4o,label):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"

    if label == 'act':
        sys_pmt = "提取str1中所有动词,以Python的list格式输出,不要输出任何其他内容或超出此格式的文字."
    elif label == 'obj':    
        sys_pmt = "提取str1中所有名词,以Python的list格式输出,不要输出任何其他内容或超出此格式的文字."

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

    # 加入重试机制
    max_retries = 10
    for attempt in range(max_retries):
        try:
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
                time.sleep(2)  # 等待2秒后重试
                exist = pred_af_node_gpt_cn(af_list,GPT_4o,label)
                return exist
            
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")
            time.sleep(2)  # 等待2秒后重试


    # 最终失败后递归调用或返回None
    print("多次重试后仍失败，递归调用 pred_af_gpt_cn")
    exist = pred_af_node_gpt_cn(uc,model_name)
    return exist

def find_tp_gpt_cn(node_list,model_name,af_list):
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"

    # 因为ncet的所有af只有两个step，而且这两个step的act和obj各不相同，所以直接让模型统计出现次数即可。
    sys_pmt = "统计表B中每个词在表A中的出现次数(相同或语义相似均视为'出现')."\
    "严格按Python列表格式,按照表B中词语的顺序依次输出对应的出现次数:" \
    "['次数1', '次数2', ...].不输出列表之外的任何内容."\
    
    user_pmt = "表A:" + node_list
    user_pmt += ".表B:'" + str(af_list)

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

    # 加入重试机制
    max_retries = 10
    for attempt in range(max_retries):
        try:
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
                time.sleep(2)  # 等待2秒后重试
                exist = find_tp_gpt_cn(node_list,model_name,af_list)
                return exist
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")
            time.sleep(2)  # 等待2秒后重试

    # 最终失败后递归调用或返回None
    print("多次重试后仍失败，递归调用 pred_af_gpt_cn")
    exist = find_tp_gpt_cn(node_list,model_name,af_list)
    return exist


def find_tp_ernie_cn(node_list,model_name,af_list):
    chat_comp = qianfan.ChatCompletion() # ERNIE 模型没有 sys_pmt    
    user_pmt = "表A:" + node_list + ".表B:'" + str(af_list)
    user_pmt +=  ".找出表A中也出现在表B中的词(相同或语义相似)."\
    "每个词的输出次数等于其在两个表中出现次数的最小值." \
    "严格按Python列表格式输出这些单词:" \
    "['词1', '词2', ...]"\
    "不要去重、不要添加新单词,不要输出列表之外的任何内容"
    

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"].strip()  # 去除两端空格
    if '[' in exist and ']' in exist:
        exist = re.search(r'\[.*\]', exist).group()
        return exist
    elif '[' not in exist:
        print(f'ernie api 调用错误！！！重新调用 pred_af_given_bp_ernie_cn')
        exist = find_tp_ernie_cn(node_list,model_name,af_list)
        exist = re.search(r'\[.*\]', exist).group()
        return exist


def get_pred_and_gt(uc,uc_gt):
    a_pred = uc['pred_af']  # 用作检查
    a_gt = uc_gt['Exc. Flow'] # 用作检查

    pred_,ground_truth= [0] * len(uc['Basic flow']),[0] * len(uc['Basic flow']) # uc_gt的一些bf不够，没有与uc的对齐，用uc的。
    if isinstance(uc['pred_af'],dict):
        for bp_local in uc['pred_af'].keys():
            if int(bp_local) < len(pred_):
                pred_[int(bp_local)] = 1
            else:
                print(f"Error: bp_local {bp_local} out of range for Basic flow length {len(uc_gt['Basic flow'])}.")
    elif uc['pred_af'] != 'false':
        print(f"Error: uc['pred_af'] is neither 'false' nor dict.")


    if len(uc_gt['Exc. Flow'])>0:  # =0的情况则全部为负例
        for items in uc_gt['Exc. Flow']:
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

if __name__ == '__main__':
    task_name = 'count_pred_af_node'
    print(f'*** task_name: {task_name} , Starting time: {datetime.now()}  !!!!! ***')

    if task_name =='get_pred_bp_af_by_llm_ncet': # 让大模型判断有无分支点、分支点位置，并生成对应分支流
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json"
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_NCET_integrated.json" 
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_bf_af.json"
        out_path_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_bf_af.json"

        for ref_path, out_path,model_name in [[Ernie_integ_path,out_path_Ernie,ERNIE_4_turbo],[GPT_integ_path,out_path_gpt,GPT_4o]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("ernie",[ref_path, out_path,model_name]): 
                break
            
            uc_ref_list = module.local_id_to_global_id(read_uc_from_stand_json(ref_path))  # 更新global id     

            uc_ref_list= get_test_sub_graph(uc_ref_list,GROUPING_UC_20[-32:]) # 只需要test所涉及的子图的数据即可
            
            uc_ref_list = get_pred_bp_af_by_llm_cn(out_path,model_name,uc_ref_list)

    elif task_name == 'get_pred_af_given_bp_by_llm_ncet':  # 给定分支点，让llm输出分支流。
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json"
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_NCET_integrated.json" 
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp.json"
        out_path_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp.json"

        branching_point_path = '../0_Data/5_branching_point/2_ncet_bp/NCET_with_bp.json'  
        for ref_path, out_path,model_name in [[Ernie_integ_path,out_path_Ernie,ERNIE_4_turbo],[GPT_integ_path,out_path_gpt,GPT_4o]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("ernie",[ref_path, out_path,model_name]): 
                break
            
            # 只需要test所涉及的子图的数据即可
            uc_bp_info= get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path)),GROUPING_UC_20[-32:]) 

            uc_ref_list = get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(ref_path)),GROUPING_UC_20[-32:]) 

            uc_bp_info = get_pred_af_given_bp_by_llm_ncet(uc_ref_list,out_path,model_name,uc_bp_info) 

    elif task_name == "get_pred_af_node_given_bp_ncet": # 给定分支点的情况，获取其中af的node
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp.json"
        in_path_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp.json"

        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp_extra_node.json"
        out_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp_extra_node.json"

        for in_path, out_path in [[in_path_Ernie,out_path_Ernie],[in_path_gpt,out_path_gpt]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("ernie",[in_path, out_path]): 
                break
            
            uc_list = read_uc_from_stand_json(in_path)
            for sub in uc_list:
                for uc in sub:
                    print(f'*** uc global id: {uc["id"]} ***')
                    if len(uc['pred_af'])>0:
                        uc['pred_af_act'],uc['pred_af_obj'] = [],[]
                        for items in uc['pred_af']:
                            for bp_local,af_list in items.items():
                                uc['pred_af_act'].append({str(bp_local):pred_af_node_gpt_cn(af_list,GPT_4o,'act')})
                                uc['pred_af_obj'].append({str(bp_local):pred_af_node_gpt_cn(af_list,GPT_4o,'obj')})
            
            write_uc_to_stand_json(out_path,uc_list)

    elif task_name == 'find_tp_in_llm_af_given_bp_ncet': # 从af_given_bp的node中，找到tp。
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp_extra_node.json"
        in_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp_extra_node.json"

        alt_node_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        alt_node_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'

        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp_extra_node_withtp.json"
        out_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp_extra_node_withtp.json" 

        for in_path, out_path,alt_path in [[in_path_gpt,out_path_gpt,alt_node_path_gpt],[in_path_Ernie,out_path_Ernie,alt_node_path_ernie]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method("gpt",[in_path, out_path,alt_path]): 
                break

            uc_bp_info= get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(alt_path)),GROUPING_UC_20[-32:]) 
            uc_list = read_uc_from_stand_json(in_path)
            for sub,sub_gt in zip(uc_list,uc_bp_info):
                for uc,uc_gt in zip(sub,sub_gt):
                    print(f'*** uc id: {uc["id"]} ***')
                    if len(uc['pred_af'])>0:
                        if uc['id'] != uc_gt['id']:
                            print(f'uc -- uc_gt do not match！！')
                            break

                        uc['pred_af_act_tp'] ,  uc['pred_af_obj_tp'] = {},{}
                        for i in range(len(uc_gt['AF obj'])):
                            bp = next(iter(uc['pred_af_act'][i].keys()))
                            uc['pred_af_act_tp'][bp]= find_tp_gpt_cn(next(iter(uc['pred_af_act'][i].values())),GPT_4o,uc_gt['AF act'][i])
                            uc['pred_af_obj_tp'][bp]= find_tp_gpt_cn(next(iter(uc['pred_af_obj'][i].values())),GPT_4o,uc_gt['AF obj'][i])
            
            write_uc_to_stand_json(out_path,uc_list)
    
    elif task_name == "metrics_given_bp": # 给定bp的情况下，需要判定llm生成af的p/r/f1/auc。给定 gold BP 时的 AF 效果（Oracle_AF）
        alt_node_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        alt_node_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'

        in_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp_extra_node_withtp.json"
        in_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp_extra_node_withtp.json" 

        out_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp_metrics.json"
        out_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp_metrics.json" 

        for in_path, out_path,alt_path in [[in_path_Ernie,out_path_Ernie,alt_node_path_ernie],[in_path_gpt,out_path_gpt,alt_node_path_gpt]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            lable = "ernie"
            if not module.check_all_under_same_method(lable,[in_path, out_path,alt_path]): 
                break
            uc_af_info= get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(alt_path)),GROUPING_UC_20[-32:]) 
            uc_list = read_uc_from_stand_json(in_path)
            p_list,r_list,f1_list,auc_list = [],[],[],[]
            for sub,sub_gt in zip(uc_list,uc_af_info):
                for uc,uc_gt in zip(sub,sub_gt):
                    if uc['id'] != uc_gt['id']:
                        print(f'uc -- uc_gt do not match！！')
                        break
                    if "pred_af_act_tp" not in uc:
                        continue
                    
                    # 只需要计算tp中有几个0即可，有1个0，则tp=1；有两个0，tp=0.因为该数据集中的af都是两个step。
                    tp,fp,fn,uc['p'],uc['r'],uc['f1'], uc['auc'] = 0,0,0,[],[],[],[]
                    for index, ((act_key,act_tp), (obj_key,obj_tp)) in enumerate(zip(uc['pred_af_act_tp'].items(),uc['pred_af_obj_tp'].items())):
                        if act_key != obj_key:
                            print(f'Error: act_key and obj_key do not match!!!')
                            break
                        tp = 2 - act_tp.count('0')  # act_tp中有几个0
                        tp += 2 - obj_tp.count('0')  # obj_tp中有几个
                        #计算有几个逗号，则有几个af node-1，所以最后+2
                        fp = str(uc['pred_af_act'][index].values()).count(',') + str(uc['pred_af_obj'][index].values()).count(',') + str(uc['pred_af_act'][index].values()).count('，') + str(uc['pred_af_obj'][index].values()).count('，') + 2 - tp  
                        fn = 2 * 2 - tp 
                        p = tp/(tp+fp) if (tp+fp)>0 else 0
                        r = tp/(tp+fn) if (tp+fn)>0 else 0
                        uc['p'].append(tp/(tp+fp) if (tp+fp)>0 else 0)
                        uc['r'].append(tp/(tp+fn) if (tp+fn)>0 else 0)
                        uc['f1'].append((2*p*r)/(p+r) if (p+r)>0 else 0)
                        uc['auc'].append(0.5*tp/4)

                        if p !=0 and r != 0 and 0.5*tp/4 != 0:
                            p_list.append(p)
                            r_list.append(r)
                            f1_list.append((2*p*r)/(p+r) if (p+r)>0 else 0)
                            auc_list.append(0.5*tp/4)
            print(f'{lable} : p: {sum(p_list)/len(p_list)}, r: {sum(r_list)/len(r_list)}, f1: {sum(f1_list)/len(f1_list)}, auc: {sum(auc_list)/len(auc_list)}')
            write_uc_to_stand_json(out_path,uc_list)

    elif task_name == "count_pred_af_node": # 统计每个uc中llm预测的af node数量,即论文中用到的数据：大模型方法生成的af平均多少个step。
        alt_node_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        alt_node_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'

        in_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp_extra_node_withtp.json"
        in_path_Ernie =  "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp_extra_node_withtp.json" 

        for in_path,alt_path in [[in_path_gpt,alt_node_path_gpt],[in_path_Ernie,alt_node_path_ernie]]:
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            lable = "gpt"
            if not module.check_all_under_same_method(lable,[in_path,alt_path]): 
                break
            uc_af_info= get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(alt_path)),GROUPING_UC_20[-32:]) 
            uc_list = read_uc_from_stand_json(in_path)
            num_act_pred,num_obj_pred,num_act_gt,num_obj_gt = [],[],[],[]
            for sub,sub_gt in zip(uc_list,uc_af_info):
                for uc,uc_gt in zip(sub,sub_gt):
                    if uc['id'] != uc_gt['id']:
                        print(f'uc -- uc_gt do not match！！')
                        break
                    if "pred_af_act_tp" not in uc:
                        continue
                    
                    for index, ((act_key,act_tp), (obj_key,obj_tp)) in enumerate(zip(uc['pred_af_act_tp'].items(),uc['pred_af_obj_tp'].items())):
                        if act_key != obj_key:
                            print(f'Error: act_key and obj_key do not match!!!')
                            break

                        num_act_pred.append(len(uc['pred_af_act'][index][act_key].split(',')))
                        num_obj_pred.append(len(uc['pred_af_obj'][index][obj_key].split(',')))
                        num_act_gt.append(len(uc_gt['AF act'][index]))
                        num_obj_gt.append(len(uc_gt['AF obj'][index]))

                    if len(num_act_pred) != len(num_obj_pred) or len(num_act_pred) != len(num_act_gt) or len(num_act_pred) != len(num_obj_gt):
                        print("Error: Length mismatch among predicted and ground truth action/object counts.")

            print(f'For {in_path}, Average predicted AF action nodes: {sum(num_act_pred)/len(num_act_pred) if len(num_act_pred)>0 else 0}, Average predicted AF object nodes: {sum(num_obj_pred)/len(num_obj_pred) if len(num_obj_pred)>0 else 0}')
            print(f'For {in_path}, Average ground truth AF action nodes: {sum(num_act_gt)/len(num_act_gt) if len(num_act_gt)>0 else 0}, Average ground truth AF object nodes: {sum(num_obj_gt)/len(num_obj_gt) if len(num_obj_gt)>0 else 0}')





    elif task_name == "metrics_wo_bp":  # 没有给定bp的情况下，需要判定llm生成bp的准确率，包括macro和micro。
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_bf_af.json"
        in_path_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_bf_af.json"
        branching_point_path = '../0_Data/5_branching_point/2_ncet_bp/NCET_with_bp.json'  

        for in_path in [in_path_Ernie,in_path_gpt]:
            lable = "ernie"
            # 检测文件是同一个method的. 更换string即可判断出入路径是否统一
            if not module.check_all_under_same_method(lable,[in_path]): 
                break
            uc_af_info= get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path)),GROUPING_UC_20[-32:]) 
            uc_list = read_uc_from_stand_json(in_path)
            a_pred_all,a_ground_truth_all,macro_list_p,macro_list_r,macro_list_f1 = [],[],[],[],[]
            for sub,sub_gt in zip(uc_list,uc_af_info):
                for uc,uc_gt in zip(sub,sub_gt):
                    if uc['id'] != uc_gt['id']:
                        print(f'uc -- uc_gt do not match！！')
                        break

                    # 1.micro平均：用全部uc的全部bf中的step作为候选集，其中的分支点为正例，其中的非分支点是负例。最后整个数据集给出一组数据。
                    pred_,ground_truth= get_pred_and_gt(uc,uc_gt)          
                    a_pred_all += pred_
                    a_ground_truth_all += ground_truth

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
            # print(a_ground_truth_all)
            # print(a_pred_all)

            bp_p,bp_r,bp_f1= precision_recall_f1(a_ground_truth_all,a_pred_all)
            
            print(f'{lable} Micro: bp_p: {bp_p}, bp_r: {bp_r}, bp_f1: {bp_f1}')

            print(f'{lable} Macro: bp_p: {sum(macro_list_p)/len(macro_list_p)}, bp_r: {sum(macro_list_r)/len(macro_list_r)}, bp_f1: {sum(macro_list_f1)/len(macro_list_f1)}')





    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')