# 用于使用LLM（GPT-4o和ERNIE 4.0Turbo）进行分支流生成的代码,专用于NCE-T数据集。
from utils import read_uc_from_stand_json, write_uc_to_stand_json,clean_and_convert,clean_string,find_list_in_string,flatten_list,delete_error,check_all_under_same_method,eval_Ernie_pred_ncet,count_node,calculate_all_dataset
from imports import ERNIE_4_turbo, GPT_4o,json,qianfan,requests,re,ERROR_WORD_LIST,inspect,datetime

def get_pred_af_by_llm_ncet(uc_list, out_path, model_name):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        # json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list[1830:]:
            print(f"*** uc id: {uc['id']}, Name: {uc['Name']} ***")

            if "Exc. Flow" in uc and len(uc["Exc. Flow"]) > 0:
                # 使用LLM生成分支流
                if model_name == ERNIE_4_turbo:
                    uc['pred_af'] = pred_af_ernie_cn(uc,model_name)
                elif model_name == GPT_4o:
                    uc['pred_af'] = pred_af_gpt_cn(uc,model_name)
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行

    return uc_list

def pred_af_ernie_cn(uc,model_name):
    chat_comp = qianfan.ChatCompletion()

    name = clean_string(uc['Name'])
    desc = clean_string(str(uc['Brief Description']))
    
    user_pmt = "用例描述:'" + name + "." + desc
    user_pmt += ".用例基本流:'" + str(uc['Basic flow']) + "'.生成该用例的分支流,以Python的list格式输出.不要重复用例描述和基本流"

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"]
    exist = clean_and_convert(exist)

    return exist


def pred_af_gpt_cn(uc,model_name):
    url = "https://api.gptsapi.net/v1/chat/completions"

    sys_pmt = "生成用例的分支流,以Python的list格式输出.不要重复用例描述和基本流"

    name = clean_string(uc['Name'])
    desc = clean_string(str(uc['Brief Description']))

    user_pmt = "用例描述:'" + name + "." + desc
    user_pmt += ".用例基本流:'" + str(uc['Basic flow'])

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-NQe10708c5d9ccb970298750dd3b6def3e706feea58rkx7b"  # 请确保将此处的API密钥替换为实际的密钥
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
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
            if response.status_code == 200:
                result = response.json()
                exist = result["choices"][0]["message"]["content"].strip() #去除两端空格
                exist = clean_and_convert(exist)
                return exist
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")

    # 最终失败后递归调用或返回None
    print("多次重试后仍失败，递归调用 pred_af_gpt_cn")
    exist = pred_af_gpt_cn(uc,model_name)
    return exist

def extract_pred_node_from_af_ncet(uc_list, model_name, out_path):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list:
            print(f"*** global id: {uc_list.index(uc)}/{len(uc_list)}, uc name: {uc['Name']} ***")                

            if len(uc['Exc. Flow']) != 0:
                if 'pred_af' not in uc or len(uc['pred_af']) == 0:
                    print(f"uc id: {uc['id']} has error") # 有分支流的uc没有进行预测
                if model_name == ERNIE_4_turbo:  # 这里选择使用ernie统一进行node提取
                    uc['pred_af_act'] = pred_af_node_ernie_ncet(uc,model_name,'act')
                    uc['pred_af_obj'] = pred_af_node_ernie_ncet(uc,model_name,'obj')
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
                    
            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行

    return uc_list

def pred_af_node_ernie_ncet(uc,model_name,label):
    chat_comp = qianfan.ChatCompletion()

    if label == 'act':    
        user_pmt = "'" + uc['pred_af'] + "'.提取该句中所有的动词,以Python的list格式输出"
    elif label == 'obj':    
        user_pmt = "'" + uc['pred_af'] + "'.提取该句中所有的名词,以Python的list格式输出"

    # 加入重试机制
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 输入llm
            resp = chat_comp.do(model=model_name, messages=[
                {
                    "role": "user",
                    "content": user_pmt
                }
            ], disable_search=True)

            exist = resp["body"]["result"]
            return exist
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")
    
    # 最终失败后递归调用或返回None
    print("多次重试后仍失败，递归调用 pred_af_node_ernie_ncet")
    exist = pred_af_node_ernie_ncet(uc,model_name)
    return exist

def find_tp_cn(uc_list_pred_node,uc_list_gt_node,out_path,model_name):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        # json_file.write('[\n')  # 手动添加逗号和换行
        for uc_pred,uc_gt in zip(uc_list_pred_node,uc_list_gt_node):

            if uc_pred['id'] != uc_gt['id']: print(f'uc列表没有对齐！')

            print(f"*** id: {uc_pred['id']}/{len(uc_list_pred_node)}, Name: {uc_pred['Name']} ***")   

            if len(uc_pred['Exc. Flow']) != 0 :
                # check 一下
                if len(uc_pred['Exc. Flow'])!=len(uc_gt['AF act']) or len(uc_pred['Exc. Flow'])!=len(uc_gt['AF obj']):
                    print(f"Exc. Flow mismatch: {uc_pred['id']} vs {uc_gt['id']}")     
                # check 结束

                alt_node_list = flatten_list(uc_gt["AF act"])
                uc_pred['tp_act'] = get_tp_in_loop_cn(uc_pred['pred_af_act'], alt_node_list,model_name)
                alt_node_list = flatten_list(uc_gt["AF obj"])
                uc_pred['tp_obj'] = get_tp_in_loop_cn(uc_pred['pred_af_obj'], alt_node_list,model_name)


            json.dump(uc_pred, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行

def get_tp_in_loop_cn(pred_af_list,af_list,model_name):
    af_list = [s.lower() for s in af_list] # 将af_list中的元素转换为小写，便于后续比较
    tp_list = []
    max_count = 5
    count = 0
    # 提取 tp 
    for pred_node in pred_af_list:
        while True: # 循环直到找到tp/false，或达到最大次数
            if model_name == ERNIE_4_turbo or model_name == GPT_4o:  # OpenRounter的余额用完了，后面都用ernie
                tp = find_tp_ernie_cn(pred_node.lower(), af_list, ERNIE_4_turbo)
            elif model_name == GPT_4o:
                tp = find_tp_gpt_cn(pred_node.lower(), af_list, model_name)
            
            if tp in af_list or 'false' in tp.lower():  # 找到或者找不到  # 是否要加“无”？
                break

            count += 1
            if count >= max_count:
                print("循环已达到最大次数，即将退出")
                break

        if 'false' not in tp.lower() and tp in af_list:  # 找到了tp
            tp_list.append(tp)
            print(f"af_list = {af_list}, pred_node = {pred_node}, tp = {tp}")
            af_list.remove(tp)  # 从af_list中移除已找到的tp
        else:
            print(f"false, and tp = {tp.lower()}")                
    return tp_list

def find_tp_ernie_cn(pred_node,af_list,model_name):
    chat_comp = qianfan.ChatCompletion()

    user_pmt = str(af_list) + \
           "该列表中如果存在与'" + pred_node + "'完全相同或语义相似的词," \
           "则输出第一个相同的或语义相似的词；如果不存在，返回'False'。不要输出任何其他信息" 
    
    # 加入重试机制
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 输入llm
            resp = chat_comp.do(model=model_name, messages=[
                {
                    "role": "user",
                    "content": user_pmt
                }
            ], disable_search=True)

            if resp['code'] != 200:
                print(f'resp["code"] ={resp["code"]} 出错！')
                continue

            exist = resp["body"]["result"]
            exist = exist.lower()  # 将可能出现的FALSE都变为小写
            exist = exist.strip()  # 去除两端空格
            exist = re.sub(r'^[\W_]+|[\W_]+$', '', exist)  # 去除两端特殊符号
            return exist
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")

    # 最终失败后递归调用或返回None
    print(f"多次重试后仍失败，递归调用 {inspect.currentframe().f_back.f_code.co_name}")
    exist = find_tp_ernie_cn(pred_node,af_list,model_name)
    return exist


def find_tp_gpt_cn(pred_node, af_list, model_name):
    url="https://openrouter.ai/api/v1/chat/completions"  # OpenRounter

    # prompt
    user_pmt = str(af_list) + \
           "该列表中如果存在与'" + pred_node + "'完全相同或语义相似的词," \
           "则输出第一个相同的或语义相似的词；如果不存在，返回'False'。不要输出任何其他信息" 

    # 定义请求头
    headers={
        "Authorization": "xxx"
    }

    # 定义请求数据
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": user_pmt}
        ]
    }

    # 加入重试机制
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 发送POST请求
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            if response.status_code != 200:
                print(f' response 不对！！ code= {response.status_code}尝试次数: {attempt+1}/{max_retries}')
            # 处理响应
            elif response.status_code == 200:
                # 如果请求成功，解析JSON响应
                result = response.json()
                exist = result["choices"][0]["message"]["content"]
                exist = exist.strip()  # 去除两端空格
                exist = re.sub(r'^[\W_]+|[\W_]+$', '', exist)  # 去除两端特殊符号
                return exist
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")
    
    # 最终失败后递归调用或返回None
    print(f"多次重试后仍失败，递归调用 {inspect.currentframe().f_back.f_code.co_name}")
    exist = find_tp_gpt_cn(pred_node, af_list, model_name)
    return exist



if __name__ == '__main__':
    
    task_name = 'calculate_ncet_metric_value_delete'
    print(f'*** task_name: {task_name} , Starting time: {datetime.now()}  !!!!! ***')
    
    if task_name =='get_pred_af_by_llm':
        in_path = 'E:/Trae_project/ConditionOfUCS/0_Data/3_cleaned_json_dataset/cleaned_NCE-T.json'

        out_path_ernie = 'E:/Trae_project/ConditionOfUCS/0_Data/baseline_data/baseline_LLM/NCET/Ernie_NCET_pred_af.json'
        out_path_gpt = 'E:/Trae_project/ConditionOfUCS/0_Data/baseline_data/baseline_LLM/NCET/GPT_NCET_pred_af.json'

        # for out_path, model_name in zip([out_path_ernie, out_path_gpt],[ERNIE_4_turbo,GPT_4o]): # ERNIE已经跑完了，所以删除了ernie的内容
        for out_path, model_name in zip([out_path_gpt],[GPT_4o]):
            uc_list = read_uc_from_stand_json(in_path)
            uc_list = get_pred_af_by_llm_ncet(uc_list, out_path, model_name)

    elif task_name == 'get_pred_af_node_ncet':
        in_path_ernie = 'E:/Trae_project/ConditionOfUCS/0_Data/baseline_data/baseline_LLM/NCET/Ernie_NCET_pred_af.json'
        in_path_gpt = 'E:/Trae_project/ConditionOfUCS/0_Data/baseline_data/baseline_LLM/NCET/GPT_NCET_pred_af.json'

        out_path_ernie = 'E:/Trae_project/ConditionOfUCS/0_Data/baseline_data/baseline_LLM/NCET/Ernie_NCET_pred_node.json'
        out_path_gpt = 'E:/Trae_project/ConditionOfUCS/0_Data/baseline_data/baseline_LLM/NCET/GPT_NCET_pred_node.json'

        for in_path, out_path in zip([in_path_ernie,in_path_gpt],[out_path_ernie,out_path_gpt]):
            uc_list = read_uc_from_stand_json(in_path)

            uc_list = extract_pred_node_from_af_ncet(uc_list, ERNIE_4_turbo, out_path)

    elif task_name == 'find_tp_in_llm_pred_af_ncet':
        pred_af_node_ernie = '../0_Data/baseline_data/baseline_LLM/NCET/Ernie_NCET_pred_node.json'
        af_node_ground_truth_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_pred_node_withtp.json"

        pred_af_node_gpt = '../0_Data/baseline_data/baseline_LLM/NCET/GPT_NCET_pred_node.json'
        af_node_ground_truth_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_pred_node_withtp.json"

        # implement_model指的是文件夹是什么模型的baseline 方法。执行tp extract统一使用ernie 4 turbo
        for pred_node_path, af_node_ground_truth, out_path, baseline, model_name in zip([pred_af_node_ernie,pred_af_node_gpt],
                                                                  [af_node_ground_truth_ernie,af_node_ground_truth_gpt],
                                                                  [out_path_ernie,out_path_gpt],
                                                                  ["Ernie",'GPT'],[ERNIE_4_turbo,GPT_4o]):  
            
            if model_name == ERNIE_4_turbo:
                continue # 分开两个pc跑程序，一个跑ernie；另一个gpt。修改这个模型名称即可。

            # 检测文件是同一个baseline的
            if not check_all_under_same_method(baseline,[pred_node_path, af_node_ground_truth, out_path,model_name]):
                break

            uc_list_pred_node = read_uc_from_stand_json(pred_node_path)
            uc_list_gt_node = read_uc_from_stand_json(af_node_ground_truth)
            if len(uc_list_pred_node) != len(uc_list_gt_node):
                print(f'pred list 与 gt list 长度不同！！！')

            uc_list_pred_node = find_list_in_string(uc_list_pred_node)  # string中提取list
            
            for uc in uc_list_pred_node:  # 删除错误词
                if "pred_af_act" in uc:
                    uc['pred_af_act'] = delete_error(uc['pred_af_act'], ERROR_WORD_LIST)
                    uc['pred_af_obj'] = delete_error(uc['pred_af_obj'], ERROR_WORD_LIST)

            find_tp_cn(uc_list_pred_node[1548:],uc_list_gt_node[1548:],out_path,model_name)

    # 因为NCET数据集中的af，额外加入了“程序终止”。evaluate_llm_pred_af任务是计算有“程序“、”终止”node的情况。
    elif task_name == 'evaluate_llm_pred_af':
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_pred_node_withtp.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_pred_node_withtp.json"

        af_gt_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        af_gt_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_eval.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_eval.json"


        for in_path,gt_path,out_path,baseline in zip([in_path_ernie,in_path_gpt],[af_gt_path_ernie,af_gt_path_gpt],[out_path_ernie,out_path_gpt],["ernie",'gpt']):
            # 检测文件是同一个baseline的
            if not check_all_under_same_method(baseline,[in_path, out_path]):
                break

            uc_list = read_uc_from_stand_json(in_path)
            uc_gt_list = read_uc_from_stand_json(gt_path)

            for uc,gt in zip(uc_list,uc_gt_list):
                if 'AF act' in gt:
                    uc['AF act'] = gt['AF act']  # 将gt node给到uc_list中
                    uc['AF obj'] = gt['AF obj']


            uc_list = eval_Ernie_pred_ncet(uc_list)
            write_uc_to_stand_json(out_path,uc_list)

    # 因为NCET数据集中的af，额外加入了“程序终止”。evaluate_llm_pred_af任务是计算有“程序“、”终止”node的情况。下面这个为配套。
    elif task_name == 'calculate_ncet_metric_value':
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_eval.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_eval.json"

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet.json"

        print(f"ernie tp 数量: {count_node(read_uc_from_stand_json(in_path_ernie),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_ernie),'tp_obj')}")
        print(f"gpt tp 数量: {count_node(read_uc_from_stand_json(in_path_gpt),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_gpt),'tp_obj')}")


        # 计算所有pub数据集的每项平均
        for in_path, out_path,baseline in zip([in_path_ernie, in_path_gpt], [out_path_ernie, out_path_gpt],['ernie','gpt']):
            # 检测文件是同一个baseline的
            if not check_all_under_same_method(baseline,[in_path, out_path]):
                break

            use_case_list = read_uc_from_stand_json(in_path)
            
            calculate_all_dataset(use_case_list, out_path)


    # 因为NCET数据集中的af，额外加入了“程序终止”。evaluate_llm_pred_af_delete任务是计算 无 “程序“、”终止”node的情况。
    elif task_name == 'evaluate_llm_pred_af_delete':
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_pred_node_withtp.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_pred_node_withtp.json"

        af_gt_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        af_gt_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_eval_delete.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_eval_delete.json"


        for in_path,gt_path,out_path,baseline in zip([in_path_ernie,in_path_gpt],[af_gt_path_ernie,af_gt_path_gpt],[out_path_ernie,out_path_gpt],["ernie",'gpt']):
            # 检测文件是同一个baseline的
            if not check_all_under_same_method(baseline,[in_path, out_path]):
                break

            uc_list = read_uc_from_stand_json(in_path)
            uc_gt_list = read_uc_from_stand_json(gt_path)

            for uc,gt in zip(uc_list,uc_gt_list):
                if 'AF act' in gt:
                    uc['AF act'] = [[item for item in sublist if item != "终止"] for sublist in gt['AF act']]  # 将gt node给到uc_list中。删除“程序“、”终止” node。
                    uc['AF obj'] = [[item for item in sublist if item != "程序"] for sublist in gt['AF obj']]

            uc_list = eval_Ernie_pred_ncet(uc_list)
            write_uc_to_stand_json(out_path,uc_list)

    # 为evaluate_llm_pred_af_delete任务的配套。
    elif task_name == 'calculate_ncet_metric_value_delete':
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_eval_delete.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_eval_delete.json"

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_delete.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/NCET/GPT_ncet_delete.json"

        print(f"ernie tp 数量: {count_node(read_uc_from_stand_json(in_path_ernie),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_ernie),'tp_obj')}")
        print(f"gpt tp 数量: {count_node(read_uc_from_stand_json(in_path_gpt),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_gpt),'tp_obj')}")


        # 计算所有pub数据集的每项平均
        for in_path, out_path,baseline in zip([in_path_ernie, in_path_gpt], [out_path_ernie, out_path_gpt],['ernie','gpt']):
            # 检测文件是同一个baseline的
            if not check_all_under_same_method(baseline,[in_path, out_path]):
                break

            use_case_list = read_uc_from_stand_json(in_path)
            
            calculate_all_dataset(use_case_list, out_path)


    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')
