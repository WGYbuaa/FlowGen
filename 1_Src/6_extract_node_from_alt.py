# 提取 node
from os import altsep
from tracemalloc import take_snapshot

from pydantic import BeforeValidator
from imports import GPT_35, GPT_4o, ERNIE_4_turbo, ERNIE_ernie35, json, qianfan,re,requests
from utils import read_uc_from_stand_json

def get_list_depth(lst):
    if not isinstance(lst, list):  # 非列表元素深度为0
        return 0
    if not lst:  # 空列表深度为1（自身算一层）
        return 1
    max_depth = 0
    for item in lst:
        if isinstance(item, list):
            # 递归计算子列表深度，取最大值
            max_depth = max(max_depth, get_list_depth(item))
    return max_depth + 1  # 当前层深度 = 子列表最大深度 + 1


# 来自DELL G7的 _2_/task0，具体位置为 utils_llm.py
def extract_node_eng_after_formalizd_ernie(steps,model_name, label):
    chat_comp = qianfan.ChatCompletion()

    # prompt
    if label == 'act':
        str1 = "str1='" + steps + "'.Extract the most important verb from str1, or return false if none is found. Do not output any other information."
    elif label == 'obj':
        str1 = "str1='" + steps + "'.Extract the most important object from str1, or return false if none is found. Do not output any other information."

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {"role": "user", "content": str1}
    ], disable_search=True)

    exist = resp["body"]["result"]
    exist = exist.strip()  # 去除两端空格
    exist = re.sub(r'^[\W_]+|[\W_]+$', '', exist)  # 去除两端特殊符号
    if 'false' in exist or 'False' in exist or not exist:
        return "None"
    else:
        return exist

# 来自DELL G7的 _2_/task0，具体位置为 utils_gpt.py
def extract_node_eng_after_formalizd_gpt(steps, model_name,label):
    url = "https://api.gptsapi.net/v1/chat/completions"

    # prompt
    if label == 'act':
        str1 = "str1='" + steps + "'.Extract the most important verb from str1, or return false if none is found. Do not output any other information."
    elif label == 'obj':
        str1 = "str1='" + steps + "'.Extract the most important object from str1, or return false if none is found. Do not output any other information."

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-NQe10708c5d9ccb970298750dd3b6def3e706feea58rkx7b"  # 请确保将此处的API密钥替换为实际的密钥
    }

    # 定义请求数据
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": str1}
        ]
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code != 200:
        print(f' response 不对！！response.status_code= {response.status_code}')
    # 处理响应
    if response.status_code == 200:
        # 如果请求成功，解析JSON响应
        result = response.json()
        exist = result["choices"][0]["message"]["content"]
        exist = exist.strip()  # 去除两端空格
        exist = re.sub(r'^[\W_]+|[\W_]+$', '', exist)  # 去除两端特殊符号
        if 'false' in exist or 'False' in exist or not exist:
            return "None"
        else:
            return exist
    return None


def extract_node_alt_flow(uc_list, model_name, out_path):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list:
            print(f'*** uc id: {uc["id"]}, dataset:{uc["dataset"]} ***')
            uc['AF act'],uc['AF obj'] = [],[]
            for alt_flow in uc['Alt. Flow']: # SIP后是3级列表，这一级是每一条alt
                act_node_list,obj_node_list = [],[]
                for alt_step in alt_flow:   # SIP后是3级列表，这一级是每一条alt中的每一个step
                    if isinstance(alt_step, str):
                        if not isinstance(alt_step, str):
                            print(f"alt_step: {alt_step} 不是字符串！！")
                        if " " not in alt_step: # 如果语句中没有空格，说明该字符串为单一单词（英文）。
                            act_node_list.append(alt_step)
                            obj_node_list.append(alt_step)
                            continue
                        if 'ERNIE' in model_name:
                            act_node_list.append(extract_node_eng_after_formalizd_ernie(alt_step,model_name, 'act'))
                            obj_node_list.append(extract_node_eng_after_formalizd_ernie(alt_step,model_name, 'obj'))
                            
                        elif 'gpt' in model_name:
                            act_node_list.append(extract_node_eng_after_formalizd_gpt(alt_step,model_name,'act'))
                            obj_node_list.append(extract_node_eng_after_formalizd_gpt(alt_step,model_name,'obj'))
                        else:
                            print("*** 模型选择错误！！ ***")
                    
                    elif isinstance(alt_step, list): # sip之后会变成三级列表
                        act_node,obj_node = [], []
                        for seg in alt_step:  # 3级列表，这一级是断句之后的seg
                            if not isinstance(seg, str):
                                print(f"seg: {seg} 不是字符串！！")
                            if " " not in seg: # 如果语句中没有空格，说明该字符串为单一单词（英文）。
                                act_node.append(seg)
                                obj_node.append(seg)
                                continue
                            if 'ERNIE' in model_name:
                                act_node.append(extract_node_eng_after_formalizd_ernie(seg,model_name, 'act'))
                                obj_node.append(extract_node_eng_after_formalizd_ernie(seg,model_name, 'obj'))
                                
                            elif 'gpt' in model_name:
                                act_node.append(extract_node_eng_after_formalizd_gpt(seg,model_name,'act'))
                                obj_node.append(extract_node_eng_after_formalizd_gpt(seg,model_name,'obj'))
                            else:
                                print("*** 模型选择错误！！ ***")
                        act_node_list.append(act_node)
                        obj_node_list.append(obj_node)


                uc['AF act'].append(act_node_list)
                uc['AF obj'].append(obj_node_list)

            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行


            # if uc['id'] == 2: # 实验几个
            #     break


    return uc_list

if __name__ == '__main__':
    task_name = 'check_result'

    print(f"*** task_name: {task_name} ***")
    


    if task_name =="extract_alt_node_AFGen":

        model_name = GPT_4o
        print(f' 当前处理所调用的模型是： {model_name} !!! ')

        if 'ERNIE' in model_name:
            in_path = "../0_Data/4_alt_flow_data/1_after_SIP/Ernie_pub_SIP.json"
            out_path = "../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json"

        elif 'gpt' in model_name:
            in_path = "../0_Data/4_alt_flow_data/1_after_SIP/GPT_pub_SIP.json"
            out_path = "../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json"
        
        uc_list = read_uc_from_stand_json(in_path)

        uc_list = extract_node_alt_flow(uc_list,model_name, out_path)

    if task_name == "extract_alt_node_wo_sip":
        model_name = GPT_4o
        print(f' 当前进行消融实验w/o SIP, 处理所调用的模型是： {model_name} !!! ')

        if 'ERNIE' in model_name:
            in_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
            out_path = "../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_pub_alt_node_woSIP.json"

        elif 'gpt' in model_name:
            in_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json"
            out_path = "../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/gpt_pub_alt_node_woSIP.json"
        
        uc_list = read_uc_from_stand_json(in_path)

        # 判断是否列表少一级
        uc_ref_list = read_uc_from_stand_json("../0_Data/4_alt_flow_data/1_after_SIP/Ernie_pub_SIP.json")
        for uc,uc_ref in zip(uc_list,uc_ref_list):
            uc_depth = get_list_depth(uc['Alt. Flow'])
            uc_ref_depth = get_list_depth(uc_ref['Alt. Flow'])
            if uc_depth + 1 !=uc_ref_depth and len(uc_ref['Alt. Flow']) != 0:
                print(f'uc_depth: {uc_depth}, uc_ref_depth: {uc_ref_depth}')
                print(f' 深度出错！uc id: {uc["id"]}, dataset:{uc["dataset"]} ')


        uc_list = extract_node_alt_flow(uc_list,model_name, out_path)


    if task_name == 'check_result':
        # 检验alt flow和 act 、obj node 的条数是否一致
        model_name = GPT_4o
        print(f' 当前 检查 的结果是： {model_name} !!! ')

        if 'ERNIE' in model_name:
            in_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
            out_path = "../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_pub_alt_node_woSIP.json"

        elif 'gpt' in model_name:
            in_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json"
            out_path = "../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/gpt_pub_alt_node_woSIP.json"

        uc_list_sip = read_uc_from_stand_json(in_path)
        uc_list_alt_node = read_uc_from_stand_json(out_path)

        for uc_sip, uc_node in zip(uc_list_sip,uc_list_alt_node):
            print(f'*** uc id: {uc_node["id"]}, dataset:{uc_node["dataset"]} ***')

            # 判断除alt flow之外的其他item
            for key in uc_sip.keys():
                if key != 'AF act' and key != 'AF obj':
                    if uc_sip[key] != uc_node[key]:
                        print(f'uc id: {uc_sip["id"]}  {key} 不一致')
            
            # 判断alt的node与句子个数之间的对应
            if len(uc_node['AF act']) != len(uc_sip["Alt. Flow"]) or len(uc_node['AF obj']) != len(uc_sip["Alt. Flow"]): # 一级比较
                print(f'uc id: {uc_sip["id"]}  alt flow 与 node 数量不一致！！')
            
            for alt_act_list,alt_obj_list,alt_flow_list in zip(uc_node['AF act'],uc_node['AF obj'],uc_sip["Alt. Flow"]):
                if len(alt_act_list) != len(alt_flow_list) or len(alt_obj_list) != len(alt_flow_list):  # 二级比较
                    print(f'alt_flow_list: {alt_flow_list}  alt flow 与 node 数量不一致！！')


                for alt_act,alt_obj,alt_step in zip(alt_act_list,alt_obj_list,alt_flow_list):
                    if isinstance(alt_act, list): # 如果是三级列表，即sip后的alt
                        if len(alt_act) != len(alt_step) or len(alt_obj) != len(alt_step):  # 三级比较
                            print(f'alt_flow_list: {alt_flow_list}  alt flow 与 node 数量不一致！！')

        print(f"全部正确！")
        
    



    print(f"*** task_name: {task_name} Finish! *** ")
