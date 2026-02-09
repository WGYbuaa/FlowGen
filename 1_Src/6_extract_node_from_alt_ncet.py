from imports import ERNIE_4_turbo, GPT_4o,json,qianfan,requests,re,ERROR_WORD_LIST,inspect,datetime
from utils import read_uc_from_stand_json,list_depth_recursive,current_function,check_all_under_same_method

def extract_alt_node_cn_llm(uc, model_name):
    if len(uc["Exc. Flow"]) == 0:  # 该uc没有"Exc. Flow"
        return []
    
    uc['AF act'],uc['AF obj'] = [],[]
    for exc_flow in uc['Exc. Flow']: # 都是二级列表
        alt_node_act,alt_node_obj = [],[]
        if "程序终止。" not in exc_flow:
            print(f'该uc中缺少“程序终止。”！！')
        for step in exc_flow:
            if isinstance(step, str) and "程序终止" in step: # 异常流的最后一句统一设置成了“程序终止”
                alt_node_act.append('终止')
                alt_node_obj.append('程序')
            elif model_name == ERNIE_4_turbo:
                alt_node_act.append(extract_node_ernie_cn(step,model_name,'act'))
                alt_node_obj.append(extract_node_ernie_cn(step,model_name,'obj'))
            elif model_name == GPT_4o:
                alt_node_act.append(extract_node_gpt_cn(step,model_name,'act'))
                alt_node_obj.append(extract_node_gpt_cn(step,model_name,'obj'))
        uc['AF act'].append(alt_node_act)
        uc['AF obj'].append(alt_node_obj)
    
    return uc


def extract_node_ernie_cn(steps,model_name,label):
    chat_comp = qianfan.ChatCompletion()

    # prompt
    if label == 'act':
        str1 = "'" + steps + "'.提取该句中最重要的动词;如果没有则返回'无'.不要输出其他任何信息"
    elif label == 'obj':
        str1 = "'" + steps + "'.提取该句中最重要的名词;如果没有则返回'无'.不要输出其他任何信息"

    # 加入重试机制
    max_retries = 5
    for attempt in range(max_retries):
        try: 
            # 输入llm
            resp = chat_comp.do(model=model_name, messages=[
                {"role": "user", "content": str1}
            ], disable_search=True)

            exist = resp["body"]["result"]
            exist = exist.strip()  # 去除两端空格
            exist = re.sub(r'^[\W_]+|[\W_]+$', '', exist)  # 去除两端特殊符号
            if 'false' in exist or 'False' in exist or '无' in exist or not exist:
                return "无"
            else:
                return exist
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")
    
    # 最终失败后递归调用或返回None
    print(f"多次重试后仍失败，递归调用 {inspect.currentframe().f_back.f_code.co_name}")
    exist = extract_node_ernie_cn(uc,model_name)
    return exist


def extract_node_gpt_cn(steps,model_name,label):
    url="https://openrouter.ai/api/v1/chat/completions"  # OpenRounter


    # prompt
    if label == 'act':
        str1 = "'" + steps + "'.提取该句中最重要的动词;如果没有则返回'无'.不要输出其他任何信息"

    elif label == 'obj':
        str1 = "'" + steps + "'.提取该句中最重要的名词;如果没有则返回'无'.不要输出其他任何信息"

    # 定义请求头
    # headers = {
    #     "Content-Type": "application/json",
    #     "Authorization": "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"  # 请确保将此处的API密钥替换为实际的密钥
    # }
    headers={
        "Authorization": "Bearer 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"
    }

    # 定义请求数据
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": str1}
        ]
    }

    # 加入重试机制
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # 发送POST请求
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            if response.status_code != 200:
                print(f' response 不对！！response.status_code= {response.status_code}')
            # 处理响应
            elif response.status_code == 200:
                # 如果请求成功，解析JSON响应
                result = response.json()
                exist = result["choices"][0]["message"]["content"]
                exist = exist.strip()  # 去除两端空格
                exist = re.sub(r'^[\W_]+|[\W_]+$', '', exist)  # 去除两端特殊符号
                if 'false' in exist or 'False' in exist or '无' in exist or not exist:
                    return "无"
                else:
                    return exist
        except Exception as e:
            print(f"请求异常: {e}, 尝试次数: {attempt+1}/{max_retries}")
    
    # 最终失败后递归调用或返回None
    print(f"多次重试后仍失败，递归调用 {current_function()}")
    exist = extract_node_gpt_cn(steps,model_name,label)
    return exist



def extract_alt_node_cn(uc_list,out_path,ERNIE_4_turbo,model_name):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        # json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list[1000:]:
            print(f"*** uc id: {uc['id']}, Name: {uc['Name']} ***")

            if "Exc. Flow" in uc and len(uc["Exc. Flow"]) > 0:
                # 使用LLM提取分支流中的node
                if model_name == ERNIE_4_turbo or model_name == GPT_4o:
                    uc = extract_alt_node_cn_llm(uc,model_name)
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行

    return uc_list


def extract_bf_node_cn_llm(bf_list,model_name,label):
    node_list = []
    for bf in bf_list:
        if model_name == ERNIE_4_turbo:
            node_list.append([extract_node_ernie_cn(bf,model_name,label)])
        elif model_name == GPT_4o:
            node_list.append([extract_node_gpt_cn(bf,model_name,label)])
    return node_list


def extract_more_bf_node_cn(uc_list,out_path,model_name,af_label):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list:
            # print(f"*** uc id: {uc['id']}, Name: {uc['Name']} ***")

            if uc['id'] in af_label:
                print(f'{af_label.index(uc["id"])} / {len(af_label)}')
                if len(uc['Basic flow']) !=1 or len(uc['BF act'])!=1 or len(uc['BF obj'])!=1:
                    print(f'该uc的bf已经拆分开了，不需要再提取node！')
                
                new_list = uc['Basic flow'][0].split('。')
                uc['BF act'],uc['BF obj'],uc['Basic flow'] = [],[],[]
                uc['Basic flow'] = new_list
                # 使用LLM提取bf中的node
                if model_name == ERNIE_4_turbo or model_name == GPT_4o:
                    uc['BF act'] = extract_bf_node_cn_llm(new_list,model_name,'act')
                    uc['BF obj'] = extract_bf_node_cn_llm(new_list,model_name,'obj')
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
            
            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行

    return uc_list

if __name__ == '__main__':
    task_name = 'get_bf_node'
    print(f'*** task_name: {task_name}, Starting time: {datetime.now()} !!!!! ***')


    if task_name =='extract_alt_node_AFGen_cn':
        in_path = '../0_Data/3_cleaned_json_dataset/cleaned_NCE-T.json'  # 与xxx_integrated.json中关于分支流内容相同

        # 工业数据集不进行SIP中前两个task,直接提取
        out_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node.json'
        out_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node.json'

        for out_path,model_name in zip([out_path_ernie,out_path_gpt],[ERNIE_4_turbo,GPT_4o]):
            uc_list = read_uc_from_stand_json(in_path)
            
            if model_name == ERNIE_4_turbo:  # 第二次不用跑ernie的了
                continue
            
            # 提取node统一用ernie
            uc_list = extract_alt_node_cn(uc_list,out_path,ERNIE_4_turbo,model_name)

    elif task_name == "check_leaks": # 再跑一遍 gt node中提取出的是“无”的情况。
        in_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node.json' # 7616个“无”
        in_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node.json' # 204个“无”

        out_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        out_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'

        for in_path,out_path,model_name in zip([in_path_gpt,in_path_ernie],[out_path_gpt,out_path_ernie],[GPT_4o,ERNIE_4_turbo]):
            
            # 判断路径一致
            if model_name == ERNIE_4_turbo:
                if not check_all_under_same_method('ERNIE',[in_path,out_path]):  
                    break
            else:
                if not check_all_under_same_method('GPT',[in_path,out_path]):  
                    break
            
            # 对“无”进行二次提取
            uc_list = read_uc_from_stand_json(in_path)
            with open(out_path, 'a', encoding='utf-8') as json_file:
                json_file.write('[\n')  # 手动添加逗号和换行
                for uc in uc_list:
                    print(f"*** uc id: {uc['id']}, Name: {uc['Name']} ***")
                    if "Exc. Flow" in uc and len(uc["Exc. Flow"]) > 0:
                        for af_act,af_obj,Exc_Flow in zip(uc['AF act'],uc['AF obj'],uc['Exc. Flow']):
                            if len(af_act) != len(af_obj) or len(af_act)!=len(Exc_Flow):
                                print(f'af act, af obj, Exc_Flow 个数不一致。')
                            for i in range(len(af_act)):
                                if af_act[i] == '无':
                                    if Exc_Flow[i] == "程序终止。":
                                        af_act[i] = '终止'
                                    else:
                                        if model_name == ERNIE_4_turbo:
                                            af_act[i] = extract_node_ernie_cn(Exc_Flow[i],model_name,'act')
                                        elif model_name == GPT_4o:
                                            af_act[i] = extract_node_gpt_cn(Exc_Flow[i],model_name,'act')
                                
                                if af_obj[i] == '无':
                                    if Exc_Flow[i] == "程序终止。":
                                        af_obj[i] = '程序'
                                    else:
                                        if model_name == ERNIE_4_turbo:
                                            af_obj[i] = extract_node_ernie_cn(Exc_Flow[i],model_name,'obj')
                                        elif model_name == GPT_4o:
                                            af_obj[i] = extract_node_gpt_cn(Exc_Flow[i],model_name,'obj')

                    json.dump(uc, json_file, ensure_ascii=False, indent=4)
                    json_file.write(',\n')  # 手动添加逗号和换行

                json_file.write(']\n')  # 手动添加逗号和换行

    elif task_name =='get_bf_node': # 发现有的bf没有被拆分开，需要重新拆分后提取node，不然没法寻找branching point的index。
        bf_node_ernie = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated_few_bfnode.json'
        bf_node_gpt = '../0_Data/4_alt_flow_data/0_raw_data/GPT_NCET_integrated_few_bfnode.json'

        out_path_ernie = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json'  # 将原来的 integrated 改名为 integrated_fewer_bfnode
        out_path_gpt = '../0_Data/4_alt_flow_data/0_raw_data/GPT_NCET_integrated.json'

        # 0、先统计一下有Exc. Flow的uc数量和位置，只补充提取这些uc中未被拆分的bf
        af_label = []
        for uc in read_uc_from_stand_json(bf_node_ernie):
            if len(uc['Exc. Flow'])>0 and len(uc['Basic flow'])==1:
                af_label.append(uc['id'])
        print(f'len(af_label)={len(af_label)}')  # 应该等于 1829 个
        
        # 1、开始提取
        for in_path, out_path,model_name in zip([bf_node_ernie,bf_node_gpt],[out_path_ernie,out_path_gpt],[ERNIE_4_turbo,GPT_4o]):
            # 检测文件是同一个method的
            if not check_all_under_same_method("GPT",[in_path, out_path,model_name]): # 这次跑gpt的，更换string即可判断出入路径是否统一
                continue

            uc_list = read_uc_from_stand_json(in_path)
            # 提取node统一用ernie
            uc_list = extract_more_bf_node_cn(uc_list,out_path,model_name,af_label)

    elif task_name =='check_bf_node': # 检查bf node是否都提取完了
        in_path_ernie = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json'

        for uc in read_uc_from_stand_json(in_path_ernie):
            # print(f"*** uc id: {uc['id']} ***")
            if len(uc['Basic flow']) != len(uc['BF act']) or len(uc['Basic flow']) != len(uc['BF obj']):
                print(f"*** Warning: UC ID {uc['id']} has inconsistent lengths among 'Basic flow', 'BF act', and 'BF obj'. ***")
            if len(uc['Exc. Flow'])>0:
                if len(uc['Basic flow']) <=1:
                    print(f"*** Warning: UC ID {uc['id']} has 'Exc. Flow' but 'Basic flow' length is {len(uc['Basic flow'])}. ***")
    
    print(f'*** task_name: {task_name} Finish! {datetime.now()}*** ')