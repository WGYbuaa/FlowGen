from imports import datetime,GPT_4o,json,requests,importlib,GROUPING_UC_20,math,random,copy
from utils import read_uc_from_stand_json,write_uc_to_stand_json,list_depth_recursive,flatten_list

def chack_data(ernie_data, gpt_data):
    for uc_ernie, uc_gpt in zip(ernie_data, gpt_data):
        assert uc_ernie['id'] == uc_gpt['id'], f"UC_ID mismatch: {uc_ernie['id']} != {uc_gpt['id']}"
        assert uc_ernie['Alt. Flow'] == uc_gpt['Alt. Flow'], f"Alt. Flow mismatch in UC_ID {uc_ernie['id']}"
        assert uc_ernie['Basic flow'] == uc_gpt['Basic flow'], f"Basic flow mismatch in UC_ID {uc_ernie['id']}"
        print(f"Data match for UC_ID {uc_ernie['id']}")

def change_alt_flow_list_to_dict(uc_list):
    for uc in uc_list:
        old_alt_flow = uc['Alt. Flow']
        uc['Alt. Flow'] = []  # 先清空，后续添加
        uc['Alt. Flow'] = [{'n_d': item} for item in old_alt_flow]  # 'n_d':n表示分支点的index，d表示该分支点是数据集原本带的。h为人加的；L为大模型加的。
        
    return uc_list

def bp_index_gpt(basic_flow,alter_flow,model_name):
    # 
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = "Bearer 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"


    sys_pmt = "Return only the integer index (starting from 0) of the branching point of the alternative flow in the basic flow."
    user_pmt = "basic flow:" + str(basic_flow) + ". alternative flow:" + str(alter_flow)

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": api_key  # 请确保将此处的API密钥替换为实际的密钥
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
        if exist.isdigit():
            return int(exist)  # 转换为整数
        else:
            return None    # 非纯数字字符串，返回None或自定义处理
    else:
        print(f"Request failed with status code {response.status_code}, response: {response.text}")




def get_branching_point_index(basic_flow,alter_flow,model_name):
    max_count = 5 # 最大调用次数
    if list_depth_recursive(basic_flow) ==2: # 如果是二级列表
        basic_flow = [' '.join(sublist) for sublist in basic_flow]
    else: 
        print('basic_flow 不是二级列表，请检查！')

    count = 0
    while True:
        if 'gpt' in model_name.lower():
            index = bp_index_gpt(basic_flow,alter_flow,model_name)

            if index is not None:
                break

    return index

def add_bp_by_gpt(uc_list,model_name,out_path):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list:
            print(f"*** uc id: {uc['id']}, dataset: {uc['dataset']} ***")
            if len(uc['Alt. Flow'])>0:
                for dicts in uc['Alt. Flow']:  
                    id_n = list(dicts.keys())[0]
                    branching_index = id_n.split('_')[0]
                    if branching_index == 'n': 
                        new_index = get_branching_point_index(uc["Basic flow"],list(dicts.values())[0],model_name) # 返回int
                        new_id_n = f"{new_index}_{'L'}"  # 只用gpt4o
                        dicts[new_id_n] = dicts.pop(id_n)
            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行
    return uc_list

def find_same_step(bf_list,af,list_label):
    bp, list_label_old = -99, list_label.copy()  
    for i, step in enumerate(bf_list):
        if step == af and list_label[i] == 0:
            list_label[i] = 1  # 标记该步骤已被使用
            bp = i
            break

    if bp == -99 or list_label_old == list_label:  # 如果最后bp没有赋值
        print(f"*** Warning: No matching step found for alternative flow step '{af}' in basic flow. ***")
    
    return bp,list_label

def find_bp_ncet(uc_list):
    error_count = 0
    for uc in uc_list:
        old_alt_flow = uc['Exc. Flow']
        uc['Exc. Flow'] = []  # 先清空，后续添加
        list_label = [0] * len(uc['Basic flow'])  # 初始化标签列表，长度与基本流相同,用以标记哪些步骤已经被标记为bp

        if len(old_alt_flow)>0:
            for af in old_alt_flow:
                if len(af)>2: 
                    print(f"UC ID {uc['id']} step>2") 
                    
                bp, list_label = find_same_step(uc['Basic flow'],af[0],list_label)  # af只有第一个步骤与基本流相同，第二个步骤为后续加的“程序终止”
                uc['Exc. Flow'].append({f"{bp}_d": af})  # 'n_d':n表示分支点的index，其中-99为错误。d表示该分支点是数据集原本带的。h为人加的；L为大模型加的。

                if bp==-99: error_count+=1
    print(f'找不到bp的地方有： {error_count} 个！')
    return uc_list


def check_data(uc_list1, uc_list2):  # uc_list1为新提取的bf node
    count2,count4,count3,count1,count5 = 0,0,0,0,0
    for uc1, uc2 in zip(uc_list1, uc_list2):
        if len(uc1['Basic flow']) != len(uc2['Basic flow']) and len(uc2['Exc. Flow']) !=0:
            count2 +=1
            if len(uc1['Basic flow']) == 1:
                count4 +=1
            elif len(uc1['Basic flow']) < len(uc2['Basic flow']):
                print(f'uc2["Basic flow"]: {uc2["Basic flow"]}')
                count1+=1
            elif len(uc1['Basic flow']) > len(uc2['Basic flow']):
                count5+=1
        if 'Exc. flow' in uc1 or 'Exc. flow' in uc2:
            if uc1['Exc. flow'] != uc2['Exc. Flow']:
                count3+=1

    print(f'分支流不相同的情况：{count3}')
    print(f"有分支流的情况下，bf数量不同的 count2: {count2}")
    print(f"新提取的bf node中，基本流只有一步的 count4: {count4}")
    print(f'拆分后基本流小于参照的 count1: {count1}')
    print(f'拆分后基本流大于参照的 count5: {count5}')
    print("*************")

def check_af_in_all_sub_graph(uc_list):
    list1 = []
    for i in range(len(uc_list)):
        count1 = 0
        for uc in uc_list[i]:
            if len(uc["Exc. Flow"])>0:
                count1 += 1
        if count1 == 0:
            list1.append(i)
        if len(uc_list[i])<10 and count1 == 0:
            print(f'第 {i} 个 sub graph 个数小于10 且 没有分支流.')

    if len(list1) ==0:
        print('所有 sub graph 都有分支流。')
    else:
        print(f'{list1}')
    return list1

def get_uc_with_af_id(uc_list):
    id_list = []
    for uc in uc_list:
        if len(uc["Exc. Flow"])>0:
            id_list.append(uc['global id'])
    return id_list

def generate_bp_review_md(uc_list, out_md_path):
    """
    生成对齐的 Markdown 审核表格，列:
    | Dataset Name | Use Case ID | Alternative Flow ID | BP (Branch Point) Predicted by LLM | Do You Agree with the LLM Prediction? (1/0) | If Not Agree, Your BP Index |
    Alternative Flow ID 从 0 开始（当前 uc 中该 af 的序号）。
    仅记录 BP Predicted by LLM 列中以 "_L" 结尾的条目。
    """
    headers = [
        "Dataset",
        "UC_ID",
        "AF_ID",
        "BP Predicted by LLM",
        "Do You Agree with the LLM Prediction? (1/0)",
        "If Not Agree, Your BP Index"
    ]

    rows = []
    for uc in uc_list:
        dataset = str(uc.get("dataset", ""))
        usecase_id = str(uc.get("id", uc.get("global id", "")))
        alt_flows = uc.get("Alt. Flow", [])
        for idx, alt_dict in enumerate(alt_flows):
            if not isinstance(alt_dict, dict) or len(alt_dict) == 0:
                continue
            af_id = list(alt_dict.keys())[0]  # 当前 id_n，如 "3_L" 或 "n_d"
            alt_id_str = str(idx)  # Alternative Flow ID 从0开始
            bp_pred = af_id  # 保持现有 id_n 填入 BP Predicted 列
            # 仅保留以 "_L" 结尾的预测结果
            if isinstance(bp_pred, str) and bp_pred.endswith("_L"):
                rows.append([dataset, usecase_id, alt_id_str, bp_pred, "1", ""])

    # 计算每列最大宽度（包含表头）
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    col_widths = [max(len(str(cell)) for cell in col) for col in cols]

    # 生成表头和分隔行（用 '-' 填充以保持对齐）
    header_line = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

    # 生成行字符串，统一对齐
    row_lines = []
    for r in rows:
        line = "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(r))) + " |"
        row_lines.append(line)

    # 写入文件
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        for line in row_lines:
            f.write(line + "\n")

    return out_md_path


if __name__ == '__main__':
    task_name = 'd_L_bp_number'
    print(f'*** task_name: {task_name} , Starting time: {datetime.now()}  !!!!! ***')
    
    if task_name =='find_branching_point':  # 提出branching point之后，人工将分支点的index加上。不过由于人工index是从1开始的，change_bp任务整体-1.
        in_file_path_ernie = '../0_Data/4_alt_flow_data/0_raw_data/ERNIE_pub_integrated.json'
        in_file_path_gpt = '../0_Data/4_alt_flow_data/0_raw_data/gpt_pub_integrated.json'

        out_file_path = '../0_Data/5_branching_point/original_data/pub_with_bp.json'

        # 0. 判断两个文件中的aft flow数据是否一致 ===> 是一致的
        # chack_data(read_uc_from_stand_json(in_file_path_ernie),read_uc_from_stand_json(in_file_path_gpt))

        # 1. 先将数据集中的"Alt. Flow"都变成列表中包含多个字典，字典的key为分支点所在step的index,以及来源；value为该分支点的分支流
        uc_list = change_alt_flow_list_to_dict(read_uc_from_stand_json(in_file_path_ernie))

        write_uc_to_stand_json(out_file_path,uc_list)

    if task_name =='change_bp': # 按照原文手动加上的分支点的index需要都-1
        uc_list = read_uc_from_stand_json('../0_Data/5_branching_point/original_data/pub_with_bp.json')
        for uc in uc_list:
            if len(uc['Alt. Flow'])>0:
                for dicts in uc['Alt. Flow']:  
                    id_n = list(dicts.keys())[0]
                    branching_index = id_n.split('_')[0]
                    if branching_index != 'n': 
                        new_index = int(branching_index) - 1
                        new_id_n = f"{new_index}_{id_n.split('_')[1]}"
                        dicts[new_id_n] = dicts.pop(id_n)
        write_uc_to_stand_json('../0_Data/5_branching_point/original_data/pub_with_bp_1.json',uc_list)  # _1的文件已经删除，更新到不_1的pub_with_initial_bp.json文件中了。
                
    if task_name =='add_bp_by_gpt': # 让gpt来找分支点
        uc_list = read_uc_from_stand_json('../0_Data/5_branching_point/original_data/pub_with_initial_bp.json')

        out_path = '../0_Data/5_branching_point/1_gpt_added_bp/pub_with_gpt_bp.json'
        uc_list = add_bp_by_gpt(uc_list,GPT_4o,out_path)

    elif task_name =='find_bp_ncet':
        out_path = '../0_Data/5_branching_point/2_ncet_bp/NCET_with_bp.json'

        # 完善于../0_Data/3_cleaned_json_dataset/cleaned_NCE-T.json'
        uc_list = read_uc_from_stand_json('../0_Data/5_branching_point/0_original_data/NCET_with_initial_bp_1.json')

        # 判断分支流是否一致。这里不需要分ernie或是gpt，因为是对于ncet数据集的统计。
        check_data(read_uc_from_stand_json('../0_Data/4_alt_flow_data/0_raw_data/ERNIE_NCET_integrated.json'),uc_list)

        # 有1829个bf不同，因为没有拆开，但是拆开就一样，或者更多一两个了，所以直接按照uc_list中的bf来找bp即可
        # 拆分后有182个BF步骤变多了，这没事儿，因为加的都是最后的“清除环境”之类，没有af，不影响bp的index。
        # # Exc. Flow变为字典，key为“index_来源”，value为该分支流的列表
        uc_list = find_bp_ncet(uc_list)


        write_uc_to_stand_json(out_path,uc_list)

        
    elif task_name == 're_group_ncet':
        in_file_path_ernie = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        # 没有分支流的sub的id
        spec = importlib.util.spec_from_file_location("module_name", "7_make_pt_file.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        uc_list = module.local_id_to_global_id(read_uc_from_stand_json(in_file_path_ernie))
        uc_list = module.group_uc_fixed(uc_list, GROUPING_UC_20)
        list1 = check_af_in_all_sub_graph(uc_list)
        group_uc = copy.deepcopy(GROUPING_UC_20)

        for id in list1:
            if id < 350:  # 一共402个sub
                id_with_af_list = get_uc_with_af_id(uc_list[id+49]) # 获取id+49的对应sub graph中哪些uc是具有af的
                if len(id_with_af_list) <2:
                    print(f' {id+49} ')
                exchange_num = int(min(len(group_uc[id]),len(id_with_af_list)) / 2) # 将后1/2个uc_with_af转换过去
                group_uc[id][-exchange_num:], group_uc[id+49][-exchange_num:] = group_uc[id+49][-exchange_num:], group_uc[id][-exchange_num:]
            else:
                id_with_af_list = get_uc_with_af_id(uc_list[id - 15]) # 获取id+49的对应sub graph中哪些uc是具有af的
                exchange_num = int(min(len(group_uc[id]),len(id_with_af_list)) / 2) # 将后1/2个uc_with_af转换过去
                group_uc[id][-exchange_num:], group_uc[id- 15][-exchange_num:] = group_uc[id- 15][-exchange_num:], group_uc[id][-exchange_num:]
        
        with open('E:/Trae_project/AFGen_pt_file/3_with_branching_point/ncet_regroup_1.json', 'w') as f:
            # 遍历外层列表
            for inner_list in group_uc:
                # 将内层列表序列化为JSON字符串，并写入文件
                json_string = json.dumps(inner_list)
                f.write(json_string + ',\n') # 每个序列化后的列表占一行

    elif task_name == 'check_regroup':
        if len(GROUPING_UC_20)!= len(GROUPING_UC_20):
            print(f'sub 总个数不够！')
        for gp,re_gp in zip(GROUPING_UC_20,GROUPING_UC_20):
            if len(gp) != len(re_gp):
                print(f'第 {GROUPING_UC_20.index(gp)} 个数不对！')
        
        flatten = flatten_list(GROUPING_UC_20)
        if len(flatten)!= 6506:
            print(f'uc 总个数不够！')
        
        for i in range(0,6505):
            if i not in flatten:
                print(f'{i} uc 不在新的列表中')

        print
        
    elif task_name == 'test':# 给邦祺制作的简易pt
        spec = importlib.util.spec_from_file_location("module_name", "7_make_pt_file.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        in_file_path_gpt = '../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json'
        uc_list = module.local_id_to_global_id(read_uc_from_stand_json(in_file_path_gpt))
        uc_with_af, uc_wo_af = [],[]
        for uc in uc_list:
            if uc['global id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10]:  # 给邦祺制作的简易pt
                if len(uc['Alt. Flow'])>0:
                    uc_with_af.append(uc['global id'])
                else:
                    uc_wo_af.append(uc['global id'])       

    elif task_name == 'd_L_bp_number': # 公开数据集中哪些数据集的bp补全，是后期加的
        uc_list = read_uc_from_stand_json('../0_Data/5_branching_point/1_gpt_added_bp/pub_with_gpt_bp.json')
        dataset_dict_by_llm, defa_num = {},0  # 第二个是原本就有的bp数量，应该等于141
        for uc in uc_list:
            if len(uc['Alt. Flow'])>0:
                print(f"*** uc id: {uc['id']}, dataset: {uc['dataset']}, af num:{len(uc['Alt. Flow'])} ***")
                for dicts in uc['Alt. Flow']:  
                    id_n = list(dicts.keys())[0]
                    branching_label = id_n.split('_')[1]
                    if branching_label == 'L': 
                        if uc['dataset'] not in dataset_dict_by_llm:
                            dataset_dict_by_llm[uc['dataset']] = 1
                        else:
                            dataset_dict_by_llm[uc['dataset']] += 1
                    else:
                        defa_num += 1
        print(f'default bp number: {defa_num}')
        print(f'dataset_dict_by_llm: {dataset_dict_by_llm}')

        # 生成供人工审核的 markdown 表格，按要求把 dataset、usecase id、Alternative Flow ID 以及当前 id_n 填入，剩下列留空
        out_md = '../0_Data/5_branching_point/1_gpt_added_bp/bp_review.md'
        generate_bp_review_md(uc_list, out_md)

        print(f'default bp number: {defa_num}')
        print(f'dataset_dict_by_llm: {dataset_dict_by_llm}')
        print(f'Generated review markdown: {out_md}')
                    

                


    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')