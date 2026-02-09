from csv import writer
from nt import write
from re import L
from unittest import result
from utils import read_uc_from_json,read_uc_from_stand_json,write_uc_to_stand_json
from imports import Path,json,groupby

def get_json_paths(directory):
    dir_path = Path(directory)
    # 递归匹配所有.json文件（含子目录）
    return [str(file) for file in dir_path.rglob('*.json')]

def find_duplicate_dicts(lst, key="data"):
    # 按指定键排序（分组前必须排序）
    sorted_lst = sorted(lst, key=lambda x: x[key])
    groups = []
    # 遍历分组结果
    for _, group in groupby(sorted_lst, key=lambda x: x[key]):
        group_list = list(group)
        if len(group_list) > 1:  # 仅保留重复项的分组
            groups.append(group_list)
    return groups


def integrate_data_111(uc_list_BFGen, cleaned_alt_data_folder):
    cleaned_file_paths = get_json_paths(cleaned_alt_data_folder)
    uc_list = []
    uc_bfgen_new_list = []
    incorrect_order_dataset = ["easyClinic","eTour"]
    no_alt_dataset = ["eANCI", "SMOS"]
    for json_file in cleaned_file_paths:
        if "SMOS" not in json_file and "eANCI" not in json_file:
            with open(json_file, 'r', encoding='utf-8') as f:
                    uc_list_alt = json.load(f)
            
            # 1. 循环cleaned列表
            for uc_alt in uc_list_alt:
                for uc_bfgen in uc_list_BFGen: # 循环bfgen列表
                    if uc_alt['dataset'] == uc_bfgen['dataset'] and uc_alt['dataset'] not in incorrect_order_dataset:
                        altname = uc_alt['Name'].replace(" ", "")
                        bfname = uc_bfgen['ucName'].replace(" ", "")
                        if uc_alt['id'] == uc_bfgen['index'] and (altname == bfname or altname in bfname):
                            uc = {}
                            uc_bfgen_new = {}
                            uc["index"] = uc_bfgen['index']
                            uc_bfgen_new["index"] = uc_bfgen['index']

                            uc['dataset'] = uc_bfgen['dataset']
                            uc_bfgen_new["dataset"] = uc_bfgen['dataset']

                            uc['Name'] = uc_bfgen['ucName']
                            uc_bfgen_new["Name"] = uc_bfgen['ucName']


                            if 'uctext' in uc_bfgen:
                                uc['Brief Description'] = uc_bfgen['uctext']
                                uc_bfgen_new['Brief Description'] = uc_bfgen['uctext']
                            elif 'Brief Description' in uc_alt:
                                uc['Brief Description'] = uc_alt["Brief Description"]

                            uc['Basic flow'] = uc_bfgen['steps']
                            uc_bfgen_new['Basic flow'] = uc_bfgen['steps']

                            if 'Alt. Flow' in uc_alt:
                                uc['Alt. Flow'] = uc_alt['Alt. Flow']
                            else:
                                uc['Alt. Flow'] = []

                            uc['BF act'] = uc_bfgen['act']
                            uc_bfgen_new['BF act'] = uc_bfgen['act']

                            uc['BF obj'] = uc_bfgen['obj']
                            uc_bfgen_new['BF obj'] = uc_bfgen['obj']


                            if 'key_name' in uc_bfgen:
                                uc['key_name'] = uc_bfgen['key_name']
                                uc_bfgen_new['key_name'] = uc_bfgen['key_name']
                            else:
                                uc['key_name'] = []

                            if 'key_act' in uc_bfgen:
                                uc['key_act'] = uc_bfgen['key_act']
                                uc_bfgen_new['key_act'] = uc_bfgen['key_act']
                            else:
                                uc['key_act'] = []
                                print(uc_bfgen['dataset'])

                            if 'key_obj' in uc_bfgen:
                                uc['key_obj'] = uc_bfgen['key_obj']
                                uc_bfgen_new['key_obj'] = uc_bfgen['key_obj']
                            else:
                                uc['key_obj'] = []
                            uc_list.append(uc)
                            uc_bfgen_new_list.append(uc_bfgen_new)
                            break                   
                        else:
                            print("123")

                    if uc_alt['dataset'] == uc_bfgen['dataset'] and uc_alt['dataset'] in incorrect_order_dataset:
                        altname = uc_alt['Name'].replace(" ", "")
                        bfname = uc_bfgen['ucName'].replace(" ", "")
                        if altname == bfname or altname in bfname:
                            uc = {}
                            uc_bfgen_new = {}
                            uc["index"] = uc_bfgen['index']
                            uc_bfgen_new["index"] = uc_bfgen['index']

                            uc['dataset'] = uc_bfgen['dataset']
                            uc_bfgen_new["dataset"] = uc_bfgen['dataset']

                            uc['Name'] = uc_bfgen['ucName']
                            uc_bfgen_new["Name"] = uc_bfgen['ucName']


                            if 'uctext' in uc_bfgen:
                                uc['Brief Description'] = uc_bfgen['uctext']
                                uc_bfgen_new['Brief Description'] = uc_bfgen['uctext']
                            elif 'Brief Description' in uc_alt:
                                uc['Brief Description'] = uc_alt["Brief Description"]

                            uc['Basic flow'] = uc_bfgen['steps']
                            uc_bfgen_new['Basic flow'] = uc_bfgen['steps']

                            if 'Alt. Flow' in uc_alt:
                                uc['Alt. Flow'] = uc_alt['Alt. Flow']
                            else:
                                uc['Alt. Flow'] = []

                            uc['BF act'] = uc_bfgen['act']
                            uc_bfgen_new['BF act'] = uc_bfgen['act']

                            uc['BF obj'] = uc_bfgen['obj']
                            uc_bfgen_new['BF obj'] = uc_bfgen['obj']


                            if 'key_name' in uc_bfgen:
                                uc['key_name'] = uc_bfgen['key_name']
                                uc_bfgen_new['key_name'] = uc_bfgen['key_name']
                            else:
                                uc['key_name'] = []

                            if 'key_act' in uc_bfgen:
                                uc['key_act'] = uc_bfgen['key_act']
                                uc_bfgen_new['key_act'] = uc_bfgen['key_act']
                            else:
                                uc['key_act'] = []
                                print(uc_bfgen['dataset'])

                            if 'key_obj' in uc_bfgen:
                                uc['key_obj'] = uc_bfgen['key_obj']
                                uc_bfgen_new['key_obj'] = uc_bfgen['key_obj']
                            else:
                                uc['key_obj'] = []
                            uc_list.append(uc)
                            uc_bfgen_new_list.append(uc_bfgen_new)
                            break    
                        else:
                            print("123")
        
        elif "SMOS" in json_file:
            for uc_bfgen in uc_list_BFGen:
                if uc_bfgen["dataset"] == "SMOS":
                    uc = {}
                    uc_bfgen_new = {}
                    uc["index"] = uc_bfgen['index']
                    uc_bfgen_new["index"] = uc_bfgen['index']

                    uc['dataset'] = uc_bfgen['dataset']
                    uc_bfgen_new["dataset"] = uc_bfgen['dataset']

                    uc['Name'] = uc_bfgen['ucName']
                    uc_bfgen_new["Name"] = uc_bfgen['ucName']


                    if 'uctext' in uc_bfgen:
                        uc['Brief Description'] = uc_bfgen['uctext']
                        uc_bfgen_new['Brief Description'] = uc_bfgen['uctext']
                    else:
                        uc['Brief Description'] = []

                    uc['Basic flow'] = uc_bfgen['steps']
                    uc_bfgen_new['Basic flow'] = uc_bfgen['steps']

                    if 'Alt. Flow' in uc_bfgen:
                        uc['Alt. Flow'] = uc_bfgen['Alt. Flow']
                    else:
                        uc['Alt. Flow'] = []

                    uc['BF act'] = uc_bfgen['act']
                    uc_bfgen_new['BF act'] = uc_bfgen['act']

                    uc['BF obj'] = uc_bfgen['obj']
                    uc_bfgen_new['BF obj'] = uc_bfgen['obj']


                    if 'key_name' in uc_bfgen:
                        uc['key_name'] = uc_bfgen['key_name']
                        uc_bfgen_new['key_name'] = uc_bfgen['key_name']
                    else:
                        uc['key_name'] = []

                    if 'key_act' in uc_bfgen:
                        uc['key_act'] = uc_bfgen['key_act']
                        uc_bfgen_new['key_act'] = uc_bfgen['key_act']
                    else:
                        uc['key_act'] = []
                        print(uc_bfgen['dataset'])

                    if 'key_obj' in uc_bfgen:
                        uc['key_obj'] = uc_bfgen['key_obj']
                        uc_bfgen_new['key_obj'] = uc_bfgen['key_obj']
                    else:
                        uc['key_obj'] = []
                    uc_list.append(uc)
                    uc_bfgen_new_list.append(uc_bfgen_new)   
            
        elif "eANCI" in json_file:
            for uc_bfgen in uc_list_BFGen:
                if uc_bfgen["dataset"] == "eANCI":
                    uc = {}
                    uc_bfgen_new = {}
                    uc["index"] = uc_bfgen['index']
                    uc_bfgen_new["index"] = uc_bfgen['index']

                    uc['dataset'] = uc_bfgen['dataset']
                    uc_bfgen_new["dataset"] = uc_bfgen['dataset']

                    uc['Name'] = uc_bfgen['ucName']
                    uc_bfgen_new["Name"] = uc_bfgen['ucName']


                    if 'uctext' in uc_bfgen:
                        uc['Brief Description'] = uc_bfgen['uctext']
                        uc_bfgen_new['Brief Description'] = uc_bfgen['uctext']
                    else:
                        uc['Brief Description'] = []

                    uc['Basic flow'] = uc_bfgen['steps']
                    uc_bfgen_new['Basic flow'] = uc_bfgen['steps']

                    if 'Alt. Flow' in uc_bfgen:
                        uc['Alt. Flow'] = uc_bfgen['Alt. Flow']
                    else:
                        uc['Alt. Flow'] = []

                    uc['BF act'] = uc_bfgen['act']
                    uc_bfgen_new['BF act'] = uc_bfgen['act']

                    uc['BF obj'] = uc_bfgen['obj']
                    uc_bfgen_new['BF obj'] = uc_bfgen['obj']


                    if 'key_name' in uc_bfgen:
                        uc['key_name'] = uc_bfgen['key_name']
                        uc_bfgen_new['key_name'] = uc_bfgen['key_name']
                    else:
                        uc['key_name'] = []

                    if 'key_act' in uc_bfgen:
                        uc['key_act'] = uc_bfgen['key_act']
                        uc_bfgen_new['key_act'] = uc_bfgen['key_act']
                    else:
                        uc['key_act'] = []
                        print(uc_bfgen['dataset'])

                    if 'key_obj' in uc_bfgen:
                        uc['key_obj'] = uc_bfgen['key_obj']
                        uc_bfgen_new['key_obj'] = uc_bfgen['key_obj']
                    else:
                        uc['key_obj'] = []
                    uc_list.append(uc)
                    uc_bfgen_new_list.append(uc_bfgen_new)
    
    
    return uc_list,uc_bfgen_new_list

def add_key(uc_list_BFGen,ref_list):
    for uc, ref in zip(uc_list_BFGen, ref_list):
        if uc['index'] == ref['index'] and uc['dataset'] == ref['dataset']:
            uc['ucName'] = ref['ucName']
        else:
            print(f"错误！{uc['index']}")
    return uc_list_BFGen

def new_key_name(dict_old):
    dict_new = {}
    dict_new['id'] = dict_old['index']
    dict_new['dataset'] = dict_old['dataset']
    dict_new['Name'] = dict_old['ucName']

    if 'uctext' in dict_old:
        dict_new['Brief Description'] = dict_old['uctext']
    else:
        dict_new['Brief Description'] = []

    dict_new['Basic flow'] = dict_old['steps']


    if 'Alt. Flow' in dict_old:
        dict_new['Alt. Flow'] = dict_old['Alt. Flow']
    else:
        dict_new['Alt. Flow'] = []


    dict_new['BF act'] = dict_old['act']
    dict_new['BF obj'] = dict_old['obj']
    if 'key_name' in dict_old:
        dict_new['key_name'] = dict_old['key_name']
    else:
        dict_new['key_name'] = []

    if 'key_act' in dict_old:
        dict_new['key_act'] = dict_old['key_act']
    else:
        dict_new['key_act'] = []

    if 'key_obj' in dict_old:
        dict_new['key_obj'] = dict_old['key_obj']
    else:
        dict_new['key_obj'] = []
    return dict_new

def integrate_data(uc_list_BFGen, cleaned_alt_data_folder):
    cleaned_file_paths_list = get_json_paths(cleaned_alt_data_folder) # 获得cleaned文件地址
    dataset_change = "keepass"  # 先定义第一个数据集，后面就不用管了
    uc_list_BFGen_new = []
    # 这次根据bfgen的uc顺序来
    for uc_bfgen in uc_list_BFGen:
        # 数据集变换的时候 更换cleaned文件
        if uc_bfgen['dataset'] != dataset_change:
            cleaned_file = [item for item in cleaned_file_paths_list if uc_bfgen['dataset'] in item][0]
            dataset_change = uc_bfgen['dataset']
        
        # 读取cleaned数据
        cleaned_file = [item for item in cleaned_file_paths_list if dataset_change in item][0]
        uc_clean_list = read_uc_from_stand_json(cleaned_file)
        
        # 如果该cleaned的用例数据集没有alt(eTour\eANCI\SMOS...)，则直接按照bfgen的内容
        if 'Alt. Flow' not in uc_clean_list[1] and 'Alt. Flow' not in uc_clean_list[3]: 
            uc_bfgen = new_key_name(uc_bfgen)
            uc_list_BFGen_new.append(uc_bfgen)
            continue


        # 如果cleaned uc 中有alt，则复制给uc_bfgen
        bfname = uc_bfgen['ucName'].replace(" ", "")
        for uc_clean in uc_clean_list:
            altname = uc_clean['Name'].replace(" ", "")

            if uc_clean['dataset'] =="easyClinic": # 这个数据集顺序不一致
                if altname == bfname or altname in bfname:
                    uc_bfgen = new_key_name(uc_bfgen)
                    if 'Alt. Flow' in uc_clean:
                        uc_bfgen['Alt. Flow'] = uc_clean['Alt. Flow']
                        uc_list_BFGen_new.append(uc_bfgen)
                        break


            else: # 其他顺序一致的dataset
                if uc_bfgen['index'] == uc_clean['id'] and (altname == bfname or altname in bfname):
                    uc_bfgen = new_key_name(uc_bfgen)
                    if 'Alt. Flow' in uc_clean:
                        uc_bfgen['Alt. Flow'] = uc_clean['Alt. Flow']
                        uc_list_BFGen_new.append(uc_bfgen)
                        break

    return uc_list_BFGen_new
        






    return uc_list

if __name__ == "__main__":
    task_name = 'check_result'

    print(f"*** task_name: {task_name} ***")

    # integrate_data_GPT 之后，判断一下ernie和gpt的文件，是否只有提取的act obj不同，其他都应该是一样的
    if task_name == 'check_result':
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json"

        uc_gpt_list = read_uc_from_stand_json(GPT_integ_path)
        uc_ernie_list = read_uc_from_stand_json(Ernie_integ_path)

        for uc_gpt, uc_ernie in zip(uc_gpt_list,uc_ernie_list):
            if uc_gpt['id'] != uc_ernie['id'] and uc_gpt['dataset'] != uc_ernie['dataset'] and uc_gpt['Name'] != uc_ernie['Name']:
                print(f"GPT用例 {uc_gpt['id']} 与 Ernie用例 {uc_ernie['id']} 不匹配")
                print(f"GPT用例 {uc_gpt['dataset']} 与 Ernie用例 {uc_ernie['dataset']} 不匹配")
                print(f"GPT用例 {uc_gpt['Name']} 与 Ernie用例 {uc_ernie['Name']} 不匹配")

            if uc_gpt['Brief Description'] != uc_ernie['Brief Description']:
                print(f"GPT用例 {uc_gpt['Brief Description']} 与 Ernie用例 {uc_ernie['Brief Description']} 不匹配")

            if uc_gpt['Basic flow'] != uc_ernie['Basic flow']:
                print(f"GPT用例 {uc_gpt['Basic flow']} 与 Ernie用例 {uc_ernie['Basic flow']} 不匹配")

            if uc_gpt['Alt. Flow'] != uc_ernie['Alt. Flow']:
                print(f"GPT用例 {uc_gpt['Alt. Flow']} 与 Ernie用例 {uc_ernie['Alt. Flow']} 不匹配")

            if uc_gpt['key_act'] == [] and uc_ernie['key_act'] != []:
                print(f"GPT用例 {uc_gpt['key_act']} 与 Ernie用例 {uc_ernie['key_act']} 不匹配")



    # 将GPT的BFGen数据与alt整合到一起，即BFGen proj中的bf等数据，加上cleaned的分支流、异常流的数据。
    # 但是这次可以用task“integrate_data”生成的Ernie的数据来了，因为都是BFGen的顺序，不存在乱序了。
    # 只需要增加alt flow 和 name 即可
    if task_name == "integrate_data_GPT":
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
        GPT_4o_path = "E:/GitHub/ASSAM/data/2_dataset_origin_node/Chatgpt_4o/4rd_after_formalized/GPT_pub_gt.json"
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json"

        uc_list_GPT_before_precessing = read_uc_from_json(GPT_4o_path)
        uc_list_Ernie_after_processing = read_uc_from_stand_json(Ernie_integ_path)

        uc_list_new = []
        # 判断两组用例是否对齐
        for uc_gpt, uc_ernie in zip(uc_list_GPT_before_precessing, uc_list_Ernie_after_processing):
            if uc_gpt['index'] != uc_ernie['id']:
                print(f"GPT用例 {uc_gpt['index']} 与 Ernie用例 {uc_ernie['id']} 不匹配")
            if 'uctext' in uc_gpt and uc_gpt['uctext'] != uc_ernie['Brief Description']:
                print(f"GPT用例 {uc_gpt['index']} 与 Ernie用例 {uc_ernie['id']} 描述不匹配")
            if uc_gpt['steps'] !=  uc_ernie['Basic flow']:
                print(f"GPT用例 {uc_gpt['index']} 与 Ernie用例 {uc_ernie['id']} 基本流不匹配")
            if uc_gpt['dataset'] != uc_ernie['dataset']:
                print(f"GPT用例 {uc_gpt['index']} 与 Ernie用例 {uc_ernie['id']} 数据集不匹配")
            
            uc_new = {}
            uc_new['id'] = uc_gpt['index']
            uc_new['dataset'] = uc_gpt['dataset']
            uc_new['Name'] = uc_ernie['Name']

            if 'uctext' in uc_gpt:
                uc_new['Brief Description'] = uc_gpt['uctext']
            elif 'Brief Description' in uc_ernie:
                uc_new['Brief Description'] = uc_ernie['Brief Description']

            uc_new['Basic flow'] = uc_gpt['steps']
            uc_new['Alt. Flow'] = uc_ernie['Alt. Flow']
            uc_new['BF act'] = uc_gpt['act']
            uc_new['BF obj'] = uc_gpt['obj']

            if 'key_name' in uc_gpt:
                uc_new['key_name'] = uc_gpt['key_name']
            else:
                uc_new['key_name'] = []

            if 'key_act' in uc_gpt and 'key_obj' in uc_gpt:
                uc_new['key_act'] = uc_gpt['key_act']
                uc_new['key_obj'] = uc_gpt['key_obj']
            else:
                uc_new['key_act'] = []
                uc_new['key_obj'] = []
                
            uc_list_new.append(uc_new)

            
        write_uc_to_stand_json(GPT_integ_path,uc_list_new)


        print("123")





    # 先统一alt格式，有的不是二级列表
    if task_name == "integrate_alt_":
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"

        # 先统一alt格式，有的不是二级列表
        uc_list = read_uc_from_stand_json(Ernie_integ_path)
        for uc in uc_list:
            list1 = []
            if uc['Alt. Flow'] != []:
                if isinstance(uc['Alt. Flow'], list):
                    if isinstance(uc['Alt. Flow'][0], dict):
                        for alt_dict in uc['Alt. Flow']:
                            if isinstance(alt_dict['text'], list):
                                list1.append(alt_dict['text'])
                            else:
                                list1.append([alt_dict['text']])
                        uc['Alt. Flow'] = list1
                    elif not isinstance(uc['Alt. Flow'][0], list):
                        print("123")
                else:
                        print("123")

        write_uc_to_stand_json(Ernie_integ_path, uc_list)
                        




    # 首先将数据整合到一起，即BFGen proj中的bf等数据，加上cleaned的分支流、异常流的数据。
    if task_name == "integrate_data":
        BFGen_data_file = "E:/GitHub/ASSAM/data/2_dataset_origin_node/Ernie-4-Turbo/2_pub_after_formalized/Ernie_pub_gt.json"
        cleaned_alt_data_folder = "../0_Data/3_cleaned_json_dataset/"
        out_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"

        uc_list_BFGen = read_uc_from_json(BFGen_data_file)
        # 因为uc_list_BFGen中的uc没有name等，找个给他们加上
        uc_list_BFGen = add_key(uc_list_BFGen, read_uc_from_json("E:/GitHub/ASSAM/data/1_dataset_origin/pub_format/pub_only_format_uctext.json"))

        # 整合出新的用例列表
        uc_list = integrate_data(uc_list_BFGen, cleaned_alt_data_folder)

        write_uc_to_stand_json(out_path, uc_list)



    # 首先将数据整合到一起，即BFGen proj中的bf等数据，加上cleaned的分支流、异常流的数据。
    # 这个task的输入文件错了，不用这个task了
    if task_name == "integrate_data_111":
        BFGen_data_file = "../0_Data/4_alt_flow_data/Ernie_pub.json"
        cleaned_alt_data_folder = "../0_Data/3_cleaned_json_dataset/"
        out_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"

        uc_list_BFGen = read_uc_from_stand_json(BFGen_data_file)

        # 整合出新的用例列表
        uc_list,uc_bfgen_new_list = integrate_data_111(uc_list_BFGen, cleaned_alt_data_folder)

        write_uc_to_stand_json(out_path, uc_list)
        write_uc_to_stand_json("../0_Data/4_alt_flow_data/Ernie_pub111.json", uc_bfgen_new_list)



    if task_name == 'get_bfgen_data':
        BFGen_data_file = "E:/GitHub/ASSAM/data/4_dataset_pred_node/ERNIE_4_Turbo_8k/with_tp/3rd_round/Ernie_pub.json"
        cleaned_alt_data_folder = "../0_Data/3_cleaned_json_dataset/"
        temp_output_path = "../0_Data/4_alt_flow_data/Ernie_pub.json"

        uc_list_BFGen = read_uc_from_json(BFGen_data_file)

        with open(temp_output_path, 'w', encoding='utf-8') as f:
            json.dump(uc_list_BFGen, f, ensure_ascii=False, indent=4)
        

    
    if task_name == "find_error":
        cleaned_itrust = "../0_Data/3_cleaned_json_dataset/cleaned_itrust.json"
        uc_list = read_uc_from_stand_json(cleaned_itrust)

        index = 0
        for uc in uc_list:
            uc['id'] = index
            index += 1
        
        write_uc_to_stand_json(cleaned_itrust,uc_list)



    print(f"*** task_name: {task_name} Finish! *** ")