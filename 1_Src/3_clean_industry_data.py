# 用于将BFGen的工业数据拿来用，但是要读出其中的分支流。
from utils import read_uc_from_json, output_uc_to_json
from imports import os, re

def extract_operate_expect_strings(file_path):
    """
    从指定文件中提取所有operate和expect语句中的字符串，并按顺序存放在列表中
    
    Args:
        file_path: 文件路径
        
    Returns:
        列表，包含所有提取的字符串
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='GBK') as f:
        content = f.read()
    
    # 使用正则表达式匹配所有operate和expect语句中的字符串
    operate_pattern = r'operate\("([^"]+)"\)\s*\{'
    expect_pattern = r'expect\("([^"]+)"\)\s*\{'
    
    # 找到所有匹配
    operate_matches = re.findall(operate_pattern, content)
    expect_matches = re.findall(expect_pattern, content)
    
    # 创建结果列表
    result = []
    
    # 遍历文件内容，按顺序添加operate和expect字符串
    lines = content.split('\n')
    for line in lines:
        operate_match = re.search(operate_pattern, line)
        if operate_match:
            result.append(operate_match.group(1))
            continue
            
        expect_match = re.search(expect_pattern, line)
        if expect_match:
            result.append(expect_match.group(1))
    
    return result

def extract_expect_strings(file_path):
    """
    从指定文件中提取所有expect语句中的字符串，并按顺序存放在二级列表中
    同时删除开头可能存在的数字和符号
    
    Args:
        file_path: 文件路径
        
    Returns:
        二级列表，包含所有提取的字符串，已去除开头的数字和符号
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='GBK') as f:
        content = f.read()
    
    # 使用正则表达式匹配所有expect语句中的字符串
    pattern = r'expect\("([^"]+)"\)\s*\{'
    matches = re.findall(pattern, content)
    
    # 处理匹配结果，删除开头可能存在的数字和符号
    result = []
    for match in matches:
        # 使用正则表达式删除开头的数字和符号（如2.1、）
        # cleaned_string = re.sub(r'^\d+(\.\d+)*[、.]\s*', '', match)
        result.append([match])
    
    return result



# 2、删除不需要的key
def get_uc_list(use_case_list_all):
    keys_to_delete = {"key_act", "key_obj","act","obj"}
    use_case_list = [
            {k: v for k, v in d.items() if k not in keys_to_delete}
            for d in use_case_list_all
        ]
    return use_case_list

def find_alt_flow(use_case_list, ncet_path):
    uc_new_list = []
    for uc in use_case_list:
        uc_new = {}
        tc_file_path = os.path.join(ncet_path, uc['ucPath'], uc['ucText'] + '.tc')
        if not os.path.exists(tc_file_path):
            print(f"文件 {tc_file_path} 不存在")
            continue
        
        # 找到了文件之后
        uc_new['id'] = uc['index']
        uc_new['dataset'] = 'NCE-T'
        uc_new['Name'] = uc['ucText']
        uc_new['Brief Description'] = uc['key_path']
        uc_new['Basic flow'] = extract_operate_expect_strings(tc_file_path)
        # 分支流这里应该将意思取反，因为basic flow是成功路径，分支流这里是异常路径。
        # 但是为了能找到分支点，即bf中分支的步骤，所以暂时不做调整。
        alt_flow = extract_expect_strings(tc_file_path)
        uc_new['Alt. Flow'] = [sublist + ['程序终止。'] for sublist in alt_flow]


        uc_new_list.append(uc_new)
    return uc_new_list



if __name__ == "__main__":
    task_name = 'read_NCET_data_find_ALT_flow'

    print(f"*** task_name: {task_name} ***")

    # 用BFGen的数据，但是标出其中的分支流
    if task_name == 'read_NCET_data_find_ALT_flow':
        # 1、读取json文件
        json_file_path = 'E:/bertTest/20240421/ControlledExper/2_dataset_origin_node/Ernie-4-Turbo/with_keyword/Ernie_NCET_ground_truth.json'
        use_case_list = read_uc_from_json(json_file_path)
        out_path = 'E:/Trae_project/ConditionOfUCS/0_Data/3_cleaned_json_dataset/cleaned_NCE-T.json'

        # 2、删除不需要的key
        use_case_list = get_uc_list(use_case_list)

        # 3、根据ucText和ucPath找到源文件，找出uc中的分支流
        ncet_path = "E:/bertTest/data/NCE-T_scripts"
        use_case_list = find_alt_flow(use_case_list, ncet_path)

        # 4、打印
        output_uc_to_json(use_case_list, out_path)


        print("任务结束")