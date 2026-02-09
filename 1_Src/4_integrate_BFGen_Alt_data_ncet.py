from utils import read_uc_from_json,read_uc_from_stand_json,write_uc_to_stand_json,split_string,is_pure_punctuation,delete_error
from imports import ERROR_WORD_LIST_BFGen,inspect,re,datetime

# 找到节点所属的步骤,用于ernie 4 turbo。这是每个step提取node，不存在没提取到node的情况。
def find_step_node_belong(node_list, node_list_ori, error_word_list):
    delimiters = ['，', ',', '。']  # 分割str的符号
    update_list, step_list, new_step_list = [], [], []
    for node_ori in node_list_ori:
        if "：" in node_ori:  # 先根据中文冒号，删除冒号前llm的描述语句
            node_ori = node_ori.split('：')[1]
        split = split_string(node_ori, delimiters)
        step_list.extend(split)

        for index in range(len(step_list)):
            if step_list[index] != "" and not step_list[index].isdigit() and not is_pure_punctuation(step_list[index]):
                item = step_list[index].strip()  # 去除两端空格
                item = re.sub(r'^[\W_]+|[\W_]+$', '', item)  # 去除两端特殊符号
                new_step_list.append(item.lower())  # 统一大小写
        update_list.append(new_step_list)
        step_list, new_step_list = [], []

    # 去除error
    update_list_new = []
    for item in update_list:
        update_list_new.append(delete_error(item, error_word_list))

    # 检验结果
    list1 = [item for sublist in update_list_new for item in (sublist if isinstance(sublist, list) else [sublist])]
    if list1 != node_list:
        print(f" ***** Error !!! node数目对不上 *****")

    return update_list_new


def extract_one_node_from_string(node_list, error_word_list):
    delimiters = ['，', ',', '。']  # 分割str的符号
    update_list, new_step_list,step_list = [],[],[]
    for node_str in node_list: # 当前每个step对应的node都是str格式。
        if "：" in node_str:  # 先根据中文冒号，删除冒号前llm的描述语句
            node_str = node_str.split('：')[1]
        split = split_string(node_str, delimiters)
        step_list.extend(split)

        for index in range(len(step_list)):
            if step_list[index] != "" and not step_list[index].isdigit() and not is_pure_punctuation(step_list[index]):
                item = step_list[index].strip()  # 去除两端空格
                item = re.sub(r'^[\W_]+|[\W_]+$', '', item)  # 去除两端特殊符号
                new_step_list.append(item.lower())  # 统一大小写
        update_list.append(new_step_list)
        step_list, new_step_list = [], []

    # 去除error
    update_list_new = []
    for item in update_list:
        update_list_new.append(delete_error(item, error_word_list))

    # 每个step只留第一个node即可
    for row in update_list_new:
        if len(row) < 1:
            row.append("无")
    result = [[row[0]] for row in update_list_new]

    return result

def find_step_node_belong_chatgpt(use_case_list, uc_list_with_step,error_word_list):
    for _1_uc, _2_uc_with_step in zip(use_case_list, uc_list_with_step):
        _1_uc['Bf act'] = find_node_from_step(_2_uc_with_step['steps'], _1_uc['Bf act'],error_word_list)
        _1_uc['Bf obj'] = find_node_from_step(_2_uc_with_step['steps'], _1_uc['Bf obj'],error_word_list)

        # 有的step没有提取出来node。因为chatgpt版本是一整个uc放进去提取的，存在某个step没提取到node的情况。
        if len(_1_uc['Bf act']) != len(_1_uc['Bf obj']) or len(_1_uc['Bf act']) != len(_1_uc['Basic flow']):
            print("error find_step_node_belong_chatgpt")

    return use_case_list

def find_node_from_step(step_list, node_list,error_word_list):
    node_list_copy = node_list[:]
    step_list_copy = step_list[:]
    # 创建len(steps)个列表，node属于哪个就放第几个list里
    nested_list = [[] for _ in range(len(step_list_copy))]
    for i in range(len(step_list_copy) - 1, -1, -1):  # node从后向前遍历，所以step也从后往前好了
        for j in range(len(node_list_copy) - 1, -1, -1):  # 从后向前遍历，可以直接删除循环中的item
            if node_list_copy[j] in step_list_copy[i]:
                step_list_copy[i] = step_list_copy[i].replace(node_list_copy[j], '', 1)
                nested_list[i].append(node_list_copy[j])
                del node_list_copy[j]

    # 为了严谨，从后往前遍历的node，再将列表倒转一下，使得node顺序正常。
    for sublist in nested_list:
        sublist.reverse()

    # 去除error
    update_list_new = []
    for item in nested_list:
        update_list_new.append(delete_error(item, error_word_list))
    # 每个step只留第一个node即可
    for row in update_list_new:
        if len(row) < 1:
            row.append("无")
    result = [[row[0]] for row in update_list_new]

    return result


if __name__ == '__main__':
    task_name = 'integrate_data_ncet_gpt'
    print(f'*** task_name: {task_name}, Starting time: {datetime.now()} !!!!! ***')

    if task_name =='integrate_data_ncet_ernie':
        in_path_key_node = 'E:/bertTest/20240421/ControlledExper/2_dataset_origin_node/Ernie-4-Turbo/with_keyword/Ernie_NCET_ground_truth.json'
        # in_path_key_node = 'E:/GitHub/ASSAM/data/2_dataset_origin_node/Ernie-4-Turbo/with_keyword/Ernie_NCET_ground_truth.json'
        in_path_step = 'E:/bertTest/20240421/ControlledExper/2_dataset_origin_node/Ernie-4-Turbo/Ernie_NCET_ground_truth.json'  # 需要根据这个找到act/obj每个node所属的step
        af_path = "../0_Data/baseline_data/baseline_LLM/NCET/Ernie_ncet_pred_node_withtp.json"
        uc_list_key_node = read_uc_from_json(in_path_key_node)
        uc_list_step = read_uc_from_json(in_path_step)
        uc_list_af = read_uc_from_stand_json(af_path)
        out_path = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json' #这里用到BFGen中node是ernie提取的。其实BF node/key node是ernie还是gpt提取的没有影响；反而应该控制该变量一致。

        # 需要判断对齐，不仅是id等，还有bf node的个数与step的个数
        for uc_key,uc_step,uc_af in zip(uc_list_key_node,uc_list_step,uc_list_af):
            if uc_key['index'] != uc_step['index'] or uc_key['index'] != uc_af['id']:
                print(f'uc id {uc_key["id"]} 不匹配！！')
            if len(uc_step['steps']) != len(uc_step['act']) or len(uc_step['steps']) != len(uc_step['obj']):
                print(f'uc id: {uc_step["index"]}! uc_step steps: {len(uc_step["Basic flow"])}; uc_step:{len(uc_step["act"])}')


        for uc_key,uc_step,uc_af in zip(uc_list_key_node,uc_list_step,uc_list_af):
            # 删除这里不需要的item
            if 'pred_af' in uc_af:
                del uc_af['pred_af'],uc_af['pred_af_act'],uc_af['pred_af_obj'],uc_af['tp_act'],uc_af['tp_obj']
            uc_af['Basic flow'] = uc_key['steps'] # 发现BFGen和AFGen对于bf的提取有一些出入，就用以前（BFGen）提取的吧，不需要重新提取了。

            # 赋值key node
            for key in ['key_act','key_obj','key_path','key_name']:
                if key in uc_key:
                    uc_af[key] = uc_key[key]
            
            # 赋值 bf node.直接从in_path_step里重新提取node吧。
            uc_af['BF act'] = extract_one_node_from_string(uc_step['act'], ERROR_WORD_LIST_BFGen) 
            uc_af['BF obj'] = extract_one_node_from_string(uc_step['obj'], ERROR_WORD_LIST_BFGen)

            if len(uc_af['BF act']) != len(uc_af['BF obj']) or len(uc_af['Basic flow']) != len(uc_af['BF obj']):
                print(f'Error! {inspect.currentframe().f_lineno}')


        write_uc_to_stand_json(out_path,uc_list_af)


    # 其实BF node/key node是ernie还是gpt提取的没有影响；反而应该控制该变量一致。但是既然有GPT提取的数据，就还是用上吧。
    elif task_name =='integrate_data_ncet_gpt':
        bf_node_path = 'E:/GitHub/ASSAM/data/4_dataset_pred_node/chatgpt_4o/2nd_round/with_tp/Chatgpt_NCE-T.json'
        uc_list = read_uc_from_json(bf_node_path)
        with_steps = "E:/bertTest/20240421/ControlledExper/2_dataset_origin_node/Ernie-4-Turbo/with_keyword/Ernie_NCET_ground_truth.json"
        out_path = '../0_Data/4_alt_flow_data/0_raw_data/GPT_NCET_integrated.json'


        for uc in uc_list:
            # 删除这里不需要的item
            if 'pred_steps' in uc:
                del uc['pred_steps'],uc['pred_act'],uc['pred_obj'],uc['tp_act'],uc['tp_obj']
            uc['id'] = uc.pop('index')
            uc['Basic flow'] = uc.pop('steps')
            uc['Name'] = uc.pop('uctext')
            uc['Brief Description'] = uc.pop('ucPath')
            uc['Bf act'] = uc.pop('act')
            uc['Bf obj'] = uc.pop('obj')
            

        uc_list = find_step_node_belong_chatgpt(uc_list,read_uc_from_json(with_steps),ERROR_WORD_LIST_BFGen)

        write_uc_to_stand_json(out_path,uc_list)

            
            
        

    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')