# baseline_LLM.py
# 用于使用LLM（GPT-4o和ERNIE 4.0Turbo）进行分支流生成的代码。专用于pub数据集。
from utils import read_uc_from_stand_json, write_uc_to_stand_json,get_json_paths,read_uc_from_json,flatten_list,metric_auc_llm_match,metric_precision_strict_llm_match,metric_recall_llm_match,metric_F1,mix_items,count_node,clean_and_convert,find_list_in_string,eval_Ernie_pred_ncet,calculate_all_dataset,calculate_all_dataset_eval
from imports import ERNIE_4_turbo, GPT_4o,json,qianfan,requests,re,PUB_GROUPING_UC_20_1


def pred_af_node_ernie(uc,model_name,label):
    chat_comp = qianfan.ChatCompletion()

    if label == 'act':    
        user_pmt = "str1='" + uc['pred_af'] + "'.Extract all the verbs from str1, output as a **List**."
    elif label == 'obj':    
        user_pmt = "str1='" + uc['pred_af'] + "'.Extract all the nouns from str1, output as a **List**."

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"]


    return exist

def extract_pred_node_from_af(uc_list, model_name, out_path):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list:
            print(f"*** global index: {uc_list.index(uc)}/{len(uc_list)}, uc index: {uc['id']}, dataset: {uc['dataset']} ***")                

            if len(uc['Alt. Flow']) != 0:
                if 'pred_af' not in uc or len(uc['pred_af']) == 0:
                    print(f"uc id: {uc['id']} has error") # 有分支流的uc没有进行预测
                if model_name == ERNIE_4_turbo:  # 这里选择使用ernie统一进行node提取
                    uc['pred_af_act'] = pred_af_node_ernie(uc,model_name,'act')
                    uc['pred_af_obj'] = pred_af_node_ernie(uc,model_name,'obj')
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
                    
            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行

    return uc_list


def pred_af_ernie(uc,model_name):
    chat_comp = qianfan.ChatCompletion()

    # ERNIE 模型没有 sys_pmt
    if 'uctext' in uc:
        user_pmt = "Use case description:'" + uc['uctext']
    
    user_pmt += ".Use case basic flow:'" + str(uc['steps']) + "'.Generate alternative flows for the use case,output as a **List**."

    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"]


    return exist

def pred_af_gpt(uc,model_name):
    url = "https://api.gptsapi.net/v1/chat/completions"

    sys_pmt = "Generate alternative flows for the use case,output as a **List**."
    if 'uctext' in uc:
        user_pmt = "Use case description:'" + uc['uctext']
    
    user_pmt += ".Use case basic flow:'" + str(uc['steps'])

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
        exist = pred_af_gpt(uc,model_name)
        return exist
    



def get_pred_af_by_LLM(uc_list, model_name,uc_ref_list,out_path_gpt):

    with open(out_path_gpt, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc,uc_ref in zip(uc_list,uc_ref_list):
            print(f"*** uc index: {uc['index']}, dataset: {uc['dataset']} ***")
            if uc['dataset'] != uc_ref['dataset'] and uc['id'] != uc_ref['id']:
                print(f"Dataset or ID mismatch: {uc['dataset']} vs {uc_ref['dataset']}, {uc['id']} vs {uc_ref['id']}")
                break
                

            if len(uc_ref['Alt. Flow']) != 0:
                # 使用LLM生成分支流
                if model_name == ERNIE_4_turbo:
                    uc_ref['pred_af'] = pred_af_ernie(uc,model_name)
                elif model_name == GPT_4o:
                    uc_ref['pred_af'] = pred_af_gpt(uc,model_name)
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")
                    
            json.dump(uc_ref, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行

    return uc_ref_list





def find_tp_ernie(pred_node,af_list,model_name):
    chat_comp = qianfan.ChatCompletion()

    user_pmt = "List1='" + str(af_list) + \
           ". Are there any words in List1 that are identical or semantically similar to '" + pred_node + "' ?" \
           "If there is, output only the first identical or semantically similar word; if there is not return 'False'. Do not output any other information" 
    # 输入llm
    resp = chat_comp.do(model=model_name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"]
    exist = exist.lower()  # 将可能出现的FALSE都变为小写
    exist = exist.strip()  # 去除两端空格
    exist = re.sub(r'^[\W_]+|[\W_]+$', '', exist)  # 去除两端特殊符号


    return exist

def get_tp_in_loop(pred_af_list,af_list,model_name):
    af_list = [s.lower() for s in af_list] # 将af_list中的元素转换为小写，便于后续比较
    tp_list = []
    max_count = 5
    count = 0
    # 提取 tp 
    for pred_node in pred_af_list:
        while True: # 循环直到找到tp/false，或达到最大次数
            tp = find_tp_ernie(pred_node.lower(), af_list, model_name)
            if tp in af_list or 'false' in tp.lower():  # 找到或者找不到
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
            print(f"false,and tp = {tp.lower()}")                
    return tp_list

def find_tp(uc_list,out_path,model_name,uc_alt_node_list):
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc,uc_alt_node in zip(uc_list,uc_alt_node_list):
            if uc['dataset'] != uc_alt_node['dataset'] or uc['id'] != uc_alt_node['id']:
                print(f"Dataset or ID mismatch: {uc['dataset']} vs {uc_alt_node['dataset']}, {uc['id']} vs {uc_alt_node['id']}")
            print(f"*** global index: {uc_list.index(uc)}/{len(uc_list)}, uc index: {uc['id']}, dataset: {uc['dataset']} ***")   

            if len(uc['Alt. Flow']) != 0:
                if model_name == ERNIE_4_turbo:  # 这里选择使用ernie统一进行tp选择
                    alt_node_list = flatten_list(uc_alt_node["AF act"])
                    uc['tp_act'] = get_tp_in_loop(uc['pred_af_act'], alt_node_list,model_name)
                    alt_node_list = flatten_list(uc_alt_node["AF obj"])
                    uc['tp_obj'] = get_tp_in_loop(uc['pred_af_obj'], alt_node_list,model_name)
                else:
                    print(f"Model {model_name} not recognized. Skipping prediction.")

            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

        json_file.write(']\n')  # 手动添加逗号和换行
    return uc_list



def mix_label_in_two_list(uc_list_1,uc_list_2,label1,label2):
    uc_list_new = []
    for uc_1, uc_2 in zip(uc_list_1,uc_list_2):
        if uc_1['id'] != uc_2['id'] and uc_1['dataset'] == uc_2['dataset']:
            print(f"Dataset or ID mismatch: {uc_1['dataset']} vs {uc_2['dataset']}, {uc_1['id']} vs {uc_2['id']}")

        for lable in [ label1, label2 ]:
            if lable != 'none':
                if lable not in uc_1 or len(uc_1[lable]) == 0:
                    if lable in uc_2 and len(uc_2[lable]) > 0:
                        uc_1[lable] = uc_2[lable]

        uc_list_new.append(uc_1)
    return uc_list_new


def count_lable_in_list(uc_list,label0,label1):
    count = 0
    for uc in uc_list:
        if label0 in uc:
            if label1 not in uc or len(uc[label1]) == 0:
                count += 1
    return count

def add_tp_value(list1,list2,list3,label_list):
    for uc1,uc2,uc3 in zip(list1,list2,list3):
        for label in label_list:
            if label in uc1:
                if label not in uc2 or label not in uc3:
                    print(f"uc id: {uc1['id']} has no {label} in uc2 or uc3")
                if len(uc1[label])== 0:
                    uc2[label] = []  # 因为_obj.json文件中本来就没有tp_act
                    if len(uc2[label]) > 0 or len(uc3[label]) > 0:
                        uc1[label] = max(uc2[label], uc3[label], key=len)

    return list1



if __name__ == '__main__':
    task_name = 'calculate_pub_metric_value_eval'
    print(f'*** task_name: {task_name} !!!!! ***')


    if task_name =='get_pred_af_by_LLM':
        # 用BFGen项目中预测时用的数据集，刚刚好
        origin_path = "E:/GitHub/ASSAM/data/2_dataset_origin_node/Ernie-4-Turbo/Ernie_pub_ground_truth.json"
        # 用作判断该用例是否有af
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json"
        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_af.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_pred_af.json"

        uc_list = read_uc_from_json(origin_path)
        uc_ref_list =read_uc_from_stand_json(GPT_integ_path)

        # 使用LLM（GPT-4o和ERNIE 4.0Turbo）进行分支流生成。一个uc结束后直接写入josn，防止出错后全部重来。
        uc_list = get_pred_af_by_LLM(uc_list, GPT_4o,uc_ref_list,out_path_gpt)
        

    if task_name == 'get_pred_af_node':
        # 紧接“get_pred_af_by_LLM”任务，从大模型预测的af，找出其中的node。
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_af.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_pred_af.json"

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_pred_node.json"

        for in_path, out_path in zip([in_path_ernie, out_path_ernie], [in_path_gpt, out_path_gpt]):

            uc_list = read_uc_from_stand_json(in_path_ernie)

            for uc in uc_list:
                if "pred_af" in uc:  # 稍微check一下
                    if len(uc['pred_af']) <=5 or uc['pred_af'] == 'none' or uc['pred_af'] == 'None':
                        print(f"uc id: {uc['id']} has no af")

            # 统一使用 ernie 4 turbo进行af的node提取
            uc_list = extract_pred_node_from_af(uc_list, ERNIE_4_turbo, out_path_ernie)


    if task_name =='find_tp_in_llm_pred_af':
        # 评估LLM预测的分支流.后面重新提取了obj
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_pred_node.json"
 
        af_node_ernie = "../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json"
        af_node_gpt = "../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json"

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node_withtp.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_pred_node_withtp.json"

        uc_list_ernie = read_uc_from_stand_json(in_path_ernie)
        uc_list_gpt = read_uc_from_stand_json(in_path_gpt)

        # pred node 当前都是字符串形式，需要转换为list
        uc_list_ernie = find_list_in_string(uc_list_ernie)
        uc_list_gpt = find_list_in_string(uc_list_gpt)

        # 需要找到 tp (true positive)，然后进行evaluate
        uc_alt_node_list_ernie = read_uc_from_stand_json(af_node_ernie)
        uc_list_ernie = find_tp(uc_list_ernie,out_path_ernie,ERNIE_4_turbo,uc_alt_node_list_ernie)

        uc_alt_node_list_gpt = read_uc_from_stand_json(af_node_gpt)
        uc_list_gpt = find_tp(uc_list_gpt,out_path_gpt,ERNIE_4_turbo,uc_alt_node_list_gpt)

    if task_name == "evaluate_llm_pred_af":
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_withtp.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_withtp.json"

        af_node_ernie = "../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json"
        af_node_gpt = "../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/gpt_pub_alt_node.json"

        uc_list_ernie = read_uc_from_stand_json(in_path_ernie)
        uc_list_gpt = read_uc_from_stand_json(in_path_gpt)

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_eval.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_eval.json"

        # in_path_ernie的数据中obj不对，所以重新生成了，在下面的文件中。==》但是这些最终只用在增加一些tp node上。
        old_ernie = "../0_Data/baseline_data/baseline_LLM/1st_error_no_lower/Ernie_pub_pred_node_withtp.json"
        old_gpt = "../0_Data/baseline_data/baseline_LLM/1st_error_no_lower/GPT_pub_pred_node_withtp.json"
        obj_path_ernie = "../0_Data/baseline_data/baseline_LLM/1st_error_no_lower/Ernie_pub_pred_node_withtp_obj.json"
        obj_path_gpt = "../0_Data/baseline_data/baseline_LLM/1st_error_no_lower/GPT_pub_pred_node_withtp_obj.json"

        uc_alt_node_list_ernie = read_uc_from_stand_json(af_node_ernie) # 这个文件里有alt node
        uc_list_ernie = mix_items(uc_list_ernie,uc_alt_node_list_ernie,['AF act','AF obj'])  # 第三个参数是混合字典时，需要保留的item
        print(f"融合前 uc_list_ernie tp 数量: {count_node(uc_list_ernie,'tp_act')}, {count_node(uc_list_ernie,'tp_obj')}")
        uc_list_ernie = add_tp_value(uc_list_ernie,read_uc_from_stand_json(obj_path_ernie),read_uc_from_stand_json(old_ernie),['tp_act','tp_obj']) # 有些tp node没出来，提取了三次，把三次的融合
        print(f"融合后 uc_list_ernie tp 数量: {count_node(uc_list_ernie,'tp_act')}, {count_node(uc_list_ernie,'tp_obj')}")
        uc_list_ernie = eval_Ernie_pred_ncet(uc_list_ernie)
        write_uc_to_stand_json(out_path_ernie,uc_list_ernie)


        uc_alt_node_list_gpt = read_uc_from_stand_json(af_node_gpt)
        uc_list_gpt = mix_items(uc_list_gpt,uc_alt_node_list_gpt,['AF act','AF obj'])
        print(f"融合前 uc_list_gpt tp 数量: {count_node(uc_list_gpt,'tp_act')}, {count_node(uc_list_gpt,'tp_obj')}")
        uc_list_gpt = add_tp_value(uc_list_gpt,read_uc_from_stand_json(obj_path_gpt),read_uc_from_stand_json(old_gpt),['tp_act','tp_obj'])
        print(f"融合后 uc_list_gpt tp 数量: {count_node(uc_list_gpt,'tp_act')}, {count_node(uc_list_gpt,'tp_obj')}")
        uc_list_gpt = eval_Ernie_pred_ncet(uc_list_gpt)
        write_uc_to_stand_json(out_path_gpt,uc_list_gpt)
        

    elif task_name == 'calculate_pub_metric_value':  # 新任务在calculate_pub_metric_value_eval，用于只计算eval数据集的指标。本任务计算所有uc的metric数据的指标。
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_eval.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_eval.json"

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub.json"

        print(f"ernie tp 数量: {count_node(read_uc_from_stand_json(in_path_ernie),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_ernie),'tp_obj')}")
        print(f"gpt tp 数量: {count_node(read_uc_from_stand_json(in_path_gpt),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_gpt),'tp_obj')}")


        # 计算所有pub数据集的每项平均
        for in_path, out_path in zip([in_path_ernie, in_path_gpt], [out_path_ernie, out_path_gpt]):
            use_case_list = read_uc_from_stand_json(in_path)
            
            calculate_all_dataset(use_case_list, out_path)

    elif task_name == 'calculate_pub_metric_value_eval':
        # 这个任务不同于calculate_pub_metric_value，因为发现baseline的数据也应该提取eval数据集的指标，而不是全部metric指标。
        # 因为训练模型时，只有eval数据集的指标才会被计算，其他的被作为train和test，不会被计算。
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/pub_dataset/Ernie_pub_eval.json"
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/pub_dataset/GPT_pub_eval.json"

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/pub_dataset/Ernie_pub_metric.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/pub_dataset/GPT_pub_metric.json"

        print(f"ernie tp 数量: {count_node(read_uc_from_stand_json(in_path_ernie),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_ernie),'tp_obj')}")
        print(f"gpt tp 数量: {count_node(read_uc_from_stand_json(in_path_gpt),'tp_act')}, {count_node(read_uc_from_stand_json(in_path_gpt),'tp_obj')}")


        # 计算所有pub数据集的每项平均
        for in_path, out_path in zip([in_path_ernie, in_path_gpt], [out_path_ernie, out_path_gpt]):
            use_case_list = read_uc_from_stand_json(in_path)
            
            calculate_all_dataset_eval(use_case_list, out_path,PUB_GROUPING_UC_20_1[33:]) # 只计算eval数据集的指标





    elif task_name == 'error_str_to_list_and_find_tp':
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_eval.json" #原本用2nd_error下的，但是下面mix_label_in_two_list跑完后，已经汇总
        in_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_eval.json"

        pred_node_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node.json"
        pred_node_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_pred_node.json"

        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_withtp.json"
        out_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_withtp.json"

        for in_path, pred_node_path,out_path in zip([in_path_ernie, in_path_gpt], [pred_node_path_ernie,pred_node_path_gpt],[out_path_ernie, out_path_gpt]):
            uc_list = read_uc_from_stand_json(in_path)
            print(f"修改错误前 uc_list tp 数量: {count_node(uc_list,'tp_act')}, {count_node(uc_list,'tp_obj')}")

            uc_pred_list = read_uc_from_stand_json(pred_node_path)

            for uc,uc_pred in zip(uc_list,uc_pred_list):
                for pred_node, tp_node, AF_node in zip(['pred_af_act', 'pred_af_obj'], ['tp_act', 'tp_obj'],['AF act', 'AF obj']):
                    if tp_node not in uc:  
                        continue # 本身没有分支流 直接下一个
                    if uc[tp_node] == 'none' or len(uc[tp_node]) == 0:  # tp act、obj不存在。需要判断pred node是否提出来了
                        if uc[pred_node] == 'none' or len(uc[pred_node]) == 0: # 如果是，则重新提取pred node
                            uc[pred_node] = convert_string_to_list(uc_pred[pred_node])  

                        # 其余情况：如果pred node提取出来；或只是没有提取出tp，则只进行重新提取tp即可。    
                            uc[tp_node] = get_tp_in_loop(uc[pred_node],uc[AF_node], ERNIE_4_turbo)  # 重新提取tp node
            
            print(f"修改错误后 uc_list tp 数量: {count_node(uc_list,'tp_act')}, {count_node(uc_list,'tp_obj')}")       
            write_uc_to_stand_json(out_path, uc_list)


    elif task_name == 'add_obj':  # 两次提取obj，都有没有提取出来的。最后提取一次。==>这是因为没有给单词同一小写。
        in_path_ernie_1 = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node_withtp_obj.json"
        in_path_ernie_2 = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node_withtp.json"

        print(f"withtp_obj.json无tp obj数量: {count_lable_in_list(read_uc_from_stand_json(in_path_ernie_1),'pred_af_obj','tp_obj')}")
        print(f"withtp.json无tp obj数量: {count_lable_in_list(read_uc_from_stand_json(in_path_ernie_2),'pred_af_obj','tp_obj')}")
        uc_list_ernie = mix_label_in_two_list(read_uc_from_stand_json(in_path_ernie_1),read_uc_from_stand_json(in_path_ernie_2),'pred_af_obj','tp_obj')
        print(f"目前无tp obj数量: {count_lable_in_list(uc_list_ernie,'pred_af_obj','tp_obj')}")

        # 给没有tp obj的uc提取一次
        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node_withtp_obj_final.json"
        
        for uc in uc_list_ernie:
            if 'pred_af_obj' in uc and len(uc['pred_af_obj']) == 0:
                uc['pred_af_obj'] = pred_af_node_ernie(uc, ERNIE_4_turbo, 'obj')
                uc['pred_af_obj'] = find_list_in_string(uc['pred_af_obj'])

        af_node_ernie = "../0_Data/4_alt_flow_data/2_get_alt_node_after_SIP/ERNIE_pub_alt_node.json"
        for uc,uc_ref in zip(uc_list_ernie, read_uc_from_stand_json(af_node_ernie)):
            if len(uc['tp_obj']) == 0:
                alt_node_list = flatten_list(uc_ref["AF obj"])
                uc['tp_obj'] = get_tp_in_loop(uc['pred_af_obj'], alt_node_list,model_name=ERNIE_4_turbo)


        print(f"withtp_obj.json无tp obj数量: {count_lable_in_list(uc_list_ernie, 'pred_af_obj','tp_obj')}")
                                 
    elif task_name == 'check_new_tp':
        new_path_ernie = "../0_Data/baseline_data/baseline_LLM/Ernie_pub_pred_node_withtp.json"
        new_path_gpt = "../0_Data/baseline_data/baseline_LLM/GPT_pub_pred_node_withtp.json"

        old_path_ernie = "../0_Data/baseline_data/baseline_LLM/no_lower/Ernie_pub_pred_node_withtp.json"
        old_path_gpt = "../0_Data/baseline_data/baseline_LLM/no_lower/GPT_pub_pred_node_withtp.json"

        new_tp_node_count = 0
        old_tp_node_count = 0

        for new_uc, old_uc in zip(read_uc_from_stand_json(new_path_gpt), read_uc_from_stand_json(old_path_gpt)):
            if 'tp_act' in new_uc:
                new_tp_node_count = new_tp_node_count + len(new_uc['tp_act']) + len(new_uc['tp_obj'])
                old_tp_node_count = old_tp_node_count + len(old_uc['tp_act']) + len(old_uc['tp_obj'])

        print(f"new tp node count: {new_tp_node_count}, old tp node count: {old_tp_node_count}")

    elif task_name == 'mix_label_in_two_list':
        in_path_ernie = "../0_Data/baseline_data/baseline_LLM/2nd_error_str_to_list/GPT_pub_eval.json"
        ref_path_ernie = "../0_Data/baseline_data/baseline_LLM/1st_error_no_lower/GPT_pub_pred_node_withtp_obj.json"
        ref_path_ernie_1 = "../0_Data/baseline_data/baseline_LLM/1st_error_no_lower/GPT_pub_pred_node_withtp.json"
        out_path_ernie = "../0_Data/baseline_data/baseline_LLM/GPT_pub_eval.json"


        uc_list = read_uc_from_stand_json(in_path_ernie)
        print(f"修改错误前 uc_list tp 数量: {count_node(uc_list,'tp_act')}, {count_node(uc_list,'tp_obj')}")
        for uc,uc_ref in zip(uc_list,read_uc_from_stand_json(ref_path_ernie)):
            if uc['dataset'] != uc_ref['dataset'] or uc['id'] != uc_ref['id']:
                print(f"Dataset or ID mismatch: {uc['dataset']} vs {uc_ref['dataset']}, {uc['id']} vs {uc_ref['id']}")
            for label in ['tp_act', 'tp_obj']:
                if label in uc and len(uc[label]) == 0:
                    if label in uc_ref and len(uc_ref[label]) > 0:
                        uc[label] = uc_ref[label]
                        uc['tp'] = [] # 清零
                        uc['p'] ,uc['r'], uc['f1'], uc['auc'] = 0, 0, 0, 0 # 清零
        print(f"修改错误前 uc_list tp 数量: {count_node(uc_list,'tp_act')}, {count_node(uc_list,'tp_obj')}")

        for uc,uc_ref in zip(uc_list,read_uc_from_stand_json(ref_path_ernie_1)):
            if uc['dataset'] != uc_ref['dataset'] or uc['id'] != uc_ref['id']:
                print(f"Dataset or ID mismatch: {uc['dataset']} vs {uc_ref['dataset']}, {uc['id']} vs {uc_ref['id']}")
            for label in ['tp_act', 'tp_obj']:
                if label in uc and len(uc[label]) == 0:
                    if label in uc_ref and len(uc_ref[label]) > 0:
                        uc[label] = uc_ref[label]
                        uc['tp'] = [] # 清零
                        uc['p'] ,uc['r'], uc['f1'], uc['auc'] = 0, 0, 0, 0 # 清零
        
        
        print(f"修改错误前 uc_list tp 数量: {count_node(uc_list,'tp_act')}, {count_node(uc_list,'tp_obj')}")
        
        write_uc_to_stand_json(out_path_ernie, uc_list)
        



    print(f'*** task_name: {task_name} Finish! *** ')