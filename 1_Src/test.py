from collections import Counter


# 首先需要得到一个uc中，uc- keyword id - bf id - af id (af id 个数/2即为step数目)


# 文件结构：[sub_graph1,sub_graph2...], pub数据集一共36个sub_graph，每个sub_graph其中按照用例存放sub_graph1 = [uc1,uc2...],用例数目不定
# 每个用例为一个字典。第一个item中key为该列表的id，value为该列表所有keyword的id，为一级列表。
# 第二个、第三个为bf act和bf obj的列表，两个内容为各自node_id,但是结构完全一致；
# 第四个、第五个为af act和af obj的列表，两个内容为各自node_id,但是结构完全一致。

# "一个用例"= {4655: [keyword_id,keyword_id,keyword_id], 
#             "BF act": [[bf step1],[bf step2]...],  # 列表长度即为bf的step个数。由于经过断句，原本的一句被断为多句，所以一句包含多个act，所以 bf step = [step1中的act_id]
#             "BF obj": [[bf step1],[bf step2]...],  
#             "AF act": [bf step1 , bf step2...], # 列表长度也是bf的step长度，这样可以直接对应哪个bf step有分支流，如果没有，则该位置上len为0。
#             "AF obj": [bf step1 , bf step2...]  # 但是由于存在一条bf 分支出多个分支流的情况，所以bf step1 = [af1,af2..]. 
#             }                                   # 由于af也经过了断句，所以af1 = [[af step1],[af step2]..]

# "用例"= {"uc_id:4655": [
#                         [keyword_id,keyword_id,keyword_id],   # keyword列表
                        
#                         [
#                             [bf_act_1,bf_act_2],  # bf 列表
#                             [bf_obj_1,bf_obj_2]
#                         ],
#                         [
#                             [                       # 与bf列表长度一致
#                                 [af_act_id,af_act_id],  # bf_act_1的af act 列表
#                                 [af_act_id,af_act_id]   # bf_act_2的af act 列表
#                             ], 
#                             [
#                                 [af_obj_id,af_obj_id], # bf_act_1的af obj 列表
#                                 [af_obj_id,af_obj_id]  # bf_act_2的af obj 列表
#                             ]
                        
#                         ]
#                         ]
#              }

from imports import GROUPING_UC_20,importlib,PUB_GROUPING_UC_20_1

in_path_gpt_wo_bp = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_af_given_bp.json"
in_path_bp ="../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_af_given_bp.json"
bp_path = "../0_Data/5_branching_point/2_ncet_bp/NCET_with_bp.json"
from utils import read_uc_from_stand_json,get_test_sub_graph
spec = importlib.util.spec_from_file_location("module_name", "7_make_pt_file.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)



in_path_gpt = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/GPT_ncet_pred_bf_af.json"
in_path_Ernie = "../0_Data/baseline_data/baseline_LLM_with_bp/ncet/Ernie_ncet_pred_bf_af.json"
branching_point_path = '../0_Data/5_branching_point/2_ncet_bp/NCET_with_bp.json'  

uc_af_info= get_test_sub_graph(module.local_id_to_global_id(read_uc_from_stand_json(branching_point_path)),GROUPING_UC_20[-32:]) 
ul1 = read_uc_from_stand_json(in_path_gpt)
ul2= read_uc_from_stand_json(in_path_Ernie)

pred1,pred2,gt1,len1,len2 = [],[],[],0,0
for sub,sub2,sub_gt in zip(ul1,ul2,uc_af_info):
    for uc,uc2,uc_gt in zip(sub,sub2,sub_gt):
        len1+= len(uc['Basic flow'])
        len2+= len(uc2['Basic flow'])
        pred_,pred_1,ground_truth= [0] * len(uc['Basic flow']),[0] * len(uc['Basic flow']),[0] * len(uc['Basic flow'])
        
        if isinstance(uc['pred_af'],dict):
            for bp_local in uc['pred_af'].keys():
                pred_[int(bp_local)] = 1
                
        if isinstance(uc2['pred_af'],dict):
            for bp_local1 in uc2['pred_af'].keys():
                pred_1[int(bp_local1)] = 1

        if len(uc_gt['Exc. Flow'])>0:  # =0的情况则全部为负例
            for items in uc_gt['Exc. Flow']:
                for bp in items.keys():
                    bp_local = bp.split('_')[0]  # 获得分支点
                    if int(bp_local) < len(ground_truth):
                        ground_truth[int(bp_local)] = 1
        
        pred1 += pred_
        pred2 += pred_1
        gt1 += ground_truth
# print(pred1)
# print(pred2)
# print(gt1)


uc_ernie = read_uc_from_stand_json('../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json')
uc_gpt  =read_uc_from_stand_json('../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json')

for i in range(len(uc_ernie)):
    if "AF act" in uc_ernie[i]:
        for al1,al2 in zip(uc_ernie[i]['AF act'],uc_gpt[i]['AF act']):
            if len(al1) != len(al2) and len(al1)!=2:
                print(f"用例 {i} 的 AF act 长度不匹配: Ernie 长度为 {len(al1)}, GPT 长度为 {len(al2)}")

        for al1,al2 in zip(uc_ernie[i]['AF obj'],uc_gpt[i]['AF obj']):
            if len(al1) != len(al2) and len(al1)!=2:
                print(f"用例 {i} 的 AF obj 长度不匹配: Ernie 长度为 {len(al1)}, GPT 长度为 {len(al2)}")

print("严格一行一个列表已保存至 'nested_list_one_line_per_list.txt'")

