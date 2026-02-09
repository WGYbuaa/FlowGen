from imports import datetime
from utils import read_uc_from_stand_json



if __name__ == '__main__':
    task_name = 'statistic_node_num'
    print(f'*** task_name: {task_name} , Starting time: {datetime.now()}  !!!!! ***')


    if task_name =='statistic_alt_num':
        ncet_ernie_path = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_NCET_integrated.json' # 按理说无论gpt还是ernie，统计数据集都应该是一样的
        pub_ernie_path = '../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json'
        pub_gpt_path = '../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json'
                           

        for in_path in [pub_ernie_path,pub_gpt_path]:
            uc_list = read_uc_from_stand_json(in_path)
            alt_num_all, alt_num, step_num,step_all_num,unique_step_num,unique_step_list,dataset = 0,0,0,0,0,[],uc_list[0]['dataset']
            duplicate = 0
            for uc in uc_list:
                if  dataset != uc['dataset']:  # 不同数据集中
                    print(f'{dataset} alt_num: {alt_num}, step_all_num: {step_num}, unique_step_num: {len(unique_step_list)}')
                    print(f'duplicate step num: {duplicate}')
                    step_all_num += step_num
                    alt_num_all += alt_num
                    unique_step_num += len(unique_step_list)
                    duplicate, alt_num, step_num,unique_step_list,dataset = 0,0,0,[],uc['dataset']

                alt_num += len(uc['Alt. Flow'])  # 分支流的个数（一个uc可能包含多条分支流）
                for alt_flow in uc['Alt. Flow']:
                    step_num += len(alt_flow)  # 每条分支流的步骤数
                    if len(alt_flow) > 0:
                        for step in alt_flow:
                            if step not in unique_step_list:
                                unique_step_list.append(step) # 不重复的步骤数目
                            else:
                                duplicate +=1

            print(f'{dataset} alt_num: {alt_num}, step_all_num: {step_num}, unique_step_num: {unique_step_num}') # 最后一个
            print(f'duplicate step num: {duplicate}')

            step_all_num += step_num
            alt_num_all += alt_num
            unique_step_num += len(unique_step_list)
            print(f'all alt num:{alt_num_all}; ALL alt step num:{step_all_num}; unique step num:{unique_step_num}')

                    

        for in_path in [ncet_ernie_path]:
            uc_list = read_uc_from_stand_json(in_path)
            alt_num, step_num,unique_step_num,unique_step_list,dataset = 0,0,0,[],uc_list[0]['key_path'][0]
            duplicate,step_all_num,alt_num_all = 0,0,0

            for uc in uc_list:
                if dataset != uc['key_path'][0]:  # 不同数据集中
                    print(f'{dataset} alt_num: {alt_num}, step_all_num: {step_num}, unique_step_num: {len(unique_step_list)}')
                    print(f'duplicate step num: {duplicate}')
                    step_all_num += step_num
                    alt_num_all += alt_num
                    unique_step_num += len(unique_step_list)
                    duplicate, alt_num, step_num,unique_step_list,dataset = 0,0,0,[],uc['key_path'][0]


                alt_num += len(uc['Exc. Flow'])
                if len(uc['Exc. Flow'])>0:
                    for exc_flow in uc['Exc. Flow']:
                        step_num += len(exc_flow)
                        for step in exc_flow:
                            if step not in unique_step_list:
                                unique_step_list.append(step)
                            else:
                                duplicate +=1
            print(f'{dataset} alt_num: {alt_num}, step_all_num: {step_num}, unique_step_num: {len(unique_step_list)}')
            print(f'duplicate step num: {duplicate}')

            step_all_num += step_num
            alt_num_all += alt_num
            unique_step_num += len(unique_step_list)
            print(f'all alt num:{alt_num_all}; ALL alt step num:{step_all_num}; unique step num:{unique_step_num}')

                    

    if task_name =='statistic_node_num':
        path_dict= {
            "stanfard_pub":'../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/Stanford_pub_alt_node.json',
            "gpt_pub":'../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/gpt_pub_alt_node_woSIP.json',
            "ERNIE_pub":'../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_pub_alt_node_woSIP.json',
            "stanfard_ncet":'../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/Stanford_NCET_alt_node.json',
            'ERNIE_ncet': '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json',
            'gpt_ncet': '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'           
        }

        for tool_name,in_path in path_dict.items():
            uc_list = read_uc_from_stand_json(in_path)
            act_node,obj_node = [],[]

            for uc in uc_list:
                if ('Alt. Flow' in uc and len(uc['Alt. Flow'])>0) or ('Exc. Flow' in uc and len(uc['Exc. Flow'])>0):
                    for step in uc['AF act']:
                        for node in step:
                            if node not in act_node:
                                act_node.append(node)
                    for step in uc['AF obj']:
                        for node in step:
                            if node not in obj_node:
                                obj_node.append(node)

            print(f'{tool_name} : act node num: {len(act_node)}; obj node num: {len(obj_node)}')
            print(f'Obj nodes: {obj_node}')
                          
    if task_name =='check':
        ERNIE_ncet ='../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/ERNIE_NCET_alt_node_1.json'
        gpt_ncet = '../0_Data/4_alt_flow_data/2_get_alt_node_wo_SIP/GPT_NCET_alt_node_1.json'  
        ernie_node_list,gpt_node_list = [],[]
        for uc in read_uc_from_stand_json(gpt_ncet):
            if 'AF obj' in uc and len(uc['AF obj']) >0:
                for step in uc['AF obj']:
                    for node in step:
                        if node not in gpt_node_list:
                            gpt_node_list.append(node)
        for uc in read_uc_from_stand_json(ERNIE_ncet):
            if 'AF obj' in uc and len(uc['AF obj']) >0:
                for step in uc['AF obj']:
                    for node in step:
                        if node not in ernie_node_list:
                            ernie_node_list.append(node)

        set_ernie = set(ernie_node_list)
        list_a_filtered = [item for item in gpt_node_list if item not in set_ernie]
        print(f'{list_a_filtered}')

        set_gpt = set(gpt_node_list)
        list_a_filtered = [item for item in ernie_node_list if item not in set_gpt]
        print(f'{list_a_filtered}')

    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')