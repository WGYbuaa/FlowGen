# SIP模块

from turtle import st
from imports import GPT_35, re, GPT_4o, ERNIE_4_turbo, ERNIE_ernie35, qianfan, REPLACEMENT_MAP,ast,json,requests
from utils import read_uc_from_stand_json

def check_missing_comma(json_str):
    try:
        json.loads(json_str)  # 尝试解析
        return False  # 无错误
    except json.JSONDecodeError as e:
        if "Expecting ',' delimiter" in e.msg:  # 明确捕获逗号缺失错误
            return True
        return False  # 其他错误


def determine_sent_complex(step, Model_Name):
    if 'ERNIE' in Model_Name:
        return determine_sent_complex_ernie(step, Model_Name)
    elif 'gpt' in Model_Name:
        return determine_sent_complex_gpt(step, Model_Name)
    


def determine_sent_complex_ernie(step, Model_Name):
    chat_comp = qianfan.ChatCompletion()

    # ERNIE 模型不能有 sys_pmt
    user_pmt = "Sentence1:'" + step + "'.Is Sentence1 a simple sentence? Return 'true' or 'false'."

    # 输入llm
    resp = chat_comp.do(model=Model_Name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"]


    if "true" in exist:
        return True
    elif "false" in exist:
        return False
    else:
        return None

def determine_sent_complex_gpt(step, Model_Name):
    url = "https://api.gptsapi.net/v1/chat/completions"

    sys_pmt = "Is the following sentence a simple sentence? Return 'true' or 'false'."
    user_pmt = step

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-NQe10708c5d9ccb970298750dd3b6def3e706feea58rkx7b"  # 请确保将此处的API密钥替换为实际的密钥
    }

    # 定义请求数据
    data = {
        "model": Model_Name,
        "messages": [
            {"role": "system", "content": sys_pmt},
            {"role": "user", "content": user_pmt}
        ]
    }

    # 发送POST请求
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # 处理响应
    if response.status_code == 200:
        # 如果请求成功，解析JSON响应
        result = response.json()
        exist = result["choices"][0]["message"]["content"]

        if "true" in exist:
            return True
        elif "false" in exist:
            return False
        else:
            return None




def sip_by_ernie(step, Model_Name):
    chat_comp = qianfan.ChatCompletion()

    # ERNIE 模型不能有 sys_pmt
    user_pmt = "str1:'" + step + "'.Split str1 into multiple sentences,surround with double quotes,output as a **List**,with one verb and one object per sentence."

    # 输入llm
    resp = chat_comp.do(model=Model_Name, messages=[
        {
            "role": "user",
            "content": user_pmt
        }
    ], disable_search=True)

    exist = resp["body"]["result"]
    exist = exist.translate(str.maketrans(REPLACEMENT_MAP))  # 将可能有的中文符号换成英文

    # 使用正则表达式匹配方括号及其中的内容
    match = re.findall(r'\[(.*?)\]', exist)
    # 如果找到了匹配项
    if match:
        last_match = match[-1]
        content = f"[{last_match}]"
        print(f"content: {content}")
        if not check_missing_comma(content):  # 没有缺少逗号
            list1 = ast.literal_eval(content)
            if isinstance(list1, list):
                return list1
    return None


def sip_by_gpt(step, Model_Name):
    url = "https://api.gptsapi.net/v1/chat/completions"

    sys_pmt = "Split the following sentence into multiple simple sentences(one verb,one object each),surround with double quotes,and output as a **Python List Object**."
    user_pmt = step

    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-NQe10708c5d9ccb970298750dd3b6def3e706feea58rkx7b"  # 请确保将此处的API密钥替换为实际的密钥
    }

    # 定义请求数据
    data = {
        "model": Model_Name,
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

        # 使用正则表达式匹配方括号及其中的内容
        match = re.findall(r'\[(.*?)\]', exist)
        match_1 = re.search(r'\[(.*?)\]', exist)
        # 如果找到了匹配项
        if match:
            last_match = match[-1]
            content = f"[{last_match}]"
            print(f"content: {content}")
            try:
                list1 = ast.literal_eval(content)
            except ValueError:
                return None
            except KeyError:
                return None
            except SyntaxError:
                return None
            if isinstance(list1, list):
                return list1
        elif match_1:
            # 提取匹配项中的内容，并去掉首尾的空格
            content = match.group(0).strip()
            print(f"content: {content}")
            try:
                list1 = ast.literal_eval(content)
            except ValueError:
                return None
            except KeyError:
                return None
            except SyntaxError:
                return None
            if isinstance(list1, list):
                return list1
    return None



def semantic_info_proc(alt_list, Model_Name):
    max_count = 5 # 最大调用次数
    alt_list_new = []
    for alt in alt_list:  # 二级列表
        alt_new = []
        for step in alt:  # 每个句子
            print(f' step : {step} ')
            step_clean = re.sub(r'[^\w]', '', step)
            if step_clean == 'None':
                alt_new.append([step])
            else:
                if isinstance(step, str):
                    if determine_sent_complex(step, Model_Name): #如果本身就是简单句,则不操作. 只有返回True才是。
                         alt_new.append([step])
                         continue
                    count = 0
                    while True: # 无限循环，直至输出false或者正确内容
                        if 'ERNIE' in Model_Name:
                            step_new = sip_by_ernie(step, Model_Name)
                        elif 'gpt' in Model_Name:
                            step_new = sip_by_gpt(step, Model_Name)

                        if step_new and isinstance(step_new, list):
                            break
                        count += 1
                        if count >= max_count:
                            print(f'step: {step} **循环已达到最大次数，退出循环')
                            alt_new.append([step])  # 实在没法做就算了
                            break
                    
                    if step_new and isinstance(step_new, list):  # SIP成功结束
                        alt_new.append(step_new)
                     
                else: # 二级列表中还不是个字符串
                    print(f'step:{step}  ** 二级列表中还不是个字符串')
        
        alt_list_new.append(alt_new)

    return alt_list_new


def get_and_write_uc_after_sip(uc_list, Model_Name):

    # 返回的内容直接写，防止出错后全部消失
    with open(out_path, 'a', encoding='utf-8') as json_file:
        json_file.write('[\n')  # 手动添加逗号和换行
        for uc in uc_list:
            print(f"*** uc id: {uc['id']}, dataset: {uc['dataset']} ***")

            if len(uc['Alt. Flow']) != 0:
                # 将 alt flow 进行 断句 + 形式化：保持一个step包含一个act+一个obj；去除多余修饰
                uc['Alt. Flow'] = semantic_info_proc(uc['Alt. Flow'], Model_Name)

            json.dump(uc, json_file, ensure_ascii=False, indent=4)
            json_file.write(',\n')  # 手动添加逗号和换行

            

        json_file.write(']\n')  # 手动添加逗号和换行
        


    return uc_list



if __name__ == '__main__':
    task_name = 'SIP_for_Alt_flow_GPT'

    # (Ernie, pub) 进行alt flow 的SIP（断句+形式化）
    if task_name =="SIP_for_Alt_flow_Ernie":
        Ernie_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/Ernie_pub_integrated.json"
        out_path = "../0_Data/4_alt_flow_data/1_after_SIP/Ernie_pub_SIP.json"

        uc_list = read_uc_from_stand_json(Ernie_integ_path)

        uc_list = get_and_write_uc_after_sip(uc_list, ERNIE_4_turbo)

    if task_name == 'chech_sip':  # 检查SIP_for_Alt_flow_GPT和SIP_for_Alt_flow_Ernie的结果
        uc_list_sip = read_uc_from_stand_json("../0_Data/4_alt_flow_data/1_after_SIP/GPT_pub_SIP.json")
        uc_list = read_uc_from_stand_json("../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json")

        for uc_sip,uc in zip(uc_list_sip,uc_list):
            # 判断除alt flow之外的其他item
            for key in uc.keys():
                if key != "Alt. Flow":
                    if uc_sip[key] != uc[key]:
                        print(f'uc id: {uc_sip["id"]}  {key} 不一致')

            # 判断 alt flow
            if len(uc_sip['Alt. Flow']) != len(uc['Alt. Flow']):
                print(f'uc id: {uc_sip["id"]}  alt flow 数量不一致')
            for alt_sip,alt in zip(uc_sip['Alt. Flow'], uc['Alt. Flow']):
                if len(alt_sip) != len(alt):
                    print(f'uc alt: {uc_sip["id"]}  alt flow 数量不一致')
        print(f'chech_sip_ernie 任务全部正确！')
     
    # (GPT, pub) 进行alt flow 的SIP（断句+形式化）
    if task_name =="SIP_for_Alt_flow_GPT":
        GPT_integ_path = "../0_Data/4_alt_flow_data/0_raw_data/GPT_pub_integrated.json"
        out_path = '../0_Data/4_alt_flow_data/1_after_SIP/GPT_pub_SIP.json'

        uc_list = read_uc_from_stand_json(GPT_integ_path)

        uc_list = get_and_write_uc_after_sip(uc_list, GPT_4o)


    print(f"*** task_name: {task_name} Finish! *** ")
