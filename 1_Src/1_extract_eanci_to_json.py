from imports import os, re, json, deepl, deepl_auth_key

def extract_use_cases_from_eanci(directory_path):
    """
    从eANCI文件夹中提取所有用例信息
    
    Args:
        directory_path: eANCI文件夹路径
        
    Returns:
        包含所有用例信息的列表，每个用例是一个字典
    """
    use_case_list = []
    
    try:
        # 获取文件夹中所有的txt文件
        file_names = [f for f in os.listdir(directory_path) if f.endswith('.txt') and f.startswith('EA')]
        # 按EA编号排序
        file_names.sort(key=lambda x: int(re.search(r'EA(\d+)', x).group(1)) if re.search(r'EA(\d+)', x) else float('inf'))
        
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(directory_path, file_name)
            
            print(f"正在处理文件 {file_name}...")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取用例名称
            name_pattern = re.compile(r'Nome caso d\'uso\s*\n(.+?)\s*\n', re.DOTALL)
            name_match = name_pattern.search(content)
            name = name_match.group(1).strip() if name_match else ""
            
            # 提取参与者
            actors_pattern = re.compile(r'Attori partecipanti\s*\n(.+?)\s*\n', re.DOTALL)
            actors_match = actors_pattern.search(content)
            actors = actors_match.group(1).strip() if actors_match else ""
            
            # 提取事件流
            flow_pattern = re.compile(r'Flusso di eventi\s*\n(.+?)\s*\nCondizione di entrata', re.DOTALL)
            flow_match = flow_pattern.search(content)
            flow_text = flow_match.group(1).strip() if flow_match else ""
            
            # 处理事件流列表
            flow_events = []
            for line in flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.?\s*', '', line)
                    if cleaned_line:
                        flow_events.append(cleaned_line)
            
            # 提取入口条件
            entry_conditions_pattern = re.compile(r'Condizione di entrata\s*\n(.+?)\s*\nCondizioni di uscita', re.DOTALL)
            entry_conditions_match = entry_conditions_pattern.search(content)
            entry_conditions_text = entry_conditions_match.group(1).strip() if entry_conditions_match else ""
            
            # 处理入口条件列表
            entry_conditions = []
            for line in entry_conditions_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除项目符号（如果有）
                    cleaned_line = re.sub(r'^[•➢\-\*]\s*', '', line)
                    if cleaned_line:
                        entry_conditions.append(cleaned_line)
            
            # 提取出口条件
            exit_conditions_pattern = re.compile(r'Condizioni di uscita\s*\n(.+?)(?=\nRequisiti di qualita|$)', re.DOTALL)
            exit_conditions_match = exit_conditions_pattern.search(content)
            exit_conditions_text = exit_conditions_match.group(1).strip() if exit_conditions_match else ""
            
            # 处理出口条件列表
            exit_conditions = []
            for line in exit_conditions_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除项目符号（如果有）
                    cleaned_line = re.sub(r'^[•➢\-\*]\s*', '', line)
                    if cleaned_line:
                        exit_conditions.append(cleaned_line)
            
            # 提取质量要求（如果有）
            quality_requirements_pattern = re.compile(r'Requisiti di qualita(.+?)$', re.DOTALL)
            quality_requirements_match = quality_requirements_pattern.search(content)
            quality_requirements_text = quality_requirements_match.group(1).strip() if quality_requirements_match else ""
            
            # 处理质量要求列表
            quality_requirements = []
            for line in quality_requirements_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除项目符号（如果有）
                    cleaned_line = re.sub(r'^[•➢\-\*]\s*', '', line)
                    if cleaned_line and cleaned_line != "Non previsti" and cleaned_line != "Non previsti.":
                        quality_requirements.append(cleaned_line)
            
            # 创建用例字典
            use_case = {
                "id": i,
                "dataset": "eANCI",
                "Nome caso d'uso": name,
                "Attori partecipanti": actors,
                "Flusso di eventi": flow_events,
                "Condizione di entrata": entry_conditions,
                "Condizioni di uscita": exit_conditions,
                "Requisiti di qualita": quality_requirements
            }
            
            use_case_list.append(use_case)
        
        return use_case_list
    
    except Exception as e:
        print(f"提取用例时出错：{str(e)}")
        return []

def translate_and_save_eanci_use_cases(input_use_cases, output_json_path, auth_key):
    """
    将eANCI用例翻译成英文并保存为JSON
    
    Args:
        input_use_cases: 提取的用例列表
        output_json_path: 输出JSON文件路径
        auth_key: DeepL API密钥
    
    Returns:
        翻译后的用例列表
    """
    translator = deepl.DeepLClient(auth_key)
    translated_use_cases = []
    
    try:
        total_cases = len(input_use_cases)
        print(f"开始翻译 {total_cases} 个用例...")
        
        for i, use_case in enumerate(input_use_cases):
            print(f"正在翻译用例 {i+1}/{total_cases}: ")
            
            # 翻译名称
            translated_name = translator.translate_text(use_case['Nome caso d\'uso'], target_lang="EN-US", source_lang="IT").text
            
            # 翻译参与者
            translated_actors = translator.translate_text(use_case['Attori partecipanti'], target_lang="EN-US", source_lang="IT").text if use_case['Attori partecipanti'] else ""
            
            # 翻译事件流
            translated_flow_events = []
            for event in use_case['Flusso di eventi']:
                if event:
                    translated_event = translator.translate_text(event, target_lang="EN-US", source_lang="IT").text
                    translated_flow_events.append(translated_event)
            
            # 翻译入口条件
            translated_entry_conditions = []
            for condition in use_case['Condizione di entrata']:
                if condition:
                    translated_condition = translator.translate_text(condition, target_lang="EN-US", source_lang="IT").text
                    translated_entry_conditions.append(translated_condition)
            
            # 翻译出口条件
            translated_exit_conditions = []
            for condition in use_case['Condizioni di uscita']:
                if condition:
                    translated_condition = translator.translate_text(condition, target_lang="EN-US", source_lang="IT").text
                    translated_exit_conditions.append(translated_condition)
            
            # 翻译质量要求
            translated_quality_requirements = []
            for requirement in use_case['Requisiti di qualita']:
                if requirement:
                    translated_requirement = translator.translate_text(requirement, target_lang="EN-US", source_lang="IT").text
                    translated_quality_requirements.append(translated_requirement)
            
            # 创建翻译后的用例字典
            translated_use_case = {
                "id": use_case['id'],
                "dataset": "eANCI",
                "Use Case Name": translated_name,
                "Participating Actors": translated_actors,
                "Flow of Events": translated_flow_events,
                "Entry Condition": translated_entry_conditions,
                "Exit Conditions": translated_exit_conditions,
                "Quality Requirements": translated_quality_requirements,
                "Original": {
                    "Nome caso d'uso": use_case['Nome caso d\'uso'],
                    "Attori partecipanti": use_case['Attori partecipanti'],
                    "Flusso di eventi": use_case['Flusso di eventi'],
                    "Condizione di entrata": use_case['Condizione di entrata'],
                    "Condizioni di uscita": use_case['Condizioni di uscita'],
                    "Requisiti di qualita": use_case['Requisiti di qualita']
                }
            }
            
            translated_use_cases.append(translated_use_case)
            
            # 每处理10个用例，保存一次中间结果
            if (i + 1) % 10 == 0:
                with open(output_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(translated_use_cases, json_file, ensure_ascii=False, indent=4)
                print(f"已保存前 {i+1} 个用例的翻译结果")
        
        # 保存最终结果
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(translated_use_cases, json_file, ensure_ascii=False, indent=4)
        
        print(f"翻译完成，结果已保存到 {output_json_path}")
        return translated_use_cases
    
    except Exception as e:
        print(f"翻译用例时出错：{str(e)}")
        # 如果出错，尝试保存已翻译的部分
        if translated_use_cases:
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(translated_use_cases, json_file, ensure_ascii=False, indent=4)
            print(f"已保存部分翻译结果到 {output_json_path}")
        return translated_use_cases

# 主函数
def main():
    # 设置路径
    eanci_dir = "e:/Trae_project/ConditionOfUCS/0_Data/0_origin_data/eANCI"
    output_json_path = "e:/Trae_project/ConditionOfUCS/0_Data/2_json_dataset/eANCI_en.json"
    
    # 提取用例
    print("开始提取eANCI用例...")
    use_cases = extract_use_cases_from_eanci(eanci_dir)
    print(f"共提取到 {len(use_cases)} 个用例")
    
    # 翻译并保存
    print("开始翻译用例...")
    translated_use_cases = translate_and_save_eanci_use_cases(use_cases, output_json_path, deepl_auth_key)
    print(f"共翻译了 {len(translated_use_cases)} 个用例")

if __name__ == "__main__":
    main()