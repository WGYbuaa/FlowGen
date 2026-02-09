from imports import os, re, json, deepl, deepl_auth_key

def extract_remaining_smos_use_cases(directory_path, start_id=62):
    """
    从SMOS文件夹中提取从指定ID开始的用例信息
    
    Args:
        directory_path: SMOS文件夹路径
        start_id: 开始提取的用例ID
        
    Returns:
        包含提取用例信息的列表，每个用例是一个字典
    """
    use_case_list = []
    
    try:
        # 获取文件夹中所有的txt文件
        file_names = [f for f in os.listdir(directory_path) if f.endswith('.txt') and f.startswith('SMOS')]
        # 按SMOS编号排序
        file_names.sort(key=lambda x: int(re.search(r'SMOS(\d+)', x).group(1)) if re.search(r'SMOS(\d+)', x) else float('inf'))
        
        # 只处理从start_id开始的文件
        file_names = [f for f in file_names if int(re.search(r'SMOS(\d+)', f).group(1)) >= start_id]
        
        for i, file_name in enumerate(file_names):
            file_id = int(re.search(r'SMOS(\d+)', file_name).group(1))
            file_path = os.path.join(directory_path, file_name)
            
            print(f"正在处理文件 {file_name}...")
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取用例名称
            name_pattern = re.compile(r'Nome?:\s*(.+?)(?:\n|$)', re.DOTALL)
            name_match = name_pattern.search(content)
            name = name_match.group(1).strip() if name_match else ""
            
            # 提取参与者
            actors_pattern = re.compile(r'Attori:\s*(.+?)(?:\n|$)', re.DOTALL)
            actors_match = actors_pattern.search(content)
            actors = actors_match.group(1).strip() if actors_match else ""
            
            # 提取描述
            description_pattern = re.compile(r'Descrizione:\s*(.+?)(?:\n|$)', re.DOTALL)
            description_match = description_pattern.search(content)
            description = description_match.group(1).strip() if description_match else ""
            
            # 提取前置条件
            preconditions_pattern = re.compile(r'Precondizioni:\s*(.+?)(?=Sequenza degli eventi|$)', re.DOTALL)
            preconditions_match = preconditions_pattern.search(content)
            preconditions_text = preconditions_match.group(1).strip() if preconditions_match else ""
            
            # 处理前置条件列表
            preconditions = []
            for line in preconditions_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除项目符号（如果有）
                    cleaned_line = re.sub(r'^[•\-\*]\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # 提取事件序列
            sequence_pattern = re.compile(r'Sequenza degli eventi.*?(?=Postcondizioni:|$)', re.DOTALL)
            sequence_match = sequence_pattern.search(content)
            sequence_text = sequence_match.group(0).strip() if sequence_match else ""
            
            # 处理事件序列 - 直接提取所有事件，忽略Utente和Sistema的区分
            sequence_events = []
            for line in sequence_text.split('\n'):
                line = line.strip()
                # 忽略标题行和Utente/Sistema标签行
                if line and not line.startswith("Sequenza degli eventi") and not line.startswith("Utente") and not line.startswith("Sistema"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.?\s*', '', line)
                    if cleaned_line:
                        sequence_events.append(cleaned_line)
            
            # 提取后置条件
            postconditions_pattern = re.compile(r'Postcondizioni:?\s*(.+?)(?=$)', re.DOTALL)
            postconditions_match = postconditions_pattern.search(content)
            postconditions_text = postconditions_match.group(1).strip() if postconditions_match else ""
            
            # 处理后置条件列表
            postconditions = []
            for line in postconditions_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除项目符号（如果有）
                    cleaned_line = re.sub(r'^[•\-\*]\s*', '', line)
                    if cleaned_line:
                        postconditions.append(cleaned_line)
            
            # 创建用例字典
            use_case = {
                "id": file_id - 1,  # 因为ID从0开始
                "dataset": "SMOS",
                "Nome": name,
                "Attori": actors,
                "Descrizione": description,
                "Precondizioni": preconditions,
                "Sequenza degli eventi": sequence_events,
                "Postcondizioni": postconditions
            }
            
            use_case_list.append(use_case)
        
        return use_case_list
    
    except Exception as e:
        print(f"提取用例时出错：{str(e)}")
        return []

def translate_and_append_smos_use_cases(input_use_cases, json_file_path, auth_key):
    """
    将SMOS用例翻译成英文并追加到现有JSON文件
    
    Args:
        input_use_cases: 提取的用例列表
        json_file_path: 现有JSON文件路径
        auth_key: DeepL API密钥
    
    Returns:
        翻译后的用例列表
    """
    translator = deepl.DeepLClient(auth_key)
    translated_use_cases = []
    
    try:
        # 读取现有JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            existing_use_cases = json.load(json_file)
        
        print(f"已读取现有JSON文件，包含 {len(existing_use_cases)} 个用例")
        
        total_cases = len(input_use_cases)
        print(f"开始翻译 {total_cases} 个新用例...")
        
        for i, use_case in enumerate(input_use_cases):
            print(f"正在翻译用例 {i+1}/{total_cases}: {use_case['Nome']}")
            
            # 翻译名称
            translated_name = translator.translate_text(use_case['Nome'], target_lang="EN-US", source_lang="IT").text
            
            # 翻译参与者
            translated_actors = translator.translate_text(use_case['Attori'], target_lang="EN-US", source_lang="IT").text if use_case['Attori'] else ""
            
            # 翻译描述
            translated_description = translator.translate_text(use_case['Descrizione'], target_lang="EN-US", source_lang="IT").text if use_case['Descrizione'] else ""
            
            # 翻译前置条件
            translated_preconditions = []
            for precondition in use_case['Precondizioni']:
                if precondition:
                    translated_precondition = translator.translate_text(precondition, target_lang="EN-US", source_lang="IT").text
                    translated_preconditions.append(translated_precondition)
            
            # 翻译事件序列
            translated_sequence_events = []
            for event in use_case['Sequenza degli eventi']:
                if event:
                    translated_event = translator.translate_text(event, target_lang="EN-US", source_lang="IT").text
                    translated_sequence_events.append(translated_event)
            
            # 翻译后置条件
            translated_postconditions = []
            for postcondition in use_case['Postcondizioni']:
                if postcondition:
                    translated_postcondition = translator.translate_text(postcondition, target_lang="EN-US", source_lang="IT").text
                    translated_postconditions.append(translated_postcondition)
            
            # 创建翻译后的用例字典
            translated_use_case = {
                "id": use_case['id'],
                "dataset": "SMOS",
                "Name": translated_name,
                "Actors": translated_actors,
                "Brief Description": translated_description,
                "Precondition": translated_preconditions,
                "Sequence of events": translated_sequence_events,
                "Postcondition": translated_postconditions,
                "Original": {
                    "Nome": use_case['Nome'],
                    "Attori": use_case['Attori'],
                    "Descrizione": use_case['Descrizione'],
                    "Precondizioni": use_case['Precondizioni'],
                    "Sequenza degli eventi": use_case['Sequenza degli eventi'],
                    "Postcondizioni": use_case['Postcondizioni']
                }
            }
            
            translated_use_cases.append(translated_use_case)
        
        # 合并现有用例和新翻译的用例
        combined_use_cases = existing_use_cases + translated_use_cases
        
        # 保存合并后的结果
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(combined_use_cases, json_file, ensure_ascii=False, indent=4)
        
        print(f"翻译完成，结果已追加到 {json_file_path}")
        print(f"文件现在包含 {len(combined_use_cases)} 个用例")
        return translated_use_cases
    
    except Exception as e:
        print(f"翻译用例时出错：{str(e)}")
        return []

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 原文件夹
    source_file_path = os.path.join(base_dir, '0_Data', '0_origin_data', 'SMOS')
    # 输出JSON文件
    output_file = os.path.join(base_dir, '0_Data', '2_json_dataset', 'SMOS_en.json')
    
    # 设置DeepL API密钥
    auth_key = deepl_auth_key
    
    # 提取剩余用例（从ID=62开始）
    print("正在从SMOS文件夹提取剩余用例...")
    remaining_use_cases = extract_remaining_smos_use_cases(source_file_path, start_id=62)
    print(f"成功提取 {len(remaining_use_cases)} 个剩余用例")
    
    # 翻译并追加用例
    if remaining_use_cases:
        translate_and_append_smos_use_cases(remaining_use_cases, output_file, auth_key)
    else:
        print("没有提取到剩余用例，无法进行翻译")