from imports import os, re, json, Document, deepl, deepl_auth_key,deepl_auth_key_1

def extract_uc_in_gamma_j(file_path):
    index = 0
    use_case_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Define regular expressions to pattern match each use case.
        use_case_pattern = re.compile(r'Use Cases Name(.*?)UC END', re.DOTALL)
        name_pattern = re.compile(r':(.*?)Goal:', re.DOTALL)
        goal_pattern = re.compile(r'Goal:(.*?)Actors:', re.DOTALL)
        actors_pattern = re.compile(r'Actors:(.*?)Preconditions:', re.DOTALL)
        precondition_pattern = re.compile(r'Preconditions:(.*?)Triggers:', re.DOTALL)
        triggers_pattern = re.compile(r'Triggers:(.*?)Basic Scenario:', re.DOTALL)
        basic_scenario_pattern = re.compile(r'Basic Scenario:(.*?)Alternative Scenario:', re.DOTALL)
        alternative_scenario_pattern = re.compile(r'Alternative Scenario:(.*?)Postconditions:', re.DOTALL)
        postcondition_pattern = re.compile(r'Postconditions:(.*?)$', re.DOTALL)

        # Find all matches of the use case pattern in the content.
        use_cases = use_case_pattern.findall(content)
        print(f"找到 {len(use_cases)} 个用例")

        for uc in use_cases:
            # Extract use case name
            name_match = name_pattern.search(uc)
            name = name_match.group(1).strip() if name_match else ""
            
            # Extract goal
            goal_match = goal_pattern.search(uc)
            goal = goal_match.group(1).strip() if goal_match else ""
            
            # Extract actors
            actors_match = actors_pattern.search(uc)
            actors_text = actors_match.group(1).strip() if actors_match else ""
            actors = [actor.strip() for actor in actors_text.split('\n') if actor.strip()]
            
            # Extract precondition
            precondition_match = precondition_pattern.search(uc)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Preconditions:"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # Extract triggers
            triggers_match = triggers_pattern.search(uc)
            triggers = triggers_match.group(1).strip() if triggers_match else ""
            
            # Extract basic scenario
            basic_scenario_match = basic_scenario_pattern.search(uc)
            basic_scenario_text = basic_scenario_match.group(1).strip() if basic_scenario_match else ""
            basic_scenario = []
            for line in basic_scenario_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Basic Scenario:"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        basic_scenario.append(cleaned_line)
            
            # Extract alternative scenario
            alternative_scenario_match = alternative_scenario_pattern.search(uc)
            alternative_scenario_text = alternative_scenario_match.group(1).strip() if alternative_scenario_match else ""
            
            # 创建一个字典来临时存储不同字母开头的场景
            letter_scenarios = {}
            
            for line in alternative_scenario_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Alternative Scenario:"):
                    # 检查是否是场景（以字母+数字+点开头）
                    letter_match = re.match(r'^([A-Z])\d+\.', line)
                    
                    if letter_match:
                        # 提取字母
                        letter = letter_match.group(1)
                        
                        # 如果这个字母还没有对应的列表，创建一个
                        if letter not in letter_scenarios:
                            letter_scenarios[letter] = []
                        
                        # 移除字母数字前缀
                        cleaned_line = re.sub(r'^[A-Z]\d+\.\s*', '', line)
                        if cleaned_line:
                            letter_scenarios[letter].append(cleaned_line)
                    elif letter_scenarios:  # 如果有当前正在处理的场景组
                        # 将这行添加到最后一个处理的字母场景中
                        last_letter = list(letter_scenarios.keys())[-1]
                        letter_scenarios[last_letter].append(line)
            
            # 将字典转换为二维列表
            alternative_scenarios = []
            for letter in sorted(letter_scenarios.keys()):
                alternative_scenarios.append(letter_scenarios[letter])
            
            # Extract postcondition
            postcondition_match = postcondition_pattern.search(uc)
            postcondition = postcondition_match.group(1).strip() if postcondition_match else ""
            
            # Create use case dictionary
            use_case_dict = {
                "id": index,
                "dataset" : "gamma j",
                "Use Case Name": name,
                "Brief Description": goal,
                "Actors": actors,
                "Precondition": preconditions,
                "Trigger": triggers,
                "Basic flow": basic_scenario,
                "Alt. Flow": alternative_scenarios,  # 二维列表，每个子列表包含相同字母开头的场景
                "Postcondition": postcondition
            }
            
            use_case_list.append(use_case_dict)
            index += 1
            print(f"已处理 {index} 个use case: {name}")
            

    return use_case_list

# 有问题直接手动改json文件了
def extract_uc_in_inventory_2(file_path):
    index = 0
    use_case_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 定义正则表达式匹配每个用例
        use_case_pattern = re.compile(r'(?:Use case name:|\d+\.\s*Use case name:)([\s\S]*?)UC END', re.DOTALL)
        
        # 找到所有匹配的用例
        use_cases = use_case_pattern.findall(content)
        print(f"找到 {len(use_cases)} 个用例")

        for uc in use_cases:
            # 提取用例名称
            name_match = re.search(r'^\s*:?\s*(.+?)(?:\n|\r\n|$)', uc)
            name = name_match.group(1).strip() if name_match else ""
            
            # 提取Actor
            actor_pattern = re.compile(r'Actor:\s*([\s\S]*?)(?:precondition:|\d+\.\d+\.\s*precondition:)', re.DOTALL)
            actor_match = actor_pattern.search(uc)
            actor_text = actor_match.group(1).strip() if actor_match else ""
            actors = [actor.strip() for actor in actor_text.split('\n') if actor.strip()]
            
            # 提取前置条件
            precondition_pattern = re.compile(r'precondition:\s*([\s\S]*?)(?:Use Case Dependencies|\d+\.\d+\.\s*Use Case Dependencies)', re.DOTALL)
            precondition_match = precondition_pattern.search(uc)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # 提取基本流程
            basic_flow_pattern = re.compile(r'Basic Flows:\s*([\s\S]*?)(?:Alternative Flows:|\d+\.\d+\.\s*Alternative Flows:)', re.DOTALL)
            basic_flow_match = basic_flow_pattern.search(uc)
            basic_flow_text = basic_flow_match.group(1).strip() if basic_flow_match else ""
            basic_flow = []
            for line in basic_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\d+\.\s*', '', line)
                    cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
                    if cleaned_line:
                        basic_flow.append(cleaned_line)
            
            # 提取备选流程
            alt_flow_pattern = re.compile(r'Alternative Flows:\s*([\s\S]*?)(?:Business Rules:|postcondition:|\d+\.\d+\.\s*Business Rules:|\d+\.\d+\.\s*postcondition:)', re.DOTALL)
            alt_flow_match = alt_flow_pattern.search(uc)
            alt_flow_text = alt_flow_match.group(1).strip() if alt_flow_match else ""
            alt_flow = []
            for line in alt_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\d+\.\s*', '', line)
                    cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
                    if cleaned_line:
                        alt_flow.append(cleaned_line)
            
            # 提取后置条件
            postcondition_pattern = re.compile(r'postcondition:\s*([\s\S]*?)(?:Open Issues:|\d+\.\d+\.\s*Open Issues:|$)', re.DOTALL)
            postcondition_match = postcondition_pattern.search(uc)
            postcondition_text = postcondition_match.group(1).strip() if postcondition_match else ""
            postconditions = []
            for line in postcondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\d+\.\s*', '', line)
                    cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
                    if cleaned_line:
                        postconditions.append(cleaned_line)
            
            # 创建用例字典
            use_case_dict = {
                "id": index,
                "dataset": "inventory_2",
                "Use Case Name": name,
                "Actor": actors,
                "Precondition": preconditions,
                "Basic Flow": basic_flow,
                "Alt. Flow": alt_flow,
                "Postcondition": postconditions
            }
            
            use_case_list.append(use_case_dict)
            index += 1
            print(f"已处理 {index} 个use case: {name}")
            
    return use_case_list


# 提出的用例有些问题，手动修改了json文件里的内容
def extract_uc_in_inventory(file_path):
    index = 0
    use_case_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Define regular expressions to pattern match each use case.
        use_case_pattern = re.compile(r'Use Cases Name:(.*?)UC END', re.DOTALL)
        identifier_pattern = re.compile(r'Identifier :(.*?)Description:', re.DOTALL)
        description_pattern = re.compile(r'Description:(.*?)Goal:', re.DOTALL)
        goal_pattern = re.compile(r'Goal:(.*?)Preconditions:', re.DOTALL)
        precondition_pattern = re.compile(r'Preconditions:(.*?)Assumptions:', re.DOTALL)
        assumptions_pattern = re.compile(r'Assumptions:(.*?)Basic Flow:', re.DOTALL)
        basic_flow_pattern = re.compile(r'Basic Flow:(.*?)Alt\. Flow:', re.DOTALL)
        alt_flow_pattern = re.compile(r'Alt\. Flow:(.*?)Exceptional Course :', re.DOTALL)
        exceptional_course_pattern = re.compile(r'Exceptional Course :(.*?)Postconditions:', re.DOTALL)
        postcondition_pattern = re.compile(r'Postconditions:(.*?)Actors:', re.DOTALL)
        actors_pattern = re.compile(r'Actors:(.*?)Included Use Cases:', re.DOTALL)
        included_use_cases_pattern = re.compile(r'Included Use Cases:(.*?)Notes:', re.DOTALL)
        notes_pattern = re.compile(r'Notes:(.*?)$', re.DOTALL)

        # Find all matches of the use case pattern in the content.
        use_cases = use_case_pattern.findall(content)
        print(f"找到 {len(use_cases)} 个用例")

        for uc in use_cases:
              
            # Extract use case name
            name = uc.split('\n')[0].strip()
            
            # Extract identifier
            identifier_match = identifier_pattern.search(uc)
            identifier = identifier_match.group(1).strip() if identifier_match else ""
            
            # Extract description
            description_match = description_pattern.search(uc)
            description = description_match.group(1).strip() if description_match else ""
            
            # Extract goal
            goal_match = goal_pattern.search(uc)
            goal = goal_match.group(1).strip() if goal_match else ""
            
            # Extract precondition
            precondition_match = precondition_pattern.search(uc)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # Extract assumptions
            assumptions_match = assumptions_pattern.search(uc)
            assumptions_text = assumptions_match.group(1).strip() if assumptions_match else ""
            assumptions = []
            for line in assumptions_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        assumptions.append(cleaned_line)
            
            # Extract basic flow
            basic_flow_match = basic_flow_pattern.search(uc)
            basic_flow_text = basic_flow_match.group(1).strip() if basic_flow_match else ""
            basic_flow = []
            for line in basic_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        basic_flow.append(cleaned_line)
            
            # Extract alt flow
            alt_flow_match = alt_flow_pattern.search(uc)
            alt_flow_text = alt_flow_match.group(1).strip() if alt_flow_match else ""
            
            # 处理Alt. Flow，可能有多个场景（A、B、C等）
            alt_flow_scenarios = []
            current_scenario = []
            
            # 检查是否有Condition:行
            condition_line = ""
            if "Condition:" in alt_flow_text:
                condition_parts = alt_flow_text.split("Condition:", 1)
                condition_line = "Condition: " + condition_parts[1].split('\n')[0].strip()
                alt_flow_text = condition_parts[1].split('\n', 1)[1] if len(condition_parts[1].split('\n')) > 1 else ""
            
            # 如果有条件行，添加到第一个场景
            if condition_line:
                current_scenario.append(condition_line)
            
            # 处理剩余的行
            for line in alt_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 检查是否是新的场景组（以字母+数字+点开头）
                    letter_match = re.match(r'^([A-Z])\d+\.', line)
                    
                    if letter_match:
                        # 如果找到新的字母，保存之前的场景组（如果有）
                        if current_scenario:
                            alt_flow_scenarios.append(current_scenario)
                            current_scenario = []
                        
                        # 移除字母数字前缀
                        cleaned_line = re.sub(r'^[A-Z]\d+\.\s*', '', line)
                        if cleaned_line:
                            current_scenario.append(cleaned_line)
                    else:
                        # 移除数字前缀（如果有）
                        cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                        if cleaned_line:
                            current_scenario.append(cleaned_line)
            
            # 添加最后一个场景组
            if current_scenario:
                alt_flow_scenarios.append(current_scenario)
            
            # Extract exceptional course
            exceptional_course_match = exceptional_course_pattern.search(uc)
            exceptional_course_text = exceptional_course_match.group(1).strip() if exceptional_course_match else ""
            exceptional_course = []
            
            # 处理Exceptional Course，可能有多个场景
            current_exception = []
            in_exception = False
            
            for line in exceptional_course_text.split('\n'):
                line = line.strip()
                if line:
                    # 检查是否是新的异常场景（以数字+点开头）
                    if re.match(r'^\d+\.\s*$', line):
                        if current_exception:
                            exceptional_course.append(current_exception)
                            current_exception = []
                        in_exception = True
                    elif in_exception:
                        # 移除数字前缀（如果有）
                        cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                        if cleaned_line:
                            current_exception.append(cleaned_line)
            
            # 添加最后一个异常场景
            if current_exception:
                exceptional_course.append(current_exception)
            
            # Extract postcondition
            postcondition_match = postcondition_pattern.search(uc)
            postcondition_text = postcondition_match.group(1).strip() if postcondition_match else ""
            postconditions = []
            for line in postcondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        postconditions.append(cleaned_line)
            
            # Extract actors
            actors_match = actors_pattern.search(uc)
            actors_text = actors_match.group(1).strip() if actors_match else ""
            actors = [actor.strip() for actor in actors_text.split('\n') if actor.strip()]
            
            # Extract included use cases
            included_use_cases_match = included_use_cases_pattern.search(uc)
            included_use_cases_text = included_use_cases_match.group(1).strip() if included_use_cases_match else ""
            included_use_cases = []
            for line in included_use_cases_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        included_use_cases.append(cleaned_line)
            
            # Extract notes
            notes_match = notes_pattern.search(uc)
            notes = notes_match.group(1).strip() if notes_match else ""
            
            # Create use case dictionary
            use_case_dict = {
                "id": index,
                "dataset": "inventory",
                "Use Case Name": name,
                "Identifier": identifier,
                "Description": description,
                "Goal": goal,
                "Precondition": preconditions,
                "Assumptions": assumptions,
                "Basic Flow": basic_flow,
                "Alt. Flow": alt_flow_scenarios,
                "Exc. Flow": exceptional_course,
                "Postcondition": postconditions,
                "Actors": actors,
                "Included Use Cases": included_use_cases,
                "Notes": notes
            }
            
            use_case_list.append(use_case_dict)
            index += 1
            print(f"已处理 {index} 个use case: {name}")
            
    return use_case_list


def extract_uc_in_pnnl(file_path):
    index = 0
    use_case_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 定义正则表达式匹配每个用例
        use_case_pattern = re.compile(r'\d+\.\d+\.\d+\s+Use case name:\s*([\s\S]*?)UC END', re.DOTALL)
        
        # 找到所有匹配的用例
        use_cases = use_case_pattern.findall(content)
        print(f"找到 {len(use_cases)} 个用例")

        for uc in use_cases:
            # 提取用例名称
            name_match = re.search(r'^\s*(.+?)(?:\n|\r\n)', uc)
            name = name_match.group(1).strip() if name_match else ""
            
            # 提取Brief Description
            brief_description_pattern = re.compile(r'Brief Description\s*([\s\S]*?)(?:\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\s+Flow of Events)', re.DOTALL)
            brief_description_match = brief_description_pattern.search(uc)
            brief_description = brief_description_match.group(1).strip() if brief_description_match else ""
            
            # 提取Basic Flow
            basic_flow_pattern = re.compile(r'Basic Flow\s*([\s\S]*?)(?:Alternative Flow|\d+\.\d+\.\d+\.\d+\s+Alternative Flow)', re.DOTALL)
            basic_flow_match = basic_flow_pattern.search(uc)
            basic_flow_text = basic_flow_match.group(1).strip() if basic_flow_match else ""
            basic_flow = []
            for line in basic_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        basic_flow.append(cleaned_line)
            
            # 提取Alternative Flow
            alt_flow_pattern = re.compile(r'Alternative Flow\s*([\s\S]*?)(?:Special Requirements|\d+\.\d+\.\d+\s+Special Requirements)', re.DOTALL)
            alt_flow_match = alt_flow_pattern.search(uc)
            alt_flow_text = alt_flow_match.group(1).strip() if alt_flow_match else ""
            alt_flow = []
            for line in alt_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^[A-Z]\d+\.\s*', '', line)
                    cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
                    if cleaned_line:
                        alt_flow.append(cleaned_line)
            
            # 提取Precondition
            precondition_pattern = re.compile(r'Preconditions\s*([\s\S]*?)(?:Postconditions|\d+\.\d+\.\d+\.\d+\s+Postconditions)', re.DOTALL)
            precondition_match = precondition_pattern.search(uc)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # 提取Postcondition
            postcondition_pattern = re.compile(r'Postconditions\s*([\s\S]*?)(?:Extension Points|\d+\.\d+\.\d+\.\d+\s+Extension Points)', re.DOTALL)
            postcondition_match = postcondition_pattern.search(uc)
            postcondition_text = postcondition_match.group(1).strip() if postcondition_match else ""
            postconditions = []
            for line in postcondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        postconditions.append(cleaned_line)
            
            # 创建用例字典
            use_case_dict = {
                "id": index,
                "dataset": "pnnl",
                "Name": name,
                "Brief Description": brief_description,
                "Precondition": preconditions,
                "Basic Flow": basic_flow,
                "Alt. Flow": alt_flow,
                "Postcondition": postconditions
            }
            
            use_case_list.append(use_case_dict)
            index += 1
            print(f"已处理 {index} 个use case: {name}")
            
    return use_case_list


def extract_uc_from_g02(file_path):
    """
    从g02-uc-cm-req.docx文件中提取用例信息
    
    Args:
        file_path: docx文件路径
        
    Returns:
        包含用例信息的列表，每个用例是一个字典
    """
    try:
        # 需要安装python-docx库
        from docx import Document
        document = Document(file_path)
        
        # 提取文档全文
        full_text = "\n".join([para.text for para in document.paragraphs])
        
        # 调试：保存文档全文到文本文件，以便检查结构
        debug_file_path = file_path.replace('.docx', '_debug.txt')
        with open(debug_file_path, 'w', encoding='utf-8') as debug_file:
            debug_file.write(full_text)
        print(f"已将文档内容保存到 {debug_file_path} 用于调试")
        
        # 尝试不同的正则表达式模式来匹配用例
        # 模式1：基于model manager.txt的结构
        use_case_pattern1 = re.compile(r'Use case name:\s*"?([^"\n]+)"?[\s\S]*?Description:\s*([\s\S]*?)\s*Primary actor:\s*([\s\S]*?)\s*Goal:\s*([\s\S]*?)\s*Action Sequence:\s*([\s\S]*?)END USE CASE', re.DOTALL)
        
        # 模式2：更宽松的模式，尝试匹配可能的用例结构
        use_case_pattern2 = re.compile(r'(?:Use [cC]ase|UC)\s*(?:name|ID)?:\s*"?([^"\n]+)"?[\s\S]*?(?:Description|Brief Description):\s*([\s\S]*?)(?:(?:Primary )?[Aa]ctor|Actors):\s*([\s\S]*?)(?:Goal|Objective):\s*([\s\S]*?)(?:Action Sequence|Basic Flow|Flow of Events):\s*([\s\S]*?)(?:END USE CASE|UC END)', re.DOTALL)
        
        # 尝试两种模式
        matches1 = list(use_case_pattern1.finditer(full_text))
        matches2 = list(use_case_pattern2.finditer(full_text))
        
        print(f"模式1找到 {len(matches1)} 个用例")
        print(f"模式2找到 {len(matches2)} 个用例")
        
        # 使用找到更多匹配的模式
        use_case_pattern = use_case_pattern1 if len(matches1) >= len(matches2) else use_case_pattern2
        matches = matches1 if len(matches1) >= len(matches2) else matches2
        
        use_cases = []
        for match in matches:
            name = match.group(1).strip()
            description = match.group(2).strip()
            actor = match.group(3).strip()
            goal = match.group(4).strip()
            action_sequence = match.group(5).strip()
            
            # 提取前置条件和后置条件（如果有）
            precondition_pattern = re.compile(r'Precondition[s]?:\s*([\s\S]*?)(?:Postcondition|Goal|Action)', re.DOTALL)
            precondition_match = precondition_pattern.search(description)
            precondition = precondition_match.group(1).strip() if precondition_match else ""
            
            postcondition_pattern = re.compile(r'Postcondition[s]?:\s*([\s\S]*?)(?:Goal|Action)', re.DOTALL)
            postcondition_match = postcondition_pattern.search(description)
            postcondition = postcondition_match.group(1).strip() if postcondition_match else ""
            
            # 提取备选流程（如果有）
            alt_flow_pattern = re.compile(r'(?:Alternative Flow[s]?:|Exceptional Course[s]?:|Exception[s]?:|Alternative[s]?:|Alternate Flow[s]?:)\s*([\s\S]*?)(?:END Action Sequence|END USE CASE|UC END)', re.DOTALL)
            alt_flow_match = alt_flow_pattern.search(action_sequence)
            alt_flow = alt_flow_match.group(1).strip() if alt_flow_match else "None"
            
            use_case = {
                "Name": name,
                "Brief Description": goal,  # 使用Goal作为Summary
                "Actors": actor,
                "Precondition": precondition,
                "Basic flow": description,
                "Alt. Flow": alt_flow,
                "Postcondition": postcondition
            }
            
            use_cases.append(use_case)
            print(f"已提取用例: {name}")
        
        # 如果仍然没有找到用例，尝试更宽松的模式
        if not use_cases:
            print("尝试更宽松的模式来匹配用例...")
            # 尝试查找可能的用例标题
            possible_uc_titles = re.findall(r'(?:Use [cC]ase|UC)\s*(?:\d+)?\s*[-:]?\s*"?([^"\n]+)"?', full_text)
            print(f"找到 {len(possible_uc_titles)} 个可能的用例标题: {possible_uc_titles}")
            
            # 尝试基于可能的标题分割文本
            if possible_uc_titles:
                for i, title in enumerate(possible_uc_titles):
                    start_idx = full_text.find(title)
                    if start_idx != -1:
                        end_idx = full_text.find(possible_uc_titles[i+1]) if i < len(possible_uc_titles)-1 else len(full_text)
                        uc_text = full_text[start_idx:end_idx]
                        
                        # 尝试从文本中提取关键信息
                        actor_match = re.search(r'(?:Actor|Actors)\s*[-:]\s*([^\n]+)', uc_text)
                        desc_match = re.search(r'(?:Description|Brief)\s*[-:]\s*([^\n]+)', uc_text)
                        
                        use_case = {
                            "Name": title.strip(),
                            "Brief Description": desc_match.group(1).strip() if desc_match else "",
                            "Actors": actor_match.group(1).strip() if actor_match else "",
                            "Precondition": "",
                            "Basic flow": "",
                            "Alt. Flow": "",
                            "Postcondition": ""
                        }
                        
                        use_cases.append(use_case)
                        print(f"已提取可能的用例: {title}")
        
        return use_cases
    
    except Exception as e:
        print(f"Error extracting use cases from {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def extract_uc_in_pnnl(file_path):
    index = 0
    use_case_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 定义正则表达式匹配每个用例
        use_case_pattern = re.compile(r'\d+\.\d+\.\d+\s+Use case name:\s*([\s\S]*?)UC END', re.DOTALL)
        
        # 找到所有匹配的用例
        use_cases = use_case_pattern.findall(content)
        print(f"找到 {len(use_cases)} 个用例")

        for uc in use_cases:
            # 提取用例名称
            name_match = re.search(r'^\s*(.+?)(?:\n|\r\n)', uc)
            name = name_match.group(1).strip() if name_match else ""
            
            # 提取Brief Description
            brief_description_pattern = re.compile(r'Brief Description\s*([\s\S]*?)(?:\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\s+Flow of Events)', re.DOTALL)
            brief_description_match = brief_description_pattern.search(uc)
            brief_description = brief_description_match.group(1).strip() if brief_description_match else ""
            
            # 提取Basic Flow
            basic_flow_pattern = re.compile(r'Basic Flow\s*([\s\S]*?)(?:Alternative Flow|\d+\.\d+\.\d+\.\d+\s+Alternative Flow)', re.DOTALL)
            basic_flow_match = basic_flow_pattern.search(uc)
            basic_flow_text = basic_flow_match.group(1).strip() if basic_flow_match else ""
            basic_flow = []
            for line in basic_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        basic_flow.append(cleaned_line)
            
            # 提取Alternative Flow
            alt_flow_pattern = re.compile(r'Alternative Flow\s*([\s\S]*?)(?:Special Requirements|\d+\.\d+\.\d+\s+Special Requirements)', re.DOTALL)
            alt_flow_match = alt_flow_pattern.search(uc)
            alt_flow_text = alt_flow_match.group(1).strip() if alt_flow_match else ""
            alt_flow = []
            for line in alt_flow_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^[A-Z]\d+\.\s*', '', line)
                    cleaned_line = re.sub(r'^\d+\.\s*', '', cleaned_line)
                    if cleaned_line:
                        alt_flow.append(cleaned_line)
            
            # 提取Precondition
            precondition_pattern = re.compile(r'Preconditions\s*([\s\S]*?)(?:Postconditions|\d+\.\d+\.\d+\.\d+\s+Postconditions)', re.DOTALL)
            precondition_match = precondition_pattern.search(uc)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # 提取Postcondition
            postcondition_pattern = re.compile(r'Postconditions\s*([\s\S]*?)(?:Extension Points|\d+\.\d+\.\d+\.\d+\s+Extension Points)', re.DOTALL)
            postcondition_match = postcondition_pattern.search(uc)
            postcondition_text = postcondition_match.group(1).strip() if postcondition_match else ""
            postconditions = []
            for line in postcondition_text.split('\n'):
                line = line.strip()
                if line:
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        postconditions.append(cleaned_line)
            
            # 创建用例字典
            use_case_dict = {
                "id": index,
                "dataset": "pnnl",
                "Name": name,
                "Brief Description": brief_description,
                "Precondition": preconditions,
                "Basic Flow": basic_flow,
                "Alt. Flow": alt_flow,
                "Postcondition": postconditions
            }
            
            use_case_list.append(use_case_dict)
            index += 1
            print(f"已处理 {index} 个use case: {name}")
            
    return use_case_list


def extract_uc_from_g02(file_path):
    """
    从g02-uc-cm-req_debug.txt文件中提取用例信息
    
    Args:
        file_path: txt文件路径
        
    Returns:
        包含用例信息的列表，每个用例是一个字典
    """
    try:
        # 读取文本文件
        with open(file_path.replace('.docx', '_debug.txt'), 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # 使用正则表达式提取用例
        # 匹配从"Use Case"开始，到下一个"Use Case"之前的内容
        use_case_pattern = re.compile(r'Use Case (\d+)\s*\nSummary: ([^\n]+)\s*\n([\s\S]*?)(?=\n\s*Use Case \d+|\nUML class diagram|$)', re.DOTALL)
        
        use_cases = []
        for i, match in enumerate(use_case_pattern.finditer(full_text)):
            uc_id = match.group(1)  # 用例编号
            summary = match.group(2).strip()  # 摘要
            content = match.group(3).strip()  # 用例内容
            
            # 提取Actors
            actors_pattern = re.compile(r'Actors?: ([^\n]+)', re.DOTALL)
            actors_match = actors_pattern.search(content)
            actors = actors_match.group(1).strip() if actors_match else ""
            
            # 提取Precondition
            precondition_pattern = re.compile(r'Precondition(?:s)?: ([^\n]+)', re.DOTALL)
            precondition_match = precondition_pattern.search(content)
            precondition = precondition_match.group(1).strip() if precondition_match else ""
            
            # 提取Description (Basic Flow)
            description_pattern = re.compile(r'Description:\s*\n([\s\S]*?)(?=\nPostcondition|\nAlternate flow|\nExceptions|$)', re.DOTALL)
            description_match = description_pattern.search(content)
            description_text = description_match.group(1).strip() if description_match else ""
            
            # 将Description文本分割成步骤列表
            basic_flow = []
            for line in description_text.split('\n'):
                line = line.strip()
                if line:
                    basic_flow.append(line)
            
            # 提取Alternate flow
            alt_flow_pattern = re.compile(r'Alternate flow:\s*\n([\s\S]*?)(?=\nPostcondition|\nExceptions|$)', re.DOTALL)
            alt_flow_match = alt_flow_pattern.search(content)
            alt_flow_text = alt_flow_match.group(1).strip() if alt_flow_match else ""
            
            # 提取Exceptions
            exceptions_pattern = re.compile(r'Exceptions:\s*\n([\s\S]*?)(?=\nPostcondition|$)', re.DOTALL)
            exceptions_match = exceptions_pattern.search(content)
            exceptions_text = exceptions_match.group(1).strip() if exceptions_match else ""
            
            # 合并Alternate flow和Exceptions作为Alt. Flow
            alt_flow = []
            if alt_flow_text:
                alt_flow.append(alt_flow_text)
            if exceptions_text:
                alt_flow.append(exceptions_text)
            
            # 提取Postcondition
            postcondition_pattern = re.compile(r'Postcondition(?:s)?: ([^\n]+)', re.DOTALL)
            postcondition_match = postcondition_pattern.search(content)
            postcondition = postcondition_match.group(1).strip() if postcondition_match else ""
            
            # 创建用例字典
            use_case = {
                "id": i,  # 使用索引作为id
                "dataset": "g02-uc-cm-req",
                "Name": f"Use Case {uc_id}",  # 使用Use Case编号作为名称
                "Brief Description": summary,  # 使用Summary作为Brief Description
                "Actors": actors,
                "Precondition": precondition,
                "Basic flow": basic_flow,
                "Alt. Flow": alt_flow,
                "Postcondition": postcondition
            }
            
            use_cases.append(use_case)
            print(f"已提取用例 {i+1}: Use Case {uc_id} - {summary}")
        
        return use_cases
    
    except Exception as e:
        print(f"Error extracting use cases from {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def extract_use_cases_from_g04(file_path):
    """
    从文本文件中提取use case信息
    
    Args:
        file_path: 文本文件路径
        
    Returns:
        包含use case信息的列表，每个use case是一个字典
    """
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 定义正则表达式模式来匹配每个use case
        use_case_pattern = re.compile(r'Use Case Name:(.*?)Table \d+:', re.DOTALL)
        
        # 查找所有匹配的use case
        use_cases_text = use_case_pattern.findall(content)
        
        use_cases = []
        for i, uc_text in enumerate(use_cases_text):
            # 提取use case名称
            name_match = re.search(r'(.+?)\n\n', uc_text.strip())
            name = name_match.group(1).strip() if name_match else ""
            
            # 提取actor
            actor_match = re.search(r'Primary actor\s*:(.+?)\n\n', uc_text, re.DOTALL)
            actor = actor_match.group(1).strip() if actor_match else ""
            
            # 提取precondition
            precondition_pattern = re.compile(r'Preconditions:\s*(.+?)\n\n', re.DOTALL)
            precondition_match = precondition_pattern.search(uc_text)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            
            # 处理precondition列表
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Preconditions:"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # 提取postcondition
            postcondition_pattern = re.compile(r'Postconditions:\s*(.+?)\n\n', re.DOTALL)
            postcondition_match = postcondition_pattern.search(uc_text)
            postcondition_text = postcondition_match.group(1).strip() if postcondition_match else ""
            
            # 处理postcondition列表
            postconditions = []
            for line in postcondition_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Postconditions:"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        postconditions.append(cleaned_line)
            
            # 提取main success scenario
            scenario_pattern = re.compile(r'Main Success Scenario:\s*(.+?)\n\n', re.DOTALL)
            scenario_match = scenario_pattern.search(uc_text)
            scenario_text = scenario_match.group(1).strip() if scenario_match else ""
            
            # 处理main success scenario列表
            scenarios = []
            for line in scenario_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Main Success Scenario:"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+(?:\.\d+)?\s*', '', line)
                    if cleaned_line:
                        scenarios.append(cleaned_line)
            
            # 创建use case字典
            use_case = {
                "id": i + 1,
                "name": name,
                "actor": actor,
                "preconditions": preconditions,
                "postconditions": postconditions,
                "main_success_scenario": scenarios
            }
            
            use_cases.append(use_case)
        
        return use_cases
    
    except Exception as e:
        print(f"提取use case时出错：{str(e)}")
        return []

def extract_use_cases_from_easyclinic(directory_path):
    """
    从easyClinic文件夹中提取所有用例信息
    
    Args:
        directory_path: easyClinic文件夹路径
        
    Returns:
        包含所有用例信息的列表，每个用例是一个字典
    """
    use_case_list = []
    
    try:
        # 获取文件夹中所有的txt文件
        file_names = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        file_names.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf'))
        
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(directory_path, file_name)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取用例名称
            name_pattern = re.compile(r'Use case:\s*(.+?)\s*\n', re.DOTALL)
            name_match = name_pattern.search(content)
            name = name_match.group(1).strip() if name_match else f"Use Case {i+1}"
            
            # 提取描述
            description_pattern = re.compile(r'Description:\s*(.+?)\s*\n\s*Preconditions:', re.DOTALL)
            description_match = description_pattern.search(content)
            description = description_match.group(1).strip() if description_match else ""
            
            # 提取前置条件
            precondition_pattern = re.compile(r'Preconditions:\s*(.+?)\s*\n\s*Postconditions:', re.DOTALL)
            precondition_match = precondition_pattern.search(content)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            
            # 处理前置条件列表
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Preconditions:"):
                    preconditions.append(line)
            
            # 提取后置条件
            postcondition_pattern = re.compile(r'Postconditions:\s*(.+?)\s*\n\s*Sub-flows:', re.DOTALL)
            postcondition_match = postcondition_pattern.search(content)
            postcondition_text = postcondition_match.group(1).strip() if postcondition_match else ""
            
            # 处理后置条件列表
            postconditions = []
            for line in postcondition_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Postconditions:"):
                    postconditions.append(line)
            
            # 提取子流程
            subflow_pattern = re.compile(r'Sub-flows:\s*(.+?)\s*\n\s*Alternative flows:', re.DOTALL)
            subflow_match = subflow_pattern.search(content)
            subflow_text = subflow_match.group(1).strip() if subflow_match else ""
            
            # 处理子流程列表
            subflows = []
            for line in subflow_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Sub-flows:"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\[?S?\d*\]?\s*', '', line)
                    if cleaned_line:
                        subflows.append(cleaned_line)
            
            # 提取备选流程
            altflow_pattern = re.compile(r'Alternative flows:\s*(.+?)\s*$', re.DOTALL)
            altflow_match = altflow_pattern.search(content)
            altflow_text = altflow_match.group(1).strip() if altflow_match else ""
            
            # 处理备选流程列表
            altflows = []
            current_altflow = ""
            
            for line in altflow_text.split('\n'):
                line = line.strip()
                if line:
                    # 如果是新的备选流程标题
                    if re.match(r'^[A-Za-z]', line) and not line.startswith("Alternative flows:") and not re.match(r'^\d+-\d+', line):
                        if current_altflow:
                            altflows.append(current_altflow)
                        current_altflow = line
                    # 如果是备选流程的步骤
                    elif re.match(r'^\d+-\d+', line) or current_altflow:
                        if current_altflow:
                            current_altflow += " " + line
                        else:
                            current_altflow = line
            
            # 添加最后一个备选流程
            if current_altflow:
                altflows.append(current_altflow)
            
            # 创建用例字典
            use_case = {
                "id": i,
                "dataset": "easyClinic",
                "Name": name,
                "Brief Description": description,
                "Precondition": preconditions,
                "Postcondition": postconditions,
                "Basic flow": subflows,
                "Alt. Flow": altflows
            }
            
            use_case_list.append(use_case)
        
        return use_case_list
    
    except Exception as e:
        print(f"提取用例时出错：{str(e)}")
        return []

def extract_use_cases_from_etour(directory_path):
    """
    从eTour文件夹中提取所有用例信息
    
    Args:
        directory_path: eTour文件夹路径
        
    Returns:
        包含所有用例信息的列表，每个用例是一个字典
    """
    use_case_list = []
    
    try:
        # 获取文件夹中所有的txt文件
        file_names = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        # 按UC编号排序
        file_names.sort(key=lambda x: int(x[2:-4]) if x[2:-4].isdigit() else float('inf'))
        
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(directory_path, file_name)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取用例名称
            name_pattern = re.compile(r'Use case name:\s*(.+?)\s*\n', re.DOTALL)
            name_match = name_pattern.search(content)
            name = name_match.group(1).strip() if name_match else f"Use Case {i+1}"
            
            # 提取描述
            description_pattern = re.compile(r'Description:\s*(.+?)\s*\n\s*Participating Actor:', re.DOTALL)
            description_match = description_pattern.search(content)
            description = description_match.group(1).strip() if description_match else ""
            
            # 提取参与者
            actor_pattern = re.compile(r'Participating Actor:\s*(.+?)\s*\n\s*Entry Operator conditions:', re.DOTALL)
            actor_match = actor_pattern.search(content)
            actor = actor_match.group(1).strip() if actor_match else ""
            
            # 提取入口条件
            entry_conditions_pattern = re.compile(r'Entry Operator conditions:\s*(.+?)\s*\n\s*Flow of events User System:', re.DOTALL)
            entry_conditions_match = entry_conditions_pattern.search(content)
            entry_conditions = entry_conditions_match.group(1).strip() if entry_conditions_match else ""
            
            # 提取事件流
            flow_pattern = re.compile(r'Flow of events User System:\s*(.+?)\s*\n\s*Exit conditions:', re.DOTALL)
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
            
            # 提取退出条件
            exit_conditions_pattern = re.compile(r'Exit conditions:\s*(.+?)\s*\n\s*Quality requirements:', re.DOTALL)
            exit_conditions_match = exit_conditions_pattern.search(content)
            exit_conditions_text = exit_conditions_match.group(1).strip() if exit_conditions_match else ""
            
            # 处理退出条件列表
            exit_conditions = []
            for line in exit_conditions_text.split('\n'):
                line = line.strip()
                if line:
                    exit_conditions.append(line)
            
            # 提取质量需求
            quality_pattern = re.compile(r'Quality requirements:\s*(.+?)\s*$', re.DOTALL)
            quality_match = quality_pattern.search(content)
            quality_requirements = quality_match.group(1).strip() if quality_match else ""
            
            # 创建用例字典
            use_case = {
                "id": i,
                "dataset": "eTour",
                "Name": name,
                "Brief Description": description,
                "Actors": actor,
                "Precondition": entry_conditions,
                "Basic flow": flow_events,
                "Postcondition": exit_conditions,
                "Quality requirements": quality_requirements
            }
            
            use_case_list.append(use_case)
        
        return use_case_list
    
    except Exception as e:
        print(f"提取用例时出错：{str(e)}")
        return []

def extract_use_cases_from_smos(directory_path):
    """
    从SMOS文件夹中提取所有用例信息
    
    Args:
        directory_path: SMOS文件夹路径
        
    Returns:
        包含所有用例信息的列表，每个用例是一个字典
    """
    use_case_list = []
    
    try:
        # 获取文件夹中所有的txt文件
        file_names = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        # 按SMOS编号排序
        file_names.sort(key=lambda x: int(re.search(r'SMOS(\d+)', x).group(1)) if re.search(r'SMOS(\d+)', x) else float('inf'))
        
        for i, file_name in enumerate(file_names):
            file_path = os.path.join(directory_path, file_name)
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取用例名称
            name_pattern = re.compile(r'Nome:\s*(.+?)(?:\n|$)', re.DOTALL)
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
            
            # 处理事件序列
            sequence_events = []
            
            # 直接提取所有事件，忽略Utente和Sistema的区分
            for line in sequence_text.split('\n'):
                line = line.strip()
                # 忽略标题行和Utente/Sistema标签行
                if line and not line.startswith("Sequenza degli eventi") and not line.startswith("Utente") and not line.startswith("Sistema"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.?\s*', '', line)
                    if cleaned_line:
                        sequence_events.append(cleaned_line)
            
            # 提取后置条件
            postconditions_pattern = re.compile(r'Postcondizioni:\s*(.+?)(?=$)', re.DOTALL)
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
                "id": i,
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

def translate_and_save_smos_use_cases(input_use_cases, output_json_path, auth_key):
    """
    将SMOS用例翻译成英文并保存为JSON
    
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
            print(f"已保存 {len(translated_use_cases)} 个已翻译的用例")
        return translated_use_cases


def extract_itrust_usecases(folder_path):
    """
    从iTrust文件夹中提取所有用例文件的内容
    
    参数:
    folder_path -- iTrust文件夹的路径
    
    返回:
    包含所有用例信息的列表
    """
    usecases = []
    
    # 获取文件夹中的所有txt文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 提取用例ID和名称
        uc_id_match = re.search(r'UC(\d+)', file_name)
        uc_id = uc_id_match.group(1) if uc_id_match else ''
        
        # 提取用例名称
        name_match = re.search(r'UC\d+\s+(.+?)\s*Use Case', content)
        name = name_match.group(1).strip() if name_match else ''
        
        # 提取前置条件
        preconditions_match = re.search(r'(\d+\.\d+)\s+Preconditions:\s*\n\n(.+?)\n\n', content, re.DOTALL)
        preconditions = preconditions_match.group(2).strip() if preconditions_match else ''
        
        # 提取主流程
        main_flow_match = re.search(r'(\d+\.\d+)\s+Main Flow:\s*\n\n(.+?)\n\n', content, re.DOTALL)
        main_flow = main_flow_match.group(2).strip() if main_flow_match else ''
        
        # 提取子流程
        sub_flows_match = re.search(r'(\d+\.\d+)\s+Sub-flows:\s*\n\n(.+?)\n\n', content, re.DOTALL)
        sub_flows_text = sub_flows_match.group(2).strip() if sub_flows_match else ''
        
        # 解析子流程
        sub_flows = []
        if sub_flows_text:
            # 使用正则表达式匹配所有子流程
            sub_flow_matches = re.findall(r'\[(S\d+)\]\s+(.+?)(?=\n\[S\d+\]|$)', sub_flows_text, re.DOTALL)
            for sf_id, sf_text in sub_flow_matches:
                sub_flows.append({
                    'id': sf_id,
                    'text': sf_text.strip()
                })
        
        # 提取替代流程
        alt_flows_match = re.search(r'(\d+\.\d+)\s+Alternative Flows:\s*\n\n(.+?)(?=\n\n\d+\.\d+|$)', content, re.DOTALL)
        alt_flows_text = alt_flows_match.group(2).strip() if alt_flows_match else ''
        
        # 解析替代流程
        alt_flows = []
        if alt_flows_text:
            # 使用正则表达式匹配所有替代流程
            alt_flow_matches = re.findall(r'\[(E\d+)\]\s+(.+?)(?=\n\[E\d+\]|$)', alt_flows_text, re.DOTALL)
            for af_id, af_text in alt_flow_matches:
                alt_flows.append({
                    'id': af_id,
                    'text': af_text.strip()
                })
        
        # 创建用例对象
        usecase = {
            'id': uc_id,
            'dataset': "iTrust",
            'Name': name,
            'Precondition': preconditions,
            'Basic flow': main_flow,
            'Sub. Flow': sub_flows,
            'Alt. Flow': alt_flows
        }
        
        usecases.append(usecase)
    
    return usecases

def extract_hats_usecases(file_path):
    index = 0
    use_case_list = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 定义正则表达式匹配每个用例
        use_case_pattern = re.compile(r'Use Cases Name:(.*?)UC END', re.DOTALL)
        
        # 找到所有匹配的用例
        use_cases = use_case_pattern.findall(content)
        print(f"找到 {len(use_cases)} 个用例")

        for uc in use_cases:
            # 提取用例名称
            name_lines = uc.strip().split('\n')
            name = name_lines[0].strip() if name_lines else ""
            
            # 提取Description
            description_pattern = re.compile(r'Description:(.*?)Preconditions :', re.DOTALL)
            description_match = description_pattern.search(uc)
            description = description_match.group(1).strip() if description_match else ""
            
            # 提取Actors
            actors_pattern = re.compile(r'Actors:(.*?)(?:Used by|Basic Flow):', re.DOTALL)
            actors_match = actors_pattern.search(uc)
            actors_text = actors_match.group(1).strip() if actors_match else ""
            actors = [actor.strip() for actor in actors_text.split('\n') if actor.strip()]
            
            # 提取Preconditions
            precondition_pattern = re.compile(r'Preconditions :(.*?)Actors:', re.DOTALL)
            precondition_match = precondition_pattern.search(uc)
            precondition_text = precondition_match.group(1).strip() if precondition_match else ""
            preconditions = []
            for line in precondition_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Preconditions :"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        preconditions.append(cleaned_line)
            
            # 提取Basic Flow
            basic_flow_pattern = re.compile(r'Basic Flow:(.*?)Alternative:', re.DOTALL)
            basic_flow_match = basic_flow_pattern.search(uc)
            basic_flow_text = basic_flow_match.group(1).strip() if basic_flow_match else ""
            basic_flow = []
            for line in basic_flow_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Basic Flow:"):
                    # 移除数字前缀（如果有）
                    cleaned_line = re.sub(r'^\d+\.\s*', '', line)
                    if cleaned_line:
                        basic_flow.append(cleaned_line)
            
            # 提取Alternative
            alternative_pattern = re.compile(r'Alternative:(.*?)$', re.DOTALL)
            alternative_match = alternative_pattern.search(uc)
            alternative_text = alternative_match.group(1).strip() if alternative_match else ""
            
            # 处理Alternative部分，按ALT序号分组
            alt_scenarios = {}
            current_alt = None
            
            for line in alternative_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # 检查是否是新的ALT组
                alt_match = re.match(r'^ALT (\d+):', line)
                if alt_match:
                    current_alt = alt_match.group(1)
                    alt_scenarios[current_alt] = []
                    # 提取ALT描述
                    alt_desc = re.sub(r'^ALT \d+:\s*', '', line)
                    if alt_desc:
                        alt_scenarios[current_alt].append(alt_desc)
                # 检查是否是ALT组内的步骤
                elif current_alt and re.match(r'^A\d+-\d+[A-Z]?:', line):
                    # 提取步骤描述
                    step_desc = re.sub(r'^A\d+-\d+[A-Z]?:\s*', '', line)
                    if step_desc:
                        alt_scenarios[current_alt].append(step_desc)
                # 其他行添加到当前ALT组
                elif current_alt:
                    alt_scenarios[current_alt].append(line)
            
            # 将alt_scenarios转换为列表格式
            alternative = []
            for alt_num in sorted(alt_scenarios.keys(), key=int):
                alternative.append({
                    "alt_num": alt_num,
                    "steps": alt_scenarios[alt_num]
                })
            
            # 创建用例字典
            use_case_dict = {
                "id": index,
                "dataset": "hats",
                "Use Case Name": name,
                "Description": description,
                "Actors": actors,
                "Precondition": preconditions,
                "Basic flow": basic_flow,
                "Alternative": alternative
            }
            
            use_case_list.append(use_case_dict)
            index += 1
            print(f"已处理 {index} 个use case: {name}")
            
    return use_case_list

def extract_viper_usecases(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 使用正则表达式匹配所有use case
    use_case_pattern = r'USE CASE #\s*\d+\s*NAME:\s*([^\n]+)\s*\n\s*Goal in Context\s*([^\n]+).*?\n\s*Scope & Level.*?\n\s*Preconditions\s*([^\n]+).*?\n\s*Success End Condition\s*([^\n]+).*?\n\s*Failed End Condition\s*([^\n]+).*?(?:Failed End Condition\s*Action\s*([\s\S]*?)\s*END DESCRIPTION).*?(?:EXTENSIONS\s*Step\s*Branching Action\s*([\s\S]*?)\s*SUB-VARIATIONS)\s*Branching Action\s*([\s\S]*?)\s*RELATED INFORMATION'
    
    use_cases = re.findall(use_case_pattern, content, re.DOTALL)
    
    result = []
    id = 0
    for uc in use_cases:
        name = uc[0].strip()
        goal_in_context = uc[1].strip()
        preconditions = uc[2].strip()
        success_end_condition = uc[3].strip()
        failed_end_condition = uc[4].strip()
        
        # 处理Action部分
        action_text = uc[5].strip() if len(uc) > 5 else ""
        action_lines = [line.strip() for line in action_text.split('\n') if line.strip()]
        actions = []
        for line in action_lines:
            # 提取步骤编号和描述
            step_match = re.match(r'\s*(\d+)\s*(.*)', line)
            if step_match:
                step_num = step_match.group(1)
                step_desc = step_match.group(2).strip()
                actions.append({"step": step_num, "description": step_desc})
        
        # 处理EXTENSIONS部分
        extensions_text = uc[6].strip() if len(uc) > 6 else ""
        extension_lines = [line.strip() for line in extensions_text.split('\n') if line.strip()]
        extensions = []
        for line in extension_lines:
            # 提取步骤和分支动作
            ext_match = re.match(r'\s*(\S+)\s*(.*)', line)
            if ext_match:
                ext_step = ext_match.group(1)
                ext_action = ext_match.group(2).strip()
                extensions.append({"step": ext_step, "branching_action": ext_action})
        
        # 处理SUB-VARIATIONS部分
        sub_variations_text = uc[7].strip() if len(uc) > 7 else ""
        sub_var_lines = [line.strip() for line in sub_variations_text.split('\n') if line.strip()]
        sub_variations = []
        for line in sub_var_lines:
            # 提取步骤和分支动作
            sub_match = re.match(r'\s*(\S+)\s*(.*)', line)
            if sub_match:
                sub_step = sub_match.group(1)
                sub_action = sub_match.group(2).strip()
                sub_variations.append({"step": sub_step, "branching_action": sub_action})
        
        use_case = {
            "id": id,
            "dataset": "viper",
            "Name": name,
            "Brief Description": goal_in_context,
            "Precondition": [preconditions],
            "Postcondition": [success_end_condition],
            "Basic flow": actions,
            "Alt. Flow": extensions + sub_variations
        }
        id += 1
        result.append(use_case)
    
    return result

def extract_model_manager_usecases(file_path):
    """从model manager.txt文件中提取use case信息"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 定义正则表达式模式来匹配use case
    # 注意：第7个use case的格式与其他use case略有不同，它以"3.6 Use case name:"开头
    use_case_pattern = re.compile(r'(?:\d+\.\d+\s+)?Use case name:\s*"?([^"\n]+)"?[\s\S]*?Description:\s*([\s\S]*?)\s*Primary actor:\s*([\s\S]*?)\s*(?:Goal:\s*([\s\S]*?)\s*)?Action Sequence:\s*([\s\S]*?)END USE CASE', re.DOTALL)
    id = 0
    # 查找所有匹配项
    use_cases = []
    for match in use_case_pattern.finditer(content):
        name = match.group(1).strip()
        description = match.group(2).strip()
        primary_actor = match.group(3).strip()
        goal = match.group(4).strip() if match.group(4) else ""
        action_sequence = match.group(5).strip()
        
        # 处理Action Sequence，将其转换为列表
        action_steps = []
        for line in action_sequence.split('\n'):
            line = line.strip()
            if line and re.match(r'^\d+\.', line):  # 如果行以数字和点开头
                # 移除行首的数字和点
                step = re.sub(r'^\d+\.\s*', '', line)
                action_steps.append(step)
        
        use_case = {
            "id": id,
            "dataset": "model manager",
            "Name": name,
            "Brief Description": description + goal,
            "Actor": [primary_actor],
            "Basic flow": action_steps
        }
        use_cases.append(use_case)
        id += 1
    
    return use_cases

def extract_keepass_usecases(file_path):
    """提取KeePass XML文件中的use case信息
    
    Args:
        file_path: KeePass XML文件的路径
        
    Returns:
        包含所有use case信息的字典列表
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 提取所有use case（从3.1到3.14）
    usecases = []
    
    # 使用正则表达式匹配每个use case的开始和结束
    pattern = r'<p id="3\.([1-9]|1[0-4])">\s*<title>(.*?)</title>.*?(?=<p id="3\.(?:[1-9]|1[0-5])">|</req_document>)'  
    matches = re.finditer(pattern, content, re.DOTALL)
    
    id = 0
    for match in matches:
        usecase_num = match.group(1)
        usecase_name = match.group(2).strip()
        usecase_content = match.group(0)
        
        # 提取Description
        description_pattern = r'<p id="3\.\d+\.1">\s*<title>Description</title>\s*<text_body>\s*(.*?)\s*</text_body>\s*</p>'
        description_match = re.search(description_pattern, usecase_content, re.DOTALL)
        description = description_match.group(1).strip() if description_match else ""
        
        # 提取Basic Data Flow
        basic_flow_pattern = r'<p id="3\.\d+\.2\.1">\s*<title>Basic Data Flow</title>\s*<text_body>\s*(.*?)\s*</text_body>\s*</p>'
        basic_flow_match = re.search(basic_flow_pattern, usecase_content, re.DOTALL)
        
        basic_flow = []
        if basic_flow_match:
            flow_text = basic_flow_match.group(1).strip()
            # 检查是否有<itemize>标签
            if '<itemize>' in flow_text:
                # 提取<item>标签中的内容
                item_pattern = r'<item>(.*?)</item>'
                items = re.findall(item_pattern, flow_text, re.DOTALL)
                basic_flow = [item.strip() for item in items]
            else:
                # 如果没有<itemize>标签，按行分割
                basic_flow = [line.strip() for line in flow_text.split('\n') if line.strip()]
        
        # 提取Alternative Data Flows（可能有多个）
        alt_flows = []
        alt_flow_sections_pattern = r'<p id="3\.\d+\.2\.2\.\d+">\s*<title>Alternative Data Flow \d+</title>\s*<text_body>\s*(.*?)\s*</text_body>\s*</p>'
        alt_flow_sections = re.finditer(alt_flow_sections_pattern, usecase_content, re.DOTALL)
        
        for alt_section in alt_flow_sections:
            alt_flow_text = alt_section.group(1).strip()
            alt_flow_items = []
            
            # 检查是否有<itemize>标签
            if '<itemize>' in alt_flow_text:
                # 提取<item>标签中的内容
                item_pattern = r'<item>(.*?)</item>'
                items = re.findall(item_pattern, alt_flow_text, re.DOTALL)
                alt_flow_items = [item.strip() for item in items]
            else:
                # 如果没有<itemize>标签，按行分割
                alt_flow_items = [line.strip() for line in alt_flow_text.split('\n') if line.strip()]
            
            if alt_flow_items:
                alt_flows.append(alt_flow_items)
        
        # 提取Functional Requirements
        func_req_pattern = r'<p id="3\.\d+\.3">\s*<title>Functional Requirements</title>\s*(.*?)\s*</p>\s*</p>'
        func_req_match = re.search(func_req_pattern, usecase_content, re.DOTALL)
        
        func_requirements = []
        if func_req_match:
            req_text = func_req_match.group(1).strip()
            # 提取<req>标签中的内容
            req_pattern = r'<req id="\d+">\s*<text_body>\s*(.*?)\s*</text_body>\s*</req>'
            reqs = re.findall(req_pattern, req_text, re.DOTALL)
            func_requirements = [req.strip() for req in reqs]
            
            # 检查是否有直接在<text_body>中的内容（不在<req>标签中）
            direct_req_pattern = r'<text_body>\s*(.*?)\s*</text_body>'
            direct_req = re.search(direct_req_pattern, req_text, re.DOTALL)
            if direct_req and not reqs:
                func_requirements = [direct_req.group(1).strip()]
        
        usecase = {
            "number": id,
            "name": usecase_name,
            "description": description,
            "basic_data_flow": basic_flow,
            "alternative_data_flows": alt_flows,
            "functional_requirements": func_requirements
        }
        
        id += 1
        usecases.append(usecase)
    
    return usecases

if __name__ == '__main__':
    task_name = 'read_data_from_txt_to_json'

    print(f"*** task_name: {task_name} ***")

    # 从txt文件提取用例，存放在json文件中
    if task_name == 'read_data_from_txt_to_json':
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # txt文件所在的文件夹
        txt_file_path = os.path.join(base_dir,'0_Data','1_preliminary_process')
        # json文件输出的文件夹
        json_output_dir = os.path.join(base_dir, '0_Data', '2_json_dataset')
        os.makedirs(json_output_dir, exist_ok=True)
        # 数据集名称列表，记得修改下面调用的函数。
        dataset_name_dict ={
            '0000 - gamma j':extract_uc_in_gamma_j, 
            '0000 - inventory':extract_uc_in_inventory, 
            '2009 - inventory 2.0':extract_uc_in_inventory_2,
            'pnnl':extract_uc_in_pnnl,
            "g02-uc-cm-req":extract_uc_from_g02,
            "g04-uc-req":extract_use_cases_from_g04,
            "hats":extract_hats_usecases,
            "viper": extract_viper_usecases,
            "model manager":extract_model_manager_usecases,
            "keepass":extract_keepass_usecases
            }# note：手动选择要处理的数据集
        # 每个数据集的项名称不一样，尽量使用RUCM的名称，比如basic flow等。一行存放一个step，或者一个actor。

        for dataset_name in dataset_name_dict.keys():
            # 手动选择要处理的数据集
            if dataset_name != 'keepass': 
                continue
            file_path = os.path.join(txt_file_path, dataset_name + ".txt")
            if dataset_name == 'keepass':
                file_path = os.path.join(base_dir,'0_Data','0_origin_data', "2008 - " + dataset_name + ".xml")
            if os.path.isfile(file_path) or os.path.isdir(file_path):
                # 更换要提取的数据集时，修改这个函数。
                use_case_list = dataset_name_dict[dataset_name](file_path)

                json_file_path = os.path.join(json_output_dir, dataset_name + '.json')
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(use_case_list, json_file, ensure_ascii=False, indent=4)
                
                print(f"成功提取 {len(use_case_list)} 个use case并保存到 {json_file_path}")

            else:
                print(f"file {file_path} does not exist")
                continue

    # 从文件夹中提取用例，存放在json文件中
    if task_name == 'read_data_from_folder_to_json':
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # txt文件所在的文件夹
        txt_file_path = os.path.join(base_dir,'0_Data','0_origin_data')
        # json文件输出的文件夹
        json_output_dir = os.path.join(base_dir, '0_Data', '2_json_dataset')
        os.makedirs(json_output_dir, exist_ok=True)
        # 数据集名称列表，记得修改下面调用的函数。
        dataset_folder_name_dict = {
            'easyClinic':extract_use_cases_from_easyclinic,
            'eTour':extract_use_cases_from_etour,
            'iTrust':extract_itrust_usecases
            }
        # 每个数据集的项名称不一样，尽量使用RUCM的名称，比如basic flow等。一行存放一个step，或者一个actor。

        for dataset_name in dataset_folder_name_dict.keys():
            # 手动选择要处理的数据集
            if dataset_name != 'iTrust': 
                continue
            file_path = os.path.join(txt_file_path, dataset_name)
            if os.path.isdir(file_path):
                # 更换要提取的数据集时，修改这个函数。
                use_case_list = dataset_folder_name_dict[dataset_name](file_path)

                json_file_path = os.path.join(json_output_dir, dataset_name + '.json')
                with open(json_file_path, 'w', encoding='utf-8') as json_file:
                    json.dump(use_case_list, json_file, ensure_ascii=False, indent=4)
                
                print(f"成功提取 {len(use_case_list)} 个use case并保存到 {json_file_path}")

            else:
                print(f"file {file_path} does not exist")
                continue
    

    # 从意大利文件夹中提取用例，存放在json文件中
    if task_name =='read_data_from_Italian_to_json':
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_name_dict = {"SMOS":"SMOS_en.json","eANCI":"eANCI_en"}

        for dataset_name in dataset_name_dict.keys():
            if dataset_name != 'SMOS': 
                continue
            # 原文件夹
            source_file_path = os.path.join(base_dir,'0_Data','0_origin_data',dataset_name)
            output_file = os.path.join(base_dir, '0_Data', '2_json_dataset', 'SMOS_en.json')

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 设置DeepL API密钥
            auth_key = deepl_auth_key # 请替换为您的DeepL API密钥
            
            # 提取用例
            print("正在从SMOS文件夹提取用例...")
            use_cases = extract_use_cases_from_smos(source_file_path)
            print(f"成功提取 {len(use_cases)} 个用例")
            
            # 翻译并保存用例
            if use_cases:
                translate_and_save_smos_use_cases(use_cases, output_file, auth_key)
            else:
                print("没有提取到用例，无法进行翻译")


    print(f"*** task_name: {task_name} is over！***")