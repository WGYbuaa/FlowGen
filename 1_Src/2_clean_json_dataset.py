import os
import json
import re
import nltk
from nltk.tokenize import sent_tokenize

# 下载nltk资源（如果尚未下载）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 辅助函数：将文本拆分成句子
def split_sentences(text):
    """
    将长文本拆分成句子列表
    """
    if not text or not isinstance(text, str):
        return []
    
    # 预处理文本，处理特殊情况
    # 1. 处理缩写词，避免错误拆分
    text = re.sub(r'(Mr\.|Mrs\.|Dr\.|Prof\.|etc\.)', r'\1<POINT>', text)
    # 2. 处理数字编号后的点，避免错误拆分
    text = re.sub(r'(\d+)\.(\s)', r'\1<POINT>\2', text)
    # 3. 处理文件扩展名，避免错误拆分
    text = re.sub(r'(\.[a-zA-Z]{2,4})\s', r'<POINT>\1 ', text)
    
    # 4. 处理大写的 OR 作为断句标志（确保小写的 or 不受影响）
    text = re.sub(r'\s+OR\s+', '. ', text)
    
    # 使用nltk拆分句子
    sentences = sent_tokenize(text)
    
    # 还原特殊标记
    sentences = [s.replace('<POINT>', '.') for s in sentences]
    
    # 清理每个句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

# eANCI_en.json 清洗函数
def clean_eanci(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 移除 Original 字段
        if "Original" in cleaned_item:
            cleaned_item.pop("Original")
        
        # 统一字段名称
        if "Use Case Name" in cleaned_item:
            cleaned_item["Name"] = cleaned_item.pop("Use Case Name")
        
        if "Participating Actors" in cleaned_item:
            actors = cleaned_item.pop("Participating Actors")
            # 将参与者拆分为列表
            if isinstance(actors, str):
                actors_list = [actor.strip() for actor in re.split(r'[,;]', actors) if actor.strip()]
                cleaned_item["Actors"] = actors_list
        
        if "Flow of Events" in cleaned_item:
            flow = cleaned_item.pop("Flow of Events")
            # 处理事件流，确保每个项目是一个完整的句子
            if isinstance(flow, list):
                # 合并列表项，然后重新拆分成句子
                combined_text = " ".join(flow)
                cleaned_item["Basic flow"] = split_sentences(combined_text)
            else:
                cleaned_item["Basic flow"] = split_sentences(flow)
        
        if "Entry Condition" in cleaned_item:
            entry_cond = cleaned_item.pop("Entry Condition")
            # 处理入口条件，确保每个项目是一个完整的句子
            if isinstance(entry_cond, list):
                # 合并列表项，然后重新拆分成句子
                combined_text = " ".join(entry_cond)
                cleaned_item["Precondition"] = split_sentences(combined_text)
            else:
                cleaned_item["Precondition"] = split_sentences(entry_cond)
        
        if "Exit Conditions" in cleaned_item:
            exit_cond = cleaned_item.pop("Exit Conditions")
            # 处理出口条件，确保每个项目是一个完整的句子
            if isinstance(exit_cond, list):
                # 合并列表项，然后重新拆分成句子
                combined_text = " ".join(exit_cond)
                cleaned_item["Postcondition"] = split_sentences(combined_text)
            else:
                cleaned_item["Postcondition"] = split_sentences(exit_cond)
        
        if "Quality Requirements" in cleaned_item:
            quality_req = cleaned_item.pop("Quality Requirements")
            # 处理质量要求，确保每个项目是一个完整的句子
            if isinstance(quality_req, list):
                # 合并列表项，然后重新拆分成句子
                combined_text = " ".join(quality_req)
                cleaned_item["Quality Requirements"] = split_sentences(combined_text)
            else:
                cleaned_item["Quality Requirements"] = split_sentences(quality_req)
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# SMOS_en.json 清洗函数
def clean_smos(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 移除 Original 字段
        if "Original" in cleaned_item:
            cleaned_item.pop("Original")
        
        # 处理Actors字段，将其转换为列表
        if "Actors" in cleaned_item and isinstance(cleaned_item["Actors"], str):
            actors = cleaned_item["Actors"]
            actors_list = [actor.strip() for actor in re.split(r'[,;]', actors) if actor.strip()]
            cleaned_item["Actors"] = actors_list
        
        # 处理Precondition字段，确保每个项目是一个完整的句子
        if "Precondition" in cleaned_item:
            precond = cleaned_item["Precondition"]
            if isinstance(precond, list):
                # 合并列表项，然后重新拆分成句子
                combined_text = " ".join(precond)
                cleaned_item["Precondition"] = split_sentences(combined_text)
        
        # 处理Sequence of events字段，确保每个项目是一个完整的句子
        if "Sequence of events" in cleaned_item:
            seq_events = cleaned_item.pop("Sequence of events")
            if isinstance(seq_events, list):
                # 清理每个事件，确保是完整的句子
                cleaned_events = []
                for event in seq_events:
                    # 移除数字前缀（如果有）
                    event = re.sub(r'^\d+\.?\s*', '', event)
                    # 拆分成句子
                    sentences = split_sentences(event)
                    cleaned_events.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_events
            else:
                cleaned_item["Basic flow"] = split_sentences(seq_events)
        
        # 处理Postcondition字段，确保每个项目是一个完整的句子
        if "Postcondition" in cleaned_item:
            postcond = cleaned_item["Postcondition"]
            if isinstance(postcond, list):
                # 合并列表项，然后重新拆分成句子
                combined_text = " ".join(postcond)
                cleaned_item["Postcondition"] = split_sentences(combined_text)
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# eTour.json 清洗函数
def clean_etour(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Actors字段，将其转换为列表
        if "Actors" in cleaned_item and isinstance(cleaned_item["Actors"], str):
            actors = cleaned_item["Actors"]
            actors_list = [actor.strip() for actor in re.split(r'[,;]', actors) if actor.strip()]
            cleaned_item["Actors"] = actors_list
        
        # 处理Precondition字段，确保是列表格式
        if "Precondition" in cleaned_item and isinstance(cleaned_item["Precondition"], str):
            precond = cleaned_item["Precondition"]
            cleaned_item["Precondition"] = split_sentences(precond)
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        # 处理Postcondition字段，确保每个项目是一个完整的句子
        if "Postcondition" in cleaned_item:
            postcond = cleaned_item["Postcondition"]
            if isinstance(postcond, list):
                # 清理每个条件，确保是完整的句子
                cleaned_postcond = []
                for cond in postcond:
                    # 拆分成句子
                    sentences = split_sentences(cond)
                    cleaned_postcond.extend(sentences)
                cleaned_item["Postcondition"] = cleaned_postcond
        
        # 处理Quality requirements字段，确保是列表格式
        if "Quality requirements" in cleaned_item and isinstance(cleaned_item["Quality requirements"], str):
            quality_req = cleaned_item["Quality requirements"]
            cleaned_item["Quality requirements"] = split_sentences(quality_req)
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# easyClinic.json 清洗函数
def clean_easyclinic(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    # 确保 flow 是字符串类型
                    if isinstance(flow, str):
                        # 移除数字前缀（如果有）
                        flow = re.sub(r'^\d+\.?\s*', '', flow)
                        # 拆分成句子
                        sentences = split_sentences(flow)
                        cleaned_alt_flow.extend(sentences)
                    else:
                        print(f"警告：在 Alt. Flow 中发现非字符串元素：{flow}")
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# g02-uc-cm-req.json 清洗函数
def clean_g02(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Actors字段，确保是列表格式
        if "Actors" in cleaned_item and not isinstance(cleaned_item["Actors"], list):
            actors = cleaned_item["Actors"]
            if isinstance(actors, str):
                actors_list = [actor.strip() for actor in re.split(r'[,;]', actors) if actor.strip()]
                cleaned_item["Actors"] = actors_list
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
            elif isinstance(basic_flow, str):
                cleaned_item["Basic flow"] = split_sentences(basic_flow)
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    # 移除数字前缀（如果有）
                    flow = re.sub(r'^\d+\.?\s*', '', flow)
                    # 拆分成句子
                    sentences = split_sentences(flow)
                    cleaned_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
            elif isinstance(alt_flow, str):
                cleaned_item["Alt. Flow"] = split_sentences(alt_flow)
        
        # 处理Postcondition字段，确保是列表格式
        if "Postcondition" in cleaned_item and isinstance(cleaned_item["Postcondition"], str):
            postcond = cleaned_item["Postcondition"]
            cleaned_item["Postcondition"] = split_sentences(postcond)
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# g04-uc-req.json 清洗函数
def clean_g04(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理actor字段，统一为Actors
        if "actor" in cleaned_item:
            cleaned_item["Actors"] = cleaned_item.pop("actor")
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# pnnl.json 清洗函数
def clean_pnnl(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Brief Description字段，确保是完整的句子
        if "Brief Description" in cleaned_item and isinstance(cleaned_item["Brief Description"], str):
            brief_desc = cleaned_item["Brief Description"]
            cleaned_item["Brief Description"] = brief_desc.strip()
        
        # 处理Basic Flow字段，确保每个项目是一个完整的句子
        if "Basic Flow" in cleaned_item:
            basic_flow = cleaned_item["Basic Flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic Flow"] = cleaned_steps
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    if isinstance(flow, str):
                        # 移除数字前缀（如果有）
                        flow = re.sub(r'^\d+\.?\s*', '', flow)
                        # 拆分成句子
                        sentences = split_sentences(flow)
                        cleaned_alt_flow.extend(sentences)
                    else:
                        print(f"警告：在 Alt. Flow 中发现非字符串元素：{flow}")
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# 0000 - gamma j.json 清洗函数
def clean_gamma_j(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list) and all(isinstance(x, list) for x in alt_flow):
                # 处理嵌套列表情况
                flattened_alt_flow = []
                for flow_group in alt_flow:
                    for flow in flow_group:
                        # 移除数字前缀（如果有）
                        flow = re.sub(r'^\d+\.?\s*', '', flow)
                        # 拆分成句子
                        sentences = split_sentences(flow)
                        flattened_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = flattened_alt_flow
            elif isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    # 移除数字前缀（如果有）
                    flow = re.sub(r'^\d+\.?\s*', '', flow)
                    # 拆分成句子
                    sentences = split_sentences(flow)
                    cleaned_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# 0000 - inventory.json 清洗函数
def clean_inventory(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Basic Flow字段，确保每个项目是一个完整的句子
        if "Basic Flow" in cleaned_item:
            basic_flow = cleaned_item["Basic Flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic Flow"] = cleaned_steps
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list) and all(isinstance(x, list) for x in alt_flow):
                # 处理嵌套列表情况
                flattened_alt_flow = []
                for flow_group in alt_flow:
                    for flow in flow_group:
                        # 移除数字前缀（如果有）
                        flow = re.sub(r'^\d+\.?\s*', '', flow)
                        # 拆分成句子
                        sentences = split_sentences(flow)
                        flattened_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = flattened_alt_flow
            elif isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    # 移除数字前缀（如果有）
                    flow = re.sub(r'^\d+\.?\s*', '', flow)
                    # 拆分成句子
                    sentences = split_sentences(flow)
                    cleaned_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
        
        # 处理Exc. Flow字段，确保格式一致
        if "Exc. Flow" in cleaned_item:
            exc_flow = cleaned_item["Exc. Flow"]
            if isinstance(exc_flow, list) and all(isinstance(x, list) for x in exc_flow):
                # 处理嵌套列表情况
                flattened_exc_flow = []
                for flow_group in exc_flow:
                    for flow in flow_group:
                        # 移除数字前缀（如果有）
                        flow = re.sub(r'^\d+\.?\s*', '', flow)
                        # 拆分成句子
                        sentences = split_sentences(flow)
                        flattened_exc_flow.extend(sentences)
                cleaned_item["Exc. Flow"] = flattened_exc_flow
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# 2009 - inventory 2.0.json 清洗函数
def clean_inventory_2(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Actor字段，统一为Actors
        if "Actor" in cleaned_item:
            cleaned_item["Actors"] = cleaned_item.pop("Actor")
        
        # 处理Basic Flow字段，确保每个项目是一个完整的句子
        if "Basic Flow" in cleaned_item:
            basic_flow = cleaned_item["Basic Flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic Flow"] = cleaned_steps
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    # 确保 flow 是字符串类型
                    if isinstance(flow, str):
                        # 移除数字前缀（如果有）
                        flow = re.sub(r'^\d+\.?\s*', '', flow)
                        # 拆分成句子
                        sentences = split_sentences(flow)
                        cleaned_alt_flow.extend(sentences)
                    else:
                        print(f"警告：在 Alt. Flow 中发现非字符串元素：{flow}")
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# viper.json 清洗函数
def clean_viper(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    if isinstance(step, dict) and "description" in step:
                        # 移除数字前缀（如果有）
                        description = re.sub(r'^\d+\.?\s*', '', step["description"])
                        # 拆分成句子
                        sentences = split_sentences(description)
                        cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    if isinstance(flow, dict) and "branching_action" in flow:
                        # 移除数字前缀（如果有）
                        branching_action = re.sub(r'^\d+\.?\s*', '', flow["branching_action"])
                        # 拆分成句子
                        sentences = split_sentences(branching_action)
                        cleaned_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
        
        # 处理Precondition字段，确保是列表格式且每个项目是一个完整的句子
        if "Precondition" in cleaned_item:
            precond = cleaned_item["Precondition"]
            if isinstance(precond, list):
                cleaned_precond = []
                for cond in precond:
                    # 移除数字前缀（如果有）
                    cond = re.sub(r'^\d+\.?\s*', '', cond)
                    # 拆分成句子
                    sentences = split_sentences(cond)
                    cleaned_precond.extend(sentences)
                cleaned_item["Precondition"] = cleaned_precond
            elif isinstance(precond, str):
                cleaned_item["Precondition"] = split_sentences(precond)
        
        # 处理Postcondition字段，确保是列表格式且每个项目是一个完整的句子
        if "Postcondition" in cleaned_item:
            postcond = cleaned_item["Postcondition"]
            if isinstance(postcond, list):
                cleaned_postcond = []
                for cond in postcond:
                    # 移除数字前缀（如果有）
                    cond = re.sub(r'^\d+\.?\s*', '', cond)
                    # 拆分成句子
                    sentences = split_sentences(cond)
                    cleaned_postcond.extend(sentences)
                cleaned_item["Postcondition"] = cleaned_postcond
            elif isinstance(postcond, str):
                cleaned_item["Postcondition"] = split_sentences(postcond)
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# hats.json 清洗函数
def clean_hats(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 统一字段名称
        if "Use Case Name" in cleaned_item:
            cleaned_item["Name"] = cleaned_item.pop("Use Case Name")
        
        if "Description" in cleaned_item:
            cleaned_item["Brief Description"] = cleaned_item.pop("Description")
        
        # 处理Actors字段，确保是列表格式
        if "Actors" in cleaned_item and not isinstance(cleaned_item["Actors"], list):
            actors = cleaned_item["Actors"]
            if isinstance(actors, str):
                actors_list = [actor.strip() for actor in re.split(r'[,;]', actors) if actor.strip()]
                cleaned_item["Actors"] = actors_list
        
        # 处理Precondition字段，确保是列表格式且每个项目是一个完整的句子
        if "Precondition" in cleaned_item:
            precond = cleaned_item["Precondition"]
            if isinstance(precond, list):
                cleaned_precond = []
                for cond in precond:
                    # 移除数字前缀（如果有）
                    cond = re.sub(r'^\d+\.?\s*', '', cond)
                    # 拆分成句子
                    sentences = split_sentences(cond)
                    cleaned_precond.extend(sentences)
                cleaned_item["Precondition"] = cleaned_precond
            elif isinstance(precond, str):
                cleaned_item["Precondition"] = split_sentences(precond)
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        # 处理Alternative字段，转换为Alt. Flow
        if "Alternative" in cleaned_item:
            alt_flow = cleaned_item.pop("Alternative")
            if isinstance(alt_flow, list):
                # 清理每个备选流程，确保是完整的句子
                cleaned_alt_flow = []
                for flow in alt_flow:
                    if isinstance(flow, dict) and "steps" in flow:
                        for step in flow["steps"]:
                            # 移除数字前缀（如果有）
                            step = re.sub(r'^\d+\.?\s*', '', step)
                            # 拆分成句子
                            sentences = split_sentences(step)
                            cleaned_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = cleaned_alt_flow
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# keepass.json 清洗函数
def clean_keepass(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Brief Description字段，确保是字符串格式
        if "Brief Description" in cleaned_item and not isinstance(cleaned_item["Brief Description"], str):
            brief_desc = cleaned_item["Brief Description"]
            if isinstance(brief_desc, list):
                cleaned_item["Brief Description"] = " ".join(brief_desc)
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        # 处理Alt. Flow字段，确保格式一致
        if "Alt. Flow" in cleaned_item:
            alt_flow = cleaned_item["Alt. Flow"]
            if isinstance(alt_flow, list) and all(isinstance(x, list) for x in alt_flow):
                # 处理嵌套列表情况
                flattened_alt_flow = []
                for flow_group in alt_flow:
                    for flow in flow_group:
                        # 移除数字前缀（如果有）
                        flow = re.sub(r'^\d+\.?\s*', '', flow)
                        # 拆分成句子
                        sentences = split_sentences(flow)
                        flattened_alt_flow.extend(sentences)
                cleaned_item["Alt. Flow"] = flattened_alt_flow
        
        # 处理Postcondition字段，确保是列表格式且每个项目是一个完整的句子
        if "Postcondition" in cleaned_item:
            postcond = cleaned_item["Postcondition"]
            if isinstance(postcond, list):
                cleaned_postcond = []
                for cond in postcond:
                    # 移除数字前缀（如果有）
                    cond = re.sub(r'^\d+\.?\s*', '', cond)
                    # 拆分成句子
                    sentences = split_sentences(cond)
                    cleaned_postcond.extend(sentences)
                cleaned_item["Postcondition"] = cleaned_postcond
            elif isinstance(postcond, str):
                cleaned_item["Postcondition"] = split_sentences(postcond)
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# model manager.json 清洗函数
def clean_model_manager(data):
    cleaned_data = []
    
    for item in data:
        cleaned_item = item.copy()
        
        # 处理Actor字段，统一为Actors
        if "Actor" in cleaned_item:
            actors = cleaned_item.pop("Actor")
            if isinstance(actors, list):
                cleaned_item["Actors"] = actors
            elif isinstance(actors, str):
                actors_list = [actor.strip() for actor in re.split(r'[,;]', actors) if actor.strip()]
                cleaned_item["Actors"] = actors_list
        
        # 处理Brief Description字段，确保是字符串格式
        if "Brief Description" in cleaned_item:
            brief_desc = cleaned_item["Brief Description"]
            if isinstance(brief_desc, list):
                cleaned_item["Brief Description"] = " ".join(brief_desc)
            elif isinstance(brief_desc, str):
                # 确保描述以标点符号结尾
                if not re.search(r'[.!?]$', brief_desc):
                    cleaned_item["Brief Description"] = brief_desc + "."
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item:
            basic_flow = cleaned_item["Basic flow"]
            if isinstance(basic_flow, list):
                # 清理每个步骤，确保是完整的句子
                cleaned_steps = []
                for step in basic_flow:
                    # 移除数字前缀（如果有）
                    step = re.sub(r'^\d+\.?\s*', '', step)
                    # 拆分成句子
                    sentences = split_sentences(step)
                    cleaned_steps.extend(sentences)
                cleaned_item["Basic flow"] = cleaned_steps
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# 主函数：处理目录下的所有JSON文件
def clean_all_json_files(directory_path, output_directory=None):
    """
    清洗和标准化目录下的所有JSON文件
    
    Args:
        directory_path: JSON文件所在目录路径
        output_directory: 输出目录路径，如果为None，则覆盖原文件
    """
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # 获取目录下所有的JSON文件
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json') and not f.startswith('README')]
    
    for file_name in json_files:
        file_path = os.path.join(directory_path, file_name)
        
        print(f"正在处理文件 {file_name}...")
        
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 根据文件名选择相应的清洗函数
        if file_name == "iTrust.json":
            cleaned_data = clean_itrust(data)
        elif file_name == "viper.json":
            cleaned_data = clean_viper(data)
        elif file_name == "hats.json":
            cleaned_data = clean_hats(data)
        elif file_name == "keepass.json":
            cleaned_data = clean_keepass(data)
        elif file_name == "model manager.json":
            cleaned_data = clean_model_manager(data)
        # if file_name == "eANCI_en.json":
        #     cleaned_data = clean_eanci(data)
        # elif file_name == "SMOS_en.json":
        #     cleaned_data = clean_smos(data)
        # elif file_name == "eTour.json":
        #     cleaned_data = clean_etour(data)
        # elif file_name == "easyClinic.json":
        #     cleaned_data = clean_easyclinic(data)
        # elif file_name == "g02-uc-cm-req.json":
        #     cleaned_data = clean_g02(data)
        # elif file_name == "g04-uc-req.json":
        #     cleaned_data = clean_g04(data)
        # elif file_name == "pnnl.json":
        #     cleaned_data = clean_pnnl(data)
        # elif file_name == "0000 - gamma j.json":
        #     cleaned_data = clean_gamma_j(data)
        # elif file_name == "0000 - inventory.json":
        #     cleaned_data = clean_inventory(data)
        # elif file_name == "2009 - inventory 2.0.json":
        #     cleaned_data = clean_inventory_2(data)
        else:
            print(f"未找到 {file_name} 的清洗函数，跳过处理")
            continue
        
        # 确定输出路径
        if output_directory:
            output_path = os.path.join(output_directory, f"cleaned_{file_name}")
        else:
            output_path = file_path
        
        # 保存清洗后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
        
        print(f"已完成 {file_name} 的清洗和标准化，保存到 {output_path}")

# 读取用dict保存的uc
def read_uc_from_json(file_path):
    use_case_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # line = line.replace("'", '"')  # 有时候数据集中会有多余的单引号或者双引号
            try:
                uc = json.loads(line)
                use_case_list.append(uc)
            except json.JSONDecodeError as e:
                print(f"错误信息: {e},  Line: {inspect.currentframe().f_lineno}, json读取失败")
    return use_case_list

# iTrust.json 清洗函数
def clean_itrust(data):
    cleaned_data = []
    
    # 按顺序重新编号
    new_id = 1
    
    for item in data:
        cleaned_item = item.copy()
        
        # 重新编写id
        cleaned_item["id"] = str(new_id)
        new_id += 1
        
        # 移除 Original 字段（如果存在）
        if "Original" in cleaned_item:
            cleaned_item.pop("Original")
        
        # 处理Precondition字段，确保是列表格式
        if "Precondition" in cleaned_item and isinstance(cleaned_item["Precondition"], str):
            precond = cleaned_item["Precondition"]
            cleaned_item["Precondition"] = split_sentences(precond)
        
        # 处理Basic flow字段，确保每个项目是一个完整的句子
        if "Basic flow" in cleaned_item and isinstance(cleaned_item["Basic flow"], str):
            basic_flow = cleaned_item["Basic flow"]
            cleaned_item["Basic flow"] = split_sentences(basic_flow)
        
        # 处理Sub. Flow字段，清理每个子流程的文本
        if "Sub. Flow" in cleaned_item and isinstance(cleaned_item["Sub. Flow"], list):
            sub_flows = cleaned_item["Sub. Flow"]
            cleaned_sub_flows = []
            
            for flow in sub_flows:
                if isinstance(flow, dict) and "id" in flow and "text" in flow:
                    cleaned_text = flow["text"]
                    
                    # 移除文本末尾可能包含的Alternative Flows标记
                    if "Alternative Flows:" in cleaned_text:
                        cleaned_text = cleaned_text.split("Alternative Flows:")[0].strip()
                    
                    # 清理文本并拆分成句子
                    sentences = split_sentences(cleaned_text)
                    
                    cleaned_sub_flows.append({
                        "id": flow["id"],
                        "text": sentences if len(sentences) > 1 else cleaned_text
                    })
            
            cleaned_item["Sub. Flow"] = cleaned_sub_flows
        
        # 处理Alt. Flow字段，清理每个备选流程的文本
        if "Alt. Flow" in cleaned_item and isinstance(cleaned_item["Alt. Flow"], list):
            alt_flows = cleaned_item["Alt. Flow"]
            cleaned_alt_flows = []
            
            for flow in alt_flows:
                if isinstance(flow, dict) and "id" in flow and "text" in flow:
                    cleaned_text = flow["text"]
                    
                    # 清理文本并拆分成句子
                    sentences = split_sentences(cleaned_text)
                    
                    cleaned_alt_flows.append({
                        "id": flow["id"],
                        "text": sentences if len(sentences) > 1 else cleaned_text
                    })
            
            cleaned_item["Alt. Flow"] = cleaned_alt_flows
        
        # 统一字段名称（如果需要）
        if "Basic flow" in cleaned_item:
            cleaned_item["Basic Flow"] = cleaned_item.pop("Basic flow")
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

def get_unique_keys_from_files(directory):
    from imports import OrderedDict
    """处理目录中所有JSON文件，输出每个文件字典的唯一键"""
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 验证数据结构：必须是列表，且元素为字典
                if not isinstance(data, list):
                    print(f"⚠️ 文件 {filename} 错误: 顶级结构不是列表（实际类型: {type(data).__name__}）")
                    continue
                
                # 收集所有唯一键（保持原始顺序）
                unique_keys = OrderedDict()
                for item in data:
                    if not isinstance(item, dict):
                        print(f"⚠️ 文件 {filename} 中有非字典元素（类型: {type(item).__name__}）")
                        continue
                    
                    for key in item.keys():
                        # 使用OrderedDict保留发现顺序
                        unique_keys[key] = None
                
                # 转换为列表输出
                key_list = list(unique_keys.keys())
                if key_list:
                    print(f"📂 文件 '{filename}' 包含的唯一键 ({len(key_list)}个):")
                    print(", ".join(key_list))
                else:
                    print(f"📂 文件 '{filename}' 没有发现任何键")
                print("─" * 60)
            
            except Exception as e:
                print(f"⛔ 处理文件 {filename} 时出错: {str(e)}")
                print("─" * 60)

if __name__ == "__main__":
    task_name = 'find_exc_flow_in_viper'

    print(f"*** task_name: {task_name} ***")

    # 从txt文件提取用例，存放在json文件中
    if task_name == 'clean_json':
        # 设置输入和输出目录
        input_directory = "e:/Trae_project/ConditionOfUCS/0_Data/2_json_dataset"
        output_directory = "e:/Trae_project/ConditionOfUCS/0_Data/3_cleaned_json_dataset"
        
        # 处理指定的JSON文件
        json_files = ["viper.json", "hats.json", "keepass.json", "model manager.json"]
        
        for file_name in json_files:
            file_path = os.path.join(input_directory, file_name)
            
            if not os.path.exists(file_path):
                print(f"文件 {file_name} 不存在，跳过处理")
                continue
                
            print(f"正在处理文件 {file_name}...")
            
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 根据文件名选择相应的清洗函数
            if file_name == "viper.json":
                cleaned_data = clean_viper(data)
            elif file_name == "hats.json":
                cleaned_data = clean_hats(data)
            elif file_name == "keepass.json":
                cleaned_data = clean_keepass(data)
            elif file_name == "model manager.json":
                cleaned_data = clean_model_manager(data)
            else:
                print(f"未找到 {file_name} 的清洗函数，跳过处理")
                continue
            
            # 确定输出路径
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                
            output_path = os.path.join(output_directory, f"cleaned_{file_name}")
            
            # 保存清洗后的数据
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
            
            print(f"已完成 {file_name} 的清洗和标准化，保存到 {output_path}")

    # 输出每个数据集的uc的key的种类，用于统一
    if task_name == "output_key":
        json_dir = "e:/Trae_project/ConditionOfUCS/0_Data/3_cleaned_json_dataset"
        get_unique_keys_from_files(json_dir)

    # 识别出viper数据集中的异常流
    if task_name == "find_exc_flow_in_viper":
        viper_path = "E:/Trae_project/ConditionOfUCS/0_Data/3_cleaned_json_dataset/cleaned_viper.json"
        # 加载viper.json文件
        with open(viper_path, 'r', encoding='utf-8') as f:
            viper_data = json.load(f)

        for uc in viper_data:
            uc['Exc. Flow'] = []
            if len(uc['Alt. Flow']) > 1 : # 按照列表存储的
                for i in range(len(uc['Alt. Flow'])-1, -1, -1):
                    if '"Error!"' in uc['Alt. Flow'][i][0]:
                        uc['Exc. Flow'].append([uc['Alt. Flow'][i][0]])
                        del uc['Alt. Flow'][i]
            elif len(uc['Alt. Flow']) == 1: # 按照字符串存储的
                if '"Error!"' in uc['Alt. Flow'][0]:
                    uc['Exc. Flow'].append([uc['Alt. Flow'][0]])
                    uc['Alt. Flow'] = []
            else:
                    print(str(uc['id']) + uc['Name'] + "alt flow 有问题")    
        
        # 保存
        with open(viper_path, 'w', encoding='utf-8') as f:
            json.dump(viper_data, f, ensure_ascii=False, indent=4)


        



