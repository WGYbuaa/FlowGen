from imports import datetime
# ...existing code...
import os
import glob
import json

def _parse_md_table_rows(md_path):
    """
    解析单个 md 表格文件，返回行项列表，每项为:
    (dataset, uc_id, af_id, bp_pred, agree_value, disagree_idx)
    兼容不同文件中列对齐和空值。
    """
    rows = []
    with open(md_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # 找到表格起始行（以 '|' 开头且包含至少一个分隔行 ---）
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and i+1 < len(lines) and lines[i+1].strip().startswith("|"):
            # header + separator detection (第二行通常是 ---)
            start = i
            break
    # 跳过 header+sep 两行
    for ln in lines[start+2:]:
        if not ln.strip().startswith("|"):
            continue
        # 分割列，去掉首尾空项
        parts = [p.strip() for p in ln.split("|")[1:-1]]
        if len(parts) < 4:
            continue
        # 我们期望列顺序至少包含前4列: Dataset, UC_ID, AF_ID, BP Predicted ...
        dataset = parts[0]
        uc_id = parts[1] if len(parts) > 1 else ""
        af_id = parts[2] if len(parts) > 2 else ""
        bp_pred = parts[3] if len(parts) > 3 else ""
        agree_val = ""
        disagree_idx = ""
        if len(parts) > 4:
            agree_val = parts[4].strip()
        if len(parts) > 5:
            disagree_idx = parts[5].strip()
        rows.append((dataset, uc_id, af_id, bp_pred, agree_val, disagree_idx))
    return rows

def aggregate_expert_votes(md_paths, out_json_path):
    """
    读取多个专家生成的 md 表（md_paths 列表），对每一条唯一项按 (dataset, UC_ID, AF_ID, BP_pred) 聚合，
    统计 agree_count / disagree_count 并输出到 out_json_path（UTF-8）。
    输出格式: 列表，每项包含 dataset, uc_id, af_id, bp_pred, agree_count, disagree_count, total_votes。
    """
    agg = {}  # key -> {"dataset":.., "uc_id":.., "af_id":.., "bp_pred":.., "agree":int, "disagree":int, "votes":int}
    for md in md_paths:
        if not os.path.isfile(md):
            continue
        for dataset, uc_id, af_id, bp_pred, agree_val, _ in _parse_md_table_rows(md):
            key = (dataset, uc_id, af_id, bp_pred)
            if key not in agg:
                agg[key] = {"dataset": dataset, "uc_id": uc_id, "af_id": af_id, "bp_pred": bp_pred,
                            "agree_count": 0, "disagree_count": 0, "votes": 0}
            # 识别 1 为同意，0 为不同意；其它（空或非 0/1）视为不计票
            v = agree_val.strip()
            if v == "1":
                agg[key]["agree_count"] += 1
                agg[key]["votes"] += 1
            elif v == "0":
                agg[key]["disagree_count"] += 1
                agg[key]["votes"] += 1
            else:
                # 不计入 votes，但保留条目（可按需修改）
                pass

    # 转换为列表并写入 json
    out_list = []
    for k, v in agg.items():
        out_list.append(v)
    # 按 dataset, uc_id, af_id 排序（便于查看）
    out_list.sort(key=lambda x: (x["dataset"], int(x["uc_id"]) if x["uc_id"].isdigit() else x["uc_id"], int(x["af_id"]) if x["af_id"].isdigit() else x["af_id"]))
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    return out_json_path

def write_aggregated_votes_md(agg_json_path, out_md_path):
    """
    从 aggregate_expert_votes 生成的 JSON 文件读取统计结果，写入对齐的 Markdown 表格文件。
    输出列: | Dataset | UC_ID | AF_ID | BP Predicted by LLM | Agree | Disagree | Votes |
    """
    with open(agg_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    headers = ["Dataset", "UC_ID", "AF_ID", "BP Predicted by LLM", "Agree", "Disagree", "Votes"]
    rows = []
    for item in data:
        rows.append([
            str(item.get("dataset", "")),
            str(item.get("uc_id", "")),
            str(item.get("af_id", "")),
            str(item.get("bp_pred", "")),
            str(item.get("agree_count", 0)),
            str(item.get("disagree_count", 0)),
            str(item.get("votes", 0))
        ])

    # 计算每列最大宽度（包含表头）
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    col_widths = [max(len(str(cell)) for cell in col) for col in cols]

    # 生成表头和分隔行（对齐）
    header_line = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

    # 生成行字符串，统一对齐
    row_lines = []
    for r in rows:
        line = "| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(r))) + " |"
        row_lines.append(line)

    # 写入 md 文件
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        for line in row_lines:
            f.write(line + "\n")

    return out_md_path
# ...existing code...
import re
import math

def _parse_aggregated_md(md_path):
    """
    解析 bp_review_aggregated.md，返回行列表：
    [(dataset, uc_id, af_id, bp_pred, agree_count:int, disagree_count:int, final_index), ...]
    """
    rows = []
    with open(md_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # 找到表格 header（第一行以 '|' 且后面有分隔行）
    start = None
    for i in range(len(lines)-1):
        if lines[i].strip().startswith("|") and lines[i+1].strip().startswith("|"):
            start = i
            break
    if start is None:
        return rows
    for ln in lines[start+2:]:
        if not ln.strip().startswith("|"):
            continue
        parts = [p.strip() for p in ln.split("|")[1:-1]]
        if len(parts) < 6:
            continue
        dataset = parts[0]
        uc_id = parts[1]
        af_id = parts[2]
        bp_pred = parts[3]
        # 从字符串中提取数字，若无则为 0
        def num_from_cell(s):
            if not s:
                return 0
            m = re.search(r"(-?\d+)", s)
            return int(m.group(1)) if m else 0
        agree = num_from_cell(parts[4]) if len(parts) > 4 else 0
        disagree = num_from_cell(parts[5]) if len(parts) > 5 else 0
        final_index = parts[6] if len(parts) > 6 else ""
        rows.append((dataset, uc_id, af_id, bp_pred, agree, disagree, final_index))
    return rows

def fleiss_kappa_from_counts(counts, n_raters=5):
    """
    counts: list of tuples (agree_count, disagree_count) per item.
    返回 Fleiss' Kappa（float），若不可计算返回 None。
    """
    N = len(counts)
    if N == 0:
        return None
    n = n_raters
    # 验证每项总和等于 n
    for a, b in counts:
        if (a + b) != n:
            raise ValueError(f"某项投票总和 != {n} (found {a}+{b})")
    # 总体类别比例
    total_agree = sum(a for a, _ in counts)
    total_disagree = sum(b for _, b in counts)
    p_agree = total_agree / (N * n)
    p_disagree = total_disagree / (N * n)
    # 每项一致性 P_i
    P_i_list = []
    for a, b in counts:
        P_i = ((a*a + b*b) - n) / (n * (n - 1))
        P_i_list.append(P_i)
    Pbar = sum(P_i_list) / N
    Pe = p_agree * p_agree + p_disagree * p_disagree
    denom = 1 - Pe
    if math.isclose(denom, 0.0):
        return None
    kappa = (Pbar - Pe) / denom
    return kappa

def compute_fleiss_kappa_for_dataset_from_md(md_path, dataset_name, n_raters=5):
    """
    从聚合的 md 文件中计算指定 dataset 的 Fleiss' Kappa。
    返回 (kappa, N_items, counts_list)。
    """
    rows = _parse_aggregated_md(md_path)
    # 过滤出目标数据集
    filtered = [(a, b) for (ds, _, _, _, a, b, _) in rows if ds.strip() == dataset_name]
    if not filtered:
        return (None, 0, [])
    # 验证并计算
    # 检查每一项 a+b == n_raters
    for a, b in filtered:
        if (a + b) != n_raters:
            raise ValueError(f"投票和不等于 {n_raters}：found {a}+{b}")
    print(f'{len(filtered)} items: {filtered}')
    kappa = fleiss_kappa_from_counts(filtered, n_raters=n_raters)
    return (kappa, len(filtered), filtered)



if __name__ == '__main__':
    task_name = 'compute_Fleiss_kappa'
    print(f'*** task_name: {task_name} , Starting time: {datetime.now()}  !!!!! ***')

    if task_name =='statistic_Exp_dataset':
        md_files = sorted(glob.glob(r"E:/Trae_project/ConditionOfUCS/0_Data/5_branching_point/1_gpt_added_bp/BP_REVIEW/bp_review_Exp_*.md"))
        agg_json = "E:/Trae_project/ConditionOfUCS/0_Data/5_branching_point/1_gpt_added_bp/BP_REVIEW/bp_review_aggregated.json"
        aggregate_expert_votes(md_files, agg_json)
        # 生成对齐的 md 汇总表
        agg_md = "E:/Trae_project/ConditionOfUCS/0_Data/5_branching_point/1_gpt_added_bp/BP_REVIEW/bp_review_aggregated.md"
        write_aggregated_votes_md(agg_json, agg_md)
        print(f'Generated aggregated json: {agg_json}')
        print(f'Generated aggregated md: {agg_md}')

    elif task_name == 'compute_Fleiss_kappa':
        md = "E:/Trae_project/ConditionOfUCS/0_Data/5_branching_point/1_gpt_added_bp/BP_REVIEW/bp_review_aggregated.md"
        for dataset in ["keepass","gamma j","0000 - inventory","hats","pnnl",  "viper","2009 - inventory 2.0","iTrust"]:
            kappa, N, counts = compute_fleiss_kappa_for_dataset_from_md(md, dataset)
            print(f" {dataset} Fleiss' Kappa:", kappa, "N_items:", N)
    print(f'*** task_name: {task_name} Finish! {datetime.now()}! *** ')