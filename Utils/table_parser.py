import pandas as pd

def parse_markdown_table(table_lines):
    """
    尝试解析简单的Markdown表格。
    
    参数:
        table_lines (list): 包含Markdown表格行的列表
        
    返回:
        pd.DataFrame 或 None: 解析成功返回DataFrame，否则返回None
    """
    import re
    header = []
    data = []
    separator_found = False
    header_parsed = False

    for i, line in enumerate(table_lines):
        line = line.strip()
        if not line:
            continue

        # 查找分隔符行 (例如, |---|---|)
        if re.match(r'^[|\s]*[-:|]+[|\s]*$', line):
            if header_parsed: # 分隔符必须在表头之后
                separator_found = True
            continue # 从数据中跳过分隔符行

        # 如果不是分隔符，解析为表头或数据
        if '|' in line:
            cells = [cell.strip() for cell in line.strip('|').split('|')]
            if not header_parsed and not separator_found:
                header = cells
                header_parsed = True
            elif header_parsed and separator_found:
                # 确保行的单元格数量与表头相同（或处理不匹配）
                if len(cells) == len(header):
                   data.append(cells)
                else:
                   print(f"警告: Markdown表格中的行数据不匹配。表头有{len(header)}列，行有{len(cells)}列。跳过行: {line}")

    if header and data:
        try:
            return pd.DataFrame(data, columns=header)
        except Exception as e:
            print(f"警告: 从解析的Markdown创建DataFrame失败: {e}")
            return None
    return None