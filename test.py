import re
import pandas as pd
from io import StringIO

def parse_markdown_table(content_lines):
    """解析Markdown格式的表格"""
    if not content_lines:
        return None
        
    # 将列表转换为字符串
    content = '\n'.join(content_lines)
    try:
        # 尝试直接作为CSV解析
        return pd.read_csv(StringIO(content))
    except:
        return None

def process_section(section_content, main_title=None):
    """处理单个章节的表格"""
    if not section_content:
        return []
            
    tables = []
    # 匹配子标题
    sub_sections = re.split(r'(?m)^(##\s*[\w_]+)', section_content)
    current_table_name = main_title if main_title else "default_table"
    content_buffer = []

    print(f"处理章节，主标题: {main_title}")

    for sub_section in sub_sections:
        sub_section = sub_section.strip()
        if not sub_section:
            continue

        if sub_section.startswith("##"):
            # 处理之前的内容
            if content_buffer:
                print(f"尝试解析表格内容: {current_table_name}")
                df = parse_markdown_table(content_buffer)
                if df is not None:
                    tables.append({"name": current_table_name, "df": df})
                    print(f"成功解析表格: {current_table_name}")
                content_buffer = []
            current_table_name = sub_section.lstrip('#').strip()
        else:
            # 保留非空且不以#开头的行
            valid_lines = [line for line in sub_section.split('\n') 
                         if line.strip() and not line.strip().startswith('#')]
            if valid_lines:
                content_buffer.extend(valid_lines)

    # 处理最后一个表格
    if content_buffer:
        print(f"处理最后一个表格: {current_table_name}")
        df = parse_markdown_table(content_buffer)
        if df is not None:
            tables.append({"name": current_table_name, "df": df})
            print(f"成功解析最后一个表格: {current_table_name}")

    return tables

def process_table_content(db_str):
    """处理表格内容"""
    if not db_str or not isinstance(db_str, str):
        print(f"警告: 输入的db_str为空或类型不正确: {type(db_str)}")
        return None
            
    parsed_tables = []
    
    # 调试信息：打印输入数据
    print("输入数据:")
    print(db_str)
    
    # 将文本按行分割
    lines = db_str.split('\n')
    main_sections = []
    current_section = []
    current_title = None
    
    # 逐行处理
    for line in lines:
        line = line.strip()
        if line.startswith('#') and not line.startswith('##'):
            # 如果遇到新的主标题，保存之前的部分
            if current_title:
                main_sections.append((current_title, '\n'.join(current_section)))
            # 开始新的部分
            current_title = line.lstrip('#').strip()
            current_section = []
        else:
            current_section.append(line)
    
    # 处理最后一个部分
    if current_title:
        main_sections.append((current_title, '\n'.join(current_section)))
    
    # 调试信息：打印分割结果
    print("\n分割后的部分:")
    for i, (title, content) in enumerate(main_sections):
        print(f"Section {i}: 标题='{title}', 内容前50个字符='{content[:50]}'")
    
    # 处理每个部分
    for title, content in main_sections:
        if title and content:
            print(f"\n处理主标题: {title}")
            tables = process_section(content, title)
            if tables:
                parsed_tables.extend(tables)
                print(f"成功解析表格，标题: {title}")
            else:
                print(f"警告: 未能解析表格内容，标题: {title}")

    return parsed_tables

def main():
    # 测试数据
    test_data = """
    #airline

    ## Air_Carriers
    Code,Description
    19393,Southwest Airlines Co.: WN
    19687,Horizon Air: QX
    19790,Delta Air Lines Inc.: DL

    ## Airports
    Code,Description
    ABQ,"Albuquerque, NM"
    ATL,"Atlanta, GA"
    BUF,"Buffalo, NY"
    """
    
    # 移除开头的空格
    test_data = '\n'.join(line.lstrip() for line in test_data.split('\n'))
    
    print("=== 开始处理表格 ===")
    parsed_tables = process_table_content(test_data)

    # 打印结果
    if parsed_tables:
        print("\n=== 解析结果 ===")
        for table in parsed_tables:
            print(f"\n表格名称: {table['name']}")
            print(table['df'])
    else:
        print("未能成功解析任何表格")

if __name__ == "__main__":
    main()