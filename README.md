# MultiTableQA

一个基于表格增强的多表问答系统，能够处理复杂的多表关联查询问题。

## 项目结构

- `multi_Table.py`: 实现表格增强功能的核心逻辑
- `open_model_evaluator.py`: 开源模型评估工具，用于评估本地模型性能
- `api_model_evaluator.py`: API模型评估工具，用于评估OpenAI/Claude等API模型性能
- `eval/`: 评估工具包，包含评估函数和工具
- `Utils/jsTool.py`: 数据加载和结果记录工具

## 运行流程

1. **初始化阶段**
   - 加载预训练的 Llama 模型
   - 应用表格增强功能
   - 配置提示类型（原始/思维链/表格增强）

2. **数据加载阶段**
   - 加载表格、问题和选项数据
   - 准备评估数据集

3. **评估循环**
   - 遍历数据集中的每个问题
   - 将表格序列化为文本格式
   - 构建适合的提示模板

4. **推理阶段**
   - 对于表格增强模式，传递表格内容给模型
   - 在生成过程中，当模型遇到高熵状态时触发表格特征注入
   - 表格特征与模型隐藏状态融合，增强表格理解能力

5. **结果记录**
   - 提取模型回答中的最终答案
   - 评估正确性并保存结果

## 使用方法

### 1. 评估不同模型

#### GLM模型评估
```bash
# GLM默认模式
python table_qa_evaluator_glm.py \
  --model_path GLM/ \
  --db_root dataset/task/scaledDB \
  --task_path dataset/task/tableQA/dataset.sqlite \
  --result_path dataset/results/tableQA/glm_default.sqlite \
  --dataset TableQA \
  --scale 8k \
  --prompt_type default \
  --db_limit 5 \
  --sample_limit 1 \
  --question_limit 5

# GLM表格增强模式
python table_qa_evaluator_glm.py \
  --model_path GLM/ \
  --db_root dataset/task/scaledDB \
  --task_path dataset/task/tableQA/dataset.sqlite \
  --result_path dataset/results/tableQA/glm_retrace.sqlite \
  --dataset TableQA \
  --scale 8k \
  --prompt_type retrace_table \
  --markdown \
  --db_limit 5 \
  --sample_limit 1 \
  --question_limit 5
```

### 2. 使用表格问答评估器
使用 table_qa_evaluator.py 评估模型在表格问答任务上的性能：

```bash


# 默认提示模式评估
python ./table_qa_evaluator.py \
  --model_path chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --db_root dataset/task/scaledDB \
  --task_path dataset/task/tableQA/dataset.sqlite \
  --result_path dataset/results/tableQA/llama3_default.sqlite \
  --dataset TableQA \
  --scale 8k \
  --db_limit 10 \
  --sample_limit 2 \
  --question_limit 1 \
  --time_sleep 10 \
   --multi_gpu
```

```bash 
# 思维链(CoT)提示模式评估
python ./table_qa_evaluator.py \
  --model_path chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --db_root dataset/task/scaledDB \
  --task_path dataset/task/tableQA/dataset.sqlite \
  --result_path dataset/results/tableQA/llama3_cot.sqlite \
  --dataset TableQA \
  --scale 16k \
  --prompt_type cot \
  --db_limit 10 \
  --sample_limit 2 \
  --question_limit 1 \
  --time_sleep 10 \
   --multi_gpu
```

```bash 
# 表格增强提示模式评估（使用Markdown格式）
python ./table_qa_evaluator_glm.py \
  --model_path GLM/ \
  --db_root dataset/task/scaledDB \
  --task_path dataset/task/tableQA/dataset.sqlite \
  --result_path dataset/results/tableQA/glm_retrace_7db.sqlite \
  --dataset TableQA \
  --scale 8k \
  --prompt_type retrace_table \
  --markdown \
  --db_limit 10 \
  --sample_limit 2 \
  --question_limit 1 \
  --time_sleep 10 \
   --multi_gpu
```

```bash 
# 调用API模型
python llama_demo.py  --db_root "dataset/task/scaledDB" --task_path "dataset/task/tableQA/dataset.sqlite" --result_path "dataset/results/tableQA/api_default.sqlite" --dataset "TableQA" --scale "8k" "16k" --prompt_type "default" --db_limit 5 --sample_limit 1 --question_limit 3
```
