import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TableLlama:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, max_seq_len: int, max_batch_size: int, device: str = "cuda:0"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.device = device
        self.table_embeddings = {}  # 用于存储表格嵌入的字典

    @classmethod
    def build(cls, model_path: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int, device: str = "cuda:0"):
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return cls(model=model, tokenizer=tokenizer, max_seq_len=max_seq_len, max_batch_size=max_batch_size, device=device)

    def process_table(self, table_id: str, table_content: str):
        """
        将表格内容转换为嵌入并存储。
        """
        table_tokens = self.tokenizer.encode(table_content, bos=True, eos=False)
        table_tensor = torch.tensor(table_tokens, dtype=torch.long).to(self.device)

        # 这里简化了嵌入过程，实际应用中可能需要更复杂的嵌入方法
        # 例如，使用模型的嵌入层或预训练的嵌入模型
        with torch.no_grad():
            table_embedding = self.model.get_input_embeddings()(table_tensor).mean(dim=0)
        
        self.table_embeddings[table_id] = table_embedding

    def generate(
        self,
        prompt_tokens: list,
        table_id: str = None,  # 添加表格ID
        max_gen_len: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ):
        """
        生成文本。如果提供了表格ID，则将表格嵌入添加到输入中。
        """
        if table_id and table_id in self.table_embeddings:
            table_embedding = self.table_embeddings[table_id]

            # 将表格嵌入添加到提示的开始或结尾
            # 这里简化处理,实际可以拼接，或者用prompt模版。
            # 这里，我们假设表格嵌入是单个向量，需要将其转换为tokens
            # 例如将向量添加到提示的末尾
            # 注意：实际操作中，可能需要将嵌入转换为可以被模型处理的格式
            # 这里使用一个简单的占位符，实际需要更复杂的方法
            # 简化版：将嵌入向量直接转换为token列表，实际需要更复杂的处理方式。
            # 由于嵌入维度和token id不一致，这里直接使用一个padding id来占位，实际需要更复杂的处理方法
            padding_id = self.tokenizer.pad_token_id
            table_tokens = [padding_id] * table_embedding.shape[0]

            for tokens in prompt_tokens:
                tokens.extend(table_tokens)

        pad_id = self.tokenizer.pad_token_id
        tokens = torch.full((len(prompt_tokens), self.max_seq_len), pad_id, dtype=torch.long).to(self.device)

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long).to(self.device)

        output = self.model.generate(
            tokens,
            max_length=max_gen_len + len(prompt_tokens[0]),
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return generated_text

# 载入模型路径
model_path = "llama/chanage_model/Meta-Llama-3___1-8B-Instruct"  # 替换为您实际的模型路径
tokenizer_path = "llama/chanage_model/tokenizer"  # 替换为实际的tokenizer路径
device = "cuda:0"  # 使用GPU进行推理
max_seq_len = 1024
max_batch_size = 8

# 使用Llama类的build方法加载模型
llama = TableLlama.build(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
    device=device,
)

# 示例表格和问题
table_id = "table1"
table_content = "Name, Age, City\nAlice, 25, New York\nBob, 30, London"
llama.process_table(table_id, table_content)  # 处理表格

messages = [
    {"role": "user", "content": "What is Alice's age?"}
]

input_ids = [llama.tokenizer.encode(msg["content"], bos=True, eos=False) for msg in messages]

# 使用Llama模型生成文本，并提供表格ID
start_time = time.time()

generation_text = llama.generate(
    prompt_tokens=input_ids,
    table_id=table_id,  # 添加表格ID
    max_gen_len=256,
    temperature=0.6,
    top_p=0.9,
)

end_time = time.time()

# 输出生成的文本
print("Generated Text:", generation_text[0])
print(f"Time taken: {end_time - start_time} seconds")