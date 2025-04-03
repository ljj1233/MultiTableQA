import torch
<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForCausalLM
from MTable import LlamaMLP, apply_table_llama
=======
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
# from MTable import LlamaMLP, apply_table_llama
>>>>>>> 367f927575cf7ea75191b338e822119fd1ccbff3

class TableLlama:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
<<<<<<< HEAD
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16
        ).to(device)
=======
        
        # 首先创建正确的配置
        config = LlamaConfig.from_pretrained(model_path)
        # 修正 rope_scaling 配置
        config.rope_scaling = {
            "type": "linear",  # 使用线性缩放
            "factor": 2.0     # 缩放因子
        }
        
        # 使用修正后的配置加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.float16
        ).to(device)
        
>>>>>>> 367f927575cf7ea75191b338e822119fd1ccbff3
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.tokenizer = self.tokenizer  # 附加tokenizer到模型实例
        
        # 应用表格注入相关的修改
<<<<<<< HEAD
        apply_table_llama(
            self,
            starting_layer=5,
            ending_layer=25,
            entropy_threshold=0.8,
            retracing_ratio=0.05
        )
=======
        # apply_table_llama(
        #     self,
        #     starting_layer=5,
        #     ending_layer=25,
        #     entropy_threshold=0.8,
        #     retracing_ratio=0.05
        # )
>>>>>>> 367f927575cf7ea75191b338e822119fd1ccbff3

    def chat(self, message: str, table_content: str = None):
        """处理单轮对话"""
        messages = [{"role": "user", "content": message}]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt"
        ).to(self.device)

        # 生成回复
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            table_content=table_content  # 传入表格内容
        )

        response = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        return response

def main():
    # 初始化模型
<<<<<<< HEAD
    model_path = "meta-llama/Llama-2-7b-chat-hf"  # 使用你的实际模型路径
=======
    model_path = "chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct"  # 使用你的实际模型路径
>>>>>>> 367f927575cf7ea75191b338e822119fd1ccbff3
    llm = TableLlama(model_path)
    
    # 示例表格内容
    table_content = """
    | Name  | Department  | Role          |
    |-------|------------|---------------|
    | Alice | Engineering| Lead Engineer |
    | Bob   | Engineering| Developer     |
    | Carol | Sales      | Manager       |
    """
    
    # 测试查询
    question = "Who works in Engineering?"
    response = llm.chat(question, table_content)
    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    main()