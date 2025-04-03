import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig,GenerationConfig
from multi_Table import apply_table_llama,apply_table_function

class TableLlama:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device = device
        apply_table_function()

        # 加载模型配置
        self.config = LlamaConfig.from_pretrained(model_path)
        self.config.rope_scaling = {
            "type": "linear",
            "factor": 2.0
        }
        
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=self.config,
            torch_dtype=torch.float16
        ).to(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token   
        
        # 初始化生成配置
        self.generation_config = GenerationConfig(
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # 应用自定义修改
        apply_table_llama(
            self.model,
            starting_layer=5,
            ending_layer=7,
            entropy_threshold=0.5,
            retracing_ratio=0.05
        )

    def chat(self, message: str, table_token: str = None):
        messages = [{"role": "user", "content": message}]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,  
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            table_token=table_token,
            tokenizer=self.tokenizer
        )

        response = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return response

def main():
    # 初始化模型
    model_path = "chanage_model/LLM-Research/Meta-Llama-3.1-8B-Instruct"  
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
    question = f"{table_content}\nWho works in Engineering?"
    response = llm.chat(question, table_content)
    print("Question:", question)
    print("Response:", response)

if __name__ == "__main__":
    main()