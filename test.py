from transformers import AutoTokenizer

# Define LlamaWithTableInjection subclass (as provided before)
# Define apply_table_llama function (as provided before)
# Define modified_llama_model_forward function (the one accepting **kwargs, provided before)

model_path = "meta-llama/Meta-Llama-3-8B-Instruct" # Your model path
tokenizer_path = model_path

# 1. Load model using the subclass
model = LlamaWithTableInjection.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 2. Attach tokenizer TO THE INSTANCE
model.set_tokenizer(tokenizer) # Crucial for register_table

# 3. Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 4. APPLY THE PATCHES (This connects generate to the modified forward)
apply_table_llama(
    model=model,
    starting_layer=5,
    ending_layer=25,
    entropy_threshold=0.8,
    retracing_ratio=0.05,
    apply_injection=True
)


messages = [{"role": "user", "content": "Who works in Engineering?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Call the OVERRIDDEN generate method on your model instance, passing the ID
output_sequences = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,
    table_id=table_id_1 # Pass the ID here
)

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(generated_text)