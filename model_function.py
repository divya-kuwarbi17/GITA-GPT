from transformers import AutoModelForCausalLM,AutoTokenizer
import torch


model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    use_auth_token="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"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",use_auth_token="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")

def generate_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    attention_mask = torch.ones(input_ids.shape)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens  = 500,
        temperature= 0.7,
        # top_p= 0.7,
        top_k = 100,
        # num_beams = 3,
        do_sample=True,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_text)

    # Remove Prompt Echo from Generated Text
    # cleaned_output_text = output_text[len(input_text):]
    # cleaned_output_text = cleaned_output_text.split("Question :")[0].strip()
    cleaned_output_text = output_text
    torch.cuda.empty_cache()
    return cleaned_output_text



