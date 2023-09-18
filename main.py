from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast

if __name__ == '__main__':
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b1")
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b1")
    prompt = "It was a dark and stormy night"
    result_length = 50
    inputs = tokenizer(prompt, return_tensors="pt")
    # Beam Search
    generate_text = model.generate(
        inputs["input_ids"],
        max_length=result_length,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )[0]
    print(generate_text)
    print(tokenizer.decode(generate_text))

    # Sampling Top-k + Top-p
    generate_text = model.generate(
        inputs["input_ids"],
        max_length=result_length,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )[0]
    print(generate_text)
    print(tokenizer.decode(generate_text))