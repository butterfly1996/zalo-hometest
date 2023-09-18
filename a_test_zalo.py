import argparse

import torch
import numpy as np
from transformers import GPT2LMHeadModel, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description='test zalo')
    parser.add_argument('--model_name',
                        type=str,
                        default="NlpHUST/gpt2-vietnamese",
                        help='model name')

    parser.add_argument('--prompt',
                        type=str,
                        required=True,
                        help='prompt')

    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=1,
                        help='number of tokens generated')

    parser.add_argument('--no_repeat_ngram_size',
                        type=int,
                        default=0,
                        help='no repeat ngram size')

    parser.add_argument('--generate_type',
                        type=str,
                        # default="greedy_search",
                        default="beam_search",
                        choices=["greedy_search", "beam_search"],
                        help='generate type')

    parser.add_argument('--num_beam',
                        type=int,
                        default=1,
                        help='num beam when generate_type="beam_search"')

    args = parser.parse_args()
    return args


class Generator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    # Split text to token and convert token to id
    def encode(self, text):
        return self.tokenizer.encode(text)

    # Convert id to text
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # Get the id with the greatest probability
    @staticmethod
    def get_top1(prob):
        score, token_id = torch.max(prob, dim=-1)
        return score, token_id

    # Get the top k id with the greatest probability
    @staticmethod
    def get_topk(prob, k=1):
        scores, token_ids = torch.topk(prob, k=k, dim=-1)
        return scores, token_ids

    # Get next token prob, returns the probability of all tokens
    def get_next_token_prob(self, input_ids: torch.Tensor):
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits
        next_token_prob = logits[-1]
        return next_token_prob

    def generate_greedy_search(self, prompt, max_new_tokens=32, no_repeat_ngram_size=0):
        # Implement here
        token_ids = self.encode(prompt)
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        # with torch.no_grad():
        #     greedy_output = self.model.generate(
        #         input_ids,
        #         max_length=max_new_tokens
        #     )
        next_token_prob = self.get_next_token_prob(input_ids=input_ids)
        greedy_output = []
        for i in range(max_new_tokens):
            score, token_id = self.get_top1(next_token_prob)
            greedy_output.append(token_id)
            print("prob: ", next_token_prob, "\ttoken generated: ", score, "\ttoken_id: ", token_id, self.tokenizer.eos_token_id)
            if token_id == self.tokenizer.eos_token_id:
                break
            token_ids.append(token_id)
            next_token_prob = self.get_next_token_prob(input_ids=torch.tensor(token_ids, dtype=torch.long))
            # next_token_prob = self.get_next_token_prob(input_ids=torch.tensor([greedy_output], dtype=torch.long))

        print("Greedy search output: ", greedy_output)
        result = self.decode(torch.tensor(greedy_output, dtype=torch.long))
        print("Generate: ", result)
        return result

    def generate_beam_search(self, prompt, max_new_tokens=32, num_beam=1, no_repeat_ngram_size=0):
        # Implement here
        token_ids = self.encode(prompt)
        bfs_q = [(1, token_ids)]
        beam_search_output = []
        # bfs beam search
        for i in range(max_new_tokens):
            tmp_score, tmp_ids = bfs_q.pop(0)
            next_token_prob = self.get_next_token_prob(input_ids=torch.tensor(tmp_ids, dtype=torch.long))
            scores, token_ids = self.get_topk(next_token_prob, k=num_beam)
            for score, token in zip(scores, token_ids):
                tmp_token_ids = tmp_ids + [token.item()]
                if not no_repeat_ngram(tmp_token_ids[len(token_ids):], no_repeat_ngram_size):
                    print("skip because of repeating ngrams")
                    continue
                if token == self.tokenizer.eos_token_id:
                    beam_search_output.append((tmp_score+np.log2(score), tmp_token_ids))
                    continue
                bfs_q.append((tmp_score+np.log2(score), tmp_token_ids))
        bfs_q.sort(key=lambda x: x[0], reverse=True)
        result = self.decode(torch.tensor(bfs_q[0][1], dtype=torch.long))
        return result


def no_repeat_ngram(sequence, n):
    if n==0:
        return True
    if len(sequence) < n:
        return True
    ngrams = [tuple(x for x in sequence[i:i + n]) for i in range(len(sequence) - n + 1)]
    return len(set(ngrams)) == len(ngrams)


def example(args, generator: Generator):
    token_ids = generator.encode(args.prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long)

    next_token_prob = generator.get_next_token_prob(input_ids=input_ids)
    score, token_id = generator.get_top1(next_token_prob)
    print(score, token_id)
    scores, token_ids = generator.get_topk(next_token_prob, k=3)
    print(scores, token_ids)


def main():
    args = get_args()
    generator = Generator(args.model_name)

    example(args, generator)
    print("Prompt: ", args.prompt)
    print("max new tokens: ", args.max_new_tokens)
    print("no repeat ngram size: ", args.no_repeat_ngram_size)
    if args.generate_type == "greedy_search":
        generator.generate_greedy_search(prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                                         no_repeat_ngram_size=args.no_repeat_ngram_size)
    else:
        generator.generate_beam_search(prompt=args.prompt, max_new_tokens=args.max_new_tokens, num_beam=args.num_beam,
                                       no_repeat_ngram_size=args.no_repeat_ngram_size)


def example_2():
    generator = Generator('NlpHUST/gpt2-vietnamese')
    print(generator.generate_beam_search(prompt="mai có nên đi làm không", max_new_tokens=64, num_beam=4,
                                         no_repeat_ngram_size=1))
    print(generator.generate_greedy_search(prompt="chó và mèo là 2 con vật", max_new_tokens=20))


if __name__ == '__main__':
    main()
