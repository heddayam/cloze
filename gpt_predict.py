
import pandas as pd
import pdb


def generate_next_word(text):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    output = generator(text, max_length=len(text.split()) + 1, temperature=0.0001)
    print(output)
    return output


# if __name__ == "__main__":
#     pass
