import minimal_llama.pref.llama_simple3 as llama_simple3
import torch
import argparse


REFERENCES = {
    "llama-7b": {
        1: [[4250, 3304, 29892, 278, 937, 11715, 29899, 14689, 6673, 310, 278, 3303, 3900, 29892, 471, 6345]],
        2: [[29892, 278, 907, 1061, 310, 1938, 290, 29892, 751, 1296, 29892, 322, 390, 482, 29892, 756]],
        3: [[4250, 3304, 29892, 278, 937, 11715, 29899, 14689, 6673, 310, 278, 3303, 3900, 29892, 471, 6345],
            [29892, 278, 907, 1061, 310, 1938, 290, 29892, 751, 1296, 29892, 322, 390, 482, 29892, 756]],
        4: [[4250, 3304, 29892, 278, 937, 11715, 29899, 14689, 6673, 310, 278, 3303, 3900, 29892, 471, 6345],
            [29892, 278, 907, 1061, 310, 1938, 290, 29892, 751, 1296, 29892, 322, 390, 482, 29892, 756]],
    },
    "llama-2-7b": {
        1: [[4250, 3304, 29892, 278, 6673, 310, 278, 3303, 3900, 29892, 338, 263, 23772, 29889, 13, 1576]],
        2: [[29892, 278, 3748, 29915, 29879, 3275, 27922, 29892, 471, 263, 13524, 310, 278, 3748, 29892, 322]],
        3: [[[4250, 3304, 29892, 278, 6673, 310, 278, 3303, 3900, 29892, 338, 263, 23772, 29889, 13, 1576],
             [29892, 278, 3748, 29915, 29879, 3275, 27922, 29892, 471, 263, 13524, 310, 278, 3748, 29892, 322]]],
        4: [[[4250, 3304, 29892, 278, 6673, 310, 278, 3303, 3900, 29892, 338, 263, 23772, 29889, 13, 1576],
             [29892, 278, 3748, 29915, 29879, 3275, 27922, 29892, 471, 263, 13524, 310, 278, 3748, 29892, 322]]],
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_path", type=str)
    parser.add_argument("--model", type=str, default="llama-7b")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda:0")

    model = llama_simple3.create_model("7b", hf_path=args.hf_path).to(device)

    input_ids1 = torch.LongTensor([[2261,  547]]).to(device)
    input_ids2 = torch.LongTensor([[2259, 1704, 29885, 547]]).to(device)
    input_ids3 = torch.LongTensor([
        [2261,  547, 0, 0],
        [2259, 1704, 29885, 547],
    ]).to(device)

    input_ids4 = torch.LongTensor([
        [2261,  547, 0, 0] + [0] * 100,
        [2259, 1704, 29885, 547] + [0] * 100,
    ]).to(device)

    out1 = model.generate(input_ids1, generation_length=16)
    out2 = model.generate(input_ids2, generation_length=16)
    out3 = model.generate(input_ids3, generation_length=16)
    out4 = model.generate(input_ids4, generation_length=16)

    if (out1.cpu() == torch.LongTensor(REFERENCES[args.model][1])).all():
        print("Test 1 passed")
    else:
        raise ValueError("Test 1 failed")

    if (out2.cpu() == torch.LongTensor(REFERENCES[args.model][2])).all():
        print("Test 2 passed")
    else:
        raise ValueError("Test 2 failed")

    if (out3.cpu() == torch.LongTensor(REFERENCES[args.model][3])).all():
        print("Test 3 passed")
    else:
        raise ValueError("Test 3 failed")

    if (out4.cpu() == torch.LongTensor(REFERENCES[args.model][4])).all():
        print("Test 4 passed")
    else:
        raise ValueError("Test 4 failed")


if __name__ == "__main__":
    main()