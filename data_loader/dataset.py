import torch
from torch.utils.data import Dataset


class CosmosQADataset(Dataset):
    """
    Returns: {
        "input_ids": [
            [C_ID + (Q_ID + A_ID)]
            ],
        "C_LEN": [],
        "Q_LEN": [],
        "A_LEN": [],
    }
    Where len(C_ID + (Q_ID + A_ID)) < MAX_LEN
    """

    def __init__(self, df, targets, tokenizer, max_len):
        self.df = df
        self.n_choices = 4  # total answer choice
        self.guids = df["id"].to_numpy()
        self.contexts = df["context"].to_numpy()
        self.questions = df["question"].to_numpy()
        self.answers = df[
            ["answer0", "answer1", "answer2", "answer3"]
        ].to_numpy()
        self.answer1s = df["answer0"].to_numpy()
        self.answer2s = df["answer1"].to_numpy()
        self.answer3s = df["answer2"].to_numpy()
        self.answer4s = df["answer3"].to_numpy()
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.contexts)

    @staticmethod
    def _truncate_and_merge(
        tokens_a,
        tokens_b,
        max_length,
        truncate=True,
        add_special_tokens=True,
        padding=True,
    ):
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        max_len_excl_spe_tok = max_length - 3
        i = tokens_a["input_ids"].size(1)
        j = tokens_b["input_ids"].size(1)
        total_padding_required = 0
        if truncate:
            while True:
                total_length = i + j
                if i <= 0 or j <= 0:
                    raise ValueError("Concat Error. One of the String Len is 0")
                if total_length <= max_len_excl_spe_tok:
                    break
                if i > j:
                    i = i - 1
                else:
                    j = j - 1
            total_padding_required = max_length - (total_length + 3)

        # Removing Extra Tokens Here
        for k, _ in tokens_a.items():
            tokens_a[k] = tokens_a[k][:, :i]

        for k, _ in tokens_b.items():
            tokens_b[k] = tokens_b[k][:, :j]

        # concatenate and padding here
        concatenated_tokens = {}
        # zero_t = torch.tensor([[0]])
        one_t = torch.tensor([[1]])
        cls_t = torch.tensor([[101] * add_special_tokens])
        sep_t = torch.tensor([[102] * add_special_tokens])
        pad_tokens = torch.tensor([[0] * total_padding_required * padding])
        for k, _ in tokens_a.items():
            if k == "input_ids":
                concatenated_tokens[k] = torch.cat(
                    (
                        cls_t,
                        tokens_a[k],
                        sep_t,
                        tokens_b[k],
                        sep_t,
                        pad_tokens,
                    ),
                    axis=1,
                )
            if k == "token_type_ids":
                n_segment_1_tokens = i + 2  # 2 extra tokens = 1 CLS + 1 SEP
                n_segment_2_tokens = j + 1  # 1 extra token = 1 SEP
                segment_1_tokens = torch.tensor([[0] * n_segment_1_tokens])
                segment_2_tokens = torch.tensor([[1] * n_segment_2_tokens])
                concatenated_tokens[k] = torch.cat(
                    (segment_1_tokens, segment_2_tokens, pad_tokens), axis=1,
                )
            if k == "attention_mask":
                concatenated_tokens[k] = torch.cat(
                    (
                        one_t,
                        tokens_a[k],
                        one_t,
                        tokens_b[k],
                        one_t,
                        pad_tokens,
                    ),
                    axis=1,
                )
        return concatenated_tokens

    def __getitem__(self, index):
        guid = self.guids[index]
        context = self.contexts[index]
        question = self.questions[index]
        answer1 = self.answer1s[index]
        answer2 = self.answer2s[index]
        answer3 = self.answer3s[index]
        answer4 = self.answer4s[index]
        target = self.targets[index]

        context_encoding = self.tokenizer.encode_plus(
            context,
            add_special_tokens=False,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        question_encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=False,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        # answer1_encoding = self.tokenizer.encode_plus(
        #     answer1,
        #     add_special_tokens=False,
        #     return_token_type_ids=False,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
        # answer2_encoding = self.tokenizer.encode_plus(
        #     answer2,
        #     add_special_tokens=False,
        #     return_token_type_ids=False,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
        # answer3_encoding = self.tokenizer.encode_plus(
        #     answer3,
        #     add_special_tokens=False,
        #     return_token_type_ids=False,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
        # answer4_encoding = self.tokenizer.encode_plus(
        #     answer1,
        #     add_special_tokens=False,
        #     return_token_type_ids=False,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )

        response_object = {}
        response_object["guid"] = guid
        response_object["c_text"] = context
        response_object["q_text"] = question
        response_object["a1_text"] = answer1
        response_object["a2_text"] = answer2
        response_object["a3_text"] = answer3
        response_object["a4_text"] = answer4
        response_object["target"] = target
        response_object["input_ids"] = []
        response_object["token_type_ids"] = []
        response_object["attention_mask"] = []
        response_object["c_len"] = []
        response_object["q_len"] = []
        response_object["a_len"] = []
        for i in range(self.n_choices):
            response_object[f"a{i}_text"] = self.answers[index][i]
            ans_encoding = self.tokenizer.encode_plus(
                response_object[f"a{i}_text"],
                add_special_tokens=False,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            q_a_concat = self._truncate_and_merge(
                question_encoding,
                ans_encoding,
                max_length=self.max_len,
                truncate=False,
                add_special_tokens=False,
                padding=False,
            )
            c_q_a_concat = self._truncate_and_merge(
                context_encoding,
                q_a_concat,
                max_length=self.max_len,
                truncate=True,
                add_special_tokens=True,
                padding=True,
            )
            c_len = context_encoding["input_ids"].size(1)
            q_len = question_encoding["input_ids"].size(1)
            a_len = ans_encoding["input_ids"].size(1)
            response_object["c_len"].append(c_len)
            response_object["q_len"].append(q_len)
            response_object["a_len"].append(a_len)

            assert c_q_a_concat["input_ids"].size(1) == self.max_len
            assert c_q_a_concat["token_type_ids"].size(1) == self.max_len
            assert c_q_a_concat["attention_mask"].size(1) == self.max_len

            response_object["input_ids"].append(
                c_q_a_concat["input_ids"].squeeze()
            )
            response_object["token_type_ids"].append(
                c_q_a_concat["token_type_ids"].squeeze()
            )
            response_object["attention_mask"].append(
                c_q_a_concat["attention_mask"].squeeze()
            )
        response_object["input_ids"] = torch.stack(
            response_object["input_ids"]
        ).type(dtype=torch.long)
        response_object["token_type_ids"] = torch.stack(
            response_object["token_type_ids"]
        ).type(dtype=torch.long)
        response_object["attention_mask"] = torch.stack(
            response_object["attention_mask"]
        ).type(dtype=torch.long)
        response_object["c_len"] = torch.tensor(
            response_object["c_len"], dtype=torch.long
        ).type(dtype=torch.long)
        response_object["q_len"] = torch.tensor(
            response_object["q_len"], dtype=torch.long
        ).type(dtype=torch.long)
        response_object["a_len"] = torch.tensor(
            response_object["a_len"], dtype=torch.long
        ).type(dtype=torch.long)
        return response_object
