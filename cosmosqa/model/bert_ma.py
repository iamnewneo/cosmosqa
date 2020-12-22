import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel


def print_debug(name, val):
    # print(f"{name}: {val}")
    pass


# BERT with multiway attention
class BertMultiwayMatch(BertPreTrainedModel):
    def __init__(self, config, num_choices=4):
        super(BertMultiwayMatch, self).__init__(config)
        # print(f"Config: {config}")
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_trans = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_fuse_p = nn.Linear(
            config.hidden_size * 2, config.hidden_size
        )
        self.linear_fuse_q = nn.Linear(
            config.hidden_size * 2, config.hidden_size
        )
        self.linear_fuse_a = nn.Linear(
            config.hidden_size * 2, config.hidden_size
        )
        self.classifier = nn.Linear(config.hidden_size * 3, 1)
        self.apply(self._init_weights)

    @staticmethod
    def get_seperate_sequence(sequence_tensor, doc_len, ques_len, option_len):
        doc_seq_output = sequence_tensor.new(sequence_tensor.size()).zero_()
        ques_seq_output = sequence_tensor.new(sequence_tensor.size()).zero_()
        option_seq_output = sequence_tensor.new(sequence_tensor.size()).zero_()
        ques_option_seq_output = sequence_tensor.new(
            sequence_tensor.size()
        ).zero_()
        doc_option_seq_output = sequence_tensor.new(
            sequence_tensor.size()
        ).zero_()
        doc_ques_seq_output = sequence_tensor.new(
            sequence_tensor.size()
        ).zero_()

        for i in range(doc_len.size(0)):
            doc_seq_output[i, : doc_len[i]] = sequence_tensor[
                i, 1 : 1 + doc_len[i]
            ]
            ques_seq_output[i, : ques_len[i]] = sequence_tensor[
                i, doc_len[i] + 2 : doc_len[i] + ques_len[i] + 2
            ]
            option_seq_output[i, : option_len[i]] = sequence_tensor[
                i,
                doc_len[i]
                + ques_len[i]
                + 2 : doc_len[i]
                + ques_len[i]
                + option_len[i]
                + 2,
            ]

            doc_ques_seq_output[i, : doc_len[i]] = sequence_tensor[
                i, 1 : 1 + doc_len[i]
            ]
            doc_ques_seq_output[
                i, doc_len[i] : doc_len[i] + ques_len[i]
            ] = sequence_tensor[
                i, doc_len[i] + 2 : doc_len[i] + ques_len[i] + 2
            ]

            doc_option_seq_output[i, : doc_len[i]] = sequence_tensor[
                i, 1 : 1 + doc_len[i]
            ]
            doc_option_seq_output[
                i, doc_len[i] : doc_len[i] + option_len[i]
            ] = sequence_tensor[
                i,
                doc_len[i]
                + ques_len[i]
                + 2 : doc_len[i]
                + ques_len[i]
                + option_len[i]
                + 2,
            ]

            ques_option_seq_output[i, : ques_len[i]] = sequence_tensor[
                i, doc_len[i] + 2 : doc_len[i] + ques_len[i] + 2
            ]
            ques_option_seq_output[
                i, ques_len[i] : ques_len[i] + option_len[i]
            ] = sequence_tensor[
                i,
                doc_len[i]
                + ques_len[i]
                + 2 : doc_len[i]
                + ques_len[i]
                + option_len[i]
                + 2,
            ]

        return (
            doc_seq_output,
            ques_seq_output,
            option_seq_output,
            doc_ques_seq_output,
            doc_option_seq_output,
            ques_option_seq_output,
        )

    def matching(
        self,
        passage_encoded,
        question_encoded,
        passage_attention_mask,
        question_attention_mask,
    ):
        # linear trans the other way
        passage_encoded_trans = self.linear_trans(passage_encoded)
        question_encoded_trans = self.linear_trans(question_encoded)
        p2q_scores = torch.matmul(
            passage_encoded_trans, question_encoded_trans.transpose(2, 1)
        )

        print_debug("\n", "")
        print_debug("passage_encoded_trans", passage_encoded_trans.size())
        print_debug("question_encoded_trans", question_encoded_trans.size())
        print_debug("p2q_scores", p2q_scores.size())
        print_debug("\n", "")

        # fp16 compatibility
        merged_attention_mask = (
            passage_attention_mask.unsqueeze(2)
            .float()
            .matmul(question_attention_mask.unsqueeze(1).float())
        )
        print_debug("\n", "")
        print_debug("merged_attention_mask", merged_attention_mask.size())
        print_debug("\n", "")
        merged_attention_mask = merged_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        merged_attention_mask = (1.0 - merged_attention_mask) * -10000.0

        p2q_scores_ = p2q_scores + merged_attention_mask
        # Normalize the attention scores to probabilities.
        p2q_w = nn.Softmax(dim=-1)(p2q_scores_)
        p2q_w_ = nn.Softmax(dim=1)(p2q_scores_)
        print_debug("\n", "")
        print_debug("p2q_scores_", p2q_scores_.size())
        print_debug("p2q_w", p2q_w.size())
        print_debug("p2q_w_", p2q_w_.size())
        print_debug("\n", "")

        # question attentive passage representation
        mp = torch.matmul(p2q_w, question_encoded)
        # passage attentive question representation
        mq = torch.matmul(p2q_w_.transpose(2, 1), passage_encoded)

        return mp, mq

    # sub and multiply
    def fusing_mlp(
        self,
        passage_encoded,
        mp_q,
        mp_a,
        mp_qa,
        question_encoded,
        mq_p,
        mq_a,
        mq_pa,
        answers_encoded,
        ma_p,
        ma_q,
        ma_pq,
    ):
        new_mp_q = torch.cat(
            [mp_q - passage_encoded, mp_q * passage_encoded], 2
        )
        new_mp_a = torch.cat(
            [mp_a - passage_encoded, mp_a * passage_encoded], 2
        )
        new_mp_qa = torch.cat(
            [mp_qa - passage_encoded, mp_qa * passage_encoded], 2
        )
        new_mq_p = torch.cat(
            [mq_p - question_encoded, mq_p * question_encoded], 2
        )
        new_mq_a = torch.cat(
            [mq_a - question_encoded, mq_a * question_encoded], 2
        )
        new_mq_pa = torch.cat(
            [mq_pa - question_encoded, mq_pa * question_encoded], 2
        )
        new_ma_p = torch.cat(
            [ma_p - answers_encoded, ma_p * answers_encoded], 2
        )
        new_ma_q = torch.cat(
            [ma_q - answers_encoded, ma_q * answers_encoded], 2
        )
        new_ma_pq = torch.cat(
            [ma_pq - answers_encoded, ma_pq * answers_encoded], 2
        )

        new_mp = torch.cat([new_mp_q, new_mp_a, new_mp_qa], 1)
        new_mq = torch.cat([new_mq_p, new_mq_a, new_mq_pa], 1)
        new_ma = torch.cat([new_ma_p, new_ma_q, new_ma_pq], 1)

        # use separate linear functions
        new_mp_ = F.relu(self.linear_fuse_p(new_mp))
        new_mq_ = F.relu(self.linear_fuse_q(new_mq))
        new_ma_ = F.relu(self.linear_fuse_a(new_ma))

        new_p_max, new_p_idx = torch.max(new_mp_, 1)
        new_q_max, new_q_idx = torch.max(new_mq_, 1)
        new_a_max, new_a_idx = torch.max(new_ma_, 1)

        new_p_max_ = new_p_max.view(-1, self.num_choices, new_p_max.size(1))
        new_q_max_ = new_q_max.view(-1, self.num_choices, new_q_max.size(1))
        new_a_max_ = new_a_max.view(-1, self.num_choices, new_a_max.size(1))

        c = torch.cat([new_p_max_, new_q_max_, new_a_max_], 2)

        return c

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        doc_len=None,
        ques_len=None,
        option_len=None,
        labels=None,
    ):
        #         flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        #         doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        #         ques_len = ques_len.view(
        #             -1, ques_len.size(0) * ques_len.size(1)
        #         ).squeeze()
        #         option_len = option_len.view(
        #             -1, option_len.size(0) * option_len.size(1)
        #         ).squeeze()
        #         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        #         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        #         sequence_output, pooled_output = self.bert.forward(
        #             flat_input_ids,
        #             flat_token_type_ids,
        #             flat_attention_mask,
        #             output_all_encoded_layers=False,
        #         )

        print_debug("\n", "")
        print_debug("input_ids", input_ids.size())
        print_debug("token_type_ids", token_type_ids.size())
        print_debug("attention_mask", attention_mask.size())
        print_debug("\n", "")

        print_debug("\n", "")
        print_debug("doc_len", doc_len.size())
        print_debug("ques_len", ques_len.size())
        print_debug("option_len", option_len.size())
        print_debug("\n", "")

        print_debug("labels", labels.size())
        print_debug("\n", "")

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        print_debug("\n", "")
        print_debug("flat_input_ids", flat_input_ids.size())
        print_debug("flat_token_type_ids", flat_token_type_ids.size())
        print_debug("flat_attention_mask", flat_attention_mask.size())
        print_debug("\n", "")

        doc_len = doc_len.view(-1, doc_len.size(0) * doc_len.size(1)).squeeze()
        ques_len = ques_len.view(
            -1, ques_len.size(0) * ques_len.size(1)
        ).squeeze()
        option_len = option_len.view(
            -1, option_len.size(0) * option_len.size(1)
        ).squeeze()

        print_debug("\n", "")
        print_debug("doc_len_squeezed", doc_len.size())
        print_debug("ques_len_squeezed", ques_len.size())
        print_debug("option_len_squeezed", option_len.size())
        print_debug("\n", "")

        sequence_output, pooled_output = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            return_dict=False,
        )
        print_debug("\n", "")
        print_debug("pooled_output", pooled_output.size())
        print_debug("sequence_output", sequence_output.size())
        print_debug("\n", "")

        (
            passage_encoded,
            question_encoded,
            answers_encoded,
            passage_question_encoded,
            passage_answer_encoded,
            question_answer_encoded,
        ) = self.get_seperate_sequence(
            sequence_tensor=sequence_output,
            doc_len=doc_len,
            ques_len=ques_len,
            option_len=option_len,
        )
        print_debug("\n", "")
        print_debug("passage_encoded", passage_encoded.size())
        print_debug("question_encoded", question_encoded.size())
        print_debug("answers_encoded", answers_encoded.size())
        print_debug("passage_question_encoded", passage_question_encoded.size())
        print_debug("passage_answer_encoded", passage_answer_encoded.size())
        print_debug("question_answer_encoded", question_answer_encoded.size())
        print_debug("\n", "")

        (
            passage_attention_mask,
            question_attention_mask,
            answers_attention_mask,
            passage_question_attention_mask,
            passage_answer_attention_mask,
            question_answer_attention_mask,
        ) = self.get_seperate_sequence(
            sequence_tensor=flat_attention_mask,
            doc_len=doc_len,
            ques_len=ques_len,
            option_len=option_len,
        )
        print_debug("\n", "")
        print_debug("passage_attention_mask", passage_attention_mask.size())
        print_debug("question_attention_mask", question_attention_mask.size())
        print_debug("answers_attention_mask", answers_attention_mask.size())
        print_debug(
            "passage_question_attention_mask",
            passage_question_attention_mask.size(),
        )
        print_debug(
            "passage_answer_attention_mask",
            passage_answer_attention_mask.size(),
        )
        print_debug(
            "question_answer_attention_mask",
            question_answer_attention_mask.size(),
        )
        print_debug("\n", "")

        # matching layer
        mp_q, mq_p = self.matching(
            passage_encoded,
            question_encoded,
            passage_attention_mask,
            question_attention_mask,
        )
        print_debug("\n", "")
        print_debug("mp_q", mp_q.size())
        print_debug("mq_p", mq_p.size())
        print_debug("\n", "")

        mp_a, ma_p = self.matching(
            passage_encoded,
            answers_encoded,
            passage_attention_mask,
            answers_attention_mask,
        )
        print_debug("\n", "")
        print_debug("mp_a", mp_a.size())
        print_debug("ma_p", ma_p.size())
        print_debug("\n", "")

        mp_qa, mqa_p = self.matching(
            passage_encoded,
            question_answer_encoded,
            passage_attention_mask,
            question_answer_attention_mask,
        )
        print_debug("\n", "")
        print_debug("mp_qa", mp_qa.size())
        print_debug("mqa_p", mqa_p.size())
        print_debug("\n", "")

        mq_a, ma_q = self.matching(
            question_encoded,
            answers_encoded,
            question_attention_mask,
            answers_attention_mask,
        )
        mq_pa, mpa_q = self.matching(
            question_encoded,
            passage_answer_encoded,
            question_attention_mask,
            passage_answer_attention_mask,
        )
        ma_pq, mpq_a = self.matching(
            answers_encoded,
            passage_question_encoded,
            answers_attention_mask,
            passage_question_attention_mask,
        )

        # MLP fuse
        c = self.fusing_mlp(
            passage_encoded,
            mp_q,
            mp_a,
            mp_qa,
            question_encoded,
            mq_p,
            mq_a,
            mq_pa,
            answers_encoded,
            ma_p,
            ma_q,
            ma_pq,
        )
        c_ = c.view(-1, c.size(2))
        logits = self.classifier(c_)
        reshaped_logits = logits.view(-1, self.num_choices)
        return reshaped_logits

        # if labels is not None:
        #     loss_fct = CrossEntropyLoss().to(self.device)
        #     loss = loss_fct(reshaped_logits, labels)
        #     return loss, reshaped_logits
        # else:
        #     return reshaped_logits
