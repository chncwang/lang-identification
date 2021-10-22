#ifndef INSNET_BENCHMARK_TRANSFORMER_MODEL_H
#define INSNET_BENCHMARK_TRANSFORMER_MODEL_H

#include <vector>
#include "insnet/insnet.h"
#include "params.h"

inline insnet::Node *wordEnc(const std::vector<int> &word, insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;

    Node *emb = insnet::embedding(graph, word, params.emb.E);
    Node *enc = insnet::transformerEncoder(*emb, params.word_enc, dropout).back();
    int dim = params.word_enc.hiddenDim();
    enc = split(*enc, dim, 0);
    return enc;
}

inline insnet::Node *segEnc(insnet::Node &input, ModelParams &params, insnet::dtype dropout) {
    using insnet::Node;
    Node *enc = insnet::transformerEncoder(input, params.seg_enc, dropout).back();
    int dim = params.word_enc.hiddenDim();
    return split(*enc, dim, 0);
}

inline std::pair<insnet::Node *, std::vector<insnet::LSTMState>> sentEnc(insnet::Node &input,
        std::vector<insnet::LSTMState> &last_states,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    using std::vector;

    Node *seg_enc = segEnc(input, params, dropout);

    vector<insnet::LSTMState> states;
    int layer = params.sent_enc.ptrs().size();
    states.reserve(layer);
    Node *last_layer = seg_enc;
    for (int i = 0; i < layer; ++i) {
        insnet::LSTMState state = insnet::lstm(last_states.at(i), *last_layer,
                *params.sent_enc.ptrs().at(i), dropout);
        states.push_back(state);
        last_layer = insnet::add({last_layer, state.hidden});
    }
    Node *output = insnet::linear(*last_layer, params.output);
    output = insnet::logSoftmax(*output);
    return std::make_pair(output, states);
}

inline std::pair<insnet::Node *, std::vector<insnet::LSTMState>> englishStyleSentEnc(
        const std::vector<std::vector<int>> &seg_words,
        insnet::Node &seg_symbol_emb,
        std::vector<insnet::LSTMState> &last_states,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    using std::vector;
    vector<Node *> seg_inputs;
    seg_inputs.push_back(&seg_symbol_emb);
    for (const auto &word_ids : seg_words) {
        Node *input = wordEnc(word_ids, graph, params, dropout);
        seg_inputs.push_back(input);
    }
    Node *merged = cat(seg_inputs);
    return sentEnc(*merged, last_states, params, dropout);
}

inline insnet::Node *englishStyleSentEnc(const std::vector<int> &sent, int word_symbol_id,
        int seg_len,
        int seg_symbol_id,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout,
        std::vector<insnet::LSTMState> &initial_state) {
    using insnet::Node;
    using std::vector;

    vector<int> word;
    vector<vector<int>> word_seg;
    word.reserve(sent.size());
    auto last_state = initial_state;
    vector<Node *> log_probs;
    log_probs.reserve(sent.size());

    Node *seg_emb = insnet::embedding(graph, seg_symbol_id, params.emb.E);

    for (int i = 0; i < sent.size(); ++i) {
        int id = sent.at(i);
        if (i == sent.size() - 1 || sent.at(i + 1) == word_symbol_id) {
            word_seg.push_back(word);
            if (word_seg.size() == seg_len - 1 || i == sent.size() - 1) {
                auto r = englishStyleSentEnc(word_seg, *seg_emb, last_state, graph, params,
                        dropout);
                last_state = r.second;
                log_probs.push_back(r.first);
                word_seg.clear();
            }
            word.clear();
        }
        word.push_back(id);
    }

    return cat(log_probs);
}

inline std::pair<insnet::Node *, std::vector<insnet::LSTMState>> chineseStyleSentEnc(
        const std::vector<int> &seg_ids,
        std::vector<insnet::LSTMState> &last_states,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    Node *input = insnet::embedding(graph, seg_ids, params.emb.E);
    return sentEnc(*input, last_states, params, dropout);
}

inline insnet::Node *chineseStyleSentEnc(const std::vector<int> &sent, int seg_len,
        int seg_symbol_id,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout,
        std::vector<insnet::LSTMState> &initial_state) {
    using insnet::Node;
    using std::vector;
    auto last_state = initial_state;
    vector<Node *> log_probs;
    log_probs.reserve(sent.size());
    for (int i = 0; i < sent.size(); i += seg_len - 1) {
        vector<int> seg_ids;
        seg_ids.reserve(seg_len);
        seg_ids.push_back(seg_symbol_id);
        for (int j = 0; j < seg_len - 1 && i + j < sent.size(); ++j) {
            seg_ids.push_back(sent.at(i + j));
        }

        auto r = chineseStyleSentEnc(seg_ids, last_state, graph, params, dropout);
        last_state = r.second;
        log_probs.push_back(r.first);
    }
    return cat(log_probs);
}

inline insnet::Node *sentEnc(const std::vector<int> &sent, int seg_len, int seg_symbol_id,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout,
        std::vector<insnet::LSTMState> &initial_state,
        int word_symbol_id) {
    using insnet::Node;
    using std::vector;
    return sent.front() == word_symbol_id ?
        englishStyleSentEnc(sent, word_symbol_id, seg_len, word_symbol_id, graph, params, dropout,
                initial_state) :
        chineseStyleSentEnc(sent, seg_len, seg_symbol_id, graph, params, dropout, initial_state);
}

#endif
