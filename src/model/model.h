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

inline std::pair<insnet::Node *, std::vector<insnet::LSTMState>> sentEnc(insnet::Node &input,
        std::vector<insnet::LSTMState> &last_states,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    using std::vector;
    vector<insnet::LSTMState> states;
    int layer = params.sent_enc.ptrs().size();
    states.reserve(layer);
    Node *last_layer = &input;
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
        const std::vector<int> &word,
        std::vector<insnet::LSTMState> &last_states,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    Node *input = wordEnc(word, graph, params, dropout);
    return sentEnc(*input, last_states, params, dropout);
}

inline insnet::Node *englishStyleSentEnc(const std::vector<int> &sent, int word_symbol_id,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout,
        std::vector<insnet::LSTMState> &initial_state) {
    using insnet::Node;
    using std::vector;

    vector<int> word;
    word.reserve(sent.size());
    auto last_state = initial_state;
    vector<Node *> log_probs;
    log_probs.reserve(sent.size());
    for (int i = 0; i < sent.size(); ++i) {
        int id = sent.at(i);
        if (i == sent.size() - 1 || sent.at(i + 1) == word_symbol_id) {
            auto r = englishStyleSentEnc(word, last_state, graph, params, dropout);
            last_state = r.second;
            log_probs.push_back(r.first);
            word.clear();
        }
        word.push_back(id);
    }

    return cat(log_probs);
}

inline std::pair<insnet::Node *, std::vector<insnet::LSTMState>> chineseStyleSentEnc(int word,
        std::vector<insnet::LSTMState> &last_states,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    Node *input = insnet::embedding(graph, word, params.emb.E);
    return sentEnc(*input, last_states, params, dropout);
}

inline insnet::Node *chineseStyleSentEnc(const std::vector<int> &sent, insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout,
        std::vector<insnet::LSTMState> &initial_state) {
    using insnet::Node;
    using std::vector;
    auto last_state = initial_state;
    vector<Node *> log_probs;
    log_probs.reserve(sent.size());
    for (int id : sent) {
        auto r = chineseStyleSentEnc(id, last_state, graph, params, dropout);
        last_state = r.second;
        log_probs.push_back(r.first);
    }
    return cat(log_probs);
}

inline insnet::Node *sentEnc(const std::vector<int> &sent, insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout,
        std::vector<insnet::LSTMState> &initial_state,
        int word_symbol_id) {
    using insnet::Node;
    using std::vector;
    return sent.front() == word_symbol_id ?
        englishStyleSentEnc(sent, word_symbol_id, graph, params, dropout, initial_state) :
        chineseStyleSentEnc(sent, graph, params, dropout, initial_state);
}

#endif
