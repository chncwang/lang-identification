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
        std::vector<int> &word,
        std::vector<insnet::LSTMState> &last_states,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    Node *input = wordEnc(word, graph, params, dropout);
    return sentEnc(*input, last_states, params, dropout);
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

#endif
