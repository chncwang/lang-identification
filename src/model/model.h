#ifndef INSNET_BENCHMARK_TRANSFORMER_MODEL_H
#define INSNET_BENCHMARK_TRANSFORMER_MODEL_H

#include <vector>
#include "insnet/insnet.h"
#include "def.h"
#include "params.h"

inline void print(const std::vector<int> ids, insnet::Vocab &vocab) {
    using std::cout;
    using std::endl;
    for (int id : ids) {
        if (id == -1) {
            cout << id <<"<-1>";
        } else if (id == vocab.from_string(WORD_SYMBOL)) {
            cout << " ";
        } else {
            cout << vocab.from_id(id);
        }
    }
    cout << endl;
    for (int id : ids) {
        cout << id << " ";
    }
    cout << endl;
}

inline insnet::Node *wordEnc(const std::vector<int> &word, insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;

    if (word.size() > 32) {
        abort();
    }

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

inline std::pair<insnet::Node *, std::vector<insnet::LSTMState>> sentEnc(
        const std::vector<std::pair<std::vector<int>, int>> &seg_words,
        insnet::Node &seg_symbol_emb,
        std::vector<insnet::LSTMState> &last_states,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout) {
    using insnet::Node;
    using std::vector;
    vector<Node *> seg_inputs;
    seg_inputs.push_back(&seg_symbol_emb);
    for (const auto &e : seg_words) {
        Node *input = e.first.empty() ? insnet::embedding(graph, e.second, params.emb.E) :
            wordEnc(e.first, graph, params, dropout);
        seg_inputs.push_back(input);
    }
    Node *merged = cat(seg_inputs);
    return sentEnc(*merged, last_states, params, dropout);
}

inline insnet::Node *sentEnc(const std::vector<int> &sent, int seg_len, int seg_symbol_id,
        insnet::Graph &graph,
        ModelParams &params,
        insnet::dtype dropout,
        std::vector<insnet::LSTMState> &initial_state,
        bool early_exit = false) {
    using insnet::Node;
    using std::make_pair;
    using std::vector;
    using std::pair;

    int word_symbol_id = params.emb.vocab.from_string(WORD_SYMBOL);

    enum State {
        IN_WORD = 0,
        IN_CHAR = 1,
    };

    vector<int> word;
    vector<pair<vector<int>, int>> word_seg;
    word.reserve(sent.size());
    auto last_state = initial_state;
    vector<Node *> log_probs;
    log_probs.reserve(sent.size());

    Node *seg_emb = insnet::embedding(graph, seg_symbol_id, params.emb.E);
    State state = IN_CHAR;

    for (int i = 0; i < sent.size(); ++i) {
        int id = sent.at(i);
        if (id == word_symbol_id) {
            state = IN_WORD;
        } else if (id == -1) {
            state = IN_CHAR;
            continue;
        }

        if (state == IN_WORD) {
            word.push_back(id);
            if (i == sent.size() - 1 || sent.at(i + 1) == word_symbol_id || sent.at(i + 1) == -1) {
                if (word.size() > 32) {
                    std::cerr << "word size:" << word.size() << std::endl;
                    print(sent, params.emb.vocab);
                    print(word, params.emb.vocab);
                    abort();
                }
                word_seg.push_back(make_pair(word, -1));
                word.clear();
            }
        } else {
            word_seg.push_back(make_pair(vector<int>(), id));
        }
        if (word_seg.size() == seg_len - 1 || i == sent.size() - 1) {
            auto r = sentEnc(word_seg, *seg_emb, last_state, graph, params, dropout);

            last_state = r.second;
            log_probs.push_back(r.first);
            word_seg.clear();

            if (early_exit) {
                graph.forward();
                int class_i = insnet::argmax({r.first}, r.first->size()).back().back();
                float prob = std::exp(r.first->getVal()[class_i]);
                if (prob > 0.9999 || log_probs.size() > 64) {
                    break;
                }
            }
        }
    }

    return cat(log_probs);
}

#endif
