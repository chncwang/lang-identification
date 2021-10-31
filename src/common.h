#ifndef LANG_ID_COMMON_H
#define LANG_ID_COMMON_H

#include "insnet/insnet.h"
#include "model/params.h"
#include "model/model.h"
#include "data_manager.h"

inline void loadModel(ModelParams &model_params, insnet::Vocab &vocab, insnet::Vocab &class_vocab,
        const std::string &filename,
        int &iter,
        int &dim,
        int &word_layer,
        int &word_head,
        int &seg_layer,
        int &seg_head,
        int &sent_layer) {
    std::cout << "loading model file..." << std::endl;
    std::ifstream is(filename.c_str());
    if (is) {
        std::cout << "loading model..." << std::endl;
        cereal::BinaryInputArchive ar(is);
        ar(iter, dim, word_layer, word_head, seg_layer, seg_head, sent_layer, class_vocab, vocab);
        model_params.init(vocab, dim, word_layer, word_head, seg_layer, seg_head, sent_layer, 1024,
                class_vocab.size());
        ar(model_params);
#if USE_GPU
        model_params.copyFromHostToDevice();
#endif
        std::cout << "model loaded" << std::endl;
    } else {
        std::cerr << fmt::format("load model fail - filename:%1%", filename) << std::endl;
        abort();
    }
}

inline void loadModel(ModelParams &model_params, insnet::Vocab &vocab, insnet::Vocab &class_vocab,
        const std::string &filename) {
    int iter, dim, word_layer, word_head, seg_layer, seg_head, sent_layer;
    loadModel(model_params, vocab,class_vocab, filename, iter, dim, word_layer, word_head,
            seg_layer, seg_head, sent_layer);
}

inline float F1(float correct, float predicted, float golden) {
    return 2 * correct / (predicted + golden + 1e-10);
}

inline std::pair<float, float> evaluate(ModelParams &params, insnet::dtype dropout,
        const std::string &dir,
        insnet::Vocab &vocab,
        insnet::Vocab &class_vocab,
        int seg_len,
        int batch_size = 1,
        float ratio = 1) {
    using namespace std;
    using namespace insnet;

    auto dataset = readDataset(dir, vocab.m_string_to_id, class_vocab.m_string_to_id, ratio);
    vector<int> ids;
    for (int i = 0; i < dataset.first.size(); ++i) {
        ids.push_back(i);
    }
    auto batch_begin = ids.begin();
    int word_symbol_id = vocab.from_string(WORD_SYMBOL);
    int seg_symbol_id = vocab.from_string(SEG_SYMBOL);
    vector<float> correct_times;
    vector<float> predicted_times;
    vector<float> golden_times;
    float correct_time = 0;
    float total_time = 0;
    cout << "class_vocab size:" << class_vocab.size() << endl;
    for (int i = 0; i < class_vocab.size(); ++i) {
        correct_times.push_back(0);
        predicted_times.push_back(0);
        golden_times.push_back(0);
    }

    float sentence_size_sum = 0;
    default_random_engine engine(0);
    shuffle(ids.begin(), ids.end(), engine);
    
    int iteration = -1;

    while (batch_begin != ids.end()) {
        ++iteration;
        auto batch_it = batch_begin;
        int seg_sum = 0;
        Graph graph(insnet::ModelStage::INFERENCE, false);
        vector<vector<int>> answers;

        vector<insnet::LSTMState> initial_states;
        initial_states.reserve(params.sent_enc.size());
        Node *zero = insnet::tensor(graph, params.word_enc.hiddenDim(), 0);
        for (int i = 0; i < params.sent_enc.size(); ++i) {
            initial_states.push_back({zero, zero});
        }

        int sentence_size = 0;
        vector<Node *> log_probs;
        vector<int> *batch_ids;
        while (seg_sum < batch_size * 0.5 && batch_it != ids.end()) {
            batch_ids = &dataset.first.at(*batch_it);
            Node *node = sentEnc(*batch_ids, seg_len, seg_symbol_id, graph, params, dropout,
                    initial_states);
            log_probs.push_back(node);

            int answer = dataset.second.at(*batch_it);
            vector<int> ans;
            int seg_num = node->size() / class_vocab.size();
            seg_sum += seg_num;
            for (int i = 0; i < seg_num; ++i) {
                ans.push_back(answer);
            }
            answers.push_back(move(ans));

            ++batch_it;
            ++sentence_size;
        }
        sentence_size_sum += sentence_size;

        graph.forward();
        auto predicted_ids = insnet::argmax(log_probs, class_vocab.size());
        for (int i = 0; i < predicted_ids.size(); ++i) {
            if (predicted_ids.at(i).back() == answers.at(i).back()) {
                correct_times.at(predicted_ids.at(i).back())++;
                correct_time++;
            }
            predicted_times.at(predicted_ids.at(i).back())++;
            int answer = answers.at(i).back();
            golden_times.at(answer)++;
            total_time++;
        }

        batch_begin = batch_it;

        if (iteration % 10 == 0) {
            float sum = 0;
            for (int i = 0; i < correct_times.size(); ++i) {
                float f = F1(correct_times.at(i), predicted_times.at(i), golden_times.at(i));
                sum += f;
            }
            cout << "macro f1:" << sum / correct_times.size() << endl;
            cout << "gold:" << class_vocab.from_id(answers.back().back()) << endl;
            for (int i = 0; i < batch_ids->size(); ++i) {
                if (batch_ids->at(i) == word_symbol_id) {
                    cout << " ";
                } else {
                    if (batch_ids->at(i) >= 0) cout << vocab.from_id(batch_ids->at(i));
                }
            }
            cout << endl;
            cout << "evaluate predicted: ";
            for (int id : predicted_ids.back()) {
                cout << class_vocab.from_id(id) << " ";
            }
            cout << endl;
        }
    }

    float sum = 0;
    int size = 0;
    for (int i = 0; i < correct_times.size(); ++i) {
        if (golden_times.at(i) > 0) {
            float f = F1(correct_times.at(i), predicted_times.at(i), golden_times.at(i));
            cout << class_vocab.from_id(i) << ":" << f << endl;
            sum += f;
            ++size;
        }
    }

    float f1 = sum / size;
    float acc = correct_time / total_time;
    return std::make_pair(f1, acc);
}


#endif
