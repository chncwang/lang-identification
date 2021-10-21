#include "cxxopts.hpp"
#include "insnet/insnet.h"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <array>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <queue>
#include "conversation_structure.h"
#include "data_manager.h"
#include "def.h"
#include "model/params.h"
#include "model/model.h"

using cxxopts::Options;
using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::priority_queue;
using std::unordered_map;
using std::unordered_set;
using std::unique_ptr;
using std::make_unique;
using std::move;
using std::pair;
using std::make_pair;
using std::default_random_engine;
using std::shuffle;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::duration;
using insnet::dtype;
using insnet::Vocab;
using insnet::Graph;
using insnet::Node;
using insnet::Profiler;

unordered_map<string, int> calWordFreq(const vector<vector<string>> &sentences) {
    unordered_map<string, int> ret;
    for (const auto &s : sentences) {
        for (const string &w : s) {
            auto it = ret.find(w);
            if (it == ret.end()) {
                ret.insert(make_pair(w, 1));
            } else {
                it->second++;
            }
        }
    }
    return ret;
}

float F1(float correct, float predicted, float golden) {
    float prec = correct / predicted;
    float recall = correct / golden;
    return 2 * prec * recall / (prec + recall);
}

int main(int argc, const char *argv[]) {
    Options options("InsNet benchmark");
    options.add_options()
        ("device_id", "device id", cxxopts::value<int>()->default_value("0"))
        ("train", "training set dir", cxxopts::value<string>())
        ("batch_size", "batch size", cxxopts::value<int>()->default_value("1"))
        ("dropout", "dropout", cxxopts::value<float>()->default_value("0.1"))
        ("lr", "learning rate", cxxopts::value<float>()->default_value("0.001"))
        ("ratio", "dataset ratio", cxxopts::value<float>()->default_value("1"))
        ("word_layer", "word layer", cxxopts::value<int>()->default_value("2"))
        ("sent_layer", "sent layer", cxxopts::value<int>()->default_value("1"))
        ("head", "head", cxxopts::value<int>()->default_value("8"))
        ("dim", "hidden dim", cxxopts::value<int>()->default_value("512"))
        ("cutoff", "cutoff", cxxopts::value<int>()->default_value("0"));

    auto args = options.parse(argc, argv);

    int device_id = args["device_id"].as<int>();
    cout << fmt::format("device_id:{}", device_id) << endl;

#if USE_GPU
    insnet::cuda::initCuda(device_id, 0);
#endif

    string train_pair_file = args["train"].as<string>();

    float ratio = args["ratio"].as<float>();
    auto char_list = charList(train_pair_file, args["cutoff"].as<int>(), ratio);
    Vocab vocab;
    vocab.init(char_list);
    cout << "vocab size:" << vocab.size() << endl;
    Vocab class_vocab;
    auto class_list = classList(train_pair_file);
    class_vocab.init(class_list);
    cout << "class size:" << class_vocab.size() << endl;

    auto train_set = readDataset(train_pair_file, vocab.m_string_to_id, class_vocab.m_string_to_id,
            ratio);

    if (train_set.first.size() != train_set.second.size()) {
        abort();
    }

    cout << "train size:" << train_set.first.size() << endl;
    ModelParams params;
    int dim = args["dim"].as<int>();
    cout << "dim:" << dim << endl;
    int word_layer = args["word_layer"].as<int>();
    cout << "word_layer:" << word_layer << endl;
    int head = args["head"].as<int>();
    cout << "head:" << head << endl;
    int sent_layer = args["sent_layer"].as<int>();
    cout << "sent_layer:" << sent_layer << endl;
    params.init(vocab, dim, word_layer, head, sent_layer, 1024, class_vocab.size());

    dtype lr = args["lr"].as<dtype>();
    cout << fmt::format("lr:{}", lr) << endl;
    insnet::AdamOptimizer optimizer(params.tunableParams(), lr);
    int iteration = -1;

    vector<int> train_ids;
    for (int i = 0; i < train_set.first.size(); ++i) {
        train_ids.push_back(i);
    }

    for (int epoch = 0; ; ++epoch) {
        default_random_engine engine(0);
        shuffle(train_ids.begin(), train_ids.end(), engine);

        auto batch_begin = train_ids.begin();
        int batch_size = args["batch_size"].as<int>();
        cout << "batch_size:" << batch_size << endl;
        dtype dropout = args["dropout"].as<dtype>();
        int word_symbol_id = vocab.from_string(WORD_SYMBOL);
        vector<float> correct_times;
        vector<float> predicted_times;
        vector<float> golden_times;
        float correct_time = 0;
        float total_time = 0;
        for (int i = 0; i < class_vocab.size(); ++i) {
            correct_times.push_back(0);
            predicted_times.push_back(0);
            golden_times.push_back(0);
        }

        float sentence_size_sum = 0;

        while (batch_begin != train_ids.end()) {
            ++iteration;
            auto batch_it = batch_begin;
            int word_sum = 0;
            Graph graph(insnet::ModelStage::TRAINING, false);
            vector<vector<int>> answers;

            vector<insnet::LSTMState> initial_states;
            initial_states.reserve(sent_layer);
            Node *zero = insnet::tensor(graph, dim, 0);
            for (int i = 0; i < sent_layer; ++i) {
                initial_states.push_back({zero, zero});
            }

            int sentence_size = 0;
            vector<Node *> log_probs;
            while (word_sum < batch_size && batch_it != train_ids.end()) {
                const auto &batch_ids = train_set.first.at(*batch_it);
                Node *node = sentEnc(batch_ids, graph, params, dropout, initial_states,
                        word_symbol_id);
                log_probs.push_back(node);

                int answer = train_set.second.at(*batch_it);
                vector<int> ans;
                int word_num = node->size() / class_vocab.size();
                word_sum += word_num;
                for (int i = 0; i < word_num; ++i) {
                    ans.push_back(answer);
                }
                answers.push_back(move(ans));

                ++batch_it;
                ++sentence_size;
            }
            sentence_size_sum += sentence_size;

            graph.forward();
            dtype loss = insnet::NLLLoss(log_probs, class_vocab.size(), answers, 1.0f);
            auto predicted_ids = insnet::argmax(log_probs, class_vocab.size());
            for (int i = 0; i < predicted_ids.size(); ++i) {
                if (predicted_ids.at(i).back() == answers.at(i).back()) {
                    correct_times.at(predicted_ids.at(i).back())++;
                    correct_time++;
                }
                predicted_times.at(predicted_ids.at(i).back())++;
                golden_times.at(answers.at(i).back())++;
                total_time++;
            }
            if (iteration % 10 == 0) {
                float sum = 0;
                for (int i = 0; i < correct_times.size(); ++i) {
                    float f1 = F1(correct_times.at(i), predicted_times.at(i), golden_times.at(i));
//                    cout << class_vocab.from_id(i) << " f1:" << f1 << endl;
                    sum += f1;
                }
                cout << fmt::format("process:{} loss:{} sentence number:{} macro F:{} acc:{}",
                        sentence_size_sum / train_ids.size(), loss,
                        sentence_size, sum / class_vocab.size(), correct_time / total_time) << endl;
            }
            graph.backward();
            optimizer.step();

            batch_begin = batch_it;
        }
    }

    return 0;
}
