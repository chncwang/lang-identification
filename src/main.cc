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
//    params.init(vocab, dim, word_layer, head, sent_layer,);

//    unique_ptr<ModelParams> params;
//    int model_type = args["model"].as<int>();
//    int hidden_dim = args["hidden_dim"].as<int>();
//    int layer = args["layer"].as<int>();
//    if (model_type == 0) {
//        int head = args["head"].as<int>();
//        params = make_unique<TransformerParams>();
//        dynamic_cast<TransformerParams &>(*params).init(src_vocab, tgt_vocab, hidden_dim, layer,
//                head);
//    }

//    dtype lr = args["lr"].as<dtype>();
//    cout << fmt::format("lr:{}", lr) << endl;
//    insnet::AdamOptimizer optimizer(params->tunableParams(), lr);
//    int iteration = -1;
//    const int BENCHMARK_BEGIN_ITER = 100;

//    for (int epoch = 0; ; ++epoch) {
//        default_random_engine engine(0);
//        shuffle(train_conversation_pairs.begin(), train_conversation_pairs.end(), engine);

//        auto batch_begin = train_conversation_pairs.begin();
//        int batch_size = args["batch_size"].as<int>();
//        dtype dropout = args["dropout"].as<dtype>();

//        decltype(high_resolution_clock::now()) begin_time;
//        int word_sum_for_benchmark = 0;

//        Profiler &profiler = Profiler::Ins();
//        bool enable_profile = args["profile"].as<bool>();
//        cout << fmt::format("enable profile:{}", enable_profile) << endl;

//        while (batch_begin != train_conversation_pairs.end()) {
//            ++iteration;
//            if (iteration == BENCHMARK_BEGIN_ITER) {
//                begin_time = high_resolution_clock::now();
//            }
//            auto batch_it = batch_begin;
//            int word_sum = 0;
//            int tgt_word_sum = 0;

//            Graph graph(insnet::ModelStage::TRAINING, false);
//            vector<vector<int>> answers;

//            if (iteration == BENCHMARK_BEGIN_ITER) {
//                profiler.SetEnabled(enable_profile);
//                profiler.BeginEvent("top");
//            }
//            profiler.BeginEvent("graph building");

//            int sentence_size = 0;
//            vector<const vector<int> *> merged_batch_src_ids;
//            vector<const vector<int> *> merged_batch_tgt_in_ids;
//            while (word_sum < batch_size && batch_it != train_conversation_pairs.end()) {
//                const auto &batch_src_ids = src_ids.at(batch_it->post_id);
//                word_sum += batch_src_ids.size();
//                const auto &batch_tgt_in_ids = tgt_in_ids.at(batch_it->response_id);
//                word_sum += batch_tgt_in_ids.size();
//                tgt_word_sum += batch_tgt_in_ids.size();

//                if (iteration >= BENCHMARK_BEGIN_ITER) {
//                    word_sum_for_benchmark += batch_tgt_in_ids.size() + batch_src_ids.size();
//                }
//                merged_batch_src_ids.push_back(&batch_src_ids);
//                merged_batch_tgt_in_ids.push_back(&batch_tgt_in_ids);

//                answers.push_back(tgt_out_ids.at(batch_it->response_id));
//                ++batch_it;
//                ++sentence_size;
//            }
//            vector<Node *> probs = transformerSeq2seq(merged_batch_src_ids,
//                    merged_batch_tgt_in_ids, graph, *params, dropout);
//            profiler.EndEvent();

//            graph.forward();
//            dtype loss = insnet::NLLLoss(probs, tgt_vocab.size(), answers, 1.0f);
//            if (iteration % 100 == 0) {
//                cout << fmt::format("loss:{} sentence number:{} ppl:{}", loss, sentence_size,
//                        std::exp(loss / tgt_word_sum)) << endl;
//            }
//            graph.backward();
//            profiler.BeginEvent("optimize");
//            optimizer.step();
//            profiler.EndCudaEvent();

//            if ((iteration % 100 == 0 && iteration >= BENCHMARK_BEGIN_ITER) ||
//                    word_sum_for_benchmark > 4000000) {
//                auto now = high_resolution_clock::now();
//                auto elapsed_time = duration_cast<milliseconds>(now -
//                        begin_time);
//                float word_count_per_sec = 1e3 * word_sum_for_benchmark /
//                    static_cast<float>(elapsed_time.count());
//                cout << fmt::format("begin:{} now:{}", begin_time.time_since_epoch().count(),
//                        now.time_since_epoch().count());
//                cout << fmt::format("epoch:{} iteration:{} word_count_per_sec:{} word count:{} time:{} step time:{}",
//                        epoch, iteration, word_count_per_sec, word_sum_for_benchmark,
//                        elapsed_time.count(),
//                        elapsed_time.count() / (iteration + 1 - BENCHMARK_BEGIN_ITER)) << endl;
//            }

//            batch_begin = batch_it;
//            if (word_sum_for_benchmark > 4000000) {
//                cout << "benchmark end" << endl;
//                profiler.EndEvent();
//                profiler.Print();
//                exit(0);
//            }
//        }
//    }

    return 0;
}
