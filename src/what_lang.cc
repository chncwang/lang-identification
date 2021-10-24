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
#include "common.h"
#include "model/params.h"
#include "model/model.h"
#include <iomanip>

using cxxopts::Options;
using std::string;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;
using std::ostringstream;
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

inline std::pair<std::vector<std::vector<int>>, std::vector<std::string>> readDataset(
        const std::string &dir_name,
        const std::unordered_map<std::string, int> &vocab) {
    std::vector<std::vector<int>> sent_ret;
    std::vector<std::string> file_name_ret;

    for (const auto &entry : std::filesystem::directory_iterator(dir_name)) {
        std::string path = entry.path();
        file_name_ret.push_back(path);
        std::ifstream ifs(path);
        std::string raw_line;
        std::string merged_content;
        while (std::getline(ifs, raw_line)) {
            merged_content += raw_line + " ";
        }
        cout << merged_content << endl;
        utf8_string line(merged_content);
        auto words = splitIntoWords(line, vocab);

        sent_ret.push_back(move(words));
    }

    return std::make_pair(sent_ret, file_name_ret);
}

int main(int argc, const char *argv[]) {
    ModelParams params;
    Vocab vocab, class_vocab;

    Options options("lang_id");
    options.add_options()
        ("model", "load model", cxxopts::value<string>()->default_value("./model"))
        ("corpus", "corpus dir", cxxopts::value<string>());

    auto args = options.parse(argc, argv);
    loadModel(params, vocab, class_vocab, args["model"].as<string>());

    auto text_info = readDataset(args["corpus"].as<string>(), vocab.m_string_to_id);

    for (int i = 0; i < text_info.first.size(); ++i) {
        insnet::Graph graph(insnet::ModelStage::INFERENCE);
        Node *zero = insnet::tensor(graph, 512, 0.0f);
        insnet::LSTMState state = {zero, zero};
        vector<insnet::LSTMState> states = {state};
        int seg_id = vocab.from_string(SEG_SYMBOL);
        Node *log_prob = sentEnc(text_info.first.at(i), 64, seg_id, graph, params, 0.1, states,
                true);
        log_prob = insnet::split(*log_prob, class_vocab.size(),
                log_prob->size() - class_vocab.size());
        graph.forward();
        int class_id = insnet::argmax({log_prob}, class_vocab.size()).front().back();
        cout << fmt::format("filename:{} class:{} prob:{}", text_info.second.at(i),
                class_vocab.from_id(class_id), std::exp(log_prob->getVal()[class_id])) << endl;
    }
    return 0;
}
