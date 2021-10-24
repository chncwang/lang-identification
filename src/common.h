#ifndef LANG_ID_COMMON_H
#define LANG_ID_COMMON_H

#include "insnet/insnet.h"
#include "model/params.h"

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

#endif
