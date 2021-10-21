#ifndef LANG_ID_PARAM_H
#define LANG_ID_PARAM_H

#include "insnet/insnet.h"

class ModelParams : insnet::TunableParamCollection
#if USE_GPU
, public insnet::cuda::TransferableComponents
#endif
{
public:
    void init(const insnet::Vocab &vocab, int dim, int word_layer, int head, int sent_layer,
            int max_sent_len,
            int class_num) {
        emb.init(vocab, dim, true);
        word_enc.init(word_layer, dim, head, max_sent_len);
        std::function<void(insnet::LSTMParams &, int)> init_lstm = [&] (insnet::LSTMParams &params,
                int layer) {
            params.init(dim, dim);
        };
        sent_enc.init(sent_layer, init_lstm);
        output.init(class_num, dim);
    }

    insnet::Embedding<insnet::SparseParam> emb;
    insnet::TransformerEncoderParams word_enc;
    insnet::ParamArray<insnet::LSTMParams> sent_enc;
    insnet::LinearParams output;

#if USE_GPU
    std::vector<insnet::cuda::Transferable *> transferablePtrs() override {
        return {&emb, &sent_enc, &word_enc};
    }
#endif

protected:
    virtual std::vector<insnet::TunableParam *> tunableComponents() override {
        return {&emb, &sent_enc, &word_enc};
    }
};

#endif
