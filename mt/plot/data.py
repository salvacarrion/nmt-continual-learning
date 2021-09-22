experiments = [

    {"id": "0.1", "name": "Europarl Large\n(630K; cs-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "cs-en", "sacrebleu_bleu": 42.5, "sacrebleu_chrf": 0.66, "sentences": 635498, "tokens": 17400566, "tokens/sentences": 28.275},
    {"id": "0.2", "name": "Europarl Large\n(630K; cs-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "cs-en", "sacrebleu_bleu": 37.1, "sacrebleu_chrf": 0.63, "sentences": 635498, "tokens": 635496, "tokens/sentences": 84.508},
    {"id": "0.3", "name": "Europarl Small\n(100K; cs-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "cs-en", "sacrebleu_bleu": 31.0, "sacrebleu_chrf": 0.57, "sentences": 100000, "tokens": 2731908, "tokens/sentences": 27.319},
    {"id": "0.4", "name": "Europarl Small\n(100K; cs-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "cs-en", "sacrebleu_bleu": 33.2, "sacrebleu_chrf": 0.6, "sentences": 100000, "tokens": 8443124, "tokens/sentences": 84.431},
    {"id": "0.5", "name": "Europarl Small\n(50K; cs-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "cs-en", "sacrebleu_bleu": 22.6, "sacrebleu_chrf": 0.49, "sentences": 50000, "tokens": 1362120, "tokens/sentences": 27.242},
    {"id": "0.6", "name": "Europarl Small\n(50K; cs-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "cs-en", "sacrebleu_bleu": 28.6, "sacrebleu_chrf": 0.56, "sentences": 50000, "tokens": 4216581, "tokens/sentences": 84.332},

    {"id": "1.1", "name": "Europarl Large\n(2M; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 44.5, "sacrebleu_chrf": 0.67, "sentences": 1948939, "tokens": 55105930, "tokens/sentences": 28.275},
    {"id": "1.2", "name": "Europarl Large\n(2M; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 39.6, "sacrebleu_chrf": 0.64, "sentences": 1948939, "tokens": 168327841, "tokens/sentences": 86.369},
    {"id": "1.3", "name": "Europarl Small\n(100K; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 33.0, "sacrebleu_chrf": 0.58, "sentences": 100000, "tokens": 2818061, "tokens/sentences": 28.181},
    {"id": "1.4", "name": "Europarl Small\n(100K; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 35.2, "sacrebleu_chrf": 0.6, "sentences": 100000, "tokens": 8621638, "tokens/sentences": 86.216},
    {"id": "1.5", "name": "Europarl Small\n(50K; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 24.4, "sacrebleu_chrf": 0.5, "sentences": 50000, "tokens": 1406811, "tokens/sentences": 28.136},
    {"id": "1.6", "name": "Europarl Small\n(50K; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 31.7, "sacrebleu_chrf": 0.58, "sentences": 50000, "tokens": 4310512, "tokens/sentences": 86.210},

    {"id": "2.1", "name": "Europarl Large\n(2M; de-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "de-en", "sacrebleu_bleu": 36.6, "sacrebleu_chrf": 0.61, "sentences": 1948939, "tokens": 55105930, "tokens/sentences": 28.275},
    {"id": "2.2", "name": "Europarl Large\n(2M; de-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "de-en", "sacrebleu_bleu": 30.7, "sacrebleu_chrf": 0.57, "sentences": 1948939, "tokens": 168327841, "tokens/sentences": 86.369},
    {"id": "2.3", "name": "Europarl Small\n(100K; de-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "de-en", "sacrebleu_bleu": 24.0, "sacrebleu_chrf": 0.5, "sentences": 100000, "tokens": 2818061, "tokens/sentences": 28.181},
    {"id": "2.4", "name": "Europarl Small\n(100K; de-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "de-en", "sacrebleu_bleu": 26.4, "sacrebleu_chrf": 0.53, "sentences": 100000, "tokens": 8621638, "tokens/sentences": 86.216},
    {"id": "2.3", "name": "Europarl Small\n(50K; de-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "de-en", "sacrebleu_bleu": 16.6, "sacrebleu_chrf": 0.43, "sentences": 50000, "tokens": 1408597, "tokens/sentences": 28.172},
    {"id": "2.4", "name": "Europarl Small\n(50K; de-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "de-en", "sacrebleu_bleu": 23.3, "sacrebleu_chrf": 0.51, "sentences": 50000, "tokens": 4320297, "tokens/sentences": 86.406},

    {"id": "3.1", "name": "Biomedical Large\n(570K; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 35.6, "sacrebleu_chrf": 0.63, "sentences": 575521, "tokens": 16596128, "tokens/sentences": 28.837},
    {"id": "3.2", "name": "Biomedical Large\n(570K; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 35.2, "sacrebleu_chrf": 0.63, "sentences": 575521, "tokens": 53033532, "tokens/sentences": 92.149},
    {"id": "3.3", "name": "Biomedical Small\n(120K; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 28.7, "sacrebleu_chrf": 0.56, "sentences": 118636, "tokens": 3415430, "tokens/sentences": 28.789},
    {"id": "3.4", "name": "Biomedical Small\n(120K; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab","lang-pair": "es-en",  "sacrebleu_bleu": 33.3, "sacrebleu_chrf": 0.61, "sentences": 118636, "tokens": 10932149, "tokens/sentences": 92.149},

    {"id": "4.1", "name": "CommonCrawl Large\n(1.8M; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 30.7, "sacrebleu_chrf": 0.52, "sentences": 1815333, "tokens": 50918607, "tokens/sentences": 28.049},
    {"id": "4.2", "name": "CommonCrawl Large\n(1.8M; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 26.5, "sacrebleu_chrf": 0.49, "sentences": 1815247, "tokens": 147347224, "tokens/sentences": 81.172},
    {"id": "4.3", "name": "CommonCrawl Small\n(100K; es-en)", "bpe-merges": "32k", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 22.6, "sacrebleu_chrf": 0.46, "sentences": 99994, "tokens": 8095831, "tokens/sentences": 80.963},
    {"id": "4.4", "name": "CommonCrawl Small\n(100K; es-en)", "bpe-merges": "64", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 15.6, "sacrebleu_chrf": 0.37, "sentences": 100000, "tokens": 2788613, "tokens/sentences": 27.886},

    {"id": "5.1", "name": "NewsCommentary Large\n(360K; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 45.4, "sacrebleu_chrf": 0.67, "sentences": 357280, "tokens": 9525425, "tokens/sentences": 26.661},
    {"id": "5.2", "name": "NewsCommentary Large\n(360K; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 42.6, "sacrebleu_chrf": 0.66, "sentences": 357280, "tokens": 30561745, "tokens/sentences": 85.540},
    {"id": "5.3", "name": "NewsCommentary Small\n(35K; es-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "es-en", "sacrebleu_bleu": 21.5, "sacrebleu_chrf": 0.48, "sentences": 35000, "tokens": 928552, "tokens/sentences": 26.530},
    {"id": "5.4", "name": "NewsCommentary Small\n(35K; es-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "es-en", "sacrebleu_bleu": 30.1, "sacrebleu_chrf": 0.57, "sentences": 35000, "tokens": 3000210, "tokens/sentences": 85.720},

    {"id": "6.1", "name": "Multi30k\n(30K; de-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "de-en", "sacrebleu_bleu": 32.9, "sacrebleu_chrf": 0.55, "sentences": 29000, "tokens": 380835, "tokens/sentences": 13.132},
    {"id": "6.2", "name": "Multi30k\n(30K; de-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "de-en", "sacrebleu_bleu": 34.3, "sacrebleu_chrf": 0.54, "sentences": 29000, "tokens": 991180, "tokens/sentences": 34.179},

    {"id": "7.1", "name": "IWLST2016\n(190K; de-en)", "bpe-merges": "32k", "bpe": "Large subword vocab", "lang-pair": "de-en", "sacrebleu_bleu": 28.1, "sacrebleu_chrf": 0.49, "sentences": 196869, "tokens": 4072858, "tokens/sentences": 20.688},
    {"id": "7.2", "name": "IWLST2016\n(190K; de-en)", "bpe-merges": "64", "bpe": "Quasi char-level vocab", "lang-pair": "de-en", "sacrebleu_bleu": 35.2, "sacrebleu_chrf": 0.63, "sentences": 196869, "tokens": 11020739, "tokens/sentences": 55.980},

]  # trH_tsH_vH_