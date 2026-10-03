"""Frozen checkpoint choices for the additional-family benchmark."""

FAMILIES = {
    'pythia_topk': dict(
        model='EleutherAI/pythia-70m-deduped', layer=3,
        release='sae_bench_pythia70m_sweep_topk_ctx128_0730',
        sae_id='blocks.3.hook_resid_post__trainer_0',
        hook='residual_post', dtype='float32',
        views={'pythia_topk': None}),
    'smol_topk': dict(
        model='HuggingFaceTB/SmolLM2-135M', layer=15,
        release='EleutherAI/sae-SmolLM2-135M-64x',
        sae_id='layers.15.mlp', hook='mlp_output', dtype='float32',
        views={'smol_topk': None}),
    'qwen_transcoder': dict(
        model='Qwen/Qwen3-0.6B', layer=14,
        release='mwhanna-qwen3-0.6b-transcoders-lowl0',
        sae_id='layer_14', hook='mlp_input_output', dtype='float32',
        views={'qwen_transcoder': None}),
    'gemma_matryoshka': dict(
        model='google/gemma-2-2b', layer=8,
        release='gemma-2-2b-res-matryoshka-dc',
        sae_id='blocks.8.hook_resid_post',
        hook='residual_post', dtype='bfloat16',
        views={'gemma_matryoshka': None,
               'gemma_matryoshka_512': 512,
               'gemma_matryoshka_2048': 2048}),
}
CHECKPOINTS = {view: (cfg['release'], cfg['sae_id'])
               for cfg in FAMILIES.values() for view in cfg['views']}
