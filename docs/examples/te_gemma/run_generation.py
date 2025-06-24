from utils import *
import transformer_engine.pytorch as te

hyperparams.model_name = (  # "/tmp/gemma-7b-hf/" # <== Add model weight location here e.g. "/path/to/downloaded/gemma/weights"
    "/perfhome/repos/ckpt/models/gemma-7b-hf/"
)
hyperparams.qkv_format = "thd"

run_generation = True
run_calibration = False

if run_calibration:
    hyperparams.fuse_qkv_params = True # This is needed by the last improvement.

    model = init_te_gemma_model(hyperparams)

    # Calibration
    with te.fp8_autocast(enabled=False, calibrating=True), \
        torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        model.train()
        run_forward_pass(model, hyperparams, num_iters=512)

    # Compute scale_fwd with enabled fp8 autocast
    with te.fp8_autocast(enabled=True), \
        torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        run_forward_pass(model, hyperparams, 1)

    # Some parameters are in pointing to the same tensors, double save is avoided here.
    dict_to_save = {k: v for k, v in model.state_dict().items() \
                    if ("_context_phase" not in k and "_generation_phase" not in k)}
    torch.save(dict_to_save, 'calibrated_weights.pth') # <== Add path to save calibrated weights.


if run_generation:

    # hyperparams.generation_cuda_graphs = False # 4.15s
    hyperparams.generation_cuda_graphs = True # 4.38s

    if hyperparams.generation_cuda_graphs:
        # It is necessary to preallocate a static buffer.
        # CUDA graphs require static input tensors for every kernel.
        # This approach may result in a slight increase in memory consumption;
        # however, the substantial speedup achieved makes it worthwhile.
        hyperparams.cuda_graphs_static_batch_size = 64
        hyperparams.cuda_graphs_static_max_seq_len = 128
        hyperparams.cuda_graphs_static_max_context_len = 128

    hyperparams.is_paged = False
    model = init_te_gemma_model(hyperparams)

    print_sample_of_generated_texts(model)
    benchmark_generation(model)

