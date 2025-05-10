import torch
import pathlib
import os

TENSOR_DUMP_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent.parent / "tensor_dumps" / "distributed"
)
tensor_dump_dir_env = os.getenv("NVTE_TEST_DISTRIBUTED_EXACT_TENSOR_DUMP_DIR")
if tensor_dump_dir_env is not None:
    TENSOR_DUMP_DIR = pathlib.Path(tensor_dump_dir_env)

# per-recipe dump directories:
PER_RECIPE_DUMP_DIRS = {
    "fp8": TENSOR_DUMP_DIR / "fp8",
    "mxfp8": TENSOR_DUMP_DIR / "mxfp8",
    "fp8_cs": TENSOR_DUMP_DIR / "fp8_cs",
    "fp8_block_scaling": TENSOR_DUMP_DIR / "fp8_block_scaling",
    "none": TENSOR_DUMP_DIR / "none",
}


def get_dump_dir(recipe):
    ret = None
    if recipe is None:
        ret = PER_RECIPE_DUMP_DIRS["none"]
    else:
        ret = PER_RECIPE_DUMP_DIRS[recipe]
    # make sure the directory exists
    ret.mkdir(parents=True, exist_ok=True)
    return ret


def maybe_dump_outputs(
    output_y,
    test_kwargs,
    prefix="",
    recipe=None,
    parallel_mode="column",
    sequence_parallel=False,
    check_rank=0,
):
    # skip if not check rank
    if torch.distributed.get_rank() != check_rank:
        return
    if test_kwargs != {}:
        return
    # only dump for fp8 block scaling
    if recipe != "fp8_block_scaling":
        return

    # encode dump name prefix
    current_seed = torch.random.initial_seed()
    world_size = torch.distributed.get_world_size()
    tensor_name_prefix = f"{prefix}_Y_World{world_size}Rank{check_rank}_ParMode_{parallel_mode}_SeqPar_{sequence_parallel}_{recipe}_{current_seed}"

    tensor_shape_string = "x".join([str(x) for x in output_y.shape])
    tensor_dtype_string = str(output_y.dtype)
    tensor_name_prefix = (
        f"{tensor_name_prefix}_Shape{tensor_shape_string}_DType{tensor_dtype_string}"
    )

    final_path = get_dump_dir(recipe) / f"{tensor_name_prefix}.pt"
    golden_path = get_dump_dir(recipe) / "golden" / f"{tensor_name_prefix}.pt"

    torch.save(output_y, final_path)

    # if golden file exists, load it and compare with zero tolerance
    if golden_path.exists():
        golden_tensor = torch.load(golden_path)
        torch.testing.assert_close(output_y, golden_tensor, atol=0, rtol=0)


def maybe_dump_gradients(
    model_dist,
    test_kwargs,
    prefix="",
    recipe=None,
    parallel_mode="column",
    sequence_parallel=False,
    check_rank=0,
):
    # skip if not check rank
    if torch.distributed.get_rank() != check_rank:
        return
    if test_kwargs != {}:
        return
    # only dump for fp8 block scaling
    if recipe != "fp8_block_scaling":
        return

    current_seed = torch.random.initial_seed()
    world_size = torch.distributed.get_world_size()

    # fetch gradients from model_dist
    for i, (name, param_d) in enumerate(model_dist.named_parameters()):
        # only check for weight gradient of linear layer for now
        if "weight" in name:
            tensor_name_prefix = f"{prefix}_{name}_World{world_size}Rank{check_rank}_ParMode_{parallel_mode}_SeqPar_{sequence_parallel}_{recipe}_{current_seed}"

            tensor_shape_string = "x".join([str(x) for x in param_d.shape])
            tensor_dtype_string = str(param_d.dtype)
            tensor_name_prefix = (
                f"{tensor_name_prefix}_Shape{tensor_shape_string}_DType{tensor_dtype_string}"
            )

            final_path = get_dump_dir(recipe) / f"{tensor_name_prefix}.pt"
            golden_path = get_dump_dir(recipe) / "golden" / f"{tensor_name_prefix}.pt"

            torch.save(param_d, final_path)

            # if golden file exists, load it and compare with zero tolerance
            if golden_path.exists():
                golden_tensor = torch.load(golden_path)
                torch.testing.assert_close(param_d, golden_tensor, atol=0, rtol=0)
