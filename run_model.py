import argparse
import sys
from logging import getLogger
from pathlib import Path

from config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from utils import get_model
from trainer import Trainer_ours


PROJECT_ROOT = Path(__file__).resolve().parent


def _config_files(model_name: str, dataset_name: str):
    files = [
        PROJECT_ROOT / "props" / "overall.yaml",
        PROJECT_ROOT / "props" / f"{dataset_name}.yaml",
        PROJECT_ROOT / "props" / f"{model_name}.yaml",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing configuration file(s):\n  " + "\n  ".join(missing)
            + "\nCheck --model/--dataset spellings and the props directory."
        )
    return [str(path) for path in files]


def run_model(
    model_name="PAEF_LightGCN",
    dataset_name="Amazon_Baby",
    fairness_type=None,
    data_path=None,
):
    """Run the paper-aligned model/trainer pair and return evaluation results.

    `fairness_type` is retained for backward-compatible launch scripts only.
    The revised PAEF-LightGCN has one provider-exposure fairness objective,
    controlled by DRM_weight (= paper lambda), so fairness_type has no effect
    on that model's loss.
    """
    props = _config_files(model_name, dataset_name)
    model_class = get_model(model_name)

    # RecBole appends the dataset name to data_path. Point it to the project
    # dataset *root* so the script works from any working directory.
    dataset_root = (
        Path(data_path).expanduser().resolve()
        if data_path is not None
        else PROJECT_ROOT / "dataset"
    )
    overrides = {"data_path": str(dataset_root)}
    if fairness_type is not None:
        overrides["fairness_type"] = fairness_type

    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=overrides,
    )

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info("Invocation: %s", sys.argv)
    logger.info("Config files: %s", props)
    logger.info("Final config: %s", config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    local_rank = config["local_rank"] if config["local_rank"] is not None else 0
    init_seed(config["seed"] + local_rank, config["reproducibility"])

    # The public dataloader.dataset property is also used by our trainer and
    # evaluation collector; avoid the private train_data._dataset attribute.
    model = model_class(config, train_data.dataset).to(config["device"])
    logger.info(model)
    trainer = Trainer_ours(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    return (
        model_name,
        dataset_name,
        {
            "best_valid_score": best_valid_score,
            "valid_score_bigger": config["valid_metric_bigger"],
            "best_valid_result": best_valid_result,
            "test_result": test_result,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run PAEF experiments")
    parser.add_argument(
        "--model", "-m", default="PAEF_LightGCN", help="Model class name"
    )
    parser.add_argument(
        "--dataset", "-d", default="Amazon_Baby", help="Dataset directory name"
    )
    parser.add_argument(
        "--fairness_type", default=None,
        help="Optional legacy flag; PAEF's provider fairness is always used",
    )
    parser.add_argument(
        "--data_path", default=None,
        help="Dataset root directory (default: <project>/dataset)",
    )
    # The custom Config class also parses additional --key=value overrides.
    args, _ = parser.parse_known_args()
    run_model(args.model, args.dataset, args.fairness_type, args.data_path)


if __name__ == "__main__":
    main()
