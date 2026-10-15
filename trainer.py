import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from logging import getLogger
from time import time
from evaluator import Evaluator, Collector
from recbole.trainer import Trainer as RecBole_Trainer
from recbole.data.dataloader import UserDataLoader
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    dict2str,
    get_tensorboard,
    set_color,
    WandbLogger,
)

class Trainer(RecBole_Trainer):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()
        self.tensorboard = get_tensorboard(self.logger)
        self.wandblogger = WandbLogger(config)
        self.learner = config["learner"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.stopping_step = config["stopping_step"]
        self.clip_grad_norm = config["clip_grad_norm"]
        self.valid_metric = config["valid_metric"].lower()
        self.valid_metric_bigger = config["valid_metric_bigger"]
        self.test_batch_size = config["eval_batch_size"]
        self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
        self.device = config["device"]
        self.checkpoint_dir = config["checkpoint_dir"]
        self.enable_amp = config["enable_amp"]
        self.enable_scaler = torch.cuda.is_available() and config["enable_scaler"]
        ensure_dir(self.checkpoint_dir)
        saved_model_file = "{}-{}.pth".format(self.config["model"], get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config["weight_decay"]

        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()
        self.eval_type = config["eval_type"]
        self.eval_collector = Collector(config)
        self.evaluator = Evaluator(config)
        self.item_tensor = None
        self.tot_item_num = None

    def train_weight_epoch(
        self, weight_epoch_idx=0, weight_data=None, show_progress=False
    ):
        self.model.train()
        total_loss = None
        iter_data = (
            tqdm(
                weight_data,
                total=len(weight_data),
                ncols=100,
                desc=set_color(f"Train weight epoch {weight_epoch_idx:>5}", "pink"),
            )
            if show_progress
            else weight_data
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            optimizer.zero_grad()
            losses = self.model.weight_loss(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
            else:
                loss = losses
                total_loss = (
                    losses.item() if total_loss is None else total_loss + losses.item()
                )
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            optimizer.step()
        self.logger.info(
            "Train Weiht Loss " + str(weight_epoch_idx) + " loss: " + str(total_loss)
        )
        return total_loss

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)
        valid_step = 0

        num_users = train_data.dataset.num(self.config["USER_ID_FIELD"])
        user_counter = train_data.dataset.user_counter

        user_popularity = np.zeros(num_users)
        for user_id, count_user in user_counter.items():
            user_popularity[user_id] = count_user

        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()

           
            train_loss = self._train_epoch(
                train_data, epoch_idx, show_progress=show_progress
            )

            self.train_loss_dict[epoch_idx] = (
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            )
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            if verbose:
                self.logger.info(train_loss_output)

            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics(
                {"epoch": epoch_idx, "train_loss": train_loss, "train_step": epoch_idx},
                head="train",
            )

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(
                    valid_data, show_progress=show_progress
                )
                (
                    self.best_valid_score,
                    self.cur_step,
                    stop_flag,
                    update_flag,
                ) = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger,
                )
                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("valid_score", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = (
                    set_color("valid result", "blue") + ": \n" + dict2str(valid_result)
                )
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar("Vaild_score", valid_score, epoch_idx)
                self.wandblogger.log_metrics(
                    {**valid_result, "valid_step": valid_step}, head="valid"
                )

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = "Finished training, best eval result in epoch %d" % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step += 1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result


class Trainer_ours(Trainer):

    def __init__(self, config, model):
        super().__init__(config, model)

        self.learning_rate_org = float(self._optional_config(
            config, "learning_rate_org", self.learning_rate
        ))
        self.learning_rate_fair = float(self._optional_config(
            config, "learning_rate_fair", self.learning_rate
        ))
        if self.learning_rate_org <= 0 or self.learning_rate_fair <= 0:
            raise ValueError("learning_rate_org and learning_rate_fair must be positive")

        required_methods = ("calculate_outer_loss", "calculate_inner_loss")
        for name in required_methods:
            if not callable(getattr(model, name, None)):
                raise TypeError(
                    "Trainer_ours requires the revised PAEF model with "
                    f"{name}(interaction)."
                )

        try:
            self.outer_params = (
                list(model.user_embedding.embedding_org.parameters())
                + list(model.item_embedding.embedding_org.parameters())
            )
            self.inner_params = (
                list(model.user_embedding.embedding_fair.parameters())
                + list(model.item_embedding.embedding_fair.parameters())
            )
        except AttributeError as error:
            raise TypeError(
                "Trainer_ours requires user_embedding/item_embedding with "
                "embedding_org and embedding_fair modules. Replace the model "
                "file with the revised paef_lightgcn.py."
            ) from error

        outer_ids = {id(p) for p in self.outer_params}
        inner_ids = {id(p) for p in self.inner_params}
        if not outer_ids or not inner_ids or outer_ids & inner_ids:
            raise ValueError("Leader and follower parameter groups must be nonempty and disjoint")
        unassigned = [name for name, param in model.named_parameters()
                      if param.requires_grad and id(param) not in outer_ids | inner_ids]
        if unassigned:
            raise ValueError(
                "Trainable parameters not assigned to a PAEF branch: "
                + ", ".join(unassigned)
            )

        self.optimizer = optim.Adam(
            [
                {"params": self.outer_params, "lr": self.learning_rate_org},
                {"params": self.inner_params, "lr": self.learning_rate_fair},
            ],
            weight_decay=self.weight_decay,
        )

        requested_steps = int(self._optional_config(config, "inner_update_steps", 1))
        if requested_steps != 1:
            self.logger.warning(
                "Ignoring inner_update_steps=%d: paper Algorithm 1 requires "
                "one fairness update after each accuracy update.", requested_steps
            )

        if self.enable_amp or self.enable_scaler:
            self.logger.warning(
                "PAEF two-stage training uses full precision (AMP/GradScaler "
                "are not activated by this trainer)."
            )

        self.last_epoch_outer_loss = 0.0
        self.last_epoch_inner_loss = 0.0

    @staticmethod
    def _optional_config(config, key, default):
        try:
            value = config[key]
        except (KeyError, TypeError, AttributeError):
            return default
        return default if value is None else value

    def _train_epoch(self, train_data, epoch_idx, loss_func=None,
                     show_progress=False):
        if loss_func is not None:
            raise ValueError("Trainer_ours requires separate outer/inner losses")

        self.model.train()
       
        if callable(getattr(self.model, "_invalidate_eval_cache", None)):
            self.model._invalidate_eval_cache()

        total_outer_loss = 0.0
        total_inner_loss = 0.0
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"PAEF epoch {epoch_idx:>5}", "pink"),
            )
            if show_progress else train_data
        )

        for interaction in iter_data:
            interaction = interaction.to(self.device)


            self.optimizer.zero_grad(set_to_none=True)
            loss_outer = self.model.calculate_outer_loss(interaction)
            self._check_nan(loss_outer)
            loss_outer.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.outer_params, **self.clip_grad_norm)
            self.optimizer.step()


            self.optimizer.zero_grad(set_to_none=True)
            loss_inner = self.model.calculate_inner_loss(interaction)
            self._check_nan(loss_inner)
            loss_inner.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.inner_params, **self.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_outer_loss += float(loss_outer.detach().item())
            total_inner_loss += float(loss_inner.detach().item())

        self.last_epoch_outer_loss = total_outer_loss
        self.last_epoch_inner_loss = total_inner_loss
        self.logger.info(
            "Epoch %d PAEF losses: accuracy/BPR=%.6f, "
            "lambda*fairness=%.6f, combined=%.6f",
            epoch_idx, total_outer_loss, total_inner_loss,
            total_outer_loss + total_inner_loss,
        )
        return total_outer_loss + total_inner_loss

   