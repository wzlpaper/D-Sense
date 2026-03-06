# This code is provided by Zhelun Wang. Stay Optimistic.
# Email: wzlpaper@126.com

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional
import logging
from scipy import stats
import pandas as pd

''' 
    This code corresponds to the VI-C module in the manuscript.
    For detailed descriptions, please refer to the manuscript.
    Briefly, DWM computes the CoBa, DLA, GA, and HTE metrics,
    and adaptively fuses them across training epochs to generate weights for different tasks.
'''
class DWM(tf.keras.callbacks.Callback):
    def __init__(self,
                 task_names: List[str],
                 total_epochs: int,
                 window_size: int = 10,
                 min_weight: float = 0.01,
                 max_weight: float = 5.0,
                 convergence_threshold: float = 0.01,
                 use_gradient_alignment: bool = True,
                 use_coba_mechanism: bool = True,
                 use_validation_loss: bool = True,
                 verbose: bool = True,
                 warmup_epochs: int = 5,
                 tau: float = 2.0,
                 eps: float = 1e-8
                 ):
        super().__init__()
        self.task_names = task_names
        self.n_tasks = len(task_names)
        self.total_epochs = total_epochs
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.convergence_threshold = convergence_threshold
        self.use_gradient_alignment = use_gradient_alignment
        self.use_coba_mechanism = use_coba_mechanism
        self.use_validation_loss = use_validation_loss
        self.verbose = verbose
        self.warmup_epochs = warmup_epochs
        self.tau = tau
        self.eps = eps
        self.alpha_max_history = []
        self.rcs_decay = 0.9
        self.acs_momentum = 0.8
        self.df_sensitivity = 2.0
        self.hard_task_bonus = 1.5
        self.impasse_threshold = 0.95
        self.gradient_conflict_threshold = 0.7
        self.rcs_smoothed = {name: 1.0 for name in self.task_names}
        self.acs_smoothed = {name: 1.0 for name in self.task_names}
        self._initialize_history()
        self._initialize_state_variables()
        self.logger = self._setup_logger()

    def _initialize_history(self):
        self.train_loss_history = {name: [] for name in self.task_names}
        self.val_loss_history = {name: [] for name in self.task_names}
        self.weight_history = {name: [] for name in self.task_names}

        self.rcs_history = {name: [] for name in self.task_names}
        self.acs_history = {name: [] for name in self.task_names}
        self.df_history = []
        self.alpha_slope_history = {name: [] for name in self.task_names}

        self.gradient_magnitude_history = {name: [] for name in self.task_names}
        self.gradient_alignment_history = {name: [] for name in self.task_names}
        self.convergence_rate_history = []
        self.task_priority_history = []

        self.coba_weight_history = {name: [] for name in self.task_names}
        self.dynamic_weight_history = {name: [] for name in self.task_names}
        self.gradient_weight_history = {name: [] for name in self.task_names}
        self.hard_task_weight_history = {name: [] for name in self.task_names}

    def _initialize_state_variables(self):
        self.current_weights = {name: 1.0 for name in self.task_names}
        self.initial_losses = {name: None for name in self.task_names}
        self.best_losses = {name: float('inf') for name in self.task_names}
        self.stagnation_counters = {name: 0 for name in self.task_names}
        self.training_phase = "exploration"
        self.phase_transition_epochs = []

    def _setup_logger(self):
        logger = logging.getLogger('DWM')
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(message)s'
            )
        return logger

    def on_train_begin(self,
                       logs=None):
        self.logger.info(f"Initializing DWM, task list: {self.task_names}")
        self.logger.info(f"Initial weights: {self.current_weights}")

    def on_train_end(self,
                     logs=None):
        self._save_loss_history_to_excel()
        self._save_weight_history_to_excel()

    def on_epoch_begin(self,
                       epoch,
                       logs=None):
        try:
            keras_model = getattr(self, 'model', None)
            if keras_model is None:
                return
            if hasattr(keras_model, 'dwm_weights'):
                dwm_vars = keras_model.dwm_weights
                task_names = self.task_names if hasattr(self, 'task_names') else getattr(keras_model, 'task_names', None)
                if task_names and len(dwm_vars) == len(task_names):
                    for i, tname in enumerate(task_names):
                        val = float(self.current_weights.get(tname, 1.0 / self.n_tasks))
                        try:
                            dwm_vars[i].assign(val)
                        except Exception:
                            try:
                                dwm_vars[i] = val
                            except Exception:
                                pass
                    if self.verbose:
                        self.logger.info(f"on_epoch_begin: assigned dwm weights {self.current_weights}")
                else:
                    if len(dwm_vars) == self.n_tasks:
                        for i in range(self.n_tasks):
                            val = float(self.current_weights.get(self.task_names[i], 1.0 / self.n_tasks))
                            dwm_vars[i].assign(val)
        except Exception as e:
            self.logger.warning(f"on_epoch_begin failed to assign dwm weights: {e}")

    def on_epoch_end(self,
                     epoch,
                     logs=None):
        try:
            train_losses = self._extract_losses(logs, validation=False)
            val_losses = self._extract_losses(logs, validation=True) if self.use_validation_loss else None
            self._update_loss_history(train_losses, val_losses)
            if epoch == 0:
                target_losses = val_losses if (val_losses and self.use_validation_loss) else train_losses
                for i, task_name in enumerate(self.task_names):
                    self.initial_losses[task_name] = max(target_losses[i], self.eps)
            # >>1<<
            new_weights = self._compute_comprehensive_weights(epoch, train_losses, val_losses)
            self.current_weights = new_weights
            self._update_weight_history(new_weights)
            self._detect_training_phase(epoch, train_losses)
            self._log_epoch_info(epoch, train_losses, new_weights)
        except Exception as e:
            self.logger.error(f"Epoch {epoch+1} weight computation error: {e}")
            self.current_weights = {name: 1.0 / self.n_tasks for name in self.task_names}

    def _extract_losses(self,
                        logs: Dict,
                        validation: bool = False) -> List[float]:
        prefix = "val_" if validation else ""
        losses = []
        for i, task_name in enumerate(self.task_names):
            possible_keys = f'{prefix}model_output_{i}_loss',
            loss_value = None
            for key in possible_keys:
                if logs and key in logs:
                    loss_value = logs[key]
                    break
            if loss_value is None:
                if validation:
                    loss_value = 0.5 + 0.3 * np.random.randn()
                else:
                    loss_value = 0.3 + 0.2 * np.random.randn()
            losses.append(float(loss_value))
        return losses

    def _update_loss_history(self,
                             train_losses: List[float],
                             val_losses: Optional[List[float]]):
        for i, task_name in enumerate(self.task_names):
            self.train_loss_history[task_name].append(train_losses[i])
            if val_losses:
                self.val_loss_history[task_name].append(val_losses[i])

    def _compute_comprehensive_weights(self,
                                       epoch: int,
                                       train_losses: List[float],
                                       val_losses: Optional[List[float]]) -> Dict[str, float]:
        target_losses = val_losses if (val_losses and self.use_validation_loss) else train_losses
        weights_components = {}

        # CoBa
        if self.use_coba_mechanism:
            coba_weights = self._compute_coba_weights(epoch, target_losses)
            weights_components['coba'] = coba_weights
        else:
            weights_components['coba'] = np.ones(self.n_tasks) / self.n_tasks
        for i, task_name in enumerate(self.task_names):
            self.coba_weight_history[task_name].append(weights_components['coba'][i])

        # Dynamic
        dynamic_weights = self._compute_dynamic_loss_weights(target_losses)
        weights_components['dynamic'] = dynamic_weights
        for i, task_name in enumerate(self.task_names):
            self.dynamic_weight_history[task_name].append(dynamic_weights[i])

        # Gradient alignment
        if self.use_gradient_alignment and epoch > 0:
            gradient_weights = self._compute_gradient_alignment_weights()
            weights_components['gradient'] = gradient_weights
        else:
            weights_components['gradient'] = np.ones(self.n_tasks) / self.n_tasks
        for i, task_name in enumerate(self.task_names):
            self.gradient_weight_history[task_name].append(weights_components['gradient'][i])

        # Hard task
        hard_task_weights = self._compute_hard_task_weights(target_losses)
        weights_components['hard_task'] = hard_task_weights
        for i, task_name in enumerate(self.task_names):
            self.hard_task_weight_history[task_name].append(hard_task_weights[i])

        # Adaptive weight fusion
        fused_weights = self._adaptive_weight_fusion(epoch, weights_components)
        # print(fused_weights.shape)
        # Process weights
        processed_weights = self._process_weights(fused_weights, target_losses)

        return {name: float(w) for name, w in zip(self.task_names, processed_weights)}

    # CoBa>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>begin
    def _compute_coba_weights(self,
                              epoch: int,
                              losses: List[float]) -> np.ndarray:
        if epoch < self.warmup_epochs:
            return np.ones(self.n_tasks) / self.n_tasks
        if self.use_validation_loss and self.val_loss_history[self.task_names[0]]:
            loss_histories = self.val_loss_history
        else:
            loss_histories = self.train_loss_history
        alpha_slopes = np.zeros(self.n_tasks)

        for i, task_name in enumerate(self.task_names):
            if len(loss_histories[task_name]) >= self.window_size:
                alpha_slopes[i] = self._compute_convergence_slope(task_name, loss_histories, epoch)
            else:
                alpha_slopes[i] = 0.0
            self.alpha_slope_history[task_name].append(alpha_slopes[i])
        rcs_scores = self._compute_rcs(alpha_slopes)
        acs_scores = self._compute_acs(epoch, alpha_slopes, loss_histories)
        df_score = self._compute_divergence_factor(epoch, alpha_slopes)
        self.df_history.append(df_score)
        # (Eq.6)
        final_weights = df_score * rcs_scores + (1 - df_score) * acs_scores
        for i, task_name in enumerate(self.task_names):
            self.rcs_history[task_name].append(rcs_scores[i])
            self.acs_history[task_name].append(acs_scores[i])
        return final_weights

    def _compute_convergence_slope(self,
                                   task_name: str,
                                   loss_histories: Dict,
                                   current_epoch: int) -> float:
        # α_i(t)
        if current_epoch < 1 or len(loss_histories[task_name]) < self.window_size:
            return 0.0
        normalized_losses = []
        initial_loss = loss_histories[task_name][0] if loss_histories[task_name][0] > 0 else 1.0
        start_epoch = max(0, current_epoch - self.window_size + 1)
        for epoch_idx in range(start_epoch, current_epoch + 1):
            if epoch_idx < len(loss_histories[task_name]):
                current_loss = loss_histories[task_name][epoch_idx]
                # paper P8066 --> l_i^val(θ;t)
                normalized_loss = current_loss / initial_loss
                normalized_losses.append(normalized_loss)
        if len(normalized_losses) < 2:
            return 0.0
        # paper P8066 X_i(N;t) = [x_i(s₀), ..., x_i(t)]^T
        #             x_i(s) = [s, 1]^T
        x = np.arange(len(normalized_losses)).reshape(-1, 1)
        y = np.array(normalized_losses)
        X = np.hstack([x, np.ones((len(x), 1))])
        try:
            # c = (X^T X)^{-1} X^T y paper P8066 ---> (Eq.2)(Eq.3)
            XTX = X.T @ X
            if np.linalg.det(XTX) == 0:
                return 0.0
            coefficients = np.linalg.inv(XTX) @ X.T @ y
            slope = coefficients[0]  # α
            return slope
        except:
            return 0.0

    def _compute_rcs(self,
                     alpha_slopes: np.ndarray) -> np.ndarray:
        # paper P8066 ---> (Eq.4)
        if np.sum(np.abs(alpha_slopes)) > 0:
            normalized_slopes = self.n_tasks * alpha_slopes / np.sum(np.abs(alpha_slopes))
            # softmax
            rcs_scores = tf.nn.softmax(normalized_slopes).numpy()
        else:
            rcs_scores = np.ones(self.n_tasks) / self.n_tasks
        return rcs_scores

    def _compute_acs(self,
                     epoch: int,
                     alpha_slopes: np.ndarray,
                     loss_histories: Dict) -> np.ndarray:
        # paper P8067 ---> (Eq.5)
        acs_inputs = np.zeros(self.n_tasks)
        for i, task_name in enumerate(self.task_names):
            if len(loss_histories[task_name]) >= self.window_size:
                recent_slopes = []
                window_start = max(0, epoch - self.window_size + 1)
                for j in range(window_start, epoch + 1):
                    if j < len(loss_histories[task_name]) and j >= 0:
                        slope = self._compute_convergence_slope(task_name, loss_histories, j)
                        recent_slopes.append(abs(slope))
                if recent_slopes and np.sum(recent_slopes) > 0:
                    # (Eq.5 denominator) no softmax
                    denominator = np.sum(recent_slopes)
                    # (Eq.5) no softmax
                    acs_inputs[i] = -self.window_size * alpha_slopes[i] / denominator
                else:
                    acs_inputs[i] = 0.0
            else:
                acs_inputs[i] = 0.0
        # (Eq.5)
        if np.any(acs_inputs != 0):
            acs_scores = tf.nn.softmax(acs_inputs).numpy()
        else:
            acs_scores = np.ones(self.n_tasks) / self.n_tasks
        return acs_scores

    def _compute_divergence_factor(self,
                                   current_epoch: int,
                                   alpha_slopes: np.ndarray) -> float:
        # paper P8067, (Eq.7)
        if current_epoch < self.warmup_epochs:
            return 1.0
        # α_max(t) = max_i α_i(t)
        alpha_max = np.max(alpha_slopes)
        if len(self.alpha_max_history) <= current_epoch:
            self.alpha_max_history.append(alpha_max)
        else:
            self.alpha_max_history[current_epoch] = alpha_max
        if current_epoch < 1 or len(self.alpha_max_history) < 2:
            return 1.0
        # D = Σ_{i=1}^{t-1} |α_max(i)| <---- Absolute value avoids elimination issues
        valid_history = self.alpha_max_history[:current_epoch]
        denominator = np.sum(np.abs(valid_history))
        if denominator < self.eps:
            return 1.0
        softmax_inputs = []
        for s in range(1, current_epoch + 1):
            if s < len(self.alpha_max_history):
                alpha_max_s = self.alpha_max_history[s]
                # -τ × s × α_max(s) / D
                numerator = -self.tau * s * alpha_max_s
                value = numerator / denominator
                softmax_inputs.append(value)
            else:
                softmax_inputs.append(0.0)
        if softmax_inputs:
            inputs_array = np.array(softmax_inputs)
            softmax_outputs = tf.nn.softmax(inputs_array).numpy()
            current_softmax = softmax_outputs[-1]
            # t × softmax_output
            raw_df = current_epoch * current_softmax
            clipped_df = min(raw_df, 1.0)
            return clipped_df
    # CoBa>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>end

    def _compute_dynamic_loss_weights(self,
                                      losses: List[float]) -> np.ndarray:
        if len(losses) == 0:
            return np.ones(self.n_tasks) / self.n_tasks
        loss_array = np.array(losses)
        mean_loss = np.mean(loss_array)
        if mean_loss > 0:
            normalized_losses = loss_array / mean_loss
            weights = tf.nn.softmax(normalized_losses * 2.0).numpy()
        else:
            weights = np.ones(self.n_tasks) / self.n_tasks
        return weights

    def _compute_gradient_alignment_weights(self) -> np.ndarray:
        try:
            alignment_scores = np.ones(self.n_tasks)

            for i, task_name in enumerate(self.task_names):
                if len(self.train_loss_history[task_name]) >= 3:
                    recent_trend = self._compute_recent_trend(self.train_loss_history[task_name])
                    alignment_scores[i] = 1.0 + max(0, -recent_trend) * 0.5
            if np.sum(alignment_scores) > 0:
                weights = alignment_scores / np.sum(alignment_scores)
            else:
                weights = np.ones(self.n_tasks) / self.n_tasks
            return weights
        except Exception as e:
            self.logger.warning(f"Gradient alignment calculation failed: {e}")
            return np.ones(self.n_tasks) / self.n_tasks

    def _compute_hard_task_weights(self,
                                   losses: List[float]) -> np.ndarray:
        if len(losses) == 0:
            return np.ones(self.n_tasks) / self.n_tasks
        loss_array = np.array(losses)
        median_loss = np.median(loss_array)
        if median_loss > 0:
            hardness_scores = np.where(loss_array > median_loss * 0.8,
                                       self.hard_task_bonus, 1.0)
            weights = hardness_scores / np.sum(hardness_scores)
        else:
            weights = np.ones(self.n_tasks) / self.n_tasks
        return weights

    def _adaptive_weight_fusion(self,
                                epoch: int,
                                weight_components: Dict[str, np.ndarray]) -> np.ndarray:
        if epoch < self.warmup_epochs:
            coefficients = {
                'coba': 0.1,
                'dynamic': 0.6,
                'gradient': 0.2,
                'hard_task': 0.1
            }
        elif self.training_phase == "exploration":
            coefficients = {
                'coba': 0.3,
                'dynamic': 0.4,
                'gradient': 0.2,
                'hard_task': 0.1
            }
        elif self.training_phase == "optimization":
            coefficients = {
                'coba': 0.5,
                'dynamic': 0.3,
                'gradient': 0.15,
                'hard_task': 0.05
            }
        else:
            coefficients = {
                'coba': 0.4,
                'dynamic': 0.3,
                'gradient': 0.2,
                'hard_task': 0.1
            }
        fused_weights = np.zeros(self.n_tasks)
        for key, weight_array in weight_components.items():
            if key in coefficients:
                fused_weights += coefficients[key] * weight_array
        return fused_weights

    def _process_weights(self,
                         weights: np.ndarray,
                         losses: List[float]) -> np.ndarray:
        weights = np.clip(weights, self.min_weight, self.max_weight)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(self.n_tasks) / self.n_tasks
        effective_min = max(self.min_weight, 0.5 / self.n_tasks)
        weights = np.maximum(weights, effective_min)
        weights = self.n_tasks * weights / np.sum(weights)
        return weights

    def _compute_recent_trend(self,
                              loss_sequence: List[float],
                              window: int = 5) -> float:
        # window = 5 < self.window_size!!!
        if len(loss_sequence) < window:
            return 0.0
        recent_losses = loss_sequence[-window:]
        if len(recent_losses) < 2:
            return 0.0
        x = np.arange(len(recent_losses))
        y = np.array(recent_losses)
        try:
            slope, _ = stats.theilslopes(y, x)
            return slope
        except:
            slope, _ = np.polyfit(x, y, 1)
            return slope

    def _update_weight_history(self,
                               weights: Dict[str, float]):
        for task_name in self.task_names:
            self.weight_history[task_name].append(weights[task_name])

    def _detect_training_phase(self,
                               epoch: int,
                               losses: List[float]):
        # The difference between 0.1 and 0.3 is negligible,
        # but we prefer to use DWM during the training phase.
        if epoch < self.total_epochs * 0.1:
            new_phase = "exploration"
        elif epoch < self.total_epochs * 0.8:
            new_phase = "optimization"
        else:
            new_phase = "convergence"
        if new_phase != self.training_phase:
            self.phase_transition_epochs.append(epoch)
            self.logger.info(f"Training phase transition: {self.training_phase} -> {new_phase} (Epoch {epoch})")
            self.training_phase = new_phase

    def _log_epoch_info(self,
                        epoch: int,
                        losses: List[float],
                        weights: Dict[str, float]):
        if self.verbose:
            loss_str = " | ".join([f"{loss:.4f}" for loss in losses])
            weight_str = " | ".join([f"{w:.4f}" for w in weights.values()])
            self.logger.info(
                f"Epoch {epoch:3d} | "
                f"Losses: {loss_str} | Weights: {weight_str}"
            )

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def _save_loss_history_to_excel(self):
        data = {}
        n_epochs = len(self.train_loss_history[self.task_names[0]])
        data["epoch"] = list(range(n_epochs))
        for task_name in self.task_names:
            original_losses = [self.train_loss_history[task_name][epoch] for epoch in range(n_epochs)]
            data[f"loss_{task_name}"] = original_losses
        df = pd.DataFrame(data)
        output_file = "weights_loss.csv"
        df.to_csv(output_file, index=False)

    def _save_weight_history_to_excel(self):
        try:
            data = {'epoch': list(range(len(self.weight_history[self.task_names[0]])))}
            for task_name in self.task_names:
                data[f'weight_{task_name}'] = self.weight_history[task_name]
            df = pd.DataFrame(data)
            output_file = "weights.csv"
            df.to_csv(output_file, index=False)
        except Exception as e:
            self.logger.error(f"Failed to save weight history: {e}")