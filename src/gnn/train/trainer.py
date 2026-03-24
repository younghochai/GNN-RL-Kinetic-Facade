import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from ..factory.scheduler_factory import SchedulerFactory
from ..factory.optimizer_factory import OptimizerFactory
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..scaler.feature_scaler import GraphTargetScaler
from IPython.display import display, clear_output

plt.ion()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        train_config: OmegaConf,
        model: nn.Module,
        optimizer_config_name: str,
        scheduler_config_name: str | None,
        save_path: str,
        device: str = 'cuda',
        scaler: GraphTargetScaler | None = None,
    ):
        """
        Trainer class for PyTorch models with optional live plotting and early stopping.
        """

        self.model = model
        self.model_config = model.config
        self.train_config = train_config

        self.loss_fn = self.create_loss_fn(train_config.loss_fn)
        self.device = device
        self.save_path = save_path
        self.patience = train_config.patience
        self.val_steps = train_config.val_steps
        self.live_plot = train_config.live_plot
        self.y_scale = train_config.y_scale
        if self.y_scale:
            assert scaler is not None, "Scaler is required when y_scale is True"
        self.device = device
        self.scaler = scaler

        # Initialize optimizer
        self.optimizer = OptimizerFactory.create_optimizer(self.model, optimizer_config_name)

        # Initialize scheduler
        self.scheduler = SchedulerFactory.create_scheduler(
            config_name=scheduler_config_name,
            optimizer=self.optimizer,
        )

        # Training state variables
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.validation_results = []
        self.learning_rates = []

        # Plot variables
        self.fig, self.ax_train, self.ax_val = None, None, None

    def create_loss_fn(self, loss_fn_name: str):
        if loss_fn_name == 'mse':
            return nn.MSELoss()
        elif loss_fn_name == 'mae':
            return nn.L1Loss()
        elif loss_fn_name == 'smooth_l1':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_fn_name}")

    # ------------------------- Core Training Methods -------------------------

    def train_one_step(self, data):
        """Run one training step and return loss."""
        self.model.train()
        data = data.to(self.device)

        self.optimizer.zero_grad()
        out = self.model(data).view(-1)
        if self.y_scale:
            y = data.scaled_y.float().view(-1)
        else:
            y = data.y.float().view(-1)
        loss = self.loss_fn(out, y)
        loss.backward()
        self.optimizer.step()

        # Track learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)

        return loss.item()

    def eval_epoch(self, loader):
        """Evaluate the model on the entire validation/test dataset."""
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data).view(-1)
                if self.y_scale:
                    y = data.scaled_y.float().view(-1)
                else:
                    y = data.y.float().view(-1)
                loss = self.loss_fn(out, y)
                total_loss += loss.item() * data.num_graphs
        return total_loss / len(loader.dataset)

    # ------------------------- Training Loop -------------------------

    def fit(self, train_loader, val_loader=None, epochs=20, live_plot_step=10):
        """Main training loop with validation, early stopping, and live plotting."""
        logger.info("Training started")
        self._setup_live_plot()
        step, progress_bar = 0, tqdm(total=epochs * len(train_loader), desc="Training")

        self.model.to(self.device)
        for epoch in range(1, epochs + 1):
            for data in train_loader:
                step += 1
                loss = self.train_one_step(data.to(self.device))
                self.train_losses.append(loss)

                if self.live_plot and step % live_plot_step == 0:
                    self._update_live_plot(filename=f"{self.save_path}/live_plot_{self.model.gnn_type}.png")

                if val_loader and step % self.val_steps == 0:
                    if self._run_validation(val_loader, step, epoch, loss):
                        progress_bar.close()
                        return  # Early stopping triggered

                progress_bar.update(1)

            # Update scheduler at the end of each epoch
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # For ReduceLROnPlateau, we need to pass the validation loss
                    if self.val_losses:
                        self.scheduler.step(self.val_losses[-1])
                    else:
                        self.scheduler.step(loss)
                else:
                    # For other schedulers, step at the end of each epoch
                    self.scheduler.step()

        progress_bar.close()
        self._finalize_training()

    def test(self, test_loader, model_path=None):
        """Test the model on the test dataset."""
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.info("Model not loaded from path")
        self.model.eval()
        test_losses = []
        predictions = []
        targets = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.model(data).view(-1)
                if self.y_scale:
                    target = data.scaled_y.float().view(-1)
                else:
                    target = data.y.float().view(-1)
                loss = self.loss_fn(out, target)
                test_losses.append(loss.item())
                if self.y_scale:
                    out = self.scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).reshape(-1)
                    target = data.y.float().view(-1).cpu().numpy()
                    predictions.extend(out)
                    targets.extend(target)
                else:
                    predictions.extend(out.cpu().numpy())
                    targets.extend(target.cpu().numpy())

        predictions = np.array(predictions)
        targets = np.array(targets)
        mean_test_loss = np.mean(test_losses)
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        correlation = np.corrcoef(predictions, targets)[0, 1]

        logger.info("=" * 50)
        logger.info("Test results")
        logger.info(f"Model path: {model_path}")
        logger.info("=" * 50)
        logger.info(f"Mean test loss: {mean_test_loss:.6f}")
        logger.info(f"MSE: {mse:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"Correlation: {correlation:.6f}")
        logger.info("=" * 50)
        return {
            'mean_test_loss': float(mean_test_loss),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation),
            'num_samples': len(predictions),
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
        }

    def _run_validation(self, val_loader, step, epoch, train_loss):
        """Run validation, check early stopping, and update plots."""
        val_loss = self.eval_epoch(val_loader)
        self.val_losses.append(val_loss)

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), f"{self.save_path}/best_model_{self.model.gnn_type}.pt")
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        self.validation_results.append({
            'step': step, 'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'is_best': is_best
        })

        # Update progress bar
        status = "Best" if is_best else f"Patience {self.early_stop_counter}/{self.patience}"
        logger.info(f"Step {step}: Train {train_loss:.4f}, Val {val_loss:.4f}, {status}")

        # Early stopping check
        if self.early_stop_counter >= self.patience:
            logger.info("Early stopping triggered")
            self._save_loss_plot(f"{self.save_path}/early_stop_loss_curve_{self.model.gnn_type}.png")
            return True

        # Update live plot
        if self.live_plot:
            self._update_live_plot(filename=f"{self.save_path}/live_plot_{self.model.gnn_type}.png")
        return False

    def _finalize_training(self):
        """Save final plots and validation results after training finishes."""
        self._save_loss_plot(f"{self.save_path}/final_loss_curve_{self.model.gnn_type}.png")
        if self.validation_results:
            pd.DataFrame(self.validation_results).to_csv("validation_results.csv", index=False)
            logger.info("Validation results saved to validation_results.csv")
        logger.info("Training finished")

    # ------------------------- Plotting -------------------------

    def _setup_live_plot(self):
        """Initialize live plot if enabled."""
        if self.live_plot:
            self.fig, (self.ax_train, self.ax_val, self.ax_lr) = plt.subplots(1, 3, figsize=(20, 6))
            if not self._is_notebook():
                plt.ion()

    def _update_live_plot(self, filename):
        """Update loss plots during training."""
        if not self.fig:
            self._setup_live_plot()
        self._plot_losses(live=True, filename=filename)

    def _save_loss_plot(self, filename):
        """Save loss plots to file."""
        self._plot_losses(live=False, filename=filename)

    def _plot_losses(self, live=False, filename=None):
        """Draw or save training/validation loss plots."""
        def plot(ax, x, y, label, color, ylabel="Loss", log_scale=True):
            ax.plot(x, y, label=label, color=color, alpha=0.8)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
            if log_scale:
                ax.set_yscale('log')

        if live:
            self.ax_train.clear()
            self.ax_val.clear()
            self.ax_lr.clear()
            plot(self.ax_train, range(len(self.train_losses)), self.train_losses, "Train Loss", "blue")
            if self.val_losses:
                steps = [i * self.val_steps for i in range(len(self.val_losses))]
                plot(self.ax_val, steps, self.val_losses, "Val Loss", "red")
            if self.learning_rates:
                plot(self.ax_lr, range(len(self.learning_rates)), self.learning_rates, "Learning Rate", "green", "Learning Rate", False)
            self.fig.tight_layout()
            clear_output(wait=True)
            if self._is_notebook():
                display(self.fig)
            else:
                plt.pause(0.01)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            plot(ax1, range(len(self.train_losses)), self.train_losses, "Train Loss", "blue")
            if self.val_losses:
                steps = [i * self.val_steps for i in range(len(self.val_losses))]
                plot(ax2, steps, self.val_losses, "Val Loss", "red")
            if self.learning_rates:
                plot(ax3, range(len(self.learning_rates)), self.learning_rates, "Learning Rate", "green", "Learning Rate", False)
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Loss curve saved as {filename}")

    # ------------------------- Utilities -------------------------

    def _is_notebook(self):
        """Check if running inside Jupyter Notebook."""
        try:
            from IPython import get_ipython
            return 'IPKernelApp' in get_ipython().config
        except Exception:
            return False
