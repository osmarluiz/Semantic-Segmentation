import sys
import torch
from tqdm import tqdm
from .meter import AverageValueMeter

class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = [f'{k} - {v:.4f}' for k, v in logs.items()]
        return ', '.join(str_logs)

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not self.verbose) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                try:
                    loss, y_pred = self.batch_update(x, y)
                except Exception as e:
                    print(f"Error during batch update: {e}")
                    continue

                loss_value = loss.detach()
                loss_meter.add(loss_value.item())
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).detach()
                    metrics_meters[metric_fn.__name__].add(metric_value.item())
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs

class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(model=model, loss=loss, metrics=metrics, stage_name='train', device=device, verbose=verbose)
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction

class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(model=model, loss=loss, metrics=metrics, stage_name='valid', device=device, verbose=verbose)

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model(x)
            loss = self.loss(prediction, y)
        return loss, prediction

class TestEpoch(Epoch):
    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(model=model, loss=loss, metrics=metrics, stage_name='test', device=device, verbose=verbose)

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model(x)
            loss = self.loss(prediction, y)
        return loss, prediction
