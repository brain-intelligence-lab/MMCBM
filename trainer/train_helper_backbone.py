# project imports
from utils.logger import CSVLogs, char_color
from trainer.trainer import SingleEpoch
from utils.EarlyStop import EarlyStopping
import warnings
from visualize.utils import ActPlot
from params import modality_model_map
from trainer.train_helper import TrainHelper, get_stage1_metrics_from_top_stage2, BatchLogger

warnings.filterwarnings("ignore")


class TrainHelperBackbone(TrainHelper):
    def __init__(self, args, model, optimizer):
        super().__init__(args, model, optimizer)
        args.clip_name = 'backbone'

    def get_epochs(self, logger):
        batch_loggers = {
            m: BatchLogger(
                model_modality=m,
                metrics=self.args.metrics,
                metrics_logger=self.metrics_logger if not self.args.plot else None,
                pred_logger=self.pred_logger,
                early_stopper=EarlyStopping(dir=f'{self.args.output_dir}/{self.args.dir_name}',
                                            name=m,
                                            logger=logger,
                                            patience=self.args.patience,
                                            mode=self.args.mode),
            ) for m in modality_model_map[self.args.modality]
        }
        trainepoch = SingleEpoch(
            model=self.model,
            loss=self.args.loss,
            optimizer=self.optimizer,
            stage_name='train',
            device=self.args.device,
            batch_loggers=batch_loggers,
            plot_fn=ActPlot(dir=f'{self.args.output_dir}/{self.args.dir_name}') if self.args.plot else None,
            mix_up_alpha=self.args.mix_up_alpha,
        )
        validepoch = SingleEpoch(
            model=self.model,
            loss=self.args.loss,
            optimizer=self.optimizer,
            stage_name='valid',
            device=self.args.device,
            batch_loggers=batch_loggers,
            plot_fn=ActPlot(dir=f'{self.args.output_dir}/{self.args.dir_name}') if self.args.plot else None,
            mix_up_alpha=self.args.mix_up_alpha,
        )
        testepoch = SingleEpoch(
            model=self.model,
            loss=self.args.loss,
            optimizer=self.optimizer,
            stage_name='test',
            device=self.args.device,
            batch_loggers=batch_loggers,
            plot_fn=ActPlot(dir=f'{self.args.output_dir}/{self.args.dir_name}') if self.args.plot else None,
            mix_up_alpha=self.args.mix_up_alpha,
        )
        return trainepoch, validepoch, testepoch

    def infer(self):
        args = self.args
        train_loader, val_loader, test_loader = self.loaders
        trainepoch, validepoch, testepoch = self.get_epochs(logger=None)
        # record learning rate
        print(char_color(f"Stage: Infer. path: {args.dir_name}", color='blue'))
        testepoch.plot_epoch(test_loader)

        validepoch(val_loader)
        testepoch(test_loader)
        dir_path = f'{args.output_dir}/{args.dir_name}'
        get_stage1_metrics_from_top_stage2(path=dir_path,
                                           file_name=f'metrics_output_{args.idx}',
                                           stage1='valid',
                                           top_k=1)
        get_stage1_metrics_from_top_stage2(path=dir_path,
                                           file_name=f'metrics_output_{args.idx}',
                                           stage1='test',
                                           stage2='valid',
                                           top_k=1)
