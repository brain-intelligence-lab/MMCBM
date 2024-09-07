# -*- encoding: utf-8 -*-
"""
@Author :   liuyang
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
"""
import pandas as pd

from utils.logger import char_color, CSVLogs, MarkDownLogs
from trainer.trainer import ConceptEpoch
from utils.EarlyStop import EarlyStopping
import warnings
from visualize.utils import ActPlot
from params import modality_model_map
from trainer.train_helper import BatchLogger, TrainHelper

warnings.filterwarnings("ignore")


class TrainHelperMMCBM(TrainHelper):
    def get_epochs(self, logger):
        args = self.args
        batch_loggers = {
            m: BatchLogger(
                model_modality=m,
                metrics=args.metrics,
                metrics_logger=self.metrics_logger if not args.plot else None,
                pred_logger=self.pred_logger,
                early_stopper=EarlyStopping(dir=f'{args.output_dir}/{args.dir_name}',
                                            name=m,
                                            logger=logger,
                                            patience=args.patience,
                                            mode=args.mode),
                wandb=args.wandb,
            ) for m in modality_model_map[args.modality]
        }
        trainepoch = ConceptEpoch(
            model=self.model,
            loss=args.loss,
            optimizer=self.optimizer,
            stage_name='train',
            device=args.device,
            batch_loggers=batch_loggers,
            plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
            concept_bank=args.concept_bank,
            pre_embeddings=args.pre_embeddings and not args.cache_data
        )
        validepoch = ConceptEpoch(
            model=self.model,
            loss=args.loss,
            optimizer=self.optimizer,
            stage_name='valid',
            device=args.device,
            batch_loggers=batch_loggers,
            plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
            concept_bank=args.concept_bank,
            pre_embeddings=args.pre_embeddings and not args.cache_data
        )
        testepoch = ConceptEpoch(
            model=self.model,
            loss=args.loss,
            optimizer=self.optimizer,
            stage_name='test',
            device=args.device,
            batch_loggers=batch_loggers,
            plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
            concept_bank=args.concept_bank,
            pre_embeddings=args.pre_embeddings and not args.cache_data
        )
        return trainepoch, validepoch, testepoch

    def infer(self):
        args = self.args
        val_loader, test_loader = self.loaders[1:]
        # md_logger = MarkDownLogs(dir_path=f'{args.output_dir}/{args.dir_name}')

        csv_logger = CSVLogs(dir_path=f'{args.output_dir}/{args.dir_name}',
                             file_name=f'predict_concepts_{args.analysis_top_k}_{args.analysis_threshold}')
        if not hasattr(args, 'ncc_fn_none'):
            args.ncc_fn_none = None
        infer_epoch = ConceptEpoch(
            model=self.model,
            loss=args.loss,
            optimizer=None,
            stage_name='infer',
            device=args.device,
            batch_loggers=None,
            concept_bank=args.concept_bank,
        )
        print(char_color(f" [Epoch: {0}/{args.epochs}], path: {args.dir_name}"))

        # start epoch
        # infer_epoch(train_loader)
        # infer_epoch(val_loader)
        infer_epoch(val_loader)
        vd = infer_epoch.get_analysis()
        vd['stage'] = 'valid'

        infer_epoch.analysis = []
        infer_epoch(test_loader)
        td = infer_epoch.get_analysis()
        td['stage'] = 'test'
        df = pd.concat([vd, td], axis=0)
        csv_logger(df)
        # infer_epoch.generate_report(md_logger, modality='MM')
