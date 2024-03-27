# project imports
from utils.logger import char_color, CSVLogs
from trainer.trainer import BatchLogger, ConceptEpoch
from utils.EarlyStop import EarlyStopping
import warnings
from visualize.utils import ActPlot
from params import modality_model_map
from trainer.train_utils import train_fold

warnings.filterwarnings("ignore")


def get_epochs(args, model, optimizer, logger, pred_logger, metrics_logger):
    batch_loggers = {
        m: BatchLogger(
            model_modality=m,
            metrics=args.metrics,
            metrics_logger=metrics_logger if not args.plot else None,
            pred_logger=pred_logger,
            early_stopper=EarlyStopping(dir=f'{args.output_dir}/{args.dir_name}',
                                        name=m,
                                        logger=logger,
                                        patience=args.patience,
                                        mode=args.mode),
        ) for m in modality_model_map[args.modality]
    }
    trainepoch = ConceptEpoch(
        model=model,
        loss=args.loss,
        optimizer=optimizer,
        stage_name='train',
        device=args.device,
        batch_loggers=batch_loggers,
        plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
        concept_bank=args.concept_bank,
    )
    validepoch = ConceptEpoch(
        model=model,
        loss=args.loss,
        optimizer=optimizer,
        stage_name='valid',
        device=args.device,
        batch_loggers=batch_loggers,
        plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
        concept_bank=args.concept_bank,
    )
    testepoch = ConceptEpoch(
        model=model,
        loss=args.loss,
        optimizer=optimizer,
        stage_name='test',
        device=args.device,
        batch_loggers=batch_loggers,
        plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
        concept_bank=args.concept_bank,
    )
    return trainepoch, validepoch, testepoch


def infer(args, model, loaders):
    train_loader, val_loader, test_loader = loaders
    from .logger import MarkDownLogs
    md_logger = MarkDownLogs(dir_path=f'{args.output_dir}/{args.dir_name}')

    csv_logger = CSVLogs(dir_path=f'{args.output_dir}/{args.dir_name}',
                         file_name=f'predict_concepts_{args.analysis_top_k}_{args.analysis_threshold}')
    if not hasattr(args, 'ncc_fn_none'):
        args.ncc_fn_none = None
    infer_epoch = ConceptEpoch(
        model=model,
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

    if not test_loader.empty():
        infer_epoch(test_loader)
    csv_logger(infer_epoch.get_analysis())
    infer_epoch.generate_report(md_logger, modality='MM')


def start_train(args, model, optimizer):
    train_fold(args=args, model=model, optimizer=optimizer, epocher_fn=get_epochs, infer_fn=infer)
