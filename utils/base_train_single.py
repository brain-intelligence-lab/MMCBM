# project imports
from .logger import CSVLogs, char_color
from .trainer import BatchLogger, SingleEpoch
from .EarlyStop import EarlyStopping
import warnings
from visualize.utils import ActPlot
from params import modality_model_map
from .train_utils import train_fold, get_stage1_metrics_from_top_stage2

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
    trainepoch = SingleEpoch(
        model=model,
        loss=args.loss,
        optimizer=optimizer,
        stage_name='train',
        device=args.device,
        batch_loggers=batch_loggers,
        plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
        mix_up_alpha=args.mix_up_alpha,
    )
    validepoch = SingleEpoch(
        model=model,
        loss=args.loss,
        optimizer=optimizer,
        stage_name='valid',
        device=args.device,
        batch_loggers=batch_loggers,
        plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
        mix_up_alpha=args.mix_up_alpha,
    )
    testepoch = SingleEpoch(
        model=model,
        loss=args.loss,
        optimizer=optimizer,
        stage_name='test',
        device=args.device,
        batch_loggers=batch_loggers,
        plot_fn=ActPlot(dir=f'{args.output_dir}/{args.dir_name}') if args.plot else None,
        mix_up_alpha=args.mix_up_alpha,
    )
    return trainepoch, validepoch, testepoch


def infer(args, model, loaders):
    train_loader, val_loader, test_loader = loaders
    metrics_logger = CSVLogs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name=f'metrics_output_{args.idx}')
    pred_logger = CSVLogs(dir_path=f'{args.output_dir}/{args.dir_name}', file_name=f'pred_output_{args.idx}')
    epochers = get_epochs(args=args, model=model, optimizer=None, logger=None, pred_logger=pred_logger,
                          metrics_logger=metrics_logger)
    trainepoch, validepoch, testepoch = epochers
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


def start_train(args, model, optimizer):
    args.clip_name = 'backbone'
    train_fold(args=args, model=model, optimizer=optimizer, epocher_fn=get_epochs, infer_fn=infer)
