import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.utils.io import ensure_dir, load_json
from src.data.dataset import LogSequenceDataset
from src.data.validation_split import combine_validation_with_synthetic_if_needed
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.training.checkpointing import save_checkpoint
from src.evaluation.scoring import score_dataset
from src.evaluation.multiscale import score_payload_multiscale
from src.evaluation.thresholds import select_validation_safe_threshold


def _notebook_display_enabled(config):
    return bool(config.get('notebook', {}).get('compact_training_log', False))


def _display_training_status(lines, enabled):
    if not enabled:
        for line in lines[-1:]:
            print(line)
        return
    try:
        from IPython.display import clear_output, display
        clear_output(wait=True)
        display({'text/plain': '\n'.join(lines)}, raw=True)
    except Exception:
        print(lines[-1])


def _raw_method_for(config, model_name):
    spot_models = set(config.get('threshold', {}).get('spot_for_models', []))
    return 'spot' if model_name in spot_models else config.get('threshold', {}).get('method', 'quantile')


def _history_path(output_dir, model_name, checkpoint_path=None):
    if checkpoint_path:
        stem = Path(checkpoint_path).stem.replace('_best', '')
        return Path(output_dir) / 'metrics' / f'{stem}_training_history.csv'
    return Path(output_dir) / 'metrics' / f'{model_name}_training_history.csv'


def _monitor_validation_metrics(model, train_loader, val_loader, val_labels, config, model_name, device, train_payload=None, val_payload=None):
    if config.get('multi_scale', {}).get('enabled', False) and train_payload is not None and val_payload is not None:
        train_scores = score_payload_multiscale(model, train_payload, train_payload, config, device)['scores']
        val_scores = score_payload_multiscale(model, val_payload, train_payload, config, device)['scores']
    else:
        train_scores = score_dataset(model, train_loader, device, config['data']['pad_id'], include_details=False)['scores']
        val_scores = score_dataset(model, val_loader, device, config['data']['pad_id'], include_details=False)['scores']
    return select_validation_safe_threshold(
        train_scores,
        val_scores,
        val_labels,
        config.get('threshold', {}),
        raw_method=_raw_method_for(config, model_name),
    )


def _threshold_changed(current, previous, min_delta, min_relative_delta):
    if current is None or previous is None:
        return False
    current = float(current)
    previous = float(previous)
    delta = abs(current - previous)
    scale = max(abs(previous), abs(current), 1e-12)
    return delta >= float(min_delta) or (delta / scale) >= float(min_relative_delta)


def _detection_score(metrics, target_recall):
    recall = float(metrics.get('recall') or 0.0)
    f1 = float(metrics.get('f1') or 0.0)
    fpr = float(metrics.get('false_positive_rate') or 0.0)
    fn = float(metrics.get('fn') or 0.0)
    recall_penalty = max(0.0, float(target_recall) - recall) * 10.0
    return f1 + recall - fpr - fn - recall_penalty


def train_model_from_config(config, model_name, checkpoint_path=None):
    device=torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    tr=load_json(f"{config['data']['splits_dir']}/train.json"); va=load_json(f"{config['data']['splits_dir']}/val.json")
    va_monitor, va_monitor_source = combine_validation_with_synthetic_if_needed(config, va)
    tr_ds=LogSequenceDataset(tr['sequences'],tr.get('labels'),max_len=config['data']['max_seq_len'],pad_id=config['data']['pad_id'],session_ids=tr.get('session_ids'))
    va_ds=LogSequenceDataset(va['sequences'],va.get('labels'),max_len=config['data']['max_seq_len'],pad_id=config['data']['pad_id'],session_ids=va.get('session_ids'))
    va_monitor_ds=LogSequenceDataset(va_monitor['sequences'],va_monitor.get('labels'),max_len=config['data']['max_seq_len'],pad_id=config['data']['pad_id'],session_ids=va_monitor.get('session_ids'))
    tr_ld=DataLoader(tr_ds,batch_size=config['training']['batch_size'],shuffle=True,num_workers=config['training'].get('num_workers',0))
    tr_eval_ld=DataLoader(tr_ds,batch_size=config['training']['batch_size'],shuffle=False,num_workers=config['training'].get('num_workers',0))
    va_ld=DataLoader(va_ds,batch_size=config['training']['batch_size'],shuffle=False,num_workers=config['training'].get('num_workers',0))
    va_monitor_ld=DataLoader(va_monitor_ds,batch_size=config['training']['batch_size'],shuffle=False,num_workers=config['training'].get('num_workers',0))
    model=build_model(model_name,config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=config['training']['weight_decay'])
    trainer=Trainer(model,opt,device,config['data']['pad_id'],config['training']['gradient_clip'])
    output_dir = config.get('project', {}).get('output_dir', 'outputs')
    ensure_dir(f'{output_dir}/metrics')
    best=float('inf'); patience=0; ckpt=checkpoint_path or f'{output_dir}/models/{model_name}_best.pt'
    best_checkpoint_score=float('inf')
    best_detection_score=float('-inf')
    last_threshold=None
    last_threshold_info=None
    history=[]; history_path=_history_path(output_dir, model_name, checkpoint_path)
    monitor_detection=bool(config['training'].get('monitor_detection_metrics', True))
    compact_log=_notebook_display_enabled(config)
    total_epochs=int(config['training']['epochs']); max_patience=int(config['training']['patience'])
    loss_min_delta=float(config['training'].get('loss_min_delta', 0.0))
    threshold_patience=int(config['training'].get('threshold_patience', 9))
    stop_patience=max(max_patience, threshold_patience if monitor_detection else max_patience)
    threshold_min_delta=float(config['training'].get('threshold_min_delta', 1e-6))
    threshold_min_relative_delta=float(config['training'].get('threshold_min_relative_delta', 0.01))
    threshold_save_loss_tolerance=float(config['training'].get('threshold_save_loss_tolerance', 0.05))
    threshold_monitor_interval=max(1, int(config['training'].get('threshold_monitor_interval', 5)))
    threshold_resets_patience=bool(config['training'].get('threshold_resets_patience', False))
    detection_min_delta=float(config['training'].get('detection_min_delta', 1e-4))
    target_recall=float(config.get('threshold', {}).get('target_recall', 0.99))
    n_params=sum(p.numel() for p in model.parameters())
    log_lines=[(
        f'{model_name}: start training | epochs={total_epochs} patience={max_patience} '
        f'threshold_patience={threshold_patience} threshold_monitor_interval={threshold_monitor_interval} '
        f'batch_size={config["training"]["batch_size"]} '
        f'device={device} params={n_params:,} val_monitor={va_monitor_source} '
        f'best_checkpoint={ckpt}'
    )]
    _display_training_status(log_lines, compact_log)
    for epoch in range(1,total_epochs+1):
        started=time.time()
        tl=trainer.train_epoch(tr_ld); vl=trainer.validate_epoch(va_ld)
        row={
            'model':model_name,
            'epoch':epoch,
            'epochs_total':total_epochs,
            'train_loss':tl,
            'val_loss':vl,
            'stop_patience':stop_patience,
            'threshold_patience':threshold_patience,
            'threshold_monitor_interval':threshold_monitor_interval,
            'best_val_loss_before_epoch':None if best == float('inf') else best,
            'best_checkpoint_score_before_epoch':None if best_checkpoint_score == float('inf') else best_checkpoint_score,
            'best_detection_score_before_epoch':None if best_detection_score == float('-inf') else best_detection_score,
            'patience_before_epoch':patience,
            'learning_rate':opt.param_groups[0]['lr'],
            'checkpoint':str(ckpt),
            'validation_monitor_source':va_monitor_source,
        }
        metric_text=''
        current_threshold=None
        threshold_moved=False
        detection_signal=False
        detection_score=None
        run_threshold_monitor = monitor_detection and va.get('labels') is not None and (
            epoch == 1 or epoch % threshold_monitor_interval == 0 or patience >= max_patience - 1
        )
        if run_threshold_monitor:
            threshold_info=_monitor_validation_metrics(model,tr_eval_ld,va_monitor_ld,va_monitor['labels'],config,model_name,device,tr,va_monitor)
            last_threshold_info=threshold_info
            current_threshold=threshold_info['threshold']
            threshold_moved=_threshold_changed(current_threshold,last_threshold,threshold_min_delta,threshold_min_relative_delta)
            vm=threshold_info['metrics']
            detection_score=_detection_score(vm,target_recall)
            detection_signal=detection_score > best_detection_score + detection_min_delta
            row.update({
                'val_precision':vm['precision'],
                'val_recall':vm['recall'],
                'val_f1':vm['f1'],
                'val_fp':vm['fp'],
                'val_fn':vm['fn'],
                'threshold':threshold_info['threshold'],
                'raw_threshold':threshold_info['raw_threshold'],
                'safe_threshold':threshold_info['safe_threshold'],
                'was_safety_applied':threshold_info['was_safety_applied'],
                'threshold_type':threshold_info['threshold_type'],
                'threshold_moved':threshold_moved,
                'previous_threshold':last_threshold,
                'detection_score':detection_score,
                'detection_signal':detection_signal,
                'threshold_monitor_ran':True,
            })
            metric_text=(
                f' val_f1={vm["f1"]:.4f} val_recall={vm["recall"]:.4f} '
                f'val_fp={vm["fp"]} val_fn={vm["fn"]} thr={threshold_info["threshold"]:.6g} '
                f'thr_move={threshold_moved} det_gain={detection_signal} safety={threshold_info["was_safety_applied"]}'
            )
        else:
            row.update({
                'threshold_moved':False,
                'detection_score':None,
                'detection_signal':False,
                'threshold_monitor_ran':False,
            })

        loss_improved = vl < (best - loss_min_delta)
        threshold_signal = bool(detection_signal)
        checkpoint_score = vl
        threshold_checkpoint = (
            threshold_signal
            and best < float('inf')
            and vl <= best * (1.0 + threshold_save_loss_tolerance)
        )
        should_save = loss_improved or threshold_checkpoint
        should_reset_patience = loss_improved or (threshold_signal and threshold_resets_patience)

        row.update({
            'loss_improved':loss_improved,
            'threshold_signal':threshold_signal,
            'threshold_checkpoint':threshold_checkpoint,
            'threshold_save_loss_tolerance':threshold_save_loss_tolerance,
            'checkpoint_score':checkpoint_score,
            'best_checkpoint_score_after_epoch':None,
        })
        if should_save:
            best=min(best,vl)
            best_checkpoint_score=min(best_checkpoint_score,checkpoint_score)
            if detection_score is not None:
                best_detection_score=max(best_detection_score,detection_score)
            patience=0 if should_reset_patience else patience
            row['is_best']=True; save_checkpoint(model,opt,epoch,row,config,ckpt)
            status='BEST saved' if loss_improved else 'DETECTION saved'
        else:
            best=min(best,vl)
            if detection_score is not None:
                best_detection_score=max(best_detection_score,detection_score)
            if should_reset_patience:
                patience=0; status=f'threshold still moving 0/{stop_patience}'
            else:
                patience += 1; status=f'no loss/threshold improvement {patience}/{stop_patience}'
            row['is_best']=False
        if current_threshold is not None:
            last_threshold=current_threshold
        row['best_val_loss_after_epoch']=best
        row['best_checkpoint_score_after_epoch']=None if best_checkpoint_score == float('inf') else best_checkpoint_score
        row['best_detection_score_after_epoch']=None if best_detection_score == float('-inf') else best_detection_score
        row['patience_after_epoch']=patience
        row['epoch_time_sec']=time.time()-started
        history.append(row)
        pd.DataFrame(history).to_csv(history_path,index=False)
        epoch_line=(
            f'{model_name} epoch={epoch:03d}/{total_epochs:03d} '
            f'train_loss={tl:.4f} val_loss={vl:.4f}{metric_text} '
            f'best_val_loss={best:.4f} {status} time={row["epoch_time_sec"]:.1f}s'
        )
        log_lines.append(epoch_line)
        if compact_log and len(log_lines) > int(config.get('notebook', {}).get('compact_training_log_lines', 12)):
            log_lines = [log_lines[0], '...', *log_lines[-(int(config.get('notebook', {}).get('compact_training_log_lines', 12))-2):]]
        _display_training_status(log_lines, compact_log)
        if patience >= stop_patience:
            stop_line=(
                f'{model_name}: early stopping at epoch {epoch}; '
                f'best_val_loss={best:.4f}; threshold_patience={threshold_patience}; history={history_path}'
            )
            log_lines.append(stop_line)
            _display_training_status(log_lines, compact_log)
            break
    done_line=f'{model_name}: training done | best_checkpoint={ckpt} | history={history_path}'
    log_lines.append(done_line)
    _display_training_status(log_lines, compact_log)
    return {'best_checkpoint': str(ckpt), 'history_path': str(history_path), 'best_val_loss': best}
