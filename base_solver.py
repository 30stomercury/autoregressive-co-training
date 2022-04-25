from tqdm import tqdm
import os
import torch
import torch.nn.functional as F


def clip_grad(parameters, g_clip=10):
    grad_norm = torch.nn.utils.clip_grad_norm_(
        parameters, g_clip
    )
    if torch.isnan(grad_norm):
        self.save('nan', './')
        raise ValueError('Grad norm is NaN at this step')


def CrossEntropy(pred, target, reduction="none"):
    """cross-entropy loss
    args
    ----
    pred : (B, T, N)
        prednet probs.
    target: (B, T, N)
        confnet probs.
    """
    N = target.size(0)
    eps = 1e-12
    # [B, T, D] -> [B, T]
    pred_stable = pred.clone().clamp(min=eps, max=1 - eps)

    cross_entropy_masked = torch.where(
        (pred > 0).to(pred.device),
        target * pred_stable.log(),
        torch.tensor(0., device=pred.device, dtype=torch.float)
    )
    cross_entropy = - torch.sum(cross_entropy_masked, -1)

    if reduction == "none":
	    return cross_entropy
    elif reduction == "sum":
        return cross_entropy.sum()
    elif reduction == "mean":
        return cross_entropy.mean()
    else:
        raise ValueError(f"{reduction} reduction not implemented")


class Stats:
    '''Collect statistics.
    '''
    def __init__(self):
        self.summary = {}
        self.summary['total_samples'] = 0
    
    def update(self, curr_stats, num_samples=0):
        '''
        args
        ----
        curr_stats: dict
        num_samples: int
        '''    
        self.summary['total_samples'] += num_samples.item()
        for ele in curr_stats:
            if ele not in self.summary:
                self.summary[ele] = 0
            self.summary[ele] += (curr_stats[ele] * num_samples).item()

    def compute_stats(self):
        average = {}
        for ele in self.summary:
            average[ele] = self.summary[ele] / self.summary['total_samples']
        return average
        

class BaseSolver:

    def __init__(
        self, 
        train_loader, 
        eval_loader, 
        config, 
        device='cpu'
    ):
        self.config = config
        self.device = device

        # steps
        self.steps = config['steps']
        self.mode = config['model']['mode']

        # Dataloader
        self.train_loader = train_loader
        self.eval_loader = eval_loader

    def forward(self, x, y, lx, mask):
        raise NotImplementedError

    def run_epoch(self, phase='test'): 
        """
        phase: (str) 
            train, eval
        """
        total_ent_loss = 0
        total_ce_loss = 0
        total_rec_loss = 0
        total_correct = 0
        total_samples = 0
        stats = Stats()

        self.model.train() if phase=='train' else self.model.eval()
        compute_batch = getattr(self, f'{phase}_batch')
        dataloader = getattr(self, f'{phase}_loader')
        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            batch = self.to_device(batch, self.device)
            idx, name, x, lx, y, ly, mask = self.to_device(batch, self.device)
            # Compute ave loss
            self.steps += 1
            losses, preds, results = compute_batch(batch)
            preds = torch.argmax(preds, -1)
            num_samples = (~mask).float().sum()
            total_ce_loss += losses['ce_loss'] * num_samples
            total_ent_loss += losses['ent'] * num_samples
            total_rec_loss += losses['rec_loss'] * num_samples
            total_samples += num_samples
            ave_ce_loss = total_ce_loss / total_samples
            ave_ent_loss = total_ent_loss / total_samples
            ave_rec_loss = total_rec_loss / total_samples
            ave_loss = ave_ce_loss-ave_ent_loss + ave_rec_loss
            stats.update(losses, num_samples)
            # Compute ave error
            targets = torch.argmax(results['latent_probs'], -1)
            total_correct += torch.sum((preds == targets).masked_fill_(mask.squeeze(-1), False))
            ave_error = 100 * (1 - total_correct / total_samples)
            # Info
            pbar.set_postfix(
                {
                    f'{phase} error': '{:.3f}'.format(ave_error),
                    f'{phase} loss': '{:.3f}'.format(ave_loss),
                    f'ent': '{:.3f}'.format(ave_ent_loss),
                    f'ce loss': '{:.3f}'.format(ave_ce_loss),
                    f'kl loss': '{:.3f}'.format(ave_ce_loss-ave_ent_loss),
                    f'rec loss': '{:.3f}'.format(ave_rec_loss),
                    'code perp': results['code_perplexity'].item(),
                    'temp': '{:.4f}'.format(results['temp']),
                    'lr': self.model_optimizer.param_groups[0]['lr'],
                }
            )
        return ave_loss.item(), ave_error.item(), ave_ent_loss.item(), ave_ce_loss.item(), ave_rec_loss.item()

    def train_batch(self, batch):
        idx, name, x, lx, y, ly, mask = batch

        # Compute logits
        preds, q, results, mask = self.forward(x, y, lx, mask)
        # Clean grads
        self.model_optimizer.zero_grad()
        # Compute loss
        losses = self.compute_loss(preds, q, y, results, mask)
        losses['loss'].backward()

        # Clip grads
        if self.config['training']['g_clip'] > 0:
            clip_grad(self.model.parameters(), self.config['training']['g_clip'])
        self.model_optimizer.step()
        losses['loss'] = losses['loss'].detach()

        return losses, preds, results

    @torch.no_grad()
    def eval_batch(self, batch):
        idx, name, x, lx, y, ly, mask = batch

        # Compute logits
        preds, q, results, mask = self.forward(x, y, lx, mask)
        # Compute loss
        losses = self.compute_loss(preds, q, y, results, mask)
        losses['loss'] = losses['loss'].detach()

        return losses, preds, results

    def compute_loss(self, preds, q, y, results, mask):
        losses = {}
        latent_probs = results['latent_probs']
        num_codes = latent_probs.size(-1)
        # mask
        mask = (~mask).float()

        # reconstruction loss
        B, T, D = y.shape

        # cross entropy 
        pred_probs = torch.softmax(preds, -1)
        ce_loss_batch = CrossEntropy(
            pred_probs, latent_probs, reduction='none'
        ) * mask.squeeze(-1)
        ce_loss = (ce_loss_batch.sum() / mask.sum())

        # entropy
        ent_loss_batch = CrossEntropy(
            latent_probs, latent_probs, reduction='none'
        ) * mask.squeeze(-1)
        ent_loss = (ent_loss_batch.sum() / mask.sum())

        if self.mode == 'gumbel':
            # compute loss based on single sample q
            rec_loss_batch = 0.5 * (q - y)**2 * mask
            rec_loss = (rec_loss_batch * mask).sum() / mask.sum() 
            loss = ce_loss - ent_loss + rec_loss

        elif self.mode == 'marginal':
            # (B, T, N, D)
            rec_losses = 0.5 * results['downstream_losses'].view(B * T, -1)
            latent_probs = latent_probs.view(-1, num_codes)
            # marginalization
            rec_loss_batch = latent_probs.unsqueeze(1).bmm(
                rec_losses.unsqueeze(-1)
            ).view(B, T) * mask.squeeze(-1)
            rec_loss = rec_loss_batch.sum() / mask.sum()
            loss = ce_loss - ent_loss + rec_loss 

        # Total loss
        losses['loss'] = loss
        losses['ent'] = ent_loss
        losses['ce_loss'] = ce_loss
        losses['kl_loss'] = ce_loss - ent_loss
        losses['rec_loss'] = rec_loss 
        return losses

    def init_optimizers(self):   
        if self.config['training']['opt'] == 'Adadelta':
            self.model_optimizer = torch.optim.Adadelta(
                self.model.parameters(), 
                lr=self.config['training']['lr']
            )
        elif self.config['training']['opt'] == 'Adam':
            self.model_optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.config['training']['lr']
            )
        else:
           raise NotImplementedError('Only support Adam and Adadelta') 

    def load(self, model_path, device='cpu'):
        if model_path is None:
            return
        optim_path = model_path.replace('model', 'optim')
        if os.path.exists(model_path):
            print('Loading model from : {}'.format(model_path))
            self.model.load_state_dict(
                torch.load(model_path, map_location=device), strict=True
            )
        if os.path.exists(optim_path):
            print('Loading model and optimizer from : {}'.format(optim_path))
            self.model_optimizer.load_state_dict(
                torch.load(optim_path, map_location=device)
            )

    def save(self, e, path):
        save_path =  os.path.join(path, 'ckpt', 'cotraining_model_{}.ckpt'.format(e))
        torch.save(self.model.state_dict(), save_path)
        optim_path = save_path.replace('model', 'optim')
        torch.save(self.model_optimizer.state_dict(), optim_path)

    def to_device(self, batch, device):
        outputs = []
        for ele in batch:
            if isinstance(ele, torch.Tensor):
                ele = ele.to(device) 
            outputs.append(ele)
        return tuple(outputs)
