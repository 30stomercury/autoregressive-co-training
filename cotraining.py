import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base_solver import BaseSolver
from modules.quantizer import GumbelQuantizer, MarginalQuantizer


class Cotraining(BaseSolver):

    def __init__(
        self,
        train_loader, 
        eval_loader, 
        config, 
        device='cpu'
    ):
        super().__init__(train_loader, eval_loader, config, device)
        
        config = config['model']

        # past view: autoregressive network
        rnn_config = config['lstm']
        in_sizes  = [rnn_config['input_size']] + [rnn_config['hidden_size']] * (rnn_config['num_layers'] - 1)
        out_sizes = [rnn_config['hidden_size']] * rnn_config['num_layers']
        self.prednet = torch.nn.ModuleList(
            [torch.nn.LSTM(input_size=in_size, hidden_size=out_size, batch_first=True)
            for (in_size, out_size) in zip(in_sizes, out_sizes)])
        self.rnn_dropout = torch.nn.Dropout(config['dropout'])
        self.rnn_residual = rnn_config['residual']
        self.postnet = torch.nn.Linear(**config['lin'])

        # future view: quantization layer
        if config['mode'] == 'marginal':
            self.confnet = MarginalQuantizer(**config['quantizer'])
        elif config['mode'] == 'gumbel':
            self.confnet = GumbelQuantizer(**config['quantizer'])
        else:
           raise NotImplementedError('Other mode not supported') 
        self.num_codes = self.confnet.num_codes

        # model
        self.model = torch.nn.ModuleList([
            self.prednet, self.postnet, self.confnet]).to(self.device)

        # init optimizers
        self.init_optimizers()

    def forward_pred(self, x, lx, mask):
        features = []
        rnn_input = x
        for i, layer in enumerate(self.prednet):
            packed_rnn_input = pack_padded_sequence(
                rnn_input, lx.cpu(), batch_first=True, enforce_sorted=False)
            packed_rnn_output, hidden = layer(packed_rnn_input)
            rnn_output, _ = pad_packed_sequence(
                packed_rnn_output, 
                batch_first=True
            )
            if i - 1 < len(self.prednet):
                rnn_output = self.rnn_dropout(rnn_output)

            if self.rnn_residual and i > 0:
                rnn_output += rnn_input

            rnn_input = rnn_output
            features.append(rnn_output)

        preds = self.postnet(rnn_output)
        preds.masked_fill_(mask, 0.0)
        preds = preds.view(
            preds.size(0), preds.size(1), -1)

        # (n_layer, n_sample, n_frame, hidden_size)
        features = torch.stack(features) 

        return preds, features, mask

    def forward_conf(self, y, mask):
        results = self.confnet(y)
        q = results['q']
        q.masked_fill_(mask, 0.0)
        self.confnet.update_temp(self.steps)

        return q, results, mask

    def forward(self, x, y, lx, mask):
        preds, features, mask = self.forward_pred(x, lx, mask)
        q, results, mask = self.forward_conf(y, mask)
        return preds, q, results, mask
