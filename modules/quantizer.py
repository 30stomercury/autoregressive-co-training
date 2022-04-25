import torch
import torch.nn.functional as F


class BaseQuantizer(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        num_codes,
        temp,
        code_dim,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_codes = num_codes
        # init codebook
        self.codebook = torch.nn.Parameter(torch.FloatTensor(num_codes, code_dim))
        torch.nn.init.normal_(self.codebook, 0.0, (1 / num_codes**0.5))

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def update_temp(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def compute_perp(self, x, B, T):
        _, k = x.max(-1)
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(B * T, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        perp = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-12), dim=-1)
        ).sum()
        return hard_x, perp

    
    def forward(self, x):
        raise NotImplementedError


class GumbelQuantizer(BaseQuantizer):
    def __init__(
        self,
        input_dim,
        num_codes,
        temp,
        code_dim,
    ):
        super().__init__(input_dim, num_codes, temp, code_dim)

    def forward(self, x):

        result = {"num_codes": self.num_codes}
        codebook = self.codebook

        B, T, D = x.shape
        x = x.reshape(-1, D)

        # p(z_t+k|x_t+k)
        # Compute Eculidean distance between x_t+k and all codes v_j,
        # note that x can also be modeled by networks, e.g., torch.nn.Linear(D, N).
        x = - (
            x.unsqueeze(1).expand(-1, self.num_codes, -1) - \
            codebook.unsqueeze(0).expand(x.size(0), -1, -1)
        ).pow(2).sum(-1)
        probs = torch.softmax(x.float(), dim=-1)

        # perplexity
        hard_x, perp = self.compute_perp(x, B, T)
        result["code_perplexity"] = perp

        if self.training:
            # ST Gumbel estimator
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)

        # selected code (single sample)
        q = torch.matmul(x, codebook.squeeze(0))
        q = q.view(bsz, tsz, -1)
        result["q"] = q
        result["latent_probs"] = probs.view(bsz, tsz, -1)
        result["temp"] = self.curr_temp

        return result


class MarginalQuantizer(BaseQuantizer):
    def __init__(
        self,
        input_dim,
        num_codes,
        temp,
        code_dim,
    ):
        super().__init__(input_dim, num_codes, temp, code_dim)

    def forward(self, x):

        result = {"num_codes": self.num_codes}
        codebook = self.codebook

        B, T, D = x.shape
        x = x.reshape(-1, D)

        # p(z_t+k|x_t+k)
        # Compute Eculidean distance between x_t+k and all codes v_j.
        x = (
            x.unsqueeze(1).expand(-1, self.num_codes, -1) - \
            codebook.unsqueeze(0).expand(x.size(0), -1, -1)
        ).pow(2).sum(-1)
        result['downstream_losses'] = x.view(B, T, -1).float()
        x = -x
        probs = torch.softmax(x.float(), dim=-1)

        # perplexity
        hard_x, perp = self.compute_perp(x, B, T)
        result["code_perplexity"] = perp

        # Selected code (single sample), q is not used during marginalization training.
        q = torch.matmul(hard_x, codebook.squeeze(0))
        q = q.view(B, T, -1)
        result["q"] = q
        result["latent_probs"] = probs.view(B, T, -1)
        result["temp"] = self.curr_temp

        return result
