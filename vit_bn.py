from math import sqrt
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = nn.BatchNorm1d(65)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim, bias = False)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x):
        a = self.linear1(x)
        b = self.activation(a)
        c = self.dropout1(b)
        d = self.linear2(c)
        return self.dropout2(d)


class LSA(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device = dots.device, dtype = torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            # nn.LayerNorm(patch_dim),
            nn.BatchNorm1d(64),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            # nn.LayerNorm(dim),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

def get_FeedForward_parameters(ff: FeedForward) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ff_weights1 = ff.linear1.weight
    ff_weights2 = ff.linear2.weight
    ff_bias2 = ff.linear2.bias
    return (ff_weights1, ff_weights2, ff_bias2)


def get_LSA_parameters(lsa: LSA) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    att_weights =  lsa.to_qkv.weight
    scale = lsa.temperature
    linear_weights = lsa.to_out[0].weight
    linear_bias = lsa.to_out[0].bias
    return (att_weights, scale, linear_weights, linear_bias)

def get_BatchNorm1d_parameters(norm: nn.BatchNorm1d) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gamma = norm.weight
    beta = norm.bias
    mean = norm.running_mean
    variance = norm.running_var
    return (gamma, beta, mean, variance)


def get_Transforemer_parameter(transformer: Transformer) -> list[torch.Tensor]:

    gamma_list = []
    beta_list = []
    ff_weights1_list = []
    ff_weights2_list = []
    ff_bias2_list = []
    att_weights_list = []
    scale_list = []
    linear_weights_list = []
    linear_bias_list = []

    for module in transformer.modules():
        if isinstance(module,nn.BatchNorm1d):
            gamma, beta, mean, variance = get_BatchNorm1d_parameters(module)
            gamma_list.append(gamma/(variance**0.5))
            beta_list.append(beta - gamma*mean/(variance**0.5))

        elif isinstance(module, FeedForward):
            ff_weights1, ff_weights2, ff_bias2 = get_FeedForward_parameters(module)
            ff_weights1_list.append(ff_weights1)
            ff_weights2_list.append(ff_weights2)
            ff_bias2_list.append(ff_bias2)

        elif isinstance(module, LSA):
            att_weights, scale, linear_weights, linear_bias = get_LSA_parameters(module)
            att_weights_list.append(att_weights)
            scale_list.append(scale)
            linear_weights_list.append(linear_weights)
            linear_bias_list.append(linear_bias)

    return [
        att_weights_list,
        scale_list,
        linear_weights_list,
        linear_bias_list,
        ff_weights1_list,
        ff_weights2_list,
        ff_bias2_list,
        gamma_list,
        beta_list
    ]


def print_matrix(mat: torch.Tensor, file: str) -> None:
  with open(file, "w") as f:
    for item in mat.flatten().tolist():
      print(item, file=f)

def print_parameter_to_file(file_name: str, tensor_list: list[torch.Tensor] | torch.Tensor) -> None:
    if isinstance(tensor_list, torch.Tensor):
        parameters = tensor_list.flatten()
    else:
        param_list = [parameter.flatten() for parameter in tensor_list]
        parameters =  torch.cat(param_list)

    with open(file_name, "w") as f:
        for item in parameters.tolist():
            print(item, file=f)


Parameter_files = [
    "headweights.txt",
    "scale.txt",
    "linearweights.txt",
    "linearbias.txt",
    "ffweights1.txt",
    "ffweights2.txt",
    "ffbias2.txt",
    "gamma.txt",
    "beta.txt"
]

input_filename = "input.txt"
result_filename = "golden_result.txt"

input = torch.rand((1, 65, 512))
model = Transformer(512, 1, 1, 384, 256)
out = model(input)
model.eval()
out = model(input)

parameters = get_Transforemer_parameter(model)

print_matrix(input, input_filename)
print_matrix(out, result_filename)


for filename, tensor_list in zip(Parameter_files, parameters):
    print_parameter_to_file(filename, tensor_list)





