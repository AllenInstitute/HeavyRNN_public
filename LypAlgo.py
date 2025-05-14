import torch
from torch.autograd import Variable
import pickle as pkl
import numpy as np

def oneStep(*params, model):
    return model(*params)

def oneStepVarQR(J, Q):
    # Linear extrapolation of the network in many directions
    Z = torch.matmul(torch.transpose(J, 1, 2), Q)

    # QR decomposition of new directions
    q, r = torch.linalg.qr(Z, "reduced")
    s = torch.diag_embed(
        torch.sign(torch.diagonal(r, dim1=1, dim2=2))
    )  # extract sign of each leading r value

    return torch.matmul(q, s), torch.diagonal(
        torch.matmul(s, r), dim1=1, dim2=2
    )  # return positive r values and corresponding vectors


def calc_LEs_an(
    input,
    initial_hidden,
    model,
    k_LE=100000,
    warmup=10,
    compute_LE=True,
    include_PR=False,
    store_all_output=False,
    store_all_states=False,
    save_all_states=False,
    get_jacobian=False,
    directory='results/states.pkl'
):
    # Calculate the exponents for a given network (with a batch of inputs)
    result = {'LE': None,
              'PR': None,
              'RNN_output': None,
              'all_hidden_states': None,
              'Jacobian_spectrum': None}
    device = model.device
    bias = model.rnn_layer.bias
    h_all = []

    x_in = Variable(input, requires_grad=False).to(device)
    hc = initial_hidden
    h0 = Variable(hc, requires_grad=False).to(device)
    num_layers, batch_size, hidden_size = h0.shape
    _, feed_seq, input_size = x_in.shape
    L = num_layers * hidden_size
    k_LE = max(min(L, k_LE), 1)

    # initialize Q for QR-based LE computation
    Q = torch.reshape(torch.eye(L), (1, L, L)).repeat(batch_size, 1, 1).to(device)
    Q = Q[:, :, :k_LE]

    ht = h0
    rvals = torch.ones(batch_size, feed_seq - warmup, k_LE).to(device)
    all_rnn_out = []

    # Warmup: run the network without accumulating LE but collecting hidden states if needed
    for xt in x_in.transpose(0, 1)[:warmup]:
        xt = torch.unsqueeze(xt, 1)
        # step
        rnn_out, states = oneStep(xt, ht, model=model)
        ht = states
        if include_PR or store_all_states:
            h_all.append(torch.squeeze(ht))
        if store_all_output:
            all_rnn_out.append(rnn_out)

    # Main loop: compute LEs and continue collecting hidden states
    t = 0
    J_all_eigen = []
    seq_post = x_in.transpose(0, 1)[warmup:]
    for idx, xt in enumerate(seq_post):
        xt = torch.unsqueeze(xt, 1)
        # forward step
        rnn_out, states = oneStep(xt, ht, model=model)
        ht = states
        if include_PR or store_all_states:
            h_all.append(torch.squeeze(ht))
        if store_all_output:
            all_rnn_out.append(rnn_out)

        # optionally collect Jacobian spectrum
        if get_jacobian and idx >= len(seq_post) - 5:
            J = rnn_jac(model.rnn_layer.all_weights, ht, xt, bias=bias)
            vals = np.linalg.eigvals(J[0].cpu().numpy())
            J_all_eigen.append(vals)

        # accumulate Lyapunov exponents
        if compute_LE:
            J = rnn_jac(model.rnn_layer.all_weights, ht, xt, bias=bias)
            Q, r = oneStepVarQR(J, Q)
            rvals[:, t, :] = r

        t += 1

    # package outputs
    if store_all_states:
        h_all_stack = torch.stack(h_all)
        result["all_hidden_states"] = h_all_stack
        if save_all_states:
            pkl.dump(h_all_stack, open(f'{directory}_allHiddenStates.pkl', "wb"))
    if get_jacobian:
        result['Jacobian_spectrum'] = J_all_eigen
    if compute_LE:
        eps = 1e-12
        safe_r = torch.clamp(rvals.detach(), min=eps)
        LEs = torch.sum(torch.log(safe_r), dim=1) / t
        result['LE'] = LEs
    if include_PR:
        # skip the warmup steps when computing participation ratio
        h_tensor = torch.stack(h_all)
        result["PR"] = participation_ratio_dimension(
            h_all=h_tensor,
            skip_init_steps=warmup
        )
    if store_all_output:
        result['RNN_output'] = torch.cat(all_rnn_out, dim=1)

    return result


def LE_stats(LE, save_file=False, file_name="LE.p"):
    mean = torch.mean(LE, dim=0)
    std  = torch.std(LE, dim=0, unbiased=False)
    if save_file:
        pkl.dump((mean, std), open(file_name, "wb"))
    return mean, std


def rnn_jac(params_array, h, x, bias):
    if bias:
        W, U, b_i, b_h = param_split(params_array, bias)
    else:
        W, U = param_split(params_array, bias)
    device = get_device(h)
    num_layers, batch_size, hidden_size = h.shape

    h_in = h.transpose(1, 2).detach()
    x_in = [x.squeeze(dim=1).t()]
    if bias:
        b = [b1 + b2 for (b1, b2) in zip(b_i, b_h)]
    else:
        b = [torch.zeros(W_i.shape[0], device=device) for W_i in W]

    J = torch.zeros(batch_size, num_layers*hidden_size, num_layers*hidden_size, device=device)
    y_out, h_out = [], []

    for layer in range(num_layers):
        if layer > 0:
            x_l = h_out[layer - 1]
            x_in.append(x_l)
        y = (W[layer] @ x_in[layer] + U[layer] @ h_in[layer] + b[layer].repeat(batch_size,1).t()).t()
        y_out.append(y)
        J_h = sech(y)**2 @ U[layer]
        start = layer*hidden_size
        end   = (layer+1)*hidden_size
        J[:, start:end, start:end] = J_h

        if layer > 0:
            J_xt = sech(y)**2 @ W[layer]
            for l in range(layer, 0, -1):
                prev_start = (l-1)*hidden_size
                prev_end   = l*hidden_size
                J[:, start:end, prev_start:prev_end] = J_xt @ J[:, prev_start:prev_end, prev_start:prev_end]

        h_out.append(torch.tanh(y).t())

    return J


def param_split(model_params, bias):
    layers = len(model_params)
    W, U, b_i, b_h = [], [], [], []
    if bias:
        param_lists = (W, U, b_i, b_h)
    else:
        param_lists = (W, U)
    grouped = []
    for idx, plist in enumerate(param_lists):
        for layer in range(layers):
            plist.append(model_params[layer][idx].detach())
        grouped.append(plist)
    return grouped


def get_device(X):
    return torch.device("cuda") if X.is_cuda else torch.device("cpu")


def sig(X):
    return 1 / (1 + torch.exp(-X))


def sigmoid(X):
    return torch.diag_embed(1 / (1 + torch.exp(-X)))


def sigmoid_p(X):
    s = sig(X)
    return torch.diag_embed(s * (1 - s))


def sech(X):
    return torch.diag_embed(1 / torch.cosh(X))


def tanh(X):
    return torch.diag_embed(torch.tanh(X))


def participation_ratio_dimension(h_all, skip_init_steps=0, division_eps=1e-12):
    """
    Participation ratio:
       D_PR = (Tr C)^2 / Tr(C^2)
    computed on the covariance of post-warmup hidden states.
    
    See Eqn 2.1, https://www.sciencedirect.com/science/article/pii/S266638992200160X
    ---
    D_PR measures the concentration of the eigenvalue distribution 
    and quantifies how many eigenmodes are needed to substantially 
    capture the data distribution, a similar notion to counting eigenmodes 
    (or principal components) that capture most of the variance.
    """
    # drop initial warmup steps
    h_sub = h_all[skip_init_steps:]
    C = torch.cov(h_sub.T)
    tr = torch.trace(C)
    num = tr**2
    den = (C*C).sum() + division_eps
    return num / den


def lyapunov_dimension(lyap_exps):
    """
    L.d. quantify the complexity of a chaotic attractor.
    
    A higher Lyapunov dimension indicates more complex and chaotic behavior. 
    This means the system is more sensitive to initial conditions, 
    and small differences in starting points can lead to vastly different outcomes.
    
    The Lyapunov dimension can be interpreted as 
    the number of effective degrees of freedom in the system. 
    It tells you how many dimensions are actively contributing 
    to the system's dynamics. 
    For example, a Lyapunov dimension of 2.5 implies that 
    the system behaves as if it has between 2 and 3 degrees of freedom.
    ---
    see Kaplan-Yorke conjecture: https://en.wikipedia.org/wiki/Kaplan%E2%80%93Yorke_conjecture
    see Lyapunov dimension: https://en.wikipedia.org/wiki/Lyapunov_dimension
    
    Lyapunov dimension is used for estimating the Hausdorff dimension of attractors.
    Since the direct numerical computation of the Hausdorff dimension of attractors
    is often complicated, Lyapunov dimension is a widely spread estimations.
    
    Hausdorff dimension is a measure of roughness (fractal dimension), 
    i.e. H.d. of a single point is 0, of line segment is 1, of square is 2, of cube is 3.
    That is, for sets of points that define a smooth shape OR a shape that has
    smaller number of corners--the shapes of traditional geometry and science--the H.d. 
    is an integer agreeing with the usual sense of dimension, 
    also known as the topological dimension.
    """
    # 1) tensor → numpy list
    if isinstance(lyap_exps, torch.Tensor):
        exps = lyap_exps.detach().cpu().numpy()
    else:
        exps = np.array(lyap_exps, copy=False)
    exps = sorted(exps.tolist(), reverse=True)

    # 2) cumulative sum of exponents
    cumsum = np.cumsum(exps)

    # 3) find first negative partial sum index
    m_neg = next((i for i, s in enumerate(cumsum) if s < 0), len(exps))

    # 4) cases
    if m_neg == 0:
        return 0.0
    if m_neg >= len(exps):
        return float(len(exps))

    j = m_neg - 1
    return (j + 1) + cumsum[j] / abs(exps[j+1])