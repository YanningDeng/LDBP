import numpy as np

# }}}
#========================================================#
# functions {{{
#========================================================#


class ssfm_parameters:
    """ handles parameters related to the split-step Fourier method (SSFM)

    Initialization is performed with a dictionary that should have the following keys:
        step_size_method
            logarithmic
            linear
            step_size
            predefined
        StPS: steps per span (only for logarithmic and linear)
        adjusting_factor: recommended is 0.4 (only for logarithmic)
        ssfm_method
            symmetric: linear->nonlinear->linear
            asymmetric: linear->nonlinear
        combine_half_steps: wether to combine half-steps of adjacent spans (only for symmetric)
        alpha: attenuation parameter; should be 0 for less steps than spans
        beta2: dispersion parameter
        gamma: nonlinear parameter
        Nsp: number of spans
        Lsp: span length [m]
        fsamp: sampling frequency
        Nsamp: length of the assumed FFT
        direction: +1 for forward, -1 for backpropagation

    computed attributes:
        model_steps
        cd_length
        nl_param
        nl_length (not used)

    Usage example:

    bw = ssfm_parameters(parameter_dict)
    for NN in range(bw.model_steps):
        u = sp.ifft(bw.get_cd_filter_freq(NN)*sp.fft(u))
        u = u*np.exp(1J*bw.nl_param[NN]*np.abs(u)**2)
    """

    def __init__(self, opts):
        # converts all dictionary entries to attributes
        self.__dict__.update(opts)

        alpha_lin = self.alpha/(10*np.log10(np.exp(1)))
        Nsp = self.Nsp
        Lsp = self.Lsp
        direction = self.direction

        if direction == +1 and self.Nsp > 1:
            raise ValueError("forward propagation valid only for 1 span")

        if self.step_size_method == 'logarithmic':
            if 'adjusting_factor' not in opts:
                self.adjusting_factor = 0.4  # 0: linear, 1: very logarithmic

        if 'combine_half_steps' not in opts:
            self.combine_half_steps = True

        if self.step_size_method == 'step_size':  # used only for subband processing
            step_size = self.step_size
            Ltot = Lsp*Nsp
            model_steps = int(np.floor(Ltot/step_size)+1)
            last_step_size = Ltot - (model_steps-1)*step_size

            cd_length = step_size*np.ones(model_steps)
            cd_length[model_steps-1] = last_step_size

            tmp = np.mod(np.cumsum(cd_length), Lsp)
            len_before = np.zeros(model_steps)
            len_after = np.zeros(model_steps)
            amplifier_location = np.zeros(model_steps)
            for NN in range(1, model_steps):
                if(tmp[NN-1] > tmp[NN]):
                    amplifier_location[NN] = 1
                    len_after[NN] = tmp[NN]
                    len_before[NN] = cd_length[NN] - len_after[NN]
            amplifier_location[0] = 1
            amplifier_location[-1] = 0

            nl_length = np.zeros(model_steps)
            eff_len_before = np.zeros(model_steps)
            for NN in range(model_steps):
                if (amplifier_location[NN] == 1) and (NN != 0):
                    h = len_after[NN]
                    eff_len_before[NN] = effective_length(
                        len_before[NN], np.abs(alpha_lin))
                else:
                    h = cd_length[NN]
                nl_length[NN] = effective_length(h, np.abs(alpha_lin))
        else:
            StPS = self.StPS
            # ====================================================== #
            # compute step sizes for one span
            # ====================================================== #
            if self.step_size_method == 'logarithmic':
                alpha_adj = self.adjusting_factor*alpha_lin
                delta = (1-np.exp(-alpha_adj*Lsp))/StPS
                if(direction == -1):
                    nn = np.arange(StPS)+1    # 1,2,...,StPS
                else:
                    nn = StPS-np.arange(StPS)  # StPS,...,2,1
                step_size = -1/(alpha_adj) * \
                    np.log((1-(StPS-nn+1)*delta)/(1-(StPS-nn)*delta))
            elif self.step_size_method == "linear":
                step_size = Lsp/StPS*np.ones(StPS)
            else:
                raise ValueError(
                    "wrong step_size_method given (should be 'linear' or 'logarithmic'): "+self.step_size_method)
            # ====================================================== #
            # compute cd_length, nl_length, amplifier_location
            # ====================================================== #
            if self.ssfm_method == "symmetric":
                if self.combine_half_steps == True:
                    model_steps = Nsp*StPS+1
                    cd_length = np.zeros(model_steps)
                    nl_length = np.zeros(model_steps)
                    for NN in range(Nsp):
                        for MM in range(StPS):
                            cd_length[NN*StPS+MM] = step_size[MM] / \
                                2 + step_size[(MM+StPS-1) % StPS]/2
                            nl_length[NN*StPS+MM] = step_size[MM]
                    cd_length[0] = step_size[0]/2
                    cd_length[model_steps-1] = step_size[StPS-1]/2

                    amplifier_location = np.zeros(model_steps)
                    amplifier_location[:-1:StPS] = 1
                else:
                    model_steps = Nsp*(StPS+1)
                    cd_length = np.concatenate(
                        [[step_size[0]/2], (step_size[0:-1]+step_size[1:])/2, [step_size[-1]/2]])
                    cd_length = np.tile(cd_length, Nsp)
                    nl_length = np.concatenate([step_size, [0]])
                    nl_length = np.tile(nl_length, Nsp)

                    amplifier_location = np.zeros(model_steps)
                    amplifier_location[::StPS+1] = 1
            elif self.ssfm_method == "asymmetric":
                model_steps = Nsp*StPS
                cd_length = np.zeros(model_steps)
                nl_length = np.zeros(model_steps)
                for NN in range(Nsp):
                    for MM in range(StPS):
                        cd_length[NN*StPS+MM] = step_size[MM]
                        nl_length[NN*StPS+MM] = effective_length(
                            step_size[MM], np.abs(alpha_lin))

                amplifier_location = np.zeros(model_steps)
                amplifier_location[::StPS] = 1
            else:
                raise ValueError(
                    "wrong split step method given (should be 'symmetric' or 'asymmetric'): "+self.ssfm_method)
        # ====================================================== #
        # compute attenuation and nl_param
        # ====================================================== #
        nl_param = direction*self.gamma*nl_length

        attenuation = np.exp(-direction*alpha_lin*cd_length/2)
        for NN in range(model_steps):
            if direction == -1 and amplifier_location[NN] == 1:
                attenuation[NN] = attenuation[NN] * \
                    np.exp(direction*alpha_lin*Lsp/2)

        # re-normalize nl_param
        for NN in range(model_steps):
            nl_param[NN] = nl_param[NN]*np.prod(attenuation[0:NN+1:])**2

        if self.step_size_method == "step_size":
            for NN in range(model_steps):
                if amplifier_location[NN] == 1:
                    nl_param[NN] = nl_param[NN] + direction * \
                        self.gamma*eff_len_before[NN]

        self.model_steps = model_steps
        self.cd_length = cd_length
        self.nl_length = nl_length
        self.nl_param = nl_param

        N = self.Nsamp
        self.fvec = np.concatenate(
            (np.linspace(0, N//2-1, N//2), np.linspace(-N//2, -1, N//2))) * self.fsamp/N

    def get_cd_filter_freq(self, NN):
        return np.exp(1j*(self.beta2/2)*(2*np.pi*self.fvec)**2*(self.direction*self.cd_length[NN]))


def get_fvec(N, fs):
    return np.concatenate((np.linspace(0, N//2-1, N//2), np.linspace(-N//2, -1, N//2))) * fs/N


def rrcosine(rolloff, delay, OS):
    """ Root-raised cosine filter for pulse shaping
    Args:
        rolloff: between 0 and 1
        delay: in symbols
        OS: oversampling factor (samples per symbol)

    Returns:
        A vector of length 2*(OS*delay)+1
    """
    rrcos = np.zeros(2*delay*OS+1)
    rrcos[delay*OS] = 1 + rolloff*(4/np.pi-1)
    for i in range(1, delay*OS+1):
        t = i/OS
        if(t == 1/4/rolloff):
            val = rolloff/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi /
                                                         (4*rolloff)) + (1-2/np.pi)*np.cos(np.pi/(4*rolloff)))
        else:
            val = (np.sin(np.pi*t*(1-rolloff)) + 4*rolloff*t *
                   np.cos(np.pi*t*(1+rolloff))) / (np.pi*t*(1-(4*rolloff*t)**2))
        rrcos[delay*OS+i] = val
        rrcos[delay*OS-i] = val
    return rrcos / np.sqrt(np.sum(rrcos**2))


def periodically_extend(x, M):
    """ Extends a numpy vector of length N to length M>N by periodically copying the elements """
    N = x.shape[0]
    y = np.zeros(M, dtype=x.dtype)
    for i in range(M):
        y[i] = x[i % N]
    return y


def line2array(line):
    ''' converts a string of comma-separated numbers to numpy array '''
    return np.array([float(v) for v in line.strip().split(",")])


def effective_length(length, alpha_lin):
    if alpha_lin == 0:
        return length
    else:
        return (1-np.exp(-alpha_lin*length))/alpha_lin


def ordered_direct_product(A, B):
    p = A.shape[0]
    q = B.shape[0]
    n = A.shape[1]
    m = B.shape[1]

    C = np.zeros([p*q, n+m])
    for i in range(q):
        C[i*q:(i+1)*q:, :n:] = A[i, :]
    for i in range(p):
        C[i*q:(i+1)*q:, n::] = B
    return C


def QAM(M):
    Msqrt = (np.sqrt(M)).astype(np.int)
    if Msqrt**2 != M:
        raise ValueError("M has to be of the form M=4^m where m>0")
    x_pam = np.expand_dims(-(Msqrt-2*np.arange(start=1,
                                               stop=Msqrt+1)+1), axis=1)
    x_qam = ordered_direct_product(x_pam, x_pam)
    const = x_qam[:, 0] + 1j * x_qam[:, 1]
    return const/np.sqrt(np.mean(np.abs(const)**2))
