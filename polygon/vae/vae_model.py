import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from polygon.utils.moses_utils import OneHotVocab
from polygon.utils.smiles_char_dict import SmilesCharDictionary

def get_vocabulary(data=None, device=None):
    """ Build vocabulary optionally from data 
    """
    if data:
        vocabulary =  OneHotVocab.from_data(data)
    else:
        # if no vocab was provided use a preset character definition
        sd = SmilesCharDictionary()
        vocabulary = OneHotVocab(sd.idx_char.values())
    if device:
        vocabulary.vectors = vocabulary.vectors.to(device)
    return vocabulary

class VAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # from defaults
        self.q_cell = "gru"
        self.q_bidir = False
        self.q_d_h = 256
        self.q_n_layers = 1
        self.q_dropout = 0.5
        self.d_cell = "gru"
        self.d_n_layers = 3
        self.d_dropout = 0
        self.d_z = 128
        self.d_d_h = 512
        self.freeze_embeddings = False
        self.vocabulary=None

        # overwrite defaults with passed parameters
        self.__dict__.update(kwargs)

        # if we were supplied with a vocabulary use that otherwise
        if self.vocabulary is None:
            self.vocabulary = get_vocabulary()


        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(self.vocabulary, ss))

        # Word embeddings layer
        n_vocab, d_emb = len(self.vocabulary), self.vocabulary.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.data.copy_(self.vocabulary.vectors)

        # if self.device == torch.device("cuda"):
        #     self.x_emb.cuda()


        if self.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        if self.q_cell == 'gru':
            self.encoder_rnn = nn.GRU(
                d_emb,
                self.q_d_h,
                num_layers=self.q_n_layers,
                batch_first=True,
                dropout=self.q_dropout if self.q_n_layers > 1 else 0,
                bidirectional=self.q_bidir
            )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = self.q_d_h * (2 if self.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, self.d_z)
        self.q_logvar = nn.Linear(q_d_last, self.d_z)

        # Decoder
        if self.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + self.d_z,
                self.d_d_h,
                num_layers=self.d_n_layers,
                batch_first=True,
                dropout=float(self.d_dropout) if self.d_n_layers > 1 else 0
            )
        else:
            raise ValueError(
                "Invalid d_cell type, should be one of the ('gru',)"
            )

        self.decoder_lat = nn.Linear(self.d_z, self.d_d_h)
        self.decoder_fc = nn.Linear(self.d_d_h, n_vocab)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_lat,
            self.decoder_rnn,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])

    @property
    def device(self):
        #return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return next(self.parameters()).device

    def string2tensor(self, string, device='model'):
        ids = self.vocabulary.string2ids(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(
            ids, dtype=torch.long,
            device=self.device if device == 'model' else device
        )

        return tensor

    def tensor2string(self, tensor):
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)
        return string

    def get_collate_device(self):
        return self.device

    def get_collate_fn(self,):
        device = self.get_collate_device()

        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = [self.string2tensor(string, device=self.device)
                       for string in data]

            return tensors

        return collate

    def forward(self, x):
        """Do the VAE forward step

        :param x: list of tensors of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)

        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)
        

        return kl_loss, recon_loss

    def encode(self, x):
        """   
        Do the VAE forward step to get the latent space of a tensor 

        :param x: list of tensors of longs, input sentence x
        :return: vector representing encoded latent space
        """
        if not isinstance(x,list):
            x = [x]
            
        # Encoder: x -> z, kl_loss
        z, kl_loss, mu = self.forward_encoder(x, return_mu=True)       

        return mu

    def decode(self, z, x=None):
        if x is not None:
            rl, y = self.forward_decoder(x,z,return_y=True)
            xr = y.argmax(2)
            # trim off eos
            #xr = [i[:(i == self.eos).nonzero()[0]] for i in xr]
            # trim off eos
            xrt = []
            for i in xr:
                try:
                    q = i[:(i==self.eos).nonzero()[0]]
                    xrt.append(q)
                except IndexError:
                    xrt.append(i)
            smiles = [self.tensor2string(i_x) for i_x in xrt]
        else:
            smiles = self.sample(z.shape[0], z=z, multinomial=True)
        return smiles

    def get_latent_space_from_smiles(self, smiles):
        """   
        Do the VAE forward step to get the latent space of a smiles 

        :param x: list of tensors of longs, input sentence x
        :return: vector representing encoded latent space
        """

        x = self.string2tensor(smiles)
        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder([x])       

        return z[0]

    def forward_encoder(self, x, return_mu=False):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: list of tensors of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        x = [self.x_emb(i_x) for i_x in x]
        x = nn.utils.rnn.pack_sequence(x)

        _, h = self.encoder_rnn(x, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        mu, logvar = self.q_mu(h), self.q_logvar(h)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        if return_mu:
             return z, kl_loss, mu
        return z, kl_loss
	
    def forward_decoder(self, x, z, return_y=False):
        """Decoder step, emulating x ~ G(z)

        :param x: list of tensors of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :return: float, recon component of loss
        """

        lengths = [len(i_x) for i_x in x]

        x = nn.utils.rnn.pad_sequence(x, batch_first=True,
                                      padding_value=self.pad)
        x_emb = self.x_emb(x)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )
        if return_y:
            return recon_loss,y
        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        return torch.randn(n_batch, self.q_mu.out_features,
                           device=self.x_emb.weight.device)

    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0, 1, size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.random.uniform(
                    0, noise_norm, size=(z.shape[0], 1))
                return z + torch.tensor(noise_amp * noise_vec, dtype=z.dtype)
        else:
            return z

    def sample(self, n_batch, max_len=100, z=None, temp=1.0, multinomial=True):
        """Generating n_batch samples in eval mode (`z` could be
        not on same device)

        :param n_batch: number of sentences to generate
        :param max_len: max len of samples
        :param z: (n_batch, d_z) of floats, latent vector z or None
        :param temp: temperature of softmax
        :return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            # Initial values
            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool,
                                   device=self.device)

            # Generating cycle
            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1)
                if multinomial:
                    w = torch.multinomial(y, 1)[:, 0]
                else:
                    w = torch.argmax(y,1)


                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                # new pytorch error with bool vs byte scaler 
                # try to convert it to byte tensor
                test_condition = torch.zeros((w.shape)).bool().to(self.device)
                test_condition = test_condition | (w==self.eos)
                #i_eos_mask = ~eos_mask & test_condition

                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]
