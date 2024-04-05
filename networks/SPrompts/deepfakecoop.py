









class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DAPL.N_CTX

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        domainnames = cfg.DATASET.SOURCE_DOMAINS + cfg.DATASET.TARGET_DOMAINS
        domainnames = [", a {} image.".format(domain) for domain in domainnames]
        n_dm = len(cfg.DATASET.SOURCE_DOMAINS) + len(cfg.DATASET.TARGET_DOMAINS)  # number of domains
        n_dmx = cfg.TRAINER.DAPL.N_DMX  # number of domain context
        n = n_dmx + n_ctx
        self.n_dm = n_dm
        self.n_dmx = n_dmx

        naive_prompt_prefix = "a photo of"

        if cfg.TRAINER.DAPL.CSC:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        print("ctx vectors size: ".format(ctx_vectors.size()))
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        prompt_prefix = " ".join(["X"] * n)

        domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
        nn.init.normal_(domain_vectors, std=0.02)
        self.domain_vectors = nn.Parameter(domain_vectors)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts = [prompt_prefix + " " + name + " " + domain + "." for domain in domainnames for name in classnames]
        naive_prompts = [naive_prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        naive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in naive_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            naive_embedding = clip_model.token_embedding(naive_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        tokenized_prompts = torch.cat([tokenized_prompts, naive_tokenized_prompts])
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = cfg.TRAINER.DAPL.CSC
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.naive_embedding = naive_embedding.to(torch.device("cuda"))

    @autocast()
    def forward(self):
        ctx = self.ctx
        ctx_dim = ctx.size(-1)
        dmx = self.domain_vectors  # dm 16 512
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)  # dm 16 512
            if not self.csc:
                ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
        else:
            ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1, -1)  # dm cls 16 512

        dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
        ctxdmx = torch.cat([ctx, dmx], dim=2).reshape(self.n_cls * self.n_dm,self.n_ctx + self.n_dmx, ctx_dim)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # naive
        neb = self.naive_embedding

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctxdmx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        prompts = torch.cat([prompts, neb], dim=0)

        return prompts
