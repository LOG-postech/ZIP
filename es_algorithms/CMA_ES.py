import cma
class shallow_cma:
    def __init__(self, cfg):
        self.opt_setting = {
            'seed': cfg.SEED,
            'popsize': cfg.TRAINER.BPTVLM.POP_SIZE,
            'maxiter': cfg.TRAINER.BPTVLM.BUDGET / cfg.TRAINER.BPTVLM.POP_SIZE,
            'verbose': -1,
        }
        if cfg.TRAINER.BPTVLM.BOUND > 0:
            self.opt_setting['bounds'] = [-1 * cfg.TRAINER.BPTVLM.BOUND, 1 * cfg.TRAINER.BPTVLM.BOUND]
        self.intrinsic_dim_L = cfg.TRAINER.BPTVLM.INTRINSIC_DIM_L
        self.intrinsic_dim_V = cfg.TRAINER.BPTVLM.INTRINSIC_DIM_V
        self.sigma = cfg.TRAINER.BPTVLM.SIGMA
        self.es = cma.CMAEvolutionStrategy((self.intrinsic_dim_L + self.intrinsic_dim_V) * [0], self.sigma , inopts=self.opt_setting)

    def ask(self):
        return self.es.ask()

    def tell(self, solutions, fitnesses):
        return self.es.tell(solutions, fitnesses)

    def stop(self):
        return self.es.stop()