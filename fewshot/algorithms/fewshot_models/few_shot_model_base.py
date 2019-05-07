class FewShotModelBase:
    def fit(self, episode_generator, **kwargs):
        raise NotImplementedError

    def predict(self, episode_generator):
        raise NotImplementedError
