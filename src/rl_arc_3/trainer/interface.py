class TrainingArguments:
    pass


class TrainerInterface:
    def train(
        self,
        resume_from_checkpoint: dict | None = None,
    ):
        raise NotImplementedError

    def eval(
        self,
    ):
        raise NotImplementedError
