__all__ = ["ActorRolloutWorker", "JEPOLewmRewardWorker", "TokenizerBridge"]


def __getattr__(name: str):
    if name == "ActorRolloutWorker":
        from .actor_rollout_worker import ActorRolloutWorker

        return ActorRolloutWorker
    if name == "TokenizerBridge":
        from .tokenizer_bridge import TokenizerBridge

        return TokenizerBridge
    if name == "JEPOLewmRewardWorker":
        from .lewm_reward_worker import JEPOLewmRewardWorker

        return JEPOLewmRewardWorker
    raise AttributeError(name)
