from bibr.config import LlmOptions


def test_batch_defaults():
    opts = LlmOptions()
    assert opts.batch_model == "claude-haiku-4-5"
    assert opts.batch_poll_interval_s == 60
