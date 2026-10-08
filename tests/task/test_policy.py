from src.task.policy import TaskPolicy, is_retryable_error

def test_policy_and_retryability():
    assert TaskPolicy(max_retries=2).max_retries == 2
    assert is_retryable_error(TimeoutError())

