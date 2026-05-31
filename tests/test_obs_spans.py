def test_init_obs_is_callable():
    from contextiq.obs import init_obs

    init_obs(enabled=False)  # no-op must not raise
