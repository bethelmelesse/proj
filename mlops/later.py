def initialize_scheduler(
    optimizer: optim.Optimizer, scheduler_args: dict
) -> optim.lr_scheduler.LambdaLR:
    print_header(text="Initializing scheduler")
    # Linear warmup, then constant. Avoids the well-known early-training
    # instability transformers show without warmup. After `warmup_steps`,
    # the lr stays at the base `optim_args["lr"]`.
    warmup_steps = scheduler_args["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / warmup_steps)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"Scheduler: LambdaLR(linear warmup, warmup_steps={warmup_steps})")
    return scheduler


scheduler_args = {"warmup_steps": 4000}
