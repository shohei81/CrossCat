import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp

from genjax import ChoiceMapBuilder as C
from genjax import Target, gen, normal, pretty, smc

key = jax.random.key(0)
pretty()

@gen
def model():
    # Initially, the prior is a pretty broad normal distribution centred at 0
    x = normal(0.0, 100.0) @ "x"
    # We add some observations, which will shift the posterior towards these values
    _ = normal(x, 1.0) @ "obs1"
    _ = normal(x, 1.0) @ "obs2"
    _ = normal(x, 1.0) @ "obs3"
    return x


# We create some data, 3 observed values at 234
obs = C["obs1"].set(234.0) ^ C["obs2"].set(234.0) ^ C["obs3"].set(234.0)

key, *sub_keys = jax.random.split(key, 1000 + 1)
sub_keys = jnp.array(sub_keys)
args = ()
jitted = jit(vmap(model.importance, in_axes=(0, None, None)))
trace, weight = jitted(sub_keys, obs, args)
print("The average weight is", logsumexp(weight) - jnp.log(len(weight)))
print("The maximum weight is", weight.max())

@gen
def proposal(obs):
    avg_val = jnp.array(obs).mean()
    std = jnp.array(obs).std()
    x = (
        normal(avg_val, 0.1 + std) @ "x"
    )  # To avoid a degenerate proposal, we add a small value to the standard deviation
    return x

def importance_sample(target, obs, proposal):
    def _inner(key, target_args, proposal_args):
        trace = proposal.simulate(key, *proposal_args)
        # the full choice map under which we evaluate the model
        # has the sampled values from the proposal and the observed values
        chm = obs ^ trace.get_choices()
        proposal_logpdf = trace.get_score()
        target_logpdf, _ = target.assess(chm, *target_args)
        importance_weight = target_logpdf - proposal_logpdf
        return (trace, importance_weight)

    return _inner

key, *sub_keys = jax.random.split(key, 1000 + 1)
sub_keys = jnp.array(sub_keys)
args_for_model = ()
args_for_proposal = (jnp.array([obs["obs1"], obs["obs2"], obs["obs3"]]),)
jitted = jit(vmap(importance_sample(model, obs, proposal), in_axes=(0, None, None)))
trace, new_weight = jitted(sub_keys, (args_for_model,), (args_for_proposal,))

print("The new average weight is", logsumexp(new_weight) - jnp.log(len(new_weight)))
print("The new maximum weight is", new_weight.max())

target_posterior = Target(model, args_for_model, obs)

