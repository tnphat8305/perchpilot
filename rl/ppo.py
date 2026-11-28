"""Minimal on-GPU PPO (rsl_rl style) for the pilot. No external RL deps."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10, 10)


def mlp(i, o, h=(256, 256, 128)):
    layers, d = [], i
    for k in h:
        layers += [nn.Linear(d, k), nn.ELU()]
        d = k
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Actor sees `obs`; critic sees `obs` + optional privileged vector (asymmetric critic)."""

    def __init__(self, obs_dim, act_dim, init_std=0.5, priv_dim=0):
        super().__init__()
        self.priv_dim = priv_dim
        self.norm = RunningNorm(obs_dim)
        self.pnorm = RunningNorm(max(priv_dim, 1))
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim + priv_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_std)).log())

    def dist(self, obs):
        mu = self.actor(self.norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs, priv=None):
        x = self.norm(obs)
        if self.priv_dim:
            x = torch.cat([x, self.pnorm(priv)], -1)
        return self.critic(x).squeeze(-1)

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor(self.norm(obs))


def load_expanded(module, sd):
    """Warm-start from a checkpoint whose observation / privileged vectors were SHORTER.

    Layout assumption (asserted by shapes): new obs = [old obs | extra obs], new critic input =
    [obs_new | priv_new] with priv_new = [old priv | extra priv]. Actor: old obs columns copied, extra
    columns zero. Critic first layer: obs block and priv block are copied to their OWN positions
    so expanding the observation block preserves the privileged input columns. New columns are zero.
    RunningNorm stats: prefix copy. Returns the list of resized keys."""
    own = module.state_dict()
    obs_old = sd["norm.mean"].numel()
    obs_new = own["norm.mean"].numel()
    assert obs_new >= obs_old, "checkpoint has a wider observation than the model"
    priv_old = sd["critic.0.weight"].shape[1] - obs_old
    priv_new = own["critic.0.weight"].shape[1] - obs_new
    assert priv_new >= priv_old, "checkpoint critic has more privileged inputs than the model"
    resized = []
    for k, v in sd.items():
        if k not in own:
            continue
        t = own[k]
        if t.shape == v.shape:
            t.copy_(v)
            continue
        resized.append(k)
        if k == "actor.0.weight":
            t.zero_()
            t[:, :obs_old] = v
        elif k == "critic.0.weight":
            t.zero_()
            t[:, :obs_old] = v[:, :obs_old]
            t[:, obs_new:obs_new + priv_old] = v[:, obs_old:]
        else:                                               # norm / pnorm buffers: prefix
            sl = tuple(slice(0, min(a, b)) for a, b in zip(t.shape, v.shape))
            t[sl] = v[sl]
    return resized


class PPO:
    def __init__(self, env, horizon=24, epochs=5, minibatches=4, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, ent=0.01, vcoef=1.0, kl_target=0.01, device="cuda", asym=False,
                 lr_max=1e-3, std_max=None, std_min=0.02):
        self.env, self.dev = env, device
        self.priv_dim = env.priv_dim if asym else 0
        self.ac = ActorCritic(env.obs_dim, env.act_dim, priv_dim=self.priv_dim).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.lr, self.H, self.E, self.M = lr, horizon, epochs, minibatches
        self.gamma, self.lam, self.clip, self.ent, self.vcoef, self.kl_t = gamma, lam, clip, ent, vcoef, kl_target
        self.lr_max, self.std_max, self.std_min = lr_max, std_max, std_min
        self.actor_frozen = False                # critic warm-up after a warm start
        N, D, A = env.N, env.obs_dim, env.act_dim
        z = lambda *s: torch.zeros(*s, device=device)
        self.b_obs, self.b_act, self.b_logp = z(horizon, N, D), z(horizon, N, A), z(horizon, N)
        self.b_rew, self.b_done, self.b_val = z(horizon, N), z(horizon, N), z(horizon, N)
        self.b_priv = z(horizon, N, max(self.priv_dim, 1))
        self.obs = env.observe()
        self.priv = env.privileged() if self.priv_dim else None

    def set_lr(self, lr):
        self.lr = lr
        for g in self.opt.param_groups:
            g["lr"] = lr

    def collect(self):
        ep = {k: 0.0 for k in ("success", "crash", "timeout", "held", "c_plate", "c_ground", "c_behind", "c_vehicle", "c_flip")}
        for t in range(self.H):
            with torch.no_grad():
                if self.priv_dim:
                    self.b_priv[t] = self.priv
                d = self.ac.dist(self.obs)
                a = d.sample()
                self.b_obs[t], self.b_act[t], self.b_logp[t] = self.obs, a, d.log_prob(a).sum(-1)
                self.b_val[t] = self.ac.value(self.obs, self.priv)
            self.obs, r, done, info = self.env.step(a)
            if self.priv_dim:
                self.priv = self.env.privileged()
            self.b_rew[t], self.b_done[t] = r, done.float()
            for k in ep:
                ep[k] += info[k].float().sum().item() if k in info else 0.0
        return ep

    def update(self):
        with torch.no_grad():
            last_v = self.ac.value(self.obs, self.priv)
            adv = torch.zeros_like(self.b_rew)
            gae = torch.zeros_like(last_v)
            for t in reversed(range(self.H)):
                nv = last_v if t == self.H - 1 else self.b_val[t + 1]
                nd = 1.0 - self.b_done[t]
                delta = self.b_rew[t] + self.gamma * nv * nd - self.b_val[t]
                gae = delta + self.gamma * self.lam * nd * gae
                adv[t] = gae
            ret = adv + self.b_val
        B = self.H * self.env.N
        obs, act, logp = self.b_obs.view(B, -1), self.b_act.view(B, -1), self.b_logp.view(B)
        priv = self.b_priv.view(B, -1) if self.priv_dim else None
        adv, ret = adv.view(B), ret.view(B)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb = B // self.M
        stats = {"kl": 0.0, "vloss": 0.0}
        for _ in range(self.E):
            perm = torch.randperm(B, device=self.dev)
            for i in range(self.M):
                j = perm[i * mb:(i + 1) * mb]
                d = self.ac.dist(obs[j])
                nlp = d.log_prob(act[j]).sum(-1)
                ratio = (nlp - logp[j]).exp()
                pl = -torch.min(ratio * adv[j], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[j]).mean()
                vl = (self.ac.value(obs[j], priv[j] if priv is not None else None) - ret[j]).pow(2).mean()
                if self.actor_frozen:
                    loss = self.vcoef * vl
                else:
                    loss = pl + self.vcoef * vl - self.ent * d.entropy().sum(-1).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.opt.step()
                with torch.no_grad():
                    if self.std_max is not None:
                        self.ac.log_std.clamp_(math.log(self.std_min), math.log(self.std_max))
                    kl = (logp[j] - nlp).mean().item()
                # adaptive lr
                if self.actor_frozen:
                    pass
                elif kl > 2 * self.kl_t:
                    self.lr = max(1e-5, self.lr / 1.5)
                elif kl < self.kl_t / 2:
                    self.lr = min(self.lr_max, self.lr * 1.5)
                for g in self.opt.param_groups:
                    g["lr"] = self.lr
                stats["kl"] = kl
                stats["vloss"] = vl.item()
        # normalizer statistics are frozen during a rollout and its update, so old/new log-probs use the
        # same normalisation (review #6); fold the rollout into the statistics only now
        with torch.no_grad():
            self.ac.norm.update(obs)
            if priv is not None:
                self.ac.pnorm.update(priv)
        return stats
